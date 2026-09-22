#!/bin/bash
# Torch-profiler capture of the best conc-1 arm at ISL 100k, on the ROCm 10 image.
#
# Produces per-rank traces directly comparable to the two reference sets:
#   ~/work/k3_traces/prof_isl100k          MI355X, ROCm 7.2.3 (427k kernel events)
#   ~/work/b300/profile_20260911_102437    B300
# Same torch format, so a three-way kernel comparison works.
#
# Config matches repro/k3_r100_c1_best.sh (iv p90 142.7, 1558.9 tok/s/gpu):
# DCP8, K=7, synthetic AL 3.84, fp8 KV, ROCM_AITER_MLA, multistream on,
# inductor graph partition. The LOAD differs: a fixed-ISL aiperf profile rather
# than the agentic replay, because a trace wants a steady shape, not a corpus.
#
# Structure, serve/load/flush, is lifted from _run_profile_nightly_k3.sh. The
# flush is the hardened one: 8 rank traces, byte-size stable three times,
# rank0 gzip-valid, only then free. Traces land incomplete without it.
#
# PREREQS (the container must already have them):
#   - the 17-patch set applied (see kimik3_fp4_mi355x_mtp.sh); in particular
#     dspark_torch_compile + fused_qk_rmsnorm_custom_op, or the compiled draft
#     graph dies in Dynamo
#   - weights at MODEL_PATH; box free and zero zombies
#
# The profiler is enabled by --profiler-config. VLLM_TORCH_PROFILER_DIR is dead
# on this build: /start_profile returning 404 means the flag is wrong, not that
# profiling is unsupported.
set -uo pipefail

CONC="${CONC:-1}"; ISL="${ISL:-100000}"; OSL="${OSL:-256}"
REQ_COUNT="${REQ_COUNT:-8}"; WARMUP="${WARMUP:-2}"
GPU_MEM="${GPU_MEM:-0.90}"; MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"; MNBT="${MNBT:-16384}"
SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-7}"; SYNTH_AL="${SYNTH_AL:-3.84}"
PORT="${PORT:-8888}"
WORKDIR="${WORKDIR:-/workspace}"

MODEL_PATH="${MODEL_PATH:-/dev/shm/hf-cache/models--moonshotai--Kimi-K3/snapshots/f831ab66814297da540d832a5235f8e904f29d06}"
DRAFT_PATH="${DRAFT_PATH:-Inferact/Kimi-K3-DSpark}"
[ -f "$MODEL_PATH/config.json" ] || { echo "!! model missing at $MODEL_PATH"; exit 1; }

TRACE_DIR="${TRACE_DIR:-$WORKDIR/prof_r100_isl${ISL}_c${CONC}}"
OUT_DIR="${OUT_DIR:-$WORKDIR/aiperf_r100_c${CONC}_${ISL}x${OSL}}"
SERVE_LOG="$WORKDIR/serve_r100_prof_c${CONC}.log"
mkdir -p "$TRACE_DIR"

AIPERF="${AIPERF:-}"
if [ -z "$AIPERF" ]; then
  command -v aiperf >/dev/null 2>&1 && AIPERF="$(command -v aiperf)"
fi
if [ -z "$AIPERF" ]; then
  for c in "$WORKDIR"/.aiperf_*/bin/aiperf /opt/.aiperf_*/bin/aiperf; do
    [ -x "$c" ] && { AIPERF="$c"; break; }
  done
fi
[ -n "$AIPERF" ] || { echo "!! no aiperf found"; exit 1; }
echo "using aiperf: $AIPERF"

if curl -sf -m5 "http://localhost:$PORT/health" >/dev/null 2>&1; then
  echo "!! a serve is already up on $PORT; stop it first"; exit 1
fi

# Untuned GEMM. Both shipped CSVs carry solidx/kernel ids from other aiter
# builds and fault this image: hipBLASLt INVALID_VALUE at the profile run, opus
# "Kernel id 205 not found" at speculator capture.
if [ ! -s /dev/shm/_empty_tuned_gemm.csv ]; then
  head -1 "$(dirname "$0")/../../../../patches/k3-dcp8/aiter/merged_bf16_tuned_gemm_rocm10_retuned.csv" \
    > /dev/shm/_empty_tuned_gemm.csv
fi
export AITER_CONFIG_GEMM_BF16=/dev/shm/_empty_tuned_gemm.csv
export K3_AMD_GATE_MULTI_STREAM="${K3_AMD_GATE_MULTI_STREAM:-1}"

echo "serve starting $(date +%T)  ISL=$ISL OSL=$OSL CONC=$CONC K=$SPEC_NUM_TOKENS"
setsid nohup vllm serve "$MODEL_PATH" --served-model-name moonshotai/Kimi-K3 \
  --host 0.0.0.0 --port "$PORT" --tensor-parallel-size 8 \
  --decode-context-parallel-size 8 --dcp-comm-backend a2a --cp-kv-cache-interleave-size 1 \
  --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$TRACE_DIR\"}" \
  --distributed-executor-backend mp --gpu-memory-utilization "$GPU_MEM" \
  --max-num-seqs "$MAX_NUM_SEQS" --max-model-len 1048576 --max-num-batched-tokens "$MNBT" \
  --trust-remote-code --load-format auto --moe-backend aiter \
  --kv-cache-dtype fp8 --attention-backend ROCM_AITER_MLA --mm-encoder-tp-mode data \
  --prefix-cache-retention-interval 1536 \
  --compilation-config "{\"mode\":3,\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"max_cudagraph_capture_size\":16,\"custom_ops\":[\"+fused_rms_norm_gated\",\"+situ_and_mul\"],\"use_inductor_graph_partition\":true,\"pass_config\":{\"fuse_rope_kvcache_cat_mla\":true}}" \
  --speculative-config "{\"model\":\"$DRAFT_PATH\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS,\"method\":\"dspark\",\"attention_backend\":\"ROCM_AITER_MLA\",\"kv_cache_dtype\":\"fp8\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_length\":$SYNTH_AL}" \
  --enable-prefix-caching --no-disable-hybrid-kv-cache-manager \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 --enable-auto-tool-choice \
  --disable-uvicorn-access-log > "$SERVE_LOG" 2>&1 &

for i in $(seq 1 360); do
  curl -sf -m5 "http://localhost:$PORT/health" >/dev/null 2>&1 && break
  pgrep -f "[v]llm serve" >/dev/null 2>&1 || { echo "serve died; tail:"; tail -40 "$SERVE_LOG"; exit 1; }
  sleep 5
done
curl -sf -m5 "http://localhost:$PORT/health" >/dev/null 2>&1 || {
  echo "!! not ready"; tail -40 "$SERVE_LOG"; exit 1; }
echo "serve ready $(date +%T)"

echo "start profiler $(date +%T)"
curl -sf -m10 -X POST "http://localhost:$PORT/start_profile" && echo " ok" \
  || { echo " start FAILED (404 here means --profiler-config did not take)"; }

"$AIPERF" profile \
  --model moonshotai/Kimi-K3 --tokenizer "$MODEL_PATH" --tokenizer-trust-remote-code \
  --url "http://127.0.0.1:$PORT" --api-key EMPTY --endpoint-type chat --streaming --use-server-token-count \
  --prompt-input-tokens-mean "$ISL" --prompt-input-tokens-stddev 0 \
  --output-tokens-mean "$OSL" --output-tokens-stddev 0 \
  --extra-inputs ignore_eos:true --extra-inputs "min_tokens:$OSL" --extra-inputs "max_tokens:$OSL" \
  --concurrency "$CONC" --request-count "$REQ_COUNT" --warmup-request-count "$WARMUP" \
  --random-seed 42 --no-gpu-telemetry --output-artifact-dir "$OUT_DIR" \
  > "$WORKDIR/profile_r100_c${CONC}_bench.log" 2>&1
rc=$?

echo "stop profiler $(date +%T)"
curl -sf -m600 -X POST "http://localhost:$PORT/stop_profile" && echo " ok" || echo " stop timeout (flushing)"

# ---- HARDENED FLUSH: 8 rank traces, size-stable x3, rank0 gzip-valid ----
echo "waiting for 8 per-rank worker traces (max 45min)..."
prev=-1; stable=0
for i in $(seq 1 270); do
  n=$(ls "$TRACE_DIR"/*rank*.pt.trace.json.gz 2>/dev/null | wc -l)
  sz=$(du -sb "$TRACE_DIR"/*rank*.pt.trace.json.gz 2>/dev/null | awk '{s+=$1} END{printf "%.0f", s+0}')
  echo "  t=$((i*10))s ranks=$n/8 bytes=$sz"
  if [ "$n" -ge 8 ]; then
    if [ "$sz" = "$prev" ]; then stable=$((stable+1)); else stable=0; fi
    prev="$sz"
    if [ "$stable" -ge 3 ]; then
      r0=$(ls "$TRACE_DIR"/*rank0*.pt.trace.json.gz 2>/dev/null | head -1)
      if [ -n "$r0" ] && gzip -t "$r0" 2>/dev/null; then
        echo "  stable + rank0 gzip-valid -> done"; break
      else stable=0; fi
    fi
  fi
  sleep 10
done

echo "bench rc=$rc  $(date +%T)"
echo "=== aiperf latency summary ==="
grep -iE "Inter Token|Time to (First|Second)|Request Latency|Output Token Throughput" \
  "$WORKDIR/profile_r100_c${CONC}_bench.log" | head

R0=$(ls "$TRACE_DIR"/*rank0*.pt.trace.json.gz 2>/dev/null | head -1)
echo "rank0 trace: ${R0:-<none>}"
if [ -n "$R0" ]; then
  python3 - "$R0" <<'PY' 2>/dev/null || true
import gzip, json, collections, sys
d = json.load(gzip.open(sys.argv[1]))
ev = d.get("traceEvents", d if isinstance(d, list) else [])
cat = collections.Counter(e.get("cat") for e in ev)
ker = cat.get("kernel", 0)
print("  events=%d kernel=%d gpu_user_annotation=%d"
      % (len(ev), ker, cat.get("gpu_user_annotation", 0)))
if ker == 0:
    print("  !! 0 kernel events: the profiler did not capture GPU work")
else:
    dur = collections.Counter()
    for e in ev:
        if e.get("cat") == "kernel":
            dur[e.get("name", "")] += e.get("dur", 0)
    tot = sum(dur.values())
    print("  total GPU kernel time: %.1f ms" % (tot / 1000))
    for n, us in sorted(dur.items(), key=lambda x: -x[1])[:5]:
        print("    %-52s %7.1f ms  %5.2f%%" % (n[:52], us / 1000, 100 * us / tot))
PY
fi
echo "########## profile COMPLETE (rc=$rc) $(date +%T) ##########"
