#!/bin/bash
# K3 Attention Benchmark — SPECULATIVE server (DSpark draft).
# Identical to _serve_k3_bench_baseline.sh EXCEPT it adds --speculative-config
# with the DSpark draft. Parameterized by NUM_SPEC (2 or 7 per the doc).
#
#   NUM_SPEC=2 PORT=8889 bash _serve_k3_bench_spec.sh
#
# Spec config follows K3_Attention_Benchmark_Instructions.md:
#   method=dspark, attention_backend=TRITON_MLA,
#   draft_sample_method=probabilistic, rejection_sample_method=block.
set -uo pipefail
cd /workspace
PORT="${PORT:-8889}"
NUM_SPEC="${NUM_SPEC:-2}"
GPU_MEM="${GPU_MEM:-0.88}"; MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"; MNBT="${MNBT:-4096}"
# Spec-decode TARGET backend.
#
# NOTE (fp8+DSpark is architecturally unrunnable on this nightly):
#   The DSpark draft uses semi-autoregressive parallel drafting, which forces
#   non-causal attention (dspark/utils.py: use_non_causal=dflash_has_any_non_causal).
#   On ROCm, ONLY TRITON_MLA advertises supports_non_causal=True; ROCM_AITER_MLA
#   does not. But TRITON_MLA sets supports_quant_query_input=False, and K3's
#   _decode_concat_cache asserts that flag whenever KV is fp8. So:
#     - draft on ROCM_AITER_MLA -> "non-causal attention not supported" (build fails)
#     - draft on TRITON_MLA + fp8 KV -> K3 fp8-query assert (after capture)
#   => DSpark REQUIRES bf16 KV + TRITON_MLA. (PR #51011 correctly fixes the TARGET's
#      fp8 verify routing, but cannot bridge the draft's non-causal requirement.)
#      The doc's reference KV was default (bf16) anyway; fp8 was our own delta.
# With the draft forced causal (dflash_config.causal=true in the draft config),
# use_non_causal=False, so the draft no longer needs TRITON_MLA and can run on the
# fp8 asm path. Target + draft both on ROCM_AITER_MLA + fp8 KV (the MI355X perf
# path). PR #51011 makes the 12-head fp8 spec verify route to the asm q-row-fold.
ATTN_BACKEND="${ATTN_BACKEND:-ROCM_AITER_MLA}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
# Real perf requires cudagraphs — NO eager. PR #51011 also fixes the persistent
# metadata gate so PIECEWISE/FULL capture completes under spec (production run
# captured 35 piecewise + 19 full graphs). Default ENFORCE_EAGER=0.
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
EAGER_ARG=(); [ "$ENFORCE_EAGER" = "1" ] && EAGER_ARG=(--enforce-eager)

export HF_HUB_CACHE=/dev/shm/hf-cache HF_HOME=/dev/shm/hf-cache
CACHE=/dev/shm/hf-cache/models--moonshotai--Kimi-K3/snapshots
MODEL_PATH="${MODEL_PATH:-$(ls -d "$CACHE"/*/ 2>/dev/null | head -1)}"; MODEL_PATH="${MODEL_PATH%/}"
[ -n "$MODEL_PATH" ] && [ -f "$MODEL_PATH/config.json" ] || { echo "!! K3 weights not staged at $CACHE"; exit 1; }

# resolve local DSpark draft snapshot (offline/portable)
DCACHE=/dev/shm/hf-cache/models--Inferact--Kimi-K3-DSpark/snapshots
DRAFT_PATH="${DRAFT_PATH:-$(ls -d "$DCACHE"/*/ 2>/dev/null | head -1)}"; DRAFT_PATH="${DRAFT_PATH%/}"
if [ -z "$DRAFT_PATH" ] || [ ! -f "$DRAFT_PATH/config.json" ]; then
  echo "!! DSpark draft not staged at $DCACHE — falling back to HF id Inferact/Kimi-K3-DSpark"
  DRAFT_PATH="Inferact/Kimi-K3-DSpark"
else
  # Refuse to start if the draft is still non-causal (cudagraph OOB source).
  if ! python3 - "$DRAFT_PATH/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
if c.get("dflash_config", {}).get("causal") is not True:
    raise SystemExit("dflash_config.causal is not true — run _k3_dspark_fp8asm_apply_patches.sh")
print("draft causal OK")
PY
  then
    echo "!! draft must be forced causal before serve — run _k3_dspark_fp8asm_apply_patches.sh"
    exit 1
  fi
fi
echo "MODEL_PATH=$MODEL_PATH"
echo "DRAFT_PATH=$DRAFT_PATH  NUM_SPEC=$NUM_SPEC  PORT=$PORT"

# DRAFT attention backend. With the draft forced causal it runs on the fp8 asm
# path like the target. ROCM_AITER_MLA (post PR #51011) accepts fp8 query and
# routes small-head verify to the asm fold.
DRAFT_BACKEND="${DRAFT_BACKEND:-ROCM_AITER_MLA}"
SPEC_CFG=$(printf '{"model":"%s","num_speculative_tokens":%s,"method":"dspark","attention_backend":"%s","draft_sample_method":"probabilistic","rejection_sample_method":"block"}' "$DRAFT_PATH" "$NUM_SPEC" "$DRAFT_BACKEND")

# cudagraph mode + optional capture-size cap. fp8-asm DSpark MUST use
# MAX_NUM_SEQS=16 (recipe default above) — not the agentic 2*CONC ladder.
# With NUM_SPEC=2, vLLM derives max_cudagraph_capture_size = 16*(1+N)*2 = 96,
# so PIECEWISE capture still walks M=96..72..8; the fp8 asm verify kernel OOBs
# when max_num_seqs is large (M~304 at 64 seqs). MAX_CG overrides the ceiling
# for experiments (batches above the cap run outside the graph, not eager).
# KV-cache memory pin (bytes). At gpu_memory_utilization=0.95 with TP8 K3-fp4,
# weights alone are ~201 GiB of the 273.6 GiB budget. vLLM's auto-sizer hands the
# leftover to KV using only the *profiled* peak-activation estimate (~5.5 GiB) as
# headroom. That estimate is far below the REAL prefill peak: MNBT=16384 chunks x
# up-to-64 concurrent 68k-token requests + DSpark verify buffers spike much
# higher, so the process OOMs to "0 MB free" — at init if KV is sized to the edge,
# or mid-benchmark once real traffic hits the peak (VllmWorker died at conc-48).
# The fix is NOT to touch the mandated knobs (gpu_mem 0.95 / seqs 64 / MNBT /
# FULL_AND_PIECEWISE) but to shrink KV so a large physical headroom remains for
# those runtime spikes. Prefix caching stores the 63,911-tok prefix ONCE, so this
# workload needs only ~350k KV tokens (~6 GiB); 32 GiB (~2M tokens) is >5x that
# while leaving ~32 GiB physical headroom for the prefill/verify activation peak.
KV_CACHE_MEMORY="${KV_CACHE_MEMORY:-34359738368}"
KVMEM_ARG=(); [ -n "$KV_CACHE_MEMORY" ] && KVMEM_ARG=(--kv-cache-memory "$KV_CACHE_MEMORY")

CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}"
MAX_CG="${MAX_CG:-}"
# cudagraph_capture_sizes — pin explicitly so DSpark decode (M = 3*conc tokens/step,
# uniform_decode_query_len = 1+num_spec = 3) lands on FULL decode graphs at every
# benchmark concurrency. vLLM derives the FULL/decode graph set as round_up(size,3)
# over this ladder; the AUTO ladder [1,2,4,8,16,24,32,40,...] leaves gaps at 12 and
# 36 (8->9,16->18 skip 12; 32->33,40->42 skip 36), so conc-4 (12 tok) and conc-12
# (36 tok) silently fall to a PIECEWISE graph -> attention runs eager every step ->
# get_mla_metadata_v1 + eager MLA on the host critical path (~75 ms ITL bubble; GPU
# at ~370 W launch-bound). Adding 12 and 36 gives exact FULL decode graphs (num_reqs
# 4 and 12), where get_mla_metadata_v1 takes the fast uniform branch. Rule to extend:
# for a new concurrency C, ensure a size s with round_up(s,3)==3*C exists (add 3*C).
# The rest of the list reproduces vLLM's auto ladder (setting this disables auto-gen).
CAPTURE_SIZES="${CAPTURE_SIZES:-1,2,4,8,12,16,24,32,36,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,272,288,304,320,336,352,368,384}"
if [ -n "$MAX_CG" ]; then
  COMPILE_CFG=$(printf '{"cudagraph_mode":"%s","custom_ops":["+fused_rms_norm_gated"],"cudagraph_capture_sizes":[%s],"max_cudagraph_capture_size":%s}' "$CUDAGRAPH_MODE" "$CAPTURE_SIZES" "$MAX_CG")
else
  COMPILE_CFG=$(printf '{"cudagraph_mode":"%s","custom_ops":["+fused_rms_norm_gated"],"cudagraph_capture_sizes":[%s]}' "$CUDAGRAPH_MODE" "$CAPTURE_SIZES")
fi

MERGED_GEMM_CSV=/opt/aiter-local/aiter/configs/merged_bf16_tuned_gemm.csv
if [ -z "${AITER_CONFIG_GEMM_BF16:-}" ] && [ -f "$MERGED_GEMM_CSV" ]; then
  export AITER_CONFIG_GEMM_BF16="$MERGED_GEMM_CSV"
fi

export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 SAFETENSORS_FAST_GPU=1 \
       GPU_ARCHS=gfx950 VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900 \
       AITER_SITUV2_A8W4=1 VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1 \
       VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
       VLLM_ROCM_AITER_MLA_ASM_PADDING="${VLLM_ROCM_AITER_MLA_ASM_PADDING:-asm}"

# Optional torch profiler. Set PROFILE_DIR=/path to enable; then drive it with
# curl -X POST /start_profile ... load ... /stop_profile (per-rank .pt.trace.json.gz
# flush to PROFILE_DIR). This build honors --profiler-config, NOT the
# VLLM_TORCH_PROFILER_DIR env. Analyze with analyze_dsv4_trace.py / backend_breakdown.py.
PROFILE_DIR="${PROFILE_DIR:-}"
PROFILE_ARG=(); [ -n "$PROFILE_DIR" ] && { mkdir -p "$PROFILE_DIR"; PROFILE_ARG=(--profiler-config.profiler=torch --profiler-config.torch_profiler_dir="$PROFILE_DIR"); }

LOG=/workspace/serve_k3_bench_spec${NUM_SPEC}.log
setsid nohup vllm serve "$MODEL_PATH" --served-model-name Kimi-K3 \
  --host 0.0.0.0 --port "$PORT" --tensor-parallel-size 8 --async-scheduling \
  --distributed-executor-backend mp --gpu-memory-utilization "$GPU_MEM" \
  --max-num-seqs "$MAX_NUM_SEQS" --max-model-len 1048576 --max-num-batched-tokens "$MNBT" \
  --trust-remote-code --load-format auto --moe-backend aiter \
  --kv-cache-dtype "$KV_CACHE_DTYPE" --attention-backend "$ATTN_BACKEND" --mm-encoder-tp-mode data \
  "${KVMEM_ARG[@]}" \
  --compilation-config "$COMPILE_CFG" \
  "${PROFILE_ARG[@]}" \
  "${EAGER_ARG[@]}" \
  --speculative-config "$SPEC_CFG" \
  --enable-prefix-caching --enable-prompt-tokens-details --no-disable-hybrid-kv-cache-manager \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 --enable-auto-tool-choice \
  --disable-uvicorn-access-log > "$LOG" 2>&1 &

echo "serving K3 spec-$NUM_SPEC (port=$PORT); log: $LOG"
for i in $(seq 1 360); do
  curl -sf -m5 "http://localhost:$PORT/health" >/dev/null 2>&1 && { echo "ready $(date +%T)"; exit 0; }
  pgrep -f "vllm serve" >/dev/null 2>&1 || { echo "serve died; tail:"; tail -50 "$LOG"; exit 1; }
  sleep 5
done
echo "!! not ready after 30min"; tail -50 "$LOG"; exit 1
