#!/bin/bash
# Profiled FIXED isl/osl run to break down the prefill/TTFT bottleneck at a real
# operating concurrency (conc16/24). Deterministic shape (stddev 0).
#   Serve  = recipe config (fused ON, ms64, mnbt 4096, ROCM_AITER_MLA, kv fp8,
#            multimodal — same as the shipped recipe) + VLLM_TORCH_PROFILER_DIR.
#            gpu-mem 0.90 for headroom (prefill run needs little KV).
#   Load   = aiperf on /v1/chat/completions — PROVEN on this serve today
#            (prefixbench + the whole 1-24 sweep use it). benchmark_serving.py on
#            /v1/completions 100%-rejects K3 on this build (see diagnostic below).
#   Profile= manual curl POST /start_profile ... /stop_profile around the run.
#   Analyze= analyze_k3_trace.py <rank0 .pt.trace.json.gz>.
# Params (env): CONC (16), ISL (16384), OSL (4), REQ_COUNT (3*CONC), WARMUP (CONC), GPU_MEM (0.90).
set -uo pipefail
cd /workspace
CONC="${CONC:-16}"; ISL="${ISL:-16384}"; OSL="${OSL:-4}"
REQ_COUNT="${REQ_COUNT:-$((CONC*3))}"; WARMUP="${WARMUP:-$CONC}"; GPU_MEM="${GPU_MEM:-0.90}"
SHM_MODEL=/dev/shm/hf-cache/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569
MODEL_PATH="${MODEL_PATH:-$SHM_MODEL}"
[ -f "$MODEL_PATH/config.json" ] || { echo "!! model missing at $MODEL_PATH"; exit 1; }

# aiperf: honor $AIPERF if valid, else auto-detect (prefer v1.0.1 pin).
if [ -n "${AIPERF:-}" ] && [ -x "$AIPERF" ]; then :
elif command -v aiperf >/dev/null 2>&1; then AIPERF="$(command -v aiperf)"
else
  AIPERF=""
  for c in /opt/.aiperf_b7b16cf8/bin/aiperf /workspace/.aiperf_v1_0_1/bin/aiperf \
           /opt/.aiperf_*/bin/aiperf /workspace/.aiperf_*/bin/aiperf; do
    [ -x "$c" ] && { AIPERF="$c"; break; }
  done
  [ -n "$AIPERF" ] || { echo "!! no aiperf found"; exit 1; }
fi
echo "using aiperf: $AIPERF"

TRACE_DIR=/workspace/kimik3_traces_c${CONC}_${ISL}x${OSL}
OUT_DIR=/workspace/k3_profile_c${CONC}_${ISL}x${OSL}
rm -rf "$TRACE_DIR" "$OUT_DIR"; mkdir -p "$TRACE_DIR" "$OUT_DIR"

if pgrep -f "vllm serve" >/dev/null 2>&1 || curl -sf -m5 http://localhost:8888/health >/dev/null 2>&1; then
  echo "!! a serve is already up; aborting"; exit 1
fi
echo "########## profile c${CONC} isl${ISL} osl${OSL} start $(date +%T) ##########"
export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 SAFETENSORS_FAST_GPU=1 \
       AITER_SITUV2_A8W4=1 AITER_BF16_FP8_MOE_BOUND=0 VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
       GPU_ARCHS=gfx950 VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900 \
       HF_HUB_CACHE=/dev/shm/hf-cache HF_HOME=/dev/shm/hf-cache
# NB: this vLLM build enables the profiler via --profiler-config (a serve flag),
# NOT the VLLM_TORCH_PROFILER_DIR env (that env is "unknown"/ignored here).
setsid nohup vllm serve "$MODEL_PATH" --served-model-name moonshotai/Kimi-K3 \
  --host 0.0.0.0 --port 8888 --tensor-parallel-size 8 --async-scheduling \
  --profiler-config.profiler=torch --profiler-config.torch_profiler_dir="$TRACE_DIR" \
  --distributed-executor-backend mp --gpu-memory-utilization "$GPU_MEM" \
  --max-num-seqs 64 --max-model-len 1048576 --max-num-batched-tokens 4096 \
  --trust-remote-code --load-format auto --moe-backend auto \
  --kv-cache-dtype fp8 --attention-backend ROCM_AITER_MLA --mm-encoder-tp-mode data \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["+fused_rms_norm_gated"]}' \
  --enable-prefix-caching --no-disable-hybrid-kv-cache-manager \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 --enable-auto-tool-choice \
  --disable-uvicorn-access-log > /workspace/serve_profile_c${CONC}.log 2>&1 &
for i in $(seq 1 240); do
  curl -sf -m5 http://localhost:8888/health >/dev/null 2>&1 && break
  pgrep -f "vllm serve" >/dev/null 2>&1 || { echo "serve died during load"; bash /workspace/_freegpu.sh; exit 1; }
  sleep 5
done
curl -sf -m5 http://localhost:8888/health >/dev/null 2>&1 || { echo "!! serve not ready; freeing"; bash /workspace/_freegpu.sh; exit 1; }
echo "serve ready $(date +%T)"

# --- diagnostic: document why benchmark_serving.py (/v1/completions) fails vs chat ---
echo "=== DIAG /v1/completions (benchmark_serving.py path) ==="
curl -s -m30 http://localhost:8888/v1/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"moonshotai/Kimi-K3\",\"prompt\":\"hello world\",\"max_tokens\":4}" | head -c 400; echo
echo "=== DIAG /v1/chat/completions (aiperf path) ==="
curl -s -m30 http://localhost:8888/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"moonshotai/Kimi-K3\",\"messages\":[{\"role\":\"user\",\"content\":\"hello world\"}],\"max_tokens\":4}" | head -c 400; echo

# --- profiled fixed-isl load via aiperf (chat) + manual torch-profiler trigger ---
echo "start profiler $(date +%T)"; curl -sf -m10 -X POST http://localhost:8888/start_profile && echo " ok" || echo " start_profile FAILED"
"$AIPERF" profile \
  --model moonshotai/Kimi-K3 --tokenizer "$MODEL_PATH" --tokenizer-trust-remote-code \
  --url http://127.0.0.1:8888 --api-key EMPTY --endpoint-type chat --streaming --use-server-token-count \
  --prompt-input-tokens-mean "$ISL" --prompt-input-tokens-stddev 0 \
  --output-tokens-mean "$OSL" --output-tokens-stddev 0 \
  --extra-inputs ignore_eos:true --extra-inputs "min_tokens:$OSL" --extra-inputs "max_tokens:$OSL" \
  --concurrency "$CONC" --request-count "$REQ_COUNT" --warmup-request-count "$WARMUP" \
  --random-seed 42 --no-gpu-telemetry --output-artifact-dir "$OUT_DIR" \
  > /workspace/profile_c${CONC}_bench.log 2>&1
rc=$?
echo "stop profiler $(date +%T)"; curl -sf -m600 -X POST http://localhost:8888/stop_profile && echo " ok" || echo " stop_profile returned/timeout (workers may still be flushing)"
# CRITICAL: the 8 worker traces (GPU kernels) flush on stop and can take minutes.
# Poll for them BEFORE killing the serve — otherwise _freegpu truncates them.
echo "waiting for per-rank worker traces to flush..."
for i in $(seq 1 90); do
  n=$(ls "$TRACE_DIR"/*rank*.pt.trace.json.gz 2>/dev/null | wc -l)
  echo "  t=$((i*10))s worker-traces=$n/8"
  [ "$n" -ge 8 ] && { echo "  all 8 worker traces written"; sleep 5; break; }
  sleep 10
done
echo "bench rc=$rc; traces -> $TRACE_DIR ; freeing $(date +%T)"
bash /workspace/_freegpu.sh
echo "########## profile c${CONC} COMPLETE (rc=$rc) $(date +%T) ##########"
