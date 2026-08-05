#!/bin/bash
# Prefixbench adapted with our LATEST update: +fused_rms_norm_gated (default ON).
# Serve is otherwise IDENTICAL to the 9K/cold-baseline serve (_run_prefixbench.sh):
#   moe aiter, ms24, max-num-batched-tokens 4096, gpu 0.95, kv fp8,
#   ROCM_AITER_MLA, hybrid KV.  Client = aiperf v1.0.1, conc 16,24, seed 42.
# Isolates fused's effect vs k3_prefixbench_fp8asm_v101_gm095 (fused OFF).
# CHAINS: waits for the in-flight 1-24 agentic sweep to release :8888 + GPUs.
set -uo pipefail
cd /workspace
SHM_MODEL=/dev/shm/hf-cache/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569
NFS_MODEL="${MODEL_SRC:-/shared_nfs/models/Kimi-K3}"
if [ -f "${MODEL_PATH:-}/config.json" ]; then
  :
elif [ -f "$SHM_MODEL/config.json" ] && [ -f "$SHM_MODEL/preprocessor_config.json" ]; then
  MODEL_PATH="$SHM_MODEL"
elif [ -f "$NFS_MODEL/config.json" ]; then
  MODEL_PATH="$NFS_MODEL"
else
  echo "!! Kimi-K3 model missing (checked $SHM_MODEL and $NFS_MODEL)"; exit 1
fi
AIPERF="${AIPERF:-/opt/.aiperf_b7b16cf8/bin/aiperf}"
[ -x "$AIPERF" ] || { echo "!! aiperf missing at $AIPERF"; exit 1; }
OUT="${OUT:-/workspace/k3_prefixbench_fp8asm_v101_fused}"

free_gpu() {
  pkill -9 -f "[a]iperf" 2>/dev/null || true
  pkill -9 -f "[v]llm serve" 2>/dev/null || true
  pkill -9 -f "[V]LLM::EngineCore" 2>/dev/null || true
}

echo "########## prefixbench-FUSED start $(date '+%T %F') ##########"
echo "waiting for GPU to free (in-flight 1-24 sweep)... $(date +%T)"
for i in $(seq 1 2160); do   # up to ~3h
  if ! pgrep -f "vllm serve" >/dev/null 2>&1 && ! curl -sf -m5 http://localhost:8888/health >/dev/null 2>&1; then
    echo "GPU free at $(date +%T)"; break
  fi
  sleep 5
done
if pgrep -f "vllm serve" >/dev/null 2>&1 || curl -sf -m5 http://localhost:8888/health >/dev/null 2>&1; then
  echo "!! serve still up after wait window; aborting to avoid collision"; exit 1
fi

export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 SAFETENSORS_FAST_GPU=1 \
       AITER_SITUV2_A8W4=1 AITER_BF16_FP8_MOE_BOUND=0 VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
       GPU_ARCHS=gfx950 VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900 \
       HF_HUB_CACHE=/dev/shm/hf-cache HF_HOME=/dev/shm/hf-cache
setsid nohup vllm serve "$MODEL_PATH" --served-model-name moonshotai/Kimi-K3 \
  --host 0.0.0.0 --port 8888 --tensor-parallel-size 8 --async-scheduling \
  --distributed-executor-backend mp --gpu-memory-utilization 0.95 \
  --max-num-seqs 24 --max-num-batched-tokens 4096 \
  --trust-remote-code --load-format auto --moe-backend aiter \
  --kv-cache-dtype fp8 --attention-backend ROCM_AITER_MLA \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["+fused_rms_norm_gated"]}' \
  --enable-prefix-caching --no-disable-hybrid-kv-cache-manager \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 --enable-auto-tool-choice \
  --disable-uvicorn-access-log > /workspace/serve_prefixbench_fused.log 2>&1 &

for i in $(seq 1 240); do
  curl -sf -m5 http://localhost:8888/health >/dev/null 2>&1 && break
  pgrep -f "vllm serve" >/dev/null 2>&1 || { echo "serve died during load"; break; }
  sleep 5
done
curl -sf -m5 http://localhost:8888/health >/dev/null 2>&1 || { echo "!! serve not ready; freeing"; free_gpu; exit 1; }
echo "serve ready (fused) $(date +%T)"

rm -rf "$OUT"; mkdir -p "$OUT"
"$AIPERF" profile \
  --model moonshotai/Kimi-K3 --tokenizer builtin \
  --url http://127.0.0.1:8888 --api-key EMPTY --endpoint-type chat --streaming --use-server-token-count \
  --num-prefix-prompts 8 --prompt-prefix-length 63240 \
  --prompt-input-tokens-mean 4760 --prompt-input-tokens-stddev 0 \
  --output-tokens-mean 350 --output-tokens-stddev 0 \
  --extra-inputs ignore_eos:true --extra-inputs min_tokens:350 --extra-inputs max_tokens:350 \
  --warmup-request-count 3 --sweep-type zip --concurrency 16,24 --request-count 80,120 \
  --random-seed 42 --no-gpu-telemetry --output-artifact-dir "$OUT" > /workspace/prefixbench_fused.log 2>&1
rc=$?
echo "aiperf rc=$rc freeing GPUs $(date +%T)"
free_gpu
echo "########## prefixbench-FUSED COMPLETE (rc=$rc) + GPUs freed $(date '+%T %F') ##########"
