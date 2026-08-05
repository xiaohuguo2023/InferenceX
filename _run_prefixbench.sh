#!/bin/bash
# Kimi-K3 fp8-ASM prefix-heavy synthetic bench (63,240-token shared prefix +
# 4,760 unique + 350 output, conc 16/24), reproducing the ~9K/12K in/tok/s/GPU run.
# Serve = the 9K serve (_serve_fp8asm.sh) config:
#   * DEFAULT --max-num-batched-tokens (NOT 4096 — 4096 chops the 63k prefill into
#     ~16 chunks and roughly halves throughput / blows up TTFT P90),
#   * --no-disable-hybrid-kv-cache-manager (K3 MLA+KDA hybrid),
#   * gpu-mem 0.95, max-num-seqs 24, moe-backend aiter, kv fp8, ROCM_AITER_MLA.
#     (the 9K reference used 0.8; raised to 0.95 — immaterial here, KV usage is ~15-28%.)
# Client = aiperf v1.0.1 (agentx-v1.0.1 @ /workspace/.aiperf_v1_0_1). v1.0.1's flag
# is --prompt-input-tokens-mean (== the older --synthetic-input-tokens-mean).
set -uo pipefail
cd /workspace
MP=/dev/shm/hf-cache/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569
AIPERF="${AIPERF:-/workspace/.aiperf_v1_0_1/bin/aiperf}"
OUT="${OUT:-/workspace/k3_prefixbench_fp8asm_v101}"

echo "########## prefixbench (9K serve config + aiperf v1.0.1) start $(date +%T) ##########"
export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 SAFETENSORS_FAST_GPU=1 \
       AITER_SITUV2_A8W4=1 AITER_BF16_FP8_MOE_BOUND=0 VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
       GPU_ARCHS=gfx950 VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900 \
       HF_HUB_CACHE=/dev/shm/hf-cache HF_HOME=/dev/shm/hf-cache
setsid nohup vllm serve "$MP" --served-model-name moonshotai/Kimi-K3 \
  --host 0.0.0.0 --port 8888 --tensor-parallel-size 8 --async-scheduling \
  --distributed-executor-backend mp --gpu-memory-utilization 0.95 \
  --max-num-seqs 24 \
  --trust-remote-code --load-format auto --moe-backend aiter \
  --kv-cache-dtype fp8 --attention-backend ROCM_AITER_MLA \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --enable-prefix-caching --no-disable-hybrid-kv-cache-manager \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 --enable-auto-tool-choice \
  --disable-uvicorn-access-log > /workspace/serve_prefixbench.log 2>&1 &

for i in $(seq 1 180); do
  curl -sf -m5 http://localhost:8888/health >/dev/null 2>&1 && break
  pgrep -f "vllm serve" >/dev/null 2>&1 || { echo "serve died"; break; }
  sleep 5
done
curl -sf -m5 http://localhost:8888/health >/dev/null 2>&1 || { echo "!! serve not ready; freeing"; bash /workspace/_freegpu.sh; exit 1; }
echo "serve ready $(date +%T)"

rm -rf "$OUT"; mkdir -p "$OUT"
"$AIPERF" profile \
  --model moonshotai/Kimi-K3 --tokenizer "$MP" --tokenizer-trust-remote-code \
  --url http://127.0.0.1:8888 --api-key EMPTY --endpoint-type chat --streaming --use-server-token-count \
  --num-prefix-prompts 8 --prompt-prefix-length 63240 \
  --prompt-input-tokens-mean 4760 --prompt-input-tokens-stddev 0 \
  --output-tokens-mean 350 --output-tokens-stddev 0 \
  --extra-inputs ignore_eos:true --extra-inputs min_tokens:350 --extra-inputs max_tokens:350 \
  --warmup-request-count 3 --sweep-type zip --concurrency 16,24 --request-count 80,120 \
  --random-seed 42 --no-gpu-telemetry --output-artifact-dir "$OUT" > /workspace/prefixbench.log 2>&1
rc=$?
echo "aiperf rc=$rc freeing GPUs $(date +%T)"
bash /workspace/_freegpu.sh
echo "########## prefixbench COMPLETE (rc=$rc) + GPUs freed $(date +%T) ##########"
