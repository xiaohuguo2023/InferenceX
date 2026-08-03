#!/bin/bash
# fp8-ASM agentic sweep with ref config + --max-context-length 131072 (keeps fp8 inside
# the mla_a8w8 <4GB-offset working range; also stabilizes the 1200s window). conc 1,4,8,16,24.
set -u
AIPERF=/workspace/.aiperf_be758d/bin/aiperf
MODEL_PATH=/dev/shm/hf-cache/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569
export HF_HUB_CACHE=/dev/shm/hf-cache HF_HOME=/dev/shm/hf-cache

killserve() {
  pkill -9 -f "vllm serve" 2>/dev/null; pkill -9 -f spawn_main 2>/dev/null
  pkill -9 -f VllmWorker 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null
  for i in $(seq 1 30); do
    pgrep -f "vllm serve|spawn_main|VllmWorker|EngineCore" >/dev/null 2>&1 || break
    pkill -9 -f spawn_main 2>/dev/null; pkill -9 -f VllmWorker 2>/dev/null; sleep 3
  done
  sleep 10
}
serve() {
  export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 SAFETENSORS_FAST_GPU=1 \
         AITER_SITUV2_A8W4=1 AITER_BF16_FP8_MOE_BOUND=0 VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
         GPU_ARCHS=gfx950 VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 \
         VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900
  setsid nohup vllm serve "$MODEL_PATH" --served-model-name moonshotai/Kimi-K3 \
    --host 0.0.0.0 --port 8888 --tensor-parallel-size 8 --async-scheduling \
    --distributed-executor-backend mp --gpu-memory-utilization 0.95 \
    --max-num-seqs 64 --max-num-batched-tokens 4096 \
    --trust-remote-code --load-format auto --moe-backend auto \
    --kv-cache-dtype fp8 --attention-backend ROCM_AITER_MLA --mm-encoder-tp-mode data \
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --enable-prefix-caching --no-disable-hybrid-kv-cache-manager \
    --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 --enable-auto-tool-choice \
    --disable-uvicorn-access-log > /workspace/serve_ref_fp8asm.log 2>&1 &
}
wait_ready() { for i in $(seq 1 144); do curl -s -m 5 http://localhost:8888/health -o /dev/null 2>/dev/null && return 0; sleep 5; done; return 1; }
run_conc() {
  local seed=42; [ "$1" = "1" ] && seed=0
  local out="/workspace/k3_fp8asm_ref_sweep_c$1"
  rm -rf "$out"; mkdir -p "$out/aiperf_artifacts"
  echo "=== fp8asm conc=$1 seed=$seed start $(date +%T) ==="
  timeout 2400 "$AIPERF" profile --scenario inferencex-agentx-mvp --url http://localhost:8888 \
    --endpoint /v1/chat/completions --endpoint-type chat --streaming --model moonshotai/Kimi-K3 \
    --concurrency "$1" --benchmark-duration 1200 --stats-interval 30 --random-seed "$seed" \
    --failed-request-threshold 0.10 --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
    --warmup-requests-per-lane 10 --warmup-grace-period 1800 --use-server-token-count --no-gpu-telemetry \
    --tokenizer-trust-remote-code --num-dataset-entries 393 --slice-duration 1.0 --max-context-length 131072 \
    --output-artifact-dir "$out/aiperf_artifacts" --public-dataset semianalysis_cc_traces_weka_062126 \
    > "/workspace/k3_fp8asm_ref_c$1.log" 2>&1
  pkill -9 -f "aiperf profile" 2>/dev/null; sleep 3
  echo "=== fp8asm conc=$1 done $(date +%T) ==="
}
echo "########## CONFIG fp8asm (kv=fp8, max-ctx=131072) start $(date +%T) ##########"
killserve; serve
if ! wait_ready; then echo "!! fp8 serve NOT ready"; killserve; exit 1; fi
echo "fp8asm serve ready $(date +%T)"
for c in 1 4 8 16 24; do run_conc "$c"; done
killserve
echo "########## fp8asm SWEEP COMPLETE $(date +%T) ##########"
