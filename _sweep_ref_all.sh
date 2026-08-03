#!/bin/bash
# Unattended agentic pareto re-sweep with the improved (conc8-experiment) config.
# For fp8-ASM (kv=fp8) and bf16-ASM (kv=auto): serve with ref flags, then run the
# pinned-aiperf agentic scenario at conc 1,4,8,16,24 @ 1200s each.
# Handles: aiperf export-hang (timeout + stdout metrics), orphan-VRAM (kill+poll).
set -u
AIPERF=/workspace/.aiperf_be758d/bin/aiperf
MODEL_PATH=/dev/shm/hf-cache/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569
export HF_HUB_CACHE=/dev/shm/hf-cache HF_HOME=/dev/shm/hf-cache

killserve() {
  pkill -9 -f "vllm serve" 2>/dev/null; pkill -9 -f spawn_main 2>/dev/null
  pkill -9 -f VllmWorker 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null
  # process-based wait: loop until no serve/worker procs remain (VRAM frees on death)
  for i in $(seq 1 30); do
    pgrep -f "vllm serve|spawn_main|VllmWorker|EngineCore" >/dev/null 2>&1 || break
    pkill -9 -f spawn_main 2>/dev/null; pkill -9 -f VllmWorker 2>/dev/null
    sleep 3
  done
  sleep 10
}

serve() {  # $1=kv dtype (fp8|auto)  $2=tag
  export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 SAFETENSORS_FAST_GPU=1 \
         AITER_SITUV2_A8W4=1 AITER_BF16_FP8_MOE_BOUND=0 VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
         GPU_ARCHS=gfx950 VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 \
         VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900
  setsid nohup vllm serve "$MODEL_PATH" --served-model-name moonshotai/Kimi-K3 \
    --host 0.0.0.0 --port 8888 --tensor-parallel-size 8 --async-scheduling \
    --distributed-executor-backend mp --gpu-memory-utilization 0.95 \
    --max-num-seqs 128 --max-num-batched-tokens 4096 \
    --trust-remote-code --load-format auto --moe-backend auto \
    --kv-cache-dtype "$1" --attention-backend ROCM_AITER_MLA --mm-encoder-tp-mode data \
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --enable-prefix-caching --no-disable-hybrid-kv-cache-manager \
    --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 --enable-auto-tool-choice \
    --disable-uvicorn-access-log > "/workspace/serve_ref_$2.log" 2>&1 &
}

wait_ready() {  # poll /health up to ~12min
  for i in $(seq 1 144); do
    curl -s -m 5 http://localhost:8888/health -o /dev/null 2>/dev/null && return 0
    sleep 5
  done
  return 1
}

run_conc() {  # $1=tag  $2=conc
  local seed=42; [ "$2" = "1" ] && seed=0
  local out="/workspace/k3_$1_ref_sweep_c$2"
  rm -rf "$out"; mkdir -p "$out/aiperf_artifacts"
  echo "=== $1 conc=$2 seed=$seed start $(date +%T) ==="
  # 1200s bench + warmup + export; timeout guards the known export-hang (metrics are in stdout)
  timeout 2400 "$AIPERF" profile --scenario inferencex-agentx-mvp --url http://localhost:8888 \
    --endpoint /v1/chat/completions --endpoint-type chat --streaming --model moonshotai/Kimi-K3 \
    --concurrency "$2" --benchmark-duration 1200 --stats-interval 30 --random-seed "$seed" \
    --failed-request-threshold 0.10 --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
    --warmup-requests-per-lane 10 --warmup-grace-period 1800 --use-server-token-count --no-gpu-telemetry \
    --tokenizer-trust-remote-code --num-dataset-entries 393 --slice-duration 1.0 \
    --output-artifact-dir "$out/aiperf_artifacts" --public-dataset semianalysis_cc_traces_weka_062126 \
    > "/workspace/k3_$1_ref_c$2.log" 2>&1
  pkill -9 -f "aiperf profile" 2>/dev/null; sleep 3
  echo "=== $1 conc=$2 done $(date +%T) ==="
}

for cfg in "fp8asm:fp8" "bf16asm:auto"; do
  tag="${cfg%%:*}"; kv="${cfg##*:}"
  echo "########## CONFIG $tag (kv=$kv) start $(date +%T) ##########"
  killserve; serve "$kv" "$tag"
  if ! wait_ready; then echo "!! serve $tag NOT ready; skipping config"; killserve; continue; fi
  echo "$tag serve ready $(date +%T)"
  for c in 1 4 8 16 24; do run_conc "$tag" "$c"; done
  killserve
  echo "########## CONFIG $tag done $(date +%T) ##########"
done
echo "########## ALL REF SWEEPS COMPLETE $(date +%T) ##########"
