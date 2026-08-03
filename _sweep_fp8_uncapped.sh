#!/bin/bash
# fp8-ASM agentic sweep — UNCAPPED (no --max-context-length), against the live
# max-num-seqs 64 / gpu-mem 0.95 serve on :8888. conc 1,4,8,16,24 @ 1200s.
export HF_HUB_CACHE=/dev/shm/hf-cache HF_HOME=/dev/shm/hf-cache
AIPERF=/workspace/.aiperf_be758d/bin/aiperf
for c in 1 4 8 16 24; do
  seed=42; [ "$c" = "1" ] && seed=0
  out=/workspace/k3_fp8asm_ref_sweep_c$c
  rm -rf "$out"; mkdir -p "$out/aiperf_artifacts"
  echo "=== fp8asm conc=$c seed=$seed start $(date +%T) ==="
  timeout 2400 "$AIPERF" profile --scenario inferencex-agentx-mvp --url http://localhost:8888 \
    --endpoint /v1/chat/completions --endpoint-type chat --streaming --model moonshotai/Kimi-K3 \
    --concurrency "$c" --benchmark-duration 1200 --stats-interval 30 --random-seed "$seed" \
    --failed-request-threshold 0.10 --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
    --warmup-requests-per-lane 10 --warmup-grace-period 1800 --use-server-token-count --no-gpu-telemetry \
    --tokenizer-trust-remote-code --num-dataset-entries 393 --slice-duration 1.0 \
    --output-artifact-dir "$out/aiperf_artifacts" --public-dataset semianalysis_cc_traces_weka_062126 \
    > /workspace/k3_fp8asm_ref_c$c.log 2>&1
  pkill -9 -f "aiperf profile" 2>/dev/null; sleep 3
  echo "=== fp8asm conc=$c done $(date +%T) ==="
done
echo "=== fp8asm UNCAPPED SWEEP COMPLETE $(date +%T) ==="
