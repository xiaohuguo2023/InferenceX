#!/bin/bash
# fp8-ASM agentic sweep — OUR asm serve config (ROCM_AITER_MLA, kv=fp8, gpu-mem 0.95,
# ms64, uncapped) + the EXACT IX-CI agentic harness (build_replay_cmd in
# benchmarks/benchmark_lib.sh). Key CI knob: --agentic-cache-warmup-duration 600
# (time-bounded warmup — replaces --warmup-requests-per-lane, which hung on the
# long-context drain at conc16/24). Runs against the already-alive serve on :8888.
set -u
AIPERF=/workspace/.aiperf_be758d/bin/aiperf
DUR=1200                 # AgentX 20-min profiling
CACHE_WARMUP=600         # IX default agentic-cache-warmup-duration
GRACE=1800               # IX warmup-grace-period
run_conc() {
  local c="$1" seed=42; [ "$c" = "1" ] && seed=0
  local out="/workspace/k3_fp8asm_ixci_c$c"
  rm -rf "$out"; mkdir -p "$out/aiperf_artifacts"
  echo "=== fp8asm-IXCI conc=$c seed=$seed start $(date +%T) ==="
  timeout 3000 "$AIPERF" profile --scenario inferencex-agentx-mvp --url http://localhost:8888 \
    --endpoint /v1/chat/completions --endpoint-type chat --streaming --model moonshotai/Kimi-K3 \
    --concurrency "$c" --benchmark-duration "$DUR" --stats-interval 30 --random-seed "$seed" \
    --failed-request-threshold 0.10 --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
    --agentic-cache-warmup-duration "$CACHE_WARMUP" --warmup-grace-period "$GRACE" \
    --use-server-token-count --no-gpu-telemetry --tokenizer-trust-remote-code \
    --num-dataset-entries 393 --slice-duration 1.0 \
    --output-artifact-dir "$out/aiperf_artifacts" --public-dataset semianalysis_cc_traces_weka_062126 \
    > "/workspace/k3_fp8asm_ixci_c$c.log" 2>&1
  pkill -9 -f "aiperf profile" 2>/dev/null; sleep 3
  echo "=== fp8asm-IXCI conc=$c done $(date +%T) ==="
}
echo "########## fp8asm-IXCI sweep start $(date +%T) ##########"
for c in 1 4 8 16 24; do run_conc "$c"; done
echo "########## fp8asm-IXCI sweep COMPLETE $(date +%T) ##########"
