#!/bin/bash
# Kimi-K3 agentic sweep — OUR ASM serve config (ROCM_AITER_MLA, kv=fp8, gpu-mem
# 0.95, ms64, uncapped) driven by the IX-CI agentic harness, matching
# benchmarks/benchmark_lib.sh:build_replay_cmd (verified against the recipe path).
#
# Harness knobs use the SAME env names + defaults as build_replay_cmd:
#   --warmup-requests-per-lane    AIPERF_WARMUP_REQUESTS_PER_LANE    (default 10)
#   --trace-idle-gap-cap-seconds  AIPERF_TRACE_IDLE_GAP_CAP_SECONDS  (default 300)
#   --warmup-grace-period         AGENTIC_WARMUP_GRACE_PERIOD        (default 1800)
#   --failed-request-threshold    AIPERF_FAILED_REQUEST_THRESHOLD    (default 0.10)
#   --benchmark-duration          DURATION                           (default 1200)
# trace-idle-gap-cap is REQUIRED: without it the cc-traces trajectories replay
# their full real-world idle gaps, so warmup never drains at high concurrency.
#
# NO shell `timeout`: aiperf self-bounds each run — warmup ends at the grace
# period or when it drains, profiling runs benchmark-duration, and
# failed-request-threshold aborts early. A tight shell timeout would kill a
# healthy run (warmup-grace 1800 + benchmark-duration 1200 already = 3000s before
# warmup-send + export). Hang protection, if needed, belongs at the job level
# with a much larger bound + diagnostic cleanup, as in IX-CI.
#
# Runs against an already-alive serve on :PORT (start it from the recipe or
# _serve_fp8_ms64.sh). Env: TAG (output prefix, default fp8asm), CONC_LIST.
set -euo pipefail
# REQUIRES aiperf pinned to the IX submodule commit (utils/aiperf @ 818c3a5a or
# newer), which allows --trace-idle-gap-cap-seconds for the inferencex-agentx-mvp
# scenario. The older be758d62 build sets forbid_trace_idle_gap_cap=True and
# rejects the flag. Build the venv with:
#   uv venv --python 3.11 /workspace/.aiperf_818c3a5a && \
#   uv pip install --python /workspace/.aiperf_818c3a5a/bin/python \
#     -r utils/agentic-benchmark/requirements.txt -e utils/aiperf \
#     "datasets>=4.7.0" "huggingface_hub[cli]>=0.25.0" urllib3 requests
AIPERF="${AIPERF:-/workspace/.aiperf_818c3a5a/bin/aiperf}"
MODEL="${MODEL:-moonshotai/Kimi-K3}"
PORT="${PORT:-8888}"
DURATION="${DURATION:-1200}"
WARMUP_REQS="${AIPERF_WARMUP_REQUESTS_PER_LANE:-10}"
IDLE_GAP_CAP="${AIPERF_TRACE_IDLE_GAP_CAP_SECONDS:-300}"
GRACE="${AGENTIC_WARMUP_GRACE_PERIOD:-1800}"
FAIL_THRESH="${AIPERF_FAILED_REQUEST_THRESHOLD:-0.10}"
TAG="${TAG:-fp8asm}"
CONC_LIST="${CONC_LIST:-1 4 8 16 24}"
OUT_ROOT="${OUT_ROOT:-/workspace}"

run_conc() {
  local c="$1" seed=42; [ "$c" = "1" ] && seed=0
  local out="$OUT_ROOT/k3_${TAG}_ixci_c$c"
  rm -rf "$out"; mkdir -p "$out/aiperf_artifacts"
  echo "=== ${TAG}-IXCI conc=$c seed=$seed start $(date +%T) ==="
  "$AIPERF" profile --scenario inferencex-agentx-mvp --url "http://localhost:$PORT" \
    --endpoint /v1/chat/completions --endpoint-type chat --streaming --model "$MODEL" \
    --concurrency "$c" --benchmark-duration "$DURATION" --stats-interval 30 --random-seed "$seed" \
    --failed-request-threshold "$FAIL_THRESH" \
    --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
    --warmup-requests-per-lane "$WARMUP_REQS" --trace-idle-gap-cap-seconds "$IDLE_GAP_CAP" \
    --warmup-grace-period "$GRACE" \
    --use-server-token-count --no-gpu-telemetry --tokenizer-trust-remote-code \
    --num-dataset-entries 393 --slice-duration 1.0 \
    --output-artifact-dir "$out/aiperf_artifacts" --public-dataset semianalysis_cc_traces_weka_062126 \
    > "$OUT_ROOT/k3_${TAG}_ixci_c$c.log" 2>&1
  echo "=== ${TAG}-IXCI conc=$c done $(date +%T) ==="
}

echo "########## ${TAG}-IXCI sweep start $(date +%T) ##########"
for c in $CONC_LIST; do run_conc "$c"; done
echo "########## ${TAG}-IXCI sweep COMPLETE $(date +%T) ##########"
