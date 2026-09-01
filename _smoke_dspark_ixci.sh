#!/bin/bash
set -uo pipefail
AIPERF=/workspace/.aiperf_venv/bin/aiperf
PORT=8890
TOK=/dev/shm/hf-cache/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569
OUT=/workspace/k3_dspark_fp8asm_smoke
rm -rf "$OUT"; mkdir -p "$OUT/aiperf_artifacts"
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
echo "=== dspark smoke start $(date +%T) ==="
"$AIPERF" profile --scenario inferencex-agentx-mvp --url "http://localhost:$PORT" \
  --endpoint /v1/chat/completions --endpoint-type chat --streaming --model Kimi-K3 \
  --concurrency 1 --benchmark-duration 120 --unsafe-override --stats-interval 30 --random-seed 42 \
  --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
  --warmup-requests-per-lane 1 --trace-idle-gap-cap-seconds 60 \
  --warmup-grace-period 120 \
  --use-server-token-count --tokenizer "$TOK" --tokenizer-trust-remote-code --no-gpu-telemetry \
  --num-dataset-entries 30 --slice-duration 1.0 \
  --output-artifact-dir "$OUT/aiperf_artifacts" --public-dataset semianalysis_cc_traces_weka_062126
echo "=== dspark smoke done rc=$? $(date +%T) ==="
