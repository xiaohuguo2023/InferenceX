#!/usr/bin/env bash
# InferenceX aiperf sweep for the native fp8-asm DSpark serve (PIECEWISE, APC-off).
# Moderate synthetic workload that fits max-model-len=65536 without prefix caching:
#   ISL=1024, OSL=256 (ignore_eos), concurrency sweep 1,8,16.
# Measures out tok/s + latency; acceptance length is read from the serve log.
#   PORT=8890 bash _bench_k3_dspark_fp8asm.sh
set -euo pipefail
AIPERF="${AIPERF:-}"
if [ -z "$AIPERF" ] || [ ! -x "$AIPERF" ]; then
  for c in /opt/.aiperf_*/bin/aiperf /workspace/.aiperf_*/bin/aiperf; do
    [ -x "$c" ] && { AIPERF="$c"; break; }
  done
fi
[ -n "$AIPERF" ] && [ -x "$AIPERF" ] || { echo "!! no aiperf found"; exit 1; }
PORT="${PORT:-8890}"
URL="http://127.0.0.1:${PORT}"
TOK=/dev/shm/hf-cache/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569
ROOT="/workspace/k3_dspark_fp8asm_bench"
rm -rf "$ROOT"; mkdir -p "$ROOT"

ISL="${ISL:-1024}"; OSL="${OSL:-256}"
# (concurrency, request-count) pairs — light counts to keep the sweep quick.
for pair in "1 12" "8 48" "16 96"; do
  read -r conc reqs <<< "$pair"
  out="$ROOT/concurrency_${conc}__requests_${reqs}"
  rm -rf "$out"; mkdir -p "$out"
  echo "=== aiperf conc=$conc reqs=$reqs ISL=$ISL OSL=$OSL $(date +%T) ==="
  "$AIPERF" profile \
    --model Kimi-K3 \
    --tokenizer "$TOK" \
    --tokenizer-trust-remote-code \
    --url "$URL" \
    --endpoint /v1/chat/completions \
    --endpoint-type chat \
    --streaming \
    --use-server-token-count \
    --synthetic-input-tokens-mean "$ISL" \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean "$OSL" \
    --output-tokens-stddev 0 \
    --extra-inputs ignore_eos:true \
    --extra-inputs min_tokens:"$OSL" \
    --extra-inputs max_tokens:"$OSL" \
    --warmup-request-count 3 \
    --concurrency "$conc" \
    --request-count "$reqs" \
    --random-seed 42 \
    --ui simple \
    --no-gpu-telemetry \
    --output-artifact-dir "$out"
done
echo "=== BENCH DONE $(date +%T) ==="
