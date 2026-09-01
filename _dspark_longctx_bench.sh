#!/usr/bin/env bash
# Long-context DSpark benchmark — mirrors the ATOM nspec report workload
# (mi355x_atom0807docker_specdecode7.md): ISL 68,089 (63,911-tok cached prefix,
# pool of 1) / OSL 350, concurrency sweep 48..1 with zip request counts.
#
# Difference vs the ATOM harness: instead of a 1 Hz /metrics sampler + window
# alignment, we snapshot vLLM /metrics immediately before and after each
# concurrency point. That yields exact per-point counter deltas for DSpark
# acceptance (overall + per draft position) with no timestamp bookkeeping.
#
#   PORT=8890 PAIRS="48 240" bash _dspark_longctx_bench.sh   # single point (conc-48 gate)
#   PORT=8890 bash _dspark_longctx_bench.sh                  # full 48..1 sweep
set -uo pipefail

PORT="${PORT:-8890}"
URL="http://127.0.0.1:${PORT}"
METRICS_URL="${URL}/metrics"
# Glob the snapshot rather than pinning a hash: the cache is re-downloaded on
# every reboot (/dev/shm has no fstab entry) and the hash changes.
TOK="${TOK:-$(ls -d /dev/shm/hf-cache/models--moonshotai--Kimi-K3/snapshots/*/ | head -1)}"
ROOT="${ROOT:-/workspace/k3_dspark_longctx_bench}"

# Workload shape (byte-identical to the ATOM report).
PREFIX_LEN="${PREFIX_LEN:-63911}"
NUM_PREFIX="${NUM_PREFIX:-1}"
ISL_SUFFIX="${ISL_SUFFIX:-4089}"     # + prefix ~= 68,000 ISL
OSL="${OSL:-350}"
WARMUP="${WARMUP:-16}"

# Resolve aiperf.
AIPERF="${AIPERF:-}"
if [ -z "$AIPERF" ] || [ ! -x "$AIPERF" ]; then
  # An -x aiperf is not enough: several stale venvs have a shebang pointing at
  # an interpreter that no longer exists (exec -> rc=126 on every point).
  for c in /opt/.aiperf_*/bin/aiperf /workspace/.aiperf_*/bin/aiperf; do
    [ -x "$c" ] || continue
    "$c" --version >/dev/null 2>&1 && { AIPERF="$c"; break; }
  done
fi
[ -n "$AIPERF" ] && [ -x "$AIPERF" ] || { echo "!! no aiperf found"; exit 1; }

# (concurrency, request-count) pairs — zip semantics, descending like the report.
# conc-48 runs FIRST so the highest-batch capture path is exercised up front.
PAIRS="${PAIRS:-48 240
32 160
24 120
16 80
12 60
8 40
4 20
2 10
1 5}"

mkdir -p "$ROOT"
echo "=== DSpark long-ctx bench  URL=$URL  ROOT=$ROOT  $(date +%T) ==="
echo "    prefix=$PREFIX_LEN suffix=$ISL_SUFFIX OSL=$OSL warmup=$WARMUP"

snap() { curl -sf -m 15 "$METRICS_URL" > "$1" 2>/dev/null || echo "!! metrics snapshot failed -> $1"; }

while read -r conc reqs; do
  [ -z "${conc:-}" ] && continue
  out="$ROOT/concurrency_${conc}__requests_${reqs}"
  rm -rf "$out"; mkdir -p "$out"
  echo "--- conc=$conc reqs=$reqs  $(date +%T) ---"

  snap "$out/metrics_before.txt"
  "$AIPERF" profile \
    --model Kimi-K3 \
    --tokenizer "$TOK" \
    --tokenizer-trust-remote-code \
    --url "$URL" \
    --endpoint /v1/chat/completions \
    --endpoint-type chat \
    --streaming \
    --use-server-token-count \
    --num-prefix-prompts "$NUM_PREFIX" \
    --prompt-prefix-length "$PREFIX_LEN" \
    --synthetic-input-tokens-mean "$ISL_SUFFIX" \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean "$OSL" \
    --output-tokens-stddev 0 \
    --extra-inputs ignore_eos:true \
    --extra-inputs min_tokens:"$OSL" \
    --extra-inputs max_tokens:"$OSL" \
    --warmup-request-count "$WARMUP" \
    --concurrency "$conc" \
    --request-count "$reqs" \
    --random-seed 42 \
    --ui simple \
    --no-gpu-telemetry \
    --output-artifact-dir "$out" 2>&1 | tail -n 40
  rc=${PIPESTATUS[0]}
  snap "$out/metrics_after.txt"
  echo "    aiperf rc=$rc  $(date +%T)"
done <<< "$PAIRS"

echo "=== BENCH DONE $(date +%T) ==="
