#!/usr/bin/env bash
# Capture a torch-profiler trace of a STEADY-STATE conc-48 DSpark decode window.
#
# The serve must already be up on $PORT with the profiler enabled
# (PROFILE_DIR=... bash _serve_k3_bench_spec.sh). This driver:
#   1. fires the mandated conc-48 long-ctx workload (ISL 68k / OSL big) via aiperf
#      in the background so 48 requests stay concurrently in decode;
#   2. waits WARM_S for prefills to drain and decode to saturate at M=144 (48*3);
#   3. wraps a short window in POST /start_profile ... /stop_profile so only
#      steady-state decode engine-steps are recorded;
#   4. lets the load finish and the 8 per-rank traces flush.
#
#   PORT=8890 PROFILE_DIR=/workspace/kimik3_traces_c48 bash _profile_c48_decode.sh
set -uo pipefail
PORT="${PORT:-8890}"
URL="http://127.0.0.1:${PORT}"
PROFILE_DIR="${PROFILE_DIR:-/workspace/kimik3_traces_c48}"
TOK="${TOK:-/dev/shm/hf-cache/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569}"
CONC="${CONC:-48}"
# enough output tokens that all 48 are still decoding through the profile window
OSL="${OSL:-600}"
PREFIX_LEN="${PREFIX_LEN:-63911}"
ISL_SUFFIX="${ISL_SUFFIX:-4089}"
WARM_S="${WARM_S:-45}"       # seconds to reach steady-state decode before profiling
PROFILE_S="${PROFILE_S:-3}"  # length of the recorded decode window

AIPERF="${AIPERF:-}"
if [ -z "$AIPERF" ] || [ ! -x "$AIPERF" ]; then
  for c in /opt/.aiperf_*/bin/aiperf /workspace/.aiperf_*/bin/aiperf; do
    [ -x "$c" ] && { AIPERF="$c"; break; }
  done
fi
[ -n "$AIPERF" ] && [ -x "$AIPERF" ] || { echo "!! no aiperf found"; exit 1; }

# DEFAULT to a SINGLE wave (REQS==CONC): all 48 prefill once, then decode
# together with NO further prefills, so a mid-decode window is PURE decode
# (the stage analyzer's ctx-based split can't separate DSpark's large-ctx decode
# from prefill, so we isolate decode at the workload level instead). Override
# REQS for a multi-wave (mixed) run.
REQS="${REQS:-$CONC}"
LOAD_LOG=/workspace/_profile_c48_load.log
echo "=== conc-$CONC decode profile  URL=$URL  OSL=$OSL  warm=${WARM_S}s window=${PROFILE_S}s  $(date +%T) ==="

"$AIPERF" profile \
  --model Kimi-K3 --tokenizer "$TOK" --tokenizer-trust-remote-code \
  --url "$URL" --endpoint /v1/chat/completions --endpoint-type chat --streaming \
  --use-server-token-count \
  --num-prefix-prompts 1 --prompt-prefix-length "$PREFIX_LEN" \
  --synthetic-input-tokens-mean "$ISL_SUFFIX" --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean "$OSL" --output-tokens-stddev 0 \
  --extra-inputs ignore_eos:true --extra-inputs min_tokens:"$OSL" --extra-inputs max_tokens:"$OSL" \
  --concurrency "$CONC" --request-count "$REQS" \
  > "$LOAD_LOG" 2>&1 &
LOAD_PID=$!
echo "aiperf load pid=$LOAD_PID (log: $LOAD_LOG)"

echo "warming ${WARM_S}s to steady-state decode..."
sleep "$WARM_S"

if ! kill -0 "$LOAD_PID" 2>/dev/null; then
  echo "!! load ended before profile window — increase OSL/REQS"; tail -20 "$LOAD_LOG"; exit 1
fi

echo "START profile $(date +%T)"
curl -s -o /dev/null -w "  start_profile HTTP %{http_code}\n" -X POST "$URL/start_profile"
sleep "$PROFILE_S"
curl -s -o /dev/null -w "  stop_profile  HTTP %{http_code}\n" -X POST "$URL/stop_profile"
echo "STOP profile $(date +%T) — traces flushing to $PROFILE_DIR"

echo "waiting for aiperf load to finish..."
wait "$LOAD_PID" 2>/dev/null
echo "=== done $(date +%T) ==="
ls -la "$PROFILE_DIR"/*.pt.trace.json.gz 2>/dev/null | tail -10
