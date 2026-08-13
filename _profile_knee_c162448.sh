#!/usr/bin/env bash
# =============================================================================
# _profile_knee_c162448.sh
#   Torch-profile the DSpark decode knee (conc-16 -> 24 -> 48, i.e. M=48/72/144)
#   on the FULLFIX nspec-2 serve, to identify the per-step bottleneck behind the
#   ITL jump (conc-16 24ms -> conc-24 47ms -> conc-48 76ms).
#
#   Method (same as task #48 _profile_c48_decode.sh): SINGLE-WAVE load (REQS=CONC)
#   so all requests prefill once (shared 63.9k prefix cached) then decode together
#   -> the profiled window is PURE steady-state decode at M=3*conc. Traces per conc
#   land in $PROFILE_ROOT/c<conc>/ and get analyzed with backend_breakdown.py +
#   analyze_dsv4_trace.py so we can diff M=48 vs M=72 vs M=144.
#
#   Runs entirely inside the container. bash _profile_knee_c162448.sh
# =============================================================================
set -uo pipefail
cd /workspace

PORT="${PORT:-8890}"
URL="http://127.0.0.1:${PORT}"
PROFILE_ROOT="${PROFILE_ROOT:-/workspace/traces_knee}"
TOK="/dev/shm/hf-cache/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
CONCS="${CONCS:-16 24 48}"
# OSL must be big enough that the single wave is STILL decoding through the window:
# run time ~= prefill(~5s) + OSL*ITL. At conc-16 (ITL~24ms) OSL=600 finishes in ~19s
# < WARM+PROFILE, so nothing got captured. 1500 gives ~40s@c16 / ~75s@c24 of decode.
OSL="${OSL:-1500}"
PREFIX_LEN=63911
ISL_SUFFIX=4089
WARM_S="${WARM_S:-12}"      # prefill drains in ~5s; 12s is safely into steady decode
PROFILE_S="${PROFILE_S:-3}"

# Mandated FULLFIX nspec-2 serve config.
export NUM_SPEC=2 GPU_MEM=0.95 MAX_NUM_SEQS=64 MNBT=16384 PORT
AIPERF=$(ls /workspace/.aiperf_*/bin/aiperf /opt/.aiperf_*/bin/aiperf 2>/dev/null | head -1)
[ -x "$AIPERF" ] || { echo "!! no aiperf"; exit 1; }

log(){ echo "[$(date +%T)] $*"; }

kill_serve(){
  for p in $(pgrep -f "vllm serve|EngineCore|VllmWorker|multiprocessing.spawn" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  for _ in $(seq 1 15); do
    sleep 4
    local u; u=$(rocm-smi --showmeminfo vram 2>/dev/null | grep -i "Used Memory" | head -1 | awk '{printf "%.0f",$NF/1073741824}')
    [ "${u:-999}" -le 20 ] 2>/dev/null && { log "VRAM drained (${u} GiB)"; return 0; }
  done
}

log "cleaning any prior serve..."
kill_serve
rm -rf "$PROFILE_ROOT"; mkdir -p "$PROFILE_ROOT"

log "starting FULLFIX nspec-2 serve with torch profiler -> $PROFILE_ROOT"
if ! PROFILE_DIR="$PROFILE_ROOT" bash /workspace/_serve_k3_bench_spec.sh; then
  log "!! serve failed"; tail -60 /workspace/serve_k3_bench_spec2.log; exit 1
fi
log "serve ready."

for CONC in $CONCS; do
  M=$((3*CONC))
  sub="$PROFILE_ROOT/c${CONC}"; mkdir -p "$sub"
  log "########## conc=$CONC (M=$M) single-wave decode profile ##########"
  LOAD_LOG="/workspace/_knee_load_c${CONC}.log"
  "$AIPERF" profile \
    --model Kimi-K3 --tokenizer "$TOK" --tokenizer-trust-remote-code \
    --url "$URL" --endpoint /v1/chat/completions --endpoint-type chat --streaming \
    --use-server-token-count \
    --num-prefix-prompts 1 --prompt-prefix-length "$PREFIX_LEN" \
    --synthetic-input-tokens-mean "$ISL_SUFFIX" --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean "$OSL" --output-tokens-stddev 0 \
    --extra-inputs ignore_eos:true --extra-inputs min_tokens:"$OSL" --extra-inputs max_tokens:"$OSL" \
    --concurrency "$CONC" --request-count "$CONC" \
    --random-seed 42 --ui simple --no-gpu-telemetry \
    --output-artifact-dir "$sub/aiperf" > "$LOAD_LOG" 2>&1 &
  LOAD_PID=$!
  log "  aiperf load pid=$LOAD_PID; warming ${WARM_S}s..."
  sleep "$WARM_S"
  if ! kill -0 "$LOAD_PID" 2>/dev/null; then log "  !! load ended early; bump OSL"; tail -15 "$LOAD_LOG"; continue; fi
  log "  START profile"
  curl -s -o /dev/null -w "    start HTTP %{http_code}\n" -X POST "$URL/start_profile"
  sleep "$PROFILE_S"
  curl -s -o /dev/null -w "    stop  HTTP %{http_code}\n" -X POST "$URL/stop_profile"
  log "  waiting load to finish + traces to flush..."
  wait "$LOAD_PID" 2>/dev/null
  # move the just-flushed per-rank traces into this conc's subdir
  for _ in $(seq 1 24); do
    n=$(ls "$PROFILE_ROOT"/*.pt.trace.json.gz 2>/dev/null | wc -l)
    [ "$n" -ge 1 ] && sleep 3 && break
    sleep 3
  done
  mv "$PROFILE_ROOT"/*.pt.trace.json.gz "$sub"/ 2>/dev/null || log "  WARN: no traces flushed for conc=$CONC"
  ntr=$(ls "$sub"/*.pt.trace.json.gz 2>/dev/null | wc -l)
  log "  conc=$CONC: $ntr trace(s) -> $sub"
done

kill_serve
log "########## ANALYSIS ##########"
for CONC in $CONCS; do
  sub="$PROFILE_ROOT/c${CONC}"
  tr=$(ls "$sub"/*.pt.trace.json.gz 2>/dev/null)
  [ -z "$tr" ] && { log "conc=$CONC: NO TRACES"; continue; }
  echo; echo "===== conc=$CONC (M=$((3*CONC))) backend breakdown ====="
  python3 /workspace/backend_breakdown.py $tr 2>/dev/null
  python3 /workspace/analyze_dsv4_trace.py --md "/workspace/_knee_c${CONC}_report.md" $tr >/dev/null 2>&1 \
    && echo "  stage report -> _knee_c${CONC}_report.md"
  # measured ITL for this conc (sanity: should reproduce the knee)
  itl=$(python3 -c "import json,glob
f=glob.glob('$sub/aiperf/profile_export_aiperf.json')
print(json.load(open(f[0]))['inter_token_latency']['p50'] if f else 'NA')" 2>/dev/null)
  echo "  measured ITL p50 = ${itl} ms"
done
log "DONE. traces under $PROFILE_ROOT ; reports _knee_c{16,24,48}_report.md"
