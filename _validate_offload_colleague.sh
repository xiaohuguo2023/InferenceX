#!/usr/bin/env bash
# =============================================================================
# _validate_offload_colleague.sh
#   Reproduce the colleague's E2E offload measurement and validate our fixes.
#
#   Colleague config (confirmed working with the PR, DSpark, 64 GiB pool):
#     ISL 8192 / OSL 1024 / conc 16, 24 sessions, TP8, DSpark,
#     64 GiB offload pool, GPU KV capped to 163,840 tokens so the 221k working
#     set evicts -> re-hits reload from host.
#
#   Capacity is NOT the blocker (64 GiB works). The bug under test is the
#   full-attention EAGLE PREFIX-VETO (_patch_offload_eagle_prefix_veto.py).
#
#   3 arms (conc 16), each cold-served, mandated DSpark decode config otherwise:
#     A baseline  : no offload                       (reference TTFT)
#     B off_fix   : offload + eagle veto FIXED       (our patch, default)
#     C off_veto  : offload + eagle veto UPSTREAM    (OFFLOAD_EAGLE_PREFIX_VETO=1)
#   B vs C isolates the patch; A vs B shows the offload TTFT win.
#
#   Waits for the K3 weight re-download to finish, applies patches, then runs.
#   bash _validate_offload_colleague.sh
# =============================================================================
set -uo pipefail

CONTAINER="${CONTAINER:-k3-dspark-benchmark}"
PORT="${PORT:-8894}"
NSPEC="${NSPEC:-2}"
CONC="${CONC:-16}"
REQ="${REQ:-192}"                 # 8x the 24 prefixes -> plenty of re-hits
WARMUP="${WARMUP:-24}"            # prime + start evicting before measurement
PREFIX_POOL="${PREFIX_POOL:-24}" # 24 "sessions"
PREFIX_LEN="${PREFIX_LEN:-8192}" # ISL 8192 reused prefix
SYN_ISL="${SYN_ISL:-64}"         # tiny per-turn suffix
OSL="${OSL:-1024}"
# GPU KV pin: ~3 GiB ~= 160-180k tokens (34 GiB ~= 2M tok on this build), just under
# the 24*8192 = 196,608-tok prefix working set -> forces eviction like the colleague's
# 163,840 cap. Actual token capacity is read back from the serve log below.
KV_CACHE_MEMORY="${KV_CACHE_MEMORY:-3221225472}"
# vLLM requires the KV pin to hold ONE request at max_model_len. The mandated 1,048,576
# needs >=16.61 GiB, so a 3 GiB pin is rejected (estimated max len 173,568). Cap
# max_model_len to the colleague's GPU-KV token budget (163,840) -- far above our
# ~9.3k-token requests -- so 3 GiB is accepted and GPU KV (~173k tok) sits below the
# 24x8192=196,608-tok working set => eviction, exactly the colleague's regime.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-163840}"
KV_OFFLOADING_SIZE_ON="${KV_OFFLOADING_SIZE_ON:-64}"
KV_OFFLOADING_BACKEND="${KV_OFFLOADING_BACKEND:-native}"
# Mirror the colleague (conc 16 / 24 sessions), NOT the agentic-sweep max_num_seqs=64.
GPU_MEM="${GPU_MEM:-0.95}"; MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"; MNBT="${MNBT:-16384}"
AIPERF="${AIPERF:-/workspace/.aiperf_v1_0_1/bin/aiperf}"
MODEL="${MODEL:-Kimi-K3}"
APPLY_PATCHES="${APPLY_PATCHES:-1}"

log() { echo "[$(date +%T)] $*"; }

wait_for_download() {
  log "waiting for K3 weight download to finish..."
  local prev=0 cur stable=0
  while pgrep -f 'hf download' >/dev/null 2>&1; do sleep 20; done
  # proc gone; confirm blob size is stable + snapshot symlinks resolve
  for _ in $(seq 1 30); do
    cur=$(du -sb /dev/shm/hf-cache/models--moonshotai--Kimi-K3/blobs 2>/dev/null | awk '{print $1}')
    [ "${cur:-0}" = "${prev:-0}" ] && stable=$((stable+1)) || stable=0
    prev="$cur"
    [ "$stable" -ge 2 ] && break
    sleep 15
  done
  local sz; sz=$(du -sh /dev/shm/hf-cache/models--moonshotai--Kimi-K3/blobs 2>/dev/null | awk '{print $1}')
  log "download settled (blobs=$sz)"
}

wait_for_gpu_free() {
  # A TP8 serve needs all 8 GPUs. The GEMM tuner (task #50) holds GPU5. Block until
  # every GPU is drained (<20 GiB used) so we never stomp the tuner. Non-destructive.
  log "waiting for all 8 GPUs to be free (tuner on GPU5 must finish)..."
  while true; do
    local busy
    busy=$(docker exec "$CONTAINER" bash -lc "rocm-smi --showmeminfo vram 2>/dev/null | grep -i 'Used Memory' | awk '{if (\$NF+0 > 21474836480) c++} END{print c+0}'" 2>/dev/null || echo 8)
    [ "${busy:-8}" = "0" ] && { log "all GPUs free"; return 0; }
    sleep 60
  done
}

# max "Used Memory" across ALL 8 GPUs, GiB (leftover VRAM often sits on a non-0
# rank, e.g. cuda:3, so a head -1 GPU0 check can falsely report "drained" while
# another rank still holds 22 GiB and the next gpu_mem=0.95 serve then fails).
_vram_max_gib() {
  docker exec "$CONTAINER" bash -lc "rocm-smi --showmeminfo vram 2>/dev/null | grep -i 'Used Memory' | awk '{g=\$NF/1073741824; if (g>m) m=g} END{printf \"%.0f\", m}'" 2>/dev/null || echo 999
}
# count of live vLLM procs; self-exclude the matching shell via $$ (its cmdline
# contains the pattern) so we never count/kill ourselves.
_vllm_procs() {
  # Bracket-regex ([v]llm) so this counting shell doesn't self-match; then RE-READ each
  # candidate's /proc/cmdline and match an UNBRACKETED token. That excludes (a) the
  # permanent [vllm] <defunct> zombies (empty cmdline; PID1 `sleep infinity` never reaps)
  # and (b) the transient $(pgrep) command-substitution subshell (its cmdline carries the
  # bracketed pattern, not "vllm") -> race-proof, so a clean serve never reads as live.
  docker exec "$CONTAINER" bash -lc 'me=$$; n=0; for p in $(pgrep -f "[v]llm|[E]ngineCore|[V]llmWorker|[m]ultiprocessing.spawn" 2>/dev/null); do [ "$p" = "$me" ] && continue; cl=$(cat /proc/$p/cmdline 2>/dev/null | tr "\0" " "); case "$cl" in *vllm*|*EngineCore*|*VllmWorker*|*multiprocessing.spawn*) n=$((n+1)) ;; esac; done; echo $n' 2>/dev/null || echo 0
}

kill_serve() {
  # GRACEFUL drain FIRST. The native KV-offload backend allocates pinned host + GPU
  # staging buffers; a hard SIGKILL skips vLLM's shutdown so those buffers are never
  # released and the ROCm driver does NOT reclaim them on process death -> ~30 GiB/GPU
  # of orphaned VRAM that ONLY a node reboot clears. SIGTERM lets each proc run its
  # cleanup (connector.close / torch HIP teardown) and free its memory. Escalate to
  # SIGKILL only if graceful stalls. Self-exclude via $$ so we never signal this shell.
  docker exec "$CONTAINER" bash -lc '
    me=$$
    for p in $(pgrep -f "[v]llm|[E]ngineCore|[V]llmWorker|[m]ultiprocessing.spawn"); do
      [ "$p" = "$me" ] && continue; kill -TERM "$p" 2>/dev/null
    done; exit 0' 2>/dev/null || true
  local clean=0 used
  for _ in $(seq 1 18); do          # up to ~90s for graceful exit + VRAM drain
    sleep 5
    used=$(_vram_max_gib)
    if [ "$(_vllm_procs)" = "0" ] && [ "${used:-999}" -le 20 ] 2>/dev/null; then
      log "graceful teardown clean (procs=0, max VRAM ${used} GiB)"; clean=1; break
    fi
  done
  if [ "$clean" != "1" ]; then      # graceful stalled -> hard-kill survivors
    log "graceful teardown incomplete (VRAM=$(_vram_max_gib) GiB) -> SIGKILL fallback"
    docker exec "$CONTAINER" bash -lc '
      me=$$
      for p in $(pgrep -f "[v]llm|[E]ngineCore|[V]llmWorker|[m]ultiprocessing.spawn"); do
        [ "$p" = "$me" ] && continue; kill -9 "$p" 2>/dev/null
      done; exit 0' 2>/dev/null || true
    for _ in $(seq 1 12); do
      sleep 5; used=$(_vram_max_gib)
      [ "${used:-999}" -le 20 ] 2>/dev/null && { log "VRAM drained after SIGKILL (max ${used} GiB)"; break; }
    done
  fi
  docker exec "$CONTAINER" bash -lc 'rm -f /dev/shm/vllm_offload_*.mmap 2>/dev/null' 2>/dev/null || true
}

metric_sum() { # $1=metrics text  $2=grep-regex
  printf '%s\n' "$1" | grep -E "$2" | grep -v '#' | awk '{s+=$NF} END{printf "%.0f", s}'
}

TOKENIZER=""
run_arm() {
  local arm="$1" off="$2" veto="$3"
  local TAG="colleague_${arm}${RUN_TAG:+_$RUN_TAG}"
  local OUT="/workspace/k3_offval_${arm}${RUN_TAG:+_$RUN_TAG}"
  log "########## ARM $arm  (offload='${off:-off}'  eagle_veto_upstream=${veto:-0}) ##########"
  kill_serve
  local extra_env=(-e NUM_SPEC="$NSPEC" -e PORT="$PORT" -e GPU_MEM="$GPU_MEM"
    -e MAX_NUM_SEQS="$MAX_NUM_SEQS" -e MNBT="$MNBT" -e KV_CACHE_MEMORY="$KV_CACHE_MEMORY"
    -e MAX_MODEL_LEN="$MAX_MODEL_LEN"
    -e KV_OFFLOADING_SIZE="$off" -e KV_OFFLOADING_BACKEND="$KV_OFFLOADING_BACKEND"
    -e OFFLOAD_EAGLE_PREFIX_VETO="${veto:-0}")
  if ! docker exec "${extra_env[@]}" "$CONTAINER" bash /workspace/_serve_k3_bench_spec.sh; then
    log "ERROR: serve failed for arm $arm"; docker exec "$CONTAINER" bash -lc "tail -50 /workspace/serve_k3_bench_spec${NSPEC}.log" || true; return 1
  fi
  # actual GPU KV token capacity (for the record)
  local kvtok; kvtok=$(docker exec "$CONTAINER" bash -lc "grep -iE 'GPU KV cache size|KV cache size|# GPU blocks' /workspace/serve_k3_bench_spec${NSPEC}.log | tail -3" 2>/dev/null || true)
  log "  GPU KV capacity: ${kvtok:-unknown}"
  [ -z "$TOKENIZER" ] && TOKENIZER=$(docker exec "$CONTAINER" bash -lc 'ls -d /dev/shm/hf-cache/models--moonshotai--Kimi-K3/snapshots/*/ 2>/dev/null | head -1' | tr -d '\r'); TOKENIZER="${TOKENIZER%/}"
  docker exec "$CONTAINER" bash -lc "rm -rf '$OUT'; mkdir -p '$OUT'"
  log "  aiperf conc=$CONC reqs=$REQ  (24x8192 prefix, ISL~$((PREFIX_LEN+SYN_ISL)), OSL=$OSL)"
  docker exec "$CONTAINER" bash -lc "
    '$AIPERF' profile --model '$MODEL' --tokenizer '$TOKENIZER' --tokenizer-trust-remote-code \
      --url 'http://localhost:$PORT' --endpoint /v1/chat/completions --endpoint-type chat --streaming \
      --use-server-token-count \
      --num-prefix-prompts $PREFIX_POOL --prompt-prefix-length $PREFIX_LEN \
      --synthetic-input-tokens-mean $SYN_ISL --synthetic-input-tokens-stddev 0 \
      --output-tokens-mean $OSL --output-tokens-stddev 0 \
      --extra-inputs ignore_eos:true --extra-inputs min_tokens:$OSL --extra-inputs max_tokens:$OSL \
      --warmup-request-count $WARMUP --concurrency $CONC --request-count $REQ \
      --random-seed 42 --ui simple --no-gpu-telemetry --output-artifact-dir '$OUT'
  " > "/tmp/aiperf_${TAG}.log" 2>&1 || log "  WARN: aiperf nonzero (see /tmp/aiperf_${TAG}.log)"
  docker exec "$CONTAINER" bash -lc "curl -s http://localhost:$PORT/metrics 2>/dev/null > '$OUT/metrics_final.txt'" || true
  local m; m=$(docker exec "$CONTAINER" bash -lc "cat '$OUT/metrics_final.txt' 2>/dev/null" || echo "")
  local hits load store
  hits=$(metric_sum "$m" 'external_prefix_cache_hits_total')
  load=$(metric_sum "$m" 'kv_offload_total_bytes_total\{.*CPU_to_GPU|prompt_tokens_by_source_total\{.*external_kv_transfer')
  store=$(metric_sum "$m" 'kv_offload_store_bytes_total')
  # TTFT p50 (ms) from aiperf export. Use python json, NOT grep: the export is
  # pretty-printed so "time_to_first_token" and "p50" sit on different lines and a
  # line-based grep returns nothing (that produced the NA in the first run).
  local ttft; ttft=$(docker exec "$CONTAINER" bash -lc "python3 -c \"import json,glob,sys
fs=glob.glob('$OUT/profile_export_aiperf.json') or glob.glob('$OUT/*_export_aiperf.json')
print(json.load(open(fs[0]))['time_to_first_token']['p50']) if fs else print('NA')\" 2>/dev/null")
  log "  [RESULT $arm conc=$CONC] TTFT_p50=${ttft:-NA}  external_hits=${hits:-0}  load_bytes=${load:-0}  store_bytes=${store:-0}"
  echo "conc=$CONC $arm ttft_p50=${ttft:-NA} hits=${hits:-0} load=${load:-0} store=${store:-0}" >> "$SUMMARY"
}

SUMMARY="/tmp/offval_summary${RUN_TAG:+_$RUN_TAG}.txt"
ARMS="${ARMS:-baseline off_fix off_veto}"
: > "$SUMMARY"
[ -n "$(pgrep -f 'hf download' 2>/dev/null)" ] && wait_for_download || log "no active download; proceeding"

if [ "$APPLY_PATCHES" = "1" ]; then
  log "applying DSpark + offload patches (draft-causal, #52047, eagle prefix-veto 6c)"
  docker exec "$CONTAINER" bash -lc "bash /workspace/_k3_dspark_fp8asm_apply_patches.sh" 2>&1 | tail -20 || log "WARN: apply-patches returned nonzero"
fi

for arm in $ARMS; do
  case "$arm" in
    baseline) run_arm baseline "" 0 ;;
    off_fix)  run_arm off_fix  "$KV_OFFLOADING_SIZE_ON" 0 ;;
    off_veto) run_arm off_veto "$KV_OFFLOADING_SIZE_ON" 1 ;;
    *) log "unknown arm '$arm' (skip)" ;;
  esac
done

kill_serve
log "########## OFFLOAD VALIDATION COMPLETE (conc=$CONC, tag=${RUN_TAG:-none}) ##########"
log "Summary:"; cat "$SUMMARY"
log "Interpretation: off_fix should show external_hits>0 AND load_bytes>0 AND TTFT < baseline;"
log "off_veto (upstream veto) should show suppressed hits / higher TTFT than off_fix."
