#!/usr/bin/env bash
# =============================================================================
# _run_offload_ab_fixed68k.sh
#   FAST DSpark baseline vs DSpark+offload A/B on the fixed-68k/350 request-count
#   shape (K3_Attention_Benchmark_Instructions.md / mi355x_atom0807docker_specdecode7.md).
#
#   WHY: the agentic trace-replay sweep is duration-based (~4h/arm). This shape is
#   REQUEST-COUNT bounded (5..240 reqs), so a full concurrency ladder finishes in
#   minutes. To make KV-offload actually engage on a synthetic shape we enlarge the
#   prefix pool (PREFIX_POOL distinct 63.9k-token prefixes) so the working set
#   overflows the ~2M-token device KV pin -> prefixes evict -> re-hits reload from
#   host (with vLLM #52047 draft-group annotation fix live) instead of recomputing.
#
#   Matrix: nspec in {2,7} x mode in {baseline(no offload), offload(on)}.
#   Real block-verify (honest acceptance); mandated config unchanged
#   (gpu-mem 0.95, seqs 64, MNBT 16384, 34 GiB KV pin, FULL_AND_PIECEWISE).
#
#   bash _run_offload_ab_fixed68k.sh
#   NSPEC_LIST="2" MODES="offload" bash _run_offload_ab_fixed68k.sh   # subset
#
#   Results: /workspace/k3_fixed68k_n{N}_{mode}/concurrency_{c}__requests_{r}/
# =============================================================================
set -uo pipefail

CONTAINER="${CONTAINER:-k3-dspark-benchmark}"
PORT="${PORT:-8893}"
NSPEC_LIST="${NSPEC_LIST:-2 7}"
MODES="${MODES:-baseline offload}"
# ATOM ladder, high->low so the prefix cache is populated/evicting from the start.
CONC_LIST="${CONC_LIST:-48 32 24 16 12 8 4 2 1}"
REQ_LIST="${REQ_LIST:-240 160 120 80 60 40 20 10 5}"     # zipped with CONC_LIST
# Prefix pool: 64 distinct 63.9k prefixes ~= 4M tokens > ~2M device pin -> eviction.
PREFIX_POOL="${PREFIX_POOL:-64}"
PREFIX_LEN="${PREFIX_LEN:-63911}"
SYN_ISL="${SYN_ISL:-4089}"                                # per-request suffix -> ~68k ISL
OSL="${OSL:-350}"
WARMUP="${WARMUP:-16}"
# Mandated DSpark config (do NOT edit).
GPU_MEM="${GPU_MEM:-0.95}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MNBT="${MNBT:-16384}"
# Pool size (offload knob). Capacity is NOT the blocker: colleague gets live reads +
# E2E gains on DSpark with a 64 GiB pool (221k-tok working set, GPU KV cap 163,840).
# The old 256 GiB "thrash" (891 GB written) was an artifact of THIS script's oversized
# 64x63.9k (~4M tok) torture shape, not a real regime. Match colleague at 64 GiB; the
# real fix is the eagle prefix-veto patch, not more pool. See memory
# k3-offload-read-path-dead-fixed68k. (For a clean test use _validate_offload_colleague.sh.)
KV_OFFLOADING_SIZE_ON="${KV_OFFLOADING_SIZE_ON:-64}"
KV_OFFLOADING_BACKEND="${KV_OFFLOADING_BACKEND:-native}"
AIPERF="${AIPERF:-/workspace/.aiperf_v1_0_1/bin/aiperf}"  # 0.12.0
MODEL="${MODEL:-Kimi-K3}"

log() { echo "[$(date +%T)] $*"; }

# max "Used Memory" across ALL 8 GPUs, GiB (leftover VRAM often sits on a non-0
# rank, so a head -1 GPU0 check can falsely report "drained").
_vram_max_gib() {
  docker exec "$CONTAINER" bash -lc "rocm-smi --showmeminfo vram 2>/dev/null | grep -i 'Used Memory' | awk '{g=\$NF/1073741824; if (g>m) m=g} END{printf \"%.0f\", m}'" 2>/dev/null || echo 999
}
# live vLLM proc count; self-exclude the matching shell via $$.
_vllm_procs() {
  # Bracket-regex ([v]llm) so this counting shell doesn't self-match; then RE-READ each
  # candidate's /proc/cmdline and match an UNBRACKETED token. That excludes (a) the
  # permanent [vllm] <defunct> zombies (empty cmdline; PID1 `sleep infinity` never reaps)
  # and (b) the transient $(pgrep) command-substitution subshell (its cmdline carries the
  # bracketed pattern, not "vllm") -> race-proof, so a clean serve never reads as live.
  docker exec "$CONTAINER" bash -lc 'me=$$; n=0; for p in $(pgrep -f "[v]llm|[E]ngineCore|[V]llmWorker|[m]ultiprocessing.spawn" 2>/dev/null); do [ "$p" = "$me" ] && continue; cl=$(cat /proc/$p/cmdline 2>/dev/null | tr "\0" " "); case "$cl" in *vllm*|*EngineCore*|*VllmWorker*|*multiprocessing.spawn*) n=$((n+1)) ;; esac; done; echo $n' 2>/dev/null || echo 0
}
kill_serve() {
  # GRACEFUL drain first (SIGTERM -> wait -> SIGKILL fallback). A hard SIGKILL of a
  # native-KV-offload serve orphans pinned GPU staging buffers the ROCm driver won't
  # reclaim (~30 GiB/GPU) -> only a reboot clears it. SIGTERM lets vLLM free them.
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
      [ "${used:-999}" -le 20 ] 2>/dev/null && { log "VRAM drained after SIGKILL (${used} GiB)"; break; }
    done
  fi
  free_offload_mmaps
}
free_offload_mmaps() {
  docker exec "$CONTAINER" bash -lc 'rm -f /dev/shm/vllm_offload_*.mmap 2>/dev/null' 2>/dev/null || true
}

# Offload-read validation gate (memory: k3-offload-eagle-annotation-52047).
check_offload_reads() {
  local m
  m=$(docker exec "$CONTAINER" bash -lc "curl -s http://localhost:${PORT}/metrics 2>/dev/null" 2>/dev/null || echo "")
  [ -z "$m" ] && { log "  [gate] WARN: /metrics unreachable"; return 0; }
  # NOTE: match *_total ONLY. The *_created siblings hold a unix timestamp (~1.79e9),
  # so a loose grep sums the clock and fabricates a huge "hits" number (bug fixed 08-13).
  # The true offload-read signal is bytes moved CPU->GPU + prompt tokens sourced from
  # external_kv_transfer; store bytes prove only that the WRITE path fired.
  local hits load store
  hits=$(printf '%s\n' "$m"  | grep -E 'external_prefix_cache_hits_total' | grep -v '#' | awk '{s+=$NF} END{printf "%.0f", s}')
  load=$(printf '%s\n' "$m"  | grep -E 'kv_offload_total_bytes_total\{.*CPU_to_GPU|prompt_tokens_by_source_total\{.*external_kv_transfer' | grep -v '#' | awk '{s+=$NF} END{printf "%.0f", s}')
  store=$(printf '%s\n' "$m" | grep -E 'kv_offload_store_bytes_total' | grep -v '#' | awk '{s+=$NF} END{printf "%.0f", s}')
  log "  [gate] external_hits=${hits:-0}  offload_load_bytes=${load:-0}  offload_store_bytes=${store:-0}"
  if [ "${hits:-0}" != "0" ] && [ "${load:-0}" != "0" ]; then
    log "  [gate] PASS: offload READ path live (#52047 working)"
  else
    log "  [gate] WARN: offload reads SUPPRESSED (hits or load == 0)"
  fi
}

# Resolve local K3 tokenizer snapshot (offline/portable), inside the container.
TOKENIZER=$(docker exec "$CONTAINER" bash -lc 'ls -d /dev/shm/hf-cache/models--moonshotai--Kimi-K3/snapshots/*/ 2>/dev/null | head -1' | tr -d '\r')
TOKENIZER="${TOKENIZER%/}"
[ -n "$TOKENIZER" ] || { log "!! K3 tokenizer snapshot not found"; exit 1; }
log "tokenizer=$TOKENIZER"

# #52047 label (offload only benefits when the fix is present).
if docker exec "$CONTAINER" bash -lc 'grep -q _annotate_eagle_groups_from_draft_spec /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py' 2>/dev/null; then
  log "vLLM #52047 draft-group annotation: PRESENT (offload arm will benefit)"
else
  log "vLLM #52047 draft-group annotation: ABSENT"
fi

read -r -a CONCS <<< "$CONC_LIST"
read -r -a REQS  <<< "$REQ_LIST"
[ "${#CONCS[@]}" -eq "${#REQS[@]}" ] || { log "!! CONC_LIST and REQ_LIST length mismatch"; exit 1; }

for n in $NSPEC_LIST; do
  for mode in $MODES; do
    if [ "$mode" = "offload" ]; then OFF="$KV_OFFLOADING_SIZE_ON"; else OFF=""; fi
    TAG="fixed68k_n${n}_${mode}"
    log "########## nspec=$n mode=$mode (offload='${OFF:-off}') ##########"
    kill_serve
    log "cold serve: nspec=$n, real block-verify, seqs=$MAX_NUM_SEQS, gpu_mem=$GPU_MEM, MNBT=$MNBT, port=$PORT"
    if ! docker exec \
        -e NUM_SPEC="$n" -e PORT="$PORT" -e GPU_MEM="$GPU_MEM" \
        -e MAX_NUM_SEQS="$MAX_NUM_SEQS" -e MNBT="$MNBT" \
        -e KV_OFFLOADING_SIZE="$OFF" -e KV_OFFLOADING_BACKEND="$KV_OFFLOADING_BACKEND" \
        "$CONTAINER" bash /workspace/_serve_k3_bench_spec.sh; then
      log "ERROR: serve failed for nspec=$n mode=$mode — skipping"
      docker exec "$CONTAINER" bash -lc "tail -40 /workspace/serve_k3_bench_spec${n}.log" || true
      continue
    fi
    OUT_ROOT="/workspace/k3_${TAG}"
    docker exec "$CONTAINER" bash -lc "mkdir -p '$OUT_ROOT'"
    for i in "${!CONCS[@]}"; do
      c="${CONCS[$i]}"; r="${REQS[$i]}"
      out="$OUT_ROOT/concurrency_${c}__requests_${r}"
      log "  point conc=$c reqs=$r  (prefix_pool=$PREFIX_POOL x ${PREFIX_LEN}tok, ISL~$((PREFIX_LEN+SYN_ISL)), OSL=$OSL)"
      docker exec "$CONTAINER" bash -lc "
        rm -rf '$out'; mkdir -p '$out'
        '$AIPERF' profile \
          --model '$MODEL' --tokenizer '$TOKENIZER' --tokenizer-trust-remote-code \
          --url 'http://localhost:$PORT' --endpoint /v1/chat/completions --endpoint-type chat --streaming \
          --use-server-token-count \
          --num-prefix-prompts $PREFIX_POOL --prompt-prefix-length $PREFIX_LEN \
          --synthetic-input-tokens-mean $SYN_ISL --synthetic-input-tokens-stddev 0 \
          --output-tokens-mean $OSL --output-tokens-stddev 0 \
          --extra-inputs ignore_eos:true --extra-inputs min_tokens:$OSL --extra-inputs max_tokens:$OSL \
          --warmup-request-count $WARMUP \
          --concurrency $c --request-count $r \
          --random-seed 42 --ui simple --no-gpu-telemetry \
          --output-artifact-dir '$out'
      " > "/tmp/aiperf_${TAG}_c${c}.log" 2>&1 || log "  WARN: aiperf conc=$c returned non-zero (see /tmp/aiperf_${TAG}_c${c}.log)"
    done
    # Snapshot spec-decode + offload counters for acceptance / gate.
    docker exec "$CONTAINER" bash -lc "curl -s http://localhost:$PORT/metrics 2>/dev/null > '$OUT_ROOT/metrics_final.txt'" || true
    [ -n "$OFF" ] && check_offload_reads
    log "nspec=$n mode=$mode complete -> $OUT_ROOT"
  done
done

kill_serve
log "########## fixed-68k A/B COMPLETE ##########"
log "Artifacts: /workspace/k3_fixed68k_n{${NSPEC_LIST// /,}}_{${MODES// /,}}/"
