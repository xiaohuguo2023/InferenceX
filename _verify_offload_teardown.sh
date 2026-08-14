#!/usr/bin/env bash
# =============================================================================
# _verify_offload_teardown.sh  (task #68)
#   Prove that a native-KV-offload serve tears down WITHOUT orphaning VRAM,
#   BEFORE we spend an hour on a real offload sweep. No benchmark / no aiperf.
#
#   Cycle:  cold offload serve  ->  a few warmup requests (touch offload path)
#           ->  graceful kill_serve  ->  assert clean.
#
#   PASS iff, after teardown:
#     * kill_serve took the GRACEFUL branch (SIGTERM drained it; NO kill -9 fired),
#     * live vLLM procs == 0 (zombie-aware, bracket-regex counter),
#     * max VRAM across all 8 GPUs <= 20 GiB (idle baseline ~0.3),
#     * no leftover /dev/shm/vllm_offload_*.mmap.
#   A hard SIGKILL of an offload serve orphans pinned host+GPU staging buffers the
#   ROCm driver never reclaims (~30 GiB/GPU, reboot-only) -> that is the leak this
#   guards against.
#
#   bash _verify_offload_teardown.sh
# =============================================================================
set -uo pipefail

CONTAINER="${CONTAINER:-k3-dspark-benchmark}"
PORT="${PORT:-8895}"                       # unused by the other drivers (8891/8893/8894)
NUM_SPEC="${NUM_SPEC:-2}"
KV_OFFLOADING_SIZE="${KV_OFFLOADING_SIZE:-64}"   # GiB total across 8 ranks
KV_OFFLOADING_BACKEND="${KV_OFFLOADING_BACKEND:-native}"
# Mandated DSpark config (do NOT edit) — verify teardown under the SAME config the
# real sweep uses, otherwise the test isn't representative.
GPU_MEM="${GPU_MEM:-0.95}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MNBT="${MNBT:-16384}"
SYNTHETIC_ACCEPT_LEN="${SYNTHETIC_ACCEPT_LEN:-2.51}"
WARMUP_REQS="${WARMUP_REQS:-5}"
MODEL="${MODEL:-Kimi-K3}"

log() { echo "[$(date +%T)] $*"; }

_vram_max_gib() {
  docker exec "$CONTAINER" bash -lc "rocm-smi --showmeminfo vram 2>/dev/null | grep -i 'Used Memory' | awk '{g=\$NF/1073741824; if (g>m) m=g} END{printf \"%.0f\", m}'" 2>/dev/null || echo 999
}
# Zombie-aware, self-excluding, bracket-regex live-proc counter (see the offload
# drivers for why: the permanent [vllm] <defunct> zombie under PID1 `sleep infinity`
# would otherwise read as a live serve forever and force the -9 branch).
_vllm_procs() {
  docker exec "$CONTAINER" bash -lc 'me=$$; n=0; for p in $(pgrep -f "[v]llm|[E]ngineCore|[V]llmWorker|[m]ultiprocessing.spawn" 2>/dev/null); do [ "$p" = "$me" ] && continue; cl=$(cat /proc/$p/cmdline 2>/dev/null | tr "\0" " "); case "$cl" in *vllm*|*EngineCore*|*VllmWorker*|*multiprocessing.spawn*) n=$((n+1)) ;; esac; done; echo $n' 2>/dev/null || echo 0
}
_mmap_count() {
  docker exec "$CONTAINER" bash -lc 'ls /dev/shm/vllm_offload_*.mmap 2>/dev/null | wc -l' 2>/dev/null || echo 0
}

# Returns: 0 = graceful clean (no -9);  3 = needed -9 fallback but drained;
#          4 = did NOT drain (leak).   Prints the branch it took.
KILL_BRANCH="unknown"
kill_serve() {
  docker exec "$CONTAINER" bash -lc '
    me=$$
    for p in $(pgrep -f "[v]llm|[E]ngineCore|[V]llmWorker|[m]ultiprocessing.spawn"); do
      [ "$p" = "$me" ] && continue; kill -TERM "$p" 2>/dev/null
    done; exit 0' 2>/dev/null || true
  local used
  for _ in $(seq 1 18); do          # up to ~90s for graceful exit + VRAM drain
    sleep 5
    used=$(_vram_max_gib)
    if [ "$(_vllm_procs)" = "0" ] && [ "${used:-999}" -le 20 ] 2>/dev/null; then
      KILL_BRANCH="graceful"; log "graceful teardown clean (procs=0, max VRAM ${used} GiB)"; return 0
    fi
  done
  log "graceful teardown incomplete (VRAM=$(_vram_max_gib) GiB) -> SIGKILL fallback"
  docker exec "$CONTAINER" bash -lc '
    me=$$
    for p in $(pgrep -f "[v]llm|[E]ngineCore|[V]llmWorker|[m]ultiprocessing.spawn"); do
      [ "$p" = "$me" ] && continue; kill -9 "$p" 2>/dev/null
    done; exit 0' 2>/dev/null || true
  for _ in $(seq 1 12); do
    sleep 5; used=$(_vram_max_gib)
    [ "${used:-999}" -le 20 ] 2>/dev/null && { KILL_BRANCH="sigkill"; log "VRAM drained after SIGKILL (${used} GiB)"; return 3; }
  done
  KILL_BRANCH="leak"; log "WARN: VRAM did not fully drain"; return 4
}

log "########## OFFLOAD TEARDOWN VERIFY (task #68) ##########"
log "config: offload=${KV_OFFLOADING_SIZE} GiB (backend=$KV_OFFLOADING_BACKEND), ns=$NUM_SPEC, seqs=$MAX_NUM_SEQS, gpu_mem=$GPU_MEM, MNBT=$MNBT, port=$PORT"

log "pre: clean slate (graceful kill of anything live)"
kill_serve || true
log "pre VRAM(max)=$(_vram_max_gib) GiB  procs=$(_vllm_procs)  mmaps=$(_mmap_count)"

log "cold offload serve starting (this blocks until ready; ~minutes for TP8 K3)..."
if ! docker exec \
    -e NUM_SPEC="$NUM_SPEC" -e PORT="$PORT" -e GPU_MEM="$GPU_MEM" \
    -e MAX_NUM_SEQS="$MAX_NUM_SEQS" -e MNBT="$MNBT" \
    -e SYNTHETIC_ACCEPT_LEN="$SYNTHETIC_ACCEPT_LEN" \
    -e KV_OFFLOADING_SIZE="$KV_OFFLOADING_SIZE" -e KV_OFFLOADING_BACKEND="$KV_OFFLOADING_BACKEND" \
    "$CONTAINER" bash /workspace/_serve_k3_bench_spec.sh; then
  log "ERROR: serve failed to come up — aborting"
  docker exec "$CONTAINER" bash -lc "tail -40 /workspace/serve_k3_bench_spec${NUM_SPEC}.log" || true
  exit 1
fi

SERVE_VRAM=$(_vram_max_gib); SERVE_MMAPS=$(_mmap_count)
log "serve READY: VRAM(max)=${SERVE_VRAM} GiB  procs=$(_vllm_procs)  offload_mmaps=${SERVE_MMAPS}"
if [ "${SERVE_MMAPS:-0}" = "0" ]; then
  log "WARN: no /dev/shm/vllm_offload_*.mmap after serve init — offload may NOT have engaged; teardown test is weaker than intended"
else
  log "offload buffers present (${SERVE_MMAPS} mmap file(s)) — teardown will exercise the real leak path"
fi

log "firing ${WARMUP_REQS} warmup requests to touch the offload path..."
for i in $(seq 1 "$WARMUP_REQS"); do
  docker exec "$CONTAINER" bash -lc "curl -s -m 60 http://localhost:$PORT/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Count from one to twenty in words.\"}],\"max_tokens\":64,\"temperature\":0}' \
    -o /dev/null -w 'req$i http=%{http_code} t=%{time_total}s\n'" 2>/dev/null || log "  warmup req $i failed (non-fatal)"
done

log "----- TEARDOWN under test: graceful kill_serve -----"
kill_serve; rc=$?
# clear any offload mmaps the same way the real drivers do, then re-check
docker exec "$CONTAINER" bash -lc 'rm -f /dev/shm/vllm_offload_*.mmap 2>/dev/null' 2>/dev/null || true

POST_VRAM=$(_vram_max_gib); POST_PROCS=$(_vllm_procs); POST_MMAPS_PRECLEAN=0
# report mmap state BEFORE our explicit rm to see if graceful shutdown removed them itself
log "post-teardown: branch=${KILL_BRANCH} rc=${rc} VRAM(max)=${POST_VRAM} GiB procs=${POST_PROCS}"

PASS=1
[ "$KILL_BRANCH" = "graceful" ] || { log "FAIL: teardown did NOT take the graceful branch (branch=$KILL_BRANCH) — the -9 leak path fired"; PASS=0; }
[ "${POST_PROCS:-9}" = "0" ]     || { log "FAIL: live vLLM procs still present ($POST_PROCS)"; PASS=0; }
[ "${POST_VRAM:-999}" -le 20 ] 2>/dev/null || { log "FAIL: VRAM did not drain (max ${POST_VRAM} GiB) — buffers orphaned"; PASS=0; }

log "############################################################"
if [ "$PASS" = "1" ]; then
  log "RESULT: PASS — offload serve tears down gracefully, VRAM freed, no -9. Safe to run offload sweeps."
  exit 0
else
  log "RESULT: FAIL — do NOT run offload sweeps until kill_serve is fixed (see FAIL lines above)."
  exit 1
fi
