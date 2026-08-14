#!/usr/bin/env bash
# free_gpu.sh — reliably stop a vLLM serve and free MI355X (gfx950) VRAM.
#
# Why not `pkill -f "vllm serve"`? It only matches the launcher bash. The procs
# that actually HOLD VRAM are the forked children, with different cmdlines:
#   - APIServer     (comm is literally "vllm")
#   - 8 TP workers  (python -c "from multiprocessing.spawn import spawn_main ...")
#   - EngineCore, resource_tracker
# So after `pkill -f "vllm serve"` the workers survive, VRAM stays ~263-283 GiB,
# and the next serve dies with "Free memory on device cuda:N (~0/287 GiB) ...
# less than gpu_memory_utilization". You MUST stop the children AND wait for the
# VRAM to actually drain (it lags the exit by several seconds).
#
# GRACEFUL-DRAIN-FIRST (offload-safe). SIGKILL is NOT safe for a native KV-offload
# serve: the benchmark container's init is `sleep infinity`, which never reaps
# children, so a SIGKILL'd offload worker becomes a zombie that PINS ~237 GiB/GPU
# in the KFD driver until reaped (accumulates; not reset/reboot-clearable). So:
#   1. Always try SIGTERM graceful drain first (wait for procs=0, zombies=0, VRAM low).
#   2. If that doesn't fully drain AND this is an OFFLOAD serve -> do NOT SIGKILL.
#      Recover by `docker restart` (reaps the zombies) then wait for the KFD async
#      GC (2-5 min). This needs host context, so run from-host for offload serves.
#   3. Only for a NON-offload serve does graceful failure fall back to SIGKILL.
# Force the old hard path on a non-offload serve with FORCE_HARD=1 (refused if an
# offload serve is detected).
#
# Usage:
#   INSIDE a container:   bash free_gpu.sh
#   FROM the host:        bash free_gpu.sh <container-name>     # e.g. k3-dspark-benchmark
#     (from-host is REQUIRED for the offload docker-restart recovery to work)
#
# Env knobs:
#   FORCE_HARD=1     skip graceful drain, SIGKILL immediately (non-offload only)
#   DRAIN_MAX_GIB    "drained" threshold in GiB (default 17)
#   GRACE_TRIES      graceful SIGTERM polls x5s (default 60 = 5 min)
#   KFD_TRIES        post-restart KFD-GC polls x5s (default 60 = 5 min)
#
# NOTE: MI355X GPUs are SHARED across containers. If VRAM won't drain here, a
# serve in a SIBLING container is holding it — run this in that container too.
set -uo pipefail

FORCE_HARD="${FORCE_HARD:-0}"
DRAIN_MAX_GIB="${DRAIN_MAX_GIB:-17}"
GRACE_TRIES="${GRACE_TRIES:-60}"
KFD_TRIES="${KFD_TRIES:-60}"

# Exit codes: 0 = GPU clear; 1 = failure; 2 = offload drain incomplete, host must
# `docker restart` (used to signal the from-host wrapper below).
RC_NEEDS_RESTART=2

# ---- from-host mode: copy self into the named container, drive it, then (only
# ---- if the in-container graceful drain signals RC_NEEDS_RESTART) restart it ---
if [ -n "${1:-}" ] && [ ! -f /.dockerenv ]; then
  ctr="$1"
  echo ">> freeing GPU inside container: $ctr"
  docker cp "$0" "$ctr:/tmp/free_gpu.sh" >/dev/null
  host_vram() {
    docker exec "$ctr" bash -lc \
      "rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used Memory/{v=\$NF/1073741824; if(v>m)m=v} END{printf \"%.0f\", m+0}'" \
      2>/dev/null
  }
  if docker exec \
        -e FORCE_HARD="$FORCE_HARD" -e DRAIN_MAX_GIB="$DRAIN_MAX_GIB" \
        -e GRACE_TRIES="$GRACE_TRIES" \
        "$ctr" bash /tmp/free_gpu.sh; then
    exit 0
  fi
  rc=$?
  if [ "$rc" != "$RC_NEEDS_RESTART" ]; then exit "$rc"; fi
  # Offload serve wouldn't fully drain — reap zombies via restart, NEVER SIGKILL.
  echo ">> offload serve did not drain; docker restart $ctr (reaps zombies, no SIGKILL) ..."
  docker restart "$ctr" >/dev/null 2>&1 || { echo "!! docker restart failed"; exit 1; }
  echo ">> waiting for KFD async GC to reclaim VRAM (lags restart by minutes) ..."
  for _ in $(seq 1 "$KFD_TRIES"); do
    sleep 5
    u="$(host_vram)"; u="${u:-99}"
    echo "   max used across GPUs: ${u} GiB"
    if [ "$u" -le "$DRAIN_MAX_GIB" ] 2>/dev/null; then
      echo ">> GPU clear (${u} GiB used) after restart. Safe to relaunch."
      exit 0
    fi
  done
  echo "!! VRAM still high after restart+wait — check a SIBLING container (shared GPUs)."
  exit 1
fi

# ============================ in-container (or bare host) =====================

vram_used_gib() {
  rocm-smi --showmeminfo vram 2>/dev/null \
    | awk '/Used Memory/{v=$NF/1073741824; if(v>m)m=v} END{printf "%.0f", m+0}'
}
live_procs() {
  { pgrep -x vllm; pgrep -f "EngineCore|VllmWorker|multiprocessing.spawn|multiprocessing-fork|resource_tracker"; } \
    2>/dev/null | sort -u | wc -l
}
# Zombies matter for the offload case: a defunct offload worker under the
# sleep-infinity init keeps pinning KFD VRAM until reaped. Gate "clean" on 0.
zombies() { ps -eo stat 2>/dev/null | grep -c '^Z'; }
# Detect a native KV-offload serve from the live serve cmdline (do this BEFORE
# stopping anything — after the procs die the flag is gone).
is_offload() {
  pgrep -af "vllm|EngineCore|VllmWorker" 2>/dev/null \
    | grep -qiE 'kv[-_]offloading|kv_connector|offloadingconnector|simplecpuoffload'
}

drained() {
  local u; u="$(vram_used_gib)"; u="${u:-99}"
  [ "$(live_procs)" -eq 0 ] && [ "$(zombies)" -eq 0 ] && [ "$u" -le "$DRAIN_MAX_GIB" ] 2>/dev/null
}

OFFLOAD=0; is_offload && OFFLOAD=1
echo ">> serve type: $([ "$OFFLOAD" = 1 ] && echo 'native KV-offload (SIGKILL-unsafe)' || echo 'non-offload')"

# Already clean? (also the no-op fast path when nothing is running)
if drained; then
  echo ">> GPU already clear ($(vram_used_gib) GiB used). Nothing to do."
  exit 0
fi

# ---- 1) graceful SIGTERM drain (always tried first) -------------------------
term_round() {
  pkill -TERM -x vllm 2>/dev/null
  for pat in EngineCore VllmWorker "multiprocessing.spawn" "multiprocessing-fork" resource_tracker; do
    pkill -TERM -f "$pat" 2>/dev/null
  done
}
if [ "$FORCE_HARD" = 1 ] && [ "$OFFLOAD" = 1 ]; then
  echo "!! FORCE_HARD=1 refused: this is an OFFLOAD serve (SIGKILL would zombie-pin VRAM)."
  echo "!! Using graceful drain instead."
  FORCE_HARD=0
fi

if [ "$FORCE_HARD" != 1 ]; then
  echo ">> graceful drain: SIGTERM to vLLM procs, waiting up to $((GRACE_TRIES*5))s ..."
  term_round
  for _ in $(seq 1 "$GRACE_TRIES"); do
    sleep 5
    if drained; then
      echo ">> graceful teardown clean (procs=0 zombies=0 VRAM=$(vram_used_gib) GiB). Safe to relaunch."
      exit 0
    fi
  done
  echo "!! graceful drain incomplete after $((GRACE_TRIES*5))s (VRAM=$(vram_used_gib) GiB, zombies=$(zombies))."
fi

# ---- 2) graceful failed ------------------------------------------------------
if [ "$OFFLOAD" = 1 ]; then
  # NEVER SIGKILL an offload serve. Signal the from-host wrapper to docker restart;
  # if we were run bare/in-container, tell the user how to recover.
  echo "!! OFFLOAD serve — NOT SIGKILLing (would zombie-pin ~237 GiB/GPU)."
  if [ -f /.dockerenv ]; then
    echo "!! Recover from the HOST:  docker restart <this-container>   then wait 2-5 min for KFD GC."
    echo "!! (Or re-run from host so this script does it for you: bash free_gpu.sh <container>)"
  fi
  exit "$RC_NEEDS_RESTART"
fi

# ---- 3) non-offload: SIGKILL fallback ---------------------------------------
echo ">> non-offload serve: SIGKILL fallback ..."
kill_round() {
  pkill -9 -x vllm 2>/dev/null
  for pat in EngineCore VllmWorker "multiprocessing.spawn" "multiprocessing-fork" resource_tracker; do
    pkill -9 -f "$pat" 2>/dev/null
  done
}
for _ in $(seq 1 25); do
  kill_round
  sleep 2
  [ "$(live_procs)" -eq 0 ] && break
done
echo ">> waiting for VRAM to drain (lags the kill by several seconds) ..."
for _ in $(seq 1 12); do
  u="$(vram_used_gib)"; u="${u:-99}"
  echo "   max used across GPUs: ${u} GiB"
  if [ "$u" -le "$DRAIN_MAX_GIB" ] 2>/dev/null; then
    echo ">> GPU clear (${u} GiB used). Safe to relaunch."
    exit 0
  fi
  sleep 8
done
echo "!! VRAM still high after wait. Likely a serve in a SIBLING container is"
echo "!! holding it (GPUs are shared) — run this in that container too."
exit 1
