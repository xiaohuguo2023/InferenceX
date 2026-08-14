#!/usr/bin/env bash
# free_gpu.sh — reliably kill a vLLM serve and free MI355X (gfx950) VRAM.
#
# Why not `pkill -f "vllm serve"`? It only matches the launcher bash. The procs
# that actually HOLD VRAM are the forked children, with different cmdlines:
#   - APIServer     (comm is literally "vllm")
#   - 8 TP workers  (python -c "from multiprocessing.spawn import spawn_main ...")
#   - EngineCore, resource_tracker
# So after `pkill -f "vllm serve"` the workers survive, VRAM stays ~263-283 GiB,
# and the next serve dies with "Free memory on device cuda:N (~0/287 GiB) ...
# less than gpu_memory_utilization". You MUST kill the children AND wait for the
# VRAM to actually drain (it lags the kill by several seconds).
#
# Usage:
#   INSIDE a container:   bash free_gpu.sh
#   FROM the host:        bash free_gpu.sh <container-name>     # e.g. xguo-k3nc
#
# NOTE: MI355X GPUs are SHARED across containers. If VRAM won't drain here, a
# serve in a SIBLING container is holding it — run this in that container too.
set -uo pipefail

# ---- from-host mode: copy self into the named container and re-exec ----------
if [ -n "${1:-}" ] && [ ! -f /.dockerenv ]; then
  ctr="$1"
  echo ">> freeing GPU inside container: $ctr"
  docker cp "$0" "$ctr:/tmp/free_gpu.sh" >/dev/null
  exec docker exec "$ctr" bash /tmp/free_gpu.sh
fi

# Deliberately exact-match the APIServer (comm == "vllm") and -f match ONLY the
# child cmdlines. We never `pkill -f vllm`, which would also match this script's
# own shell (that self-kill is why an inline one-liner returns exit 137).
kill_round() {
  pkill -9 -x vllm 2>/dev/null
  for pat in EngineCore VllmWorker "multiprocessing.spawn" "multiprocessing-fork" resource_tracker; do
    pkill -9 -f "$pat" 2>/dev/null
  done
}

live_procs() {
  { pgrep -x vllm; pgrep -f "EngineCore|VllmWorker|multiprocessing.spawn|multiprocessing-fork|resource_tracker"; } \
    2>/dev/null | sort -u | wc -l
}

echo ">> killing vLLM processes ..."
for _ in $(seq 1 25); do
  kill_round
  sleep 2
  [ "$(live_procs)" -eq 0 ] && break
done
# Any remaining `vllm <defunct>` (zombie) entries are harmless: parent is gone,
# they can't be reaped and hold NO VRAM. rocm-smi is the source of truth below.

# max "Used Memory" (bytes -> GiB) across all GPUs
vram_used_gib() {
  rocm-smi --showmeminfo vram 2>/dev/null \
    | awk '/Used Memory/{v=$NF/1073741824; if(v>m)m=v} END{printf "%.0f", m+0}'
}

echo ">> waiting for VRAM to drain (lags the kill by several seconds) ..."
for _ in $(seq 1 12); do
  u="$(vram_used_gib)"; u="${u:-99}"
  echo "   max used across GPUs: ${u} GiB"
  # drained = ~0 (fully clean) up to ~17 GiB (harmless residual; 287-17 still
  # exceeds the ~253 GiB a K3 TP8 serve needs).
  if [ "$u" -le 17 ]; then
    echo ">> GPU clear (${u} GiB used). Safe to relaunch."
    exit 0
  fi
  sleep 8
done

echo "!! VRAM still high after wait. Likely a serve in a SIBLING container is"
echo "!! holding it (GPUs are shared) — run this in that container too."
exit 1
