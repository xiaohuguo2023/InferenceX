#!/bin/bash
# Stop a benchmark cleanly: aiperf client first, then the VRAM sampler, then the
# vLLM server. Run inside the benchmark container.
#
# Order matters: killing the server first makes aiperf log a wall of connection
# errors and can leave it retrying past the sweep's own bookkeeping.

stop() {
  local label="$1" pattern="$2"
  local pids
  pids=$(pgrep -f "$pattern" 2>/dev/null | tr '\n' ' ')
  if [ -z "$pids" ]; then
    echo "[stop] $label: not running"
    return
  fi
  echo "[stop] $label: TERM -> $pids"
  pkill -TERM -f "$pattern" 2>/dev/null
  for _ in $(seq 1 20); do
    pgrep -f "$pattern" >/dev/null 2>&1 || { echo "[stop] $label: stopped"; return; }
    sleep 1
  done
  echo "[stop] $label: still up after 20s, KILL"
  pkill -KILL -f "$pattern" 2>/dev/null
}

stop "aiperf client" "aiperf"
stop "VRAM sampler" "rocm-smi.*--showmeminfo|vram_sampler"
stop "vLLM server" "vllm serve"

echo "[stop] remaining GPU processes:"
rocm-smi --showpids 2>/dev/null | tail -12 || echo "  (rocm-smi unavailable)"
