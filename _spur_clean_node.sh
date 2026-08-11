#!/usr/bin/env bash
# Hard-reset a held compute node via spur exec. Aborts if VRAM does not drain.
#
# Usage: bash _spur_clean_node.sh JOB [label]
#   MIN_FREE_GIB=250 bash _spur_clean_node.sh 50837 n249
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

JOB="${1:?usage: _spur_clean_node.sh JOB [label]}"
LABEL="${2:-job $JOB}"
export HOME="${SPUR_USER_HOME:-/home/$(id -un)}"
MIN_FREE_GIB="${MIN_FREE_GIB:-250}"
VRAM_TOTAL_B="${VRAM_TOTAL_B:-309220868096}"

echo "=== clean $LABEL (job $JOB) ==="

# spur exec must use bash -c (not -lc): login profiles break /proc and pkill.
spur exec "$JOB" bash -c "
  set -uo pipefail
  export HOME='$HOME'
  cd '$HERE'
  echo host=\$(hostname) job=$JOB
  docker exec k3-benchmark bash /workspace/_stop_bench.sh 2>/dev/null || true
  docker rm -f k3-benchmark k3-dspark-benchmark 2>/dev/null || true
  pkill -9 -f _k3_attention_bench_node.sh 2>/dev/null || true
  pkill -9 -f _pra_gsm8k_node.sh 2>/dev/null || true
  pkill -9 -f _prg_c24_node.sh 2>/dev/null || true
  pkill -9 -f _agentic_ladder.sh 2>/dev/null || true
  pkill -9 -f _fixed_arm.sh 2>/dev/null || true
  pkill -9 -f aiperf 2>/dev/null || true
  pkill -9 -f 'vllm ser[v]e' 2>/dev/null || true
  pkill -9 python3 2>/dev/null || true
  pkill -9 python 2>/dev/null || true
  sleep 10
  for i in \$(seq 1 40); do
    used=\$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/GPU\\[0\\].*Used Memory/ {print \$NF; exit}')
    if [ -n \"\$used\" ]; then
      free_gib=\$(( ($VRAM_TOTAL_B - used) / 1073741824 ))
      if [ \"\$free_gib\" -ge $MIN_FREE_GIB ]; then
        echo \"vram drained after \$((i*10))s: \${free_gib} GiB min free\"
        rocm-smi --showmeminfo vram 2>/dev/null | awk '/GPU\\[0\\].*Used/ {printf \"gpu0 used: %.1f GiB\\n\", \$NF/1073741824}'
        exit 0
      fi
    fi
    pkill -9 python3 2>/dev/null || true
    sleep 10
  done
  rocm-smi --showmeminfo vram 2>/dev/null | awk '/GPU\\[0\\].*Used/ {printf \"gpu0 used: %.1f GiB\\n\", \$NF/1073741824}'
  echo \"!! vram did NOT drain: \${free_gib:-?} GiB min free (need >= $MIN_FREE_GIB)\"
  rocm-smi --showpids 2>/dev/null | tail -15 || true
  exit 1
"
