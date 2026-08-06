#!/bin/bash
V=/usr/local/lib/python3.12/dist-packages/vllm

echo "=== flag in the running server's env ==="
pid=$(pgrep -f "vllm serve" | head -1)
echo "  server pid: ${pid:-none}"
if [ -n "$pid" ]; then
  tr '\0' '\n' < "/proc/$pid/environ" | grep -iE "ESTIMATE_CUDAGRAPHS" || echo "  NOT SET in server env"
fi

echo "=== flag in a worker's env (workers are forked children) ==="
wpid=$(pgrep -f "VLLM::Worker|Worker_TP" | head -1)
echo "  worker pid: ${wpid:-none}"
if [ -n "$wpid" ]; then
  tr '\0' '\n' < "/proc/$wpid/environ" | grep -iE "ESTIMATE_CUDAGRAPHS" || echo "  NOT SET in worker env"
fi

echo "=== envs.py definition ==="
grep -n -B2 -A6 "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS" "$V/envs.py"

echo "=== what python sees ==="
python3 -c "
import os
print('  os.environ:', os.environ.get('VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS'))
from vllm import envs
print('  envs value:', envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS)
" 2>&1 | tail -5
