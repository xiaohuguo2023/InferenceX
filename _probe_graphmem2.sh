#!/bin/bash
V=/usr/local/lib/python3.12/dist-packages/vllm

echo "=== VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: definition ==="
grep -n -A6 "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS" "$V/envs.py" | head -30

echo
echo "=== ...and its usages ==="
grep -rn "MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS" "$V" --include=*.py | grep -v envs.py | head -6

echo
echo "=== usage context ==="
f=$(grep -rln "MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS" "$V" --include=*.py | grep -v envs.py | head -1)
if [ -n "$f" ]; then
  echo "--- $f ---"
  n=$(grep -n "MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS" "$f" | head -1 | cut -d: -f1)
  sed -n "$((n-25)),$((n+35))p" "$f"
fi

echo
echo "=== VLLM_ENABLE_CUDAGRAPH_GC definition ==="
grep -n -A6 "VLLM_ENABLE_CUDAGRAPH_GC" "$V/envs.py" | head -20

echo
echo "=== empty_cache ordering around capture (gpu_model_runner 6835-6880) ==="
sed -n '6835,6880p' "$V/v1/worker/gpu_model_runner.py"
