#!/bin/bash
# Inspect how the nightly accounts for / allocates CUDA graph memory.
V=/usr/local/lib/python3.12/dist-packages/vllm

echo "=== vLLM version ==="
python3 -c "import vllm;print(vllm.__version__)" 2>/dev/null

echo "=== 'Graph capturing finished' site ==="
grep -rn "Graph capturing finished" "$V" 2>/dev/null | head -3

echo "=== how cudagraph memory is measured (context) ==="
f=$(grep -rln "Graph capturing finished" "$V" 2>/dev/null | head -1)
if [ -n "$f" ]; then
  n=$(grep -n "Graph capturing finished" "$f" | head -1 | cut -d: -f1)
  sed -n "$((n-30)),$((n+6))p" "$f"
fi

echo "=== graph memory pool creation ==="
grep -rn "graph_pool_handle\|graph_pool =\|self.graph_pool" "$V/compilation" "$V/v1/worker" 2>/dev/null | head -12

echo "=== empty_cache calls around capture ==="
grep -rn "empty_cache" "$V/v1/worker/gpu_model_runner.py" "$V/v1/worker/gpu_worker.py" 2>/dev/null | head -12

echo "=== CUDAGRAPH-related env vars available ==="
grep -oE "\"VLLM_[A-Z0-9_]*CUDAGRAPH[A-Z0-9_]*\"|\"VLLM_[A-Z0-9_]*GRAPH[A-Z0-9_]*\"" "$V/envs.py" 2>/dev/null | sort -u

echo "=== compilation config fields for cudagraph ==="
grep -nE "cudagraph_(mode|capture_sizes|copy_inputs|share_memory_pool|specialize)" "$V/config/compilation.py" 2>/dev/null | head -20
