#!/usr/bin/env bash
set -eo pipefail

# Kimi-K3 on MI355X via vLLM — serve + online-serving benchmark.
# Serve flags follow MAD's day-0 AMD recipe (benchmark/kimi_k3, vllm/vllm-openai-rocm:kimi-k3):
#   -tp 8, --moe-backend auto, --reasoning-parser kimi_k3, --language-model-only,
#   env VLLM_ROCM_USE_AITER=1 / AITER_SITUV2_A8W4=1 / AITER_BF16_FP8_MOE_BOUND=0 /
#   VLLM_USE_BREAKABLE_CUDAGRAPH=0.
# Model: moonshotai/Kimi-K3 (2.8T MoE, ~1.56 TB). Pass MODEL as a local path to skip download.

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars MODEL TP CONC ISL OSL MAX_MODEL_LEN RANDOM_RANGE_RATIO RESULT_FILENAME

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi
if [ -n "$ROCR_VISIBLE_DEVICES" ]; then export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"; fi

export VLLM_ROCM_USE_AITER=1
export AITER_SITUV2_A8W4=1
export AITER_BF16_FP8_MOE_BOUND=0
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export SAFETENSORS_FAST_GPU=1

SERVER_LOG=/workspace/server.log

# --- optional torch-profiler hook (PROFILE=1) ---
profiler_args=()
if [[ "${PROFILE:-0}" == "1" ]]; then
    export VLLM_TORCH_PROFILER_DIR="${VLLM_TORCH_PROFILER_DIR:-/workspace/kimik3_traces}"
    mkdir -p "$VLLM_TORCH_PROFILER_DIR"
    profiler_args=(--profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"$VLLM_TORCH_PROFILER_DIR\", \"torch_profiler_record_shapes\": true}")
    echo "Profiler ENABLED -> $VLLM_TORCH_PROFILER_DIR"
fi

start_gpu_monitor
set -x
vllm serve "$MODEL" --port "$PORT" \
    --dtype auto \
    --tensor-parallel-size "$TP" \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --load-format auto \
    --gpu-memory-utilization 0.95 \
    --moe-backend auto \
    --max-num-seqs "$CONC" \
    --max-num-batched-tokens 4096 \
    --max-model-len "$MAX_MODEL_LEN" \
    --reasoning-parser kimi_k3 \
    --language-model-only \
    --disable-uvicorn-access-log \
    "${profiler_args[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

run_benchmark_serving \
    --model "$MODEL" --port "$PORT" --backend vllm \
    --input-len "$ISL" --output-len "$OSL" --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 10))" --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" --result-dir /workspace/ --trust-remote-code

stop_gpu_monitor

# Wait for torch traces (one per TP worker) to finish dumping before killing the
# server — otherwise the worker traces (which hold the GPU kernels) are lost.
if [[ "${PROFILE:-0}" == "1" && -n "${VLLM_TORCH_PROFILER_DIR:-}" ]]; then
    echo "Waiting for torch traces to finish in $VLLM_TORCH_PROFILER_DIR ..."
    prev="__init__"; stable=0
    for _ in $(seq 1 180); do
        cur=$(find "$VLLM_TORCH_PROFILER_DIR" -name '*.json*' -printf '%s ' 2>/dev/null)
        if [[ -n "$cur" && "$cur" == "$prev" ]]; then stable=$((stable+1)); [[ $stable -ge 3 ]] && break; else stable=0; fi
        prev="$cur"; sleep 2
    done
    echo "Traces collected:"; find "$VLLM_TORCH_PROFILER_DIR" -name '*.json*' -printf '  %f  %s bytes\n' 2>/dev/null
fi
# Trace on disk -> safe to stop the server. Kill and WAIT until the workers
# actually release the GPU (orphaned VLLM workers otherwise hold VRAM and OOM the
# next run), mirroring gptoss_fp4_mi355x_profiling.sh's cleanup.
[[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null || true
pkill -9 -f "/usr/local/bin/vll[m]" 2>/dev/null || true
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "EngineCore" 2>/dev/null || true
pkill -9 -f "multiprocessing.spawn" 2>/dev/null || true
pkill -9 -f spawn_main 2>/dev/null || true
for _ in $(seq 1 30); do
    pgrep -f "vllm serve\|/usr/local/bin/vll[m]\|multiprocessing.spawn\|spawn_main" >/dev/null 2>&1 || break
    sleep 2
done
set +x
