#!/usr/bin/env bash
# Variant of sweep_gptoss_widegraph_mi355x.sh that DROPS the wide
# --compilation-config (compile_sizes / cudagraph_capture_sizes / max_cudagraph_capture_size)
# and lets vLLM pick default compile/capture sizes.
#
# Purpose: isolate the AITER-env effect from the wide compile/capture window.
# Compared to sweep_gptoss_widegraph_mi355x.sh this script keeps:
#   * widegraph AITER env (MOE, RMSNORM, UNIFIED_ATTENTION, FUSED_MOE_A16W4, MHA=0)
#   * --max-num-seqs 256, --async-scheduling, --block-size 64,
#     --no-enable-prefix-caching, --gpu-memory-utilization 0.95
# and removes:
#   * --compilation-config (so vLLM uses its default capture/compile sizes)
#
# Run inside xguo-comms4 container:
#   docker exec xguo-comms4 bash /home/work/InferenceX/sweep_gptoss_widegraph_default_mi355x.sh

# NOTE: do NOT set -u. benchmark_lib.sh references several env vars
# (EVAL_ONLY, RUN_EVAL, etc.) without defaults; with -u they trip immediately.
cd /home/work/InferenceX

# Defaults for benchmark_lib helpers.
export EVAL_ONLY="${EVAL_ONLY:-false}"
export RUN_EVAL="${RUN_EVAL:-false}"

source benchmarks/benchmark_lib.sh

MODEL="${MODEL:-/root/.cache/huggingface/hub/models--amd--gpt-oss-120b-w-mxfp4-a-fp8/snapshots/0e654a3aab9cf63088ffc1e0690f0067acfc4e4a}"

# --- Widegraph AITER env (verbatim from gpt_oss_mi350_serve_widegraph.sh).
export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-1}"
export VLLM_ROCM_USE_AITER_MOE="${VLLM_ROCM_USE_AITER_MOE:-1}"
export VLLM_ROCM_USE_AITER_RMSNORM="${VLLM_ROCM_USE_AITER_RMSNORM:-1}"
export VLLM_USE_AITER_UNIFIED_ATTENTION="${VLLM_USE_AITER_UNIFIED_ATTENTION:-1}"
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION="${VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION:-1}"
export VLLM_ROCM_USE_AITER_MHA="${VLLM_ROCM_USE_AITER_MHA:-0}"
export VLLM_ROCM_USE_AITER_FUSED_MOE_A16W4="${VLLM_ROCM_USE_AITER_FUSED_MOE_A16W4:-1}"
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION="${VLLM_ROCM_QUICK_REDUCE_QUANTIZATION:-INT4}"
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}"

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/home/work/.triton_cache_gpt_oss}"
mkdir -p "${TRITON_CACHE_DIR}"

MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
PORT="${PORT:-8888}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"

OUT_BASE="/workspace/sweep_widegraph_default_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT_BASE"
echo "Sweep results will be saved under $OUT_BASE"
echo "Model: $MODEL"
echo "Compilation config: <vLLM defaults>"

# Matrix matches InferenceX dashboard B200 vLLM coverage for direct comparison:
#   TP=1: CONC [4, 8, 16, 32, 64, 128]
#   TP=4: CONC [4, 8, 16, 32, 64]
#   TP=8: CONC [4, 8, 16, 32, 64]
COMBOS=(
  # ISL=1024, OSL=1024
  "1024 1024 1 4"   "1024 1024 1 8"   "1024 1024 1 16"  "1024 1024 1 32"  "1024 1024 1 64"  "1024 1024 1 128"
  "1024 1024 4 4"   "1024 1024 4 8"   "1024 1024 4 16"  "1024 1024 4 32"  "1024 1024 4 64"
  "1024 1024 8 4"   "1024 1024 8 8"   "1024 1024 8 16"  "1024 1024 8 32"  "1024 1024 8 64"
  # ISL=8192, OSL=1024
  "8192 1024 1 4"   "8192 1024 1 8"   "8192 1024 1 16"  "8192 1024 1 32"  "8192 1024 1 64"  "8192 1024 1 128"
  "8192 1024 4 4"   "8192 1024 4 8"   "8192 1024 4 16"  "8192 1024 4 32"  "8192 1024 4 64"
  "8192 1024 8 4"   "8192 1024 8 8"   "8192 1024 8 16"  "8192 1024 8 32"  "8192 1024 8 64"
)

cleanup_vllm() {
  pkill -KILL -f "vllm serve"  2>/dev/null || true
  pkill -KILL -f "VLLM::"      2>/dev/null || true
  pkill -KILL -f "gpu_monitor" 2>/dev/null || true
  sleep 5
}

run_one_combo() {
  local ISL=$1 OSL=$2 TP=$3 CONC=$4
  local MAX_MODEL_LEN=$((ISL + OSL + 256))
  local RESULT_FILENAME="gptoss_widegraph_default_mi355x_isl${ISL}_osl${OSL}_tp${TP}_conc${CONC}"
  local SERVER_LOG="/workspace/server.log"

  cleanup_vllm
  rm -f "$SERVER_LOG" /workspace/gpu_metrics.csv

  start_gpu_monitor

  set -x
  vllm serve "$MODEL" \
      --port "$PORT" \
      --max-model-len "$MAX_MODEL_LEN" \
      --tensor-parallel-size "$TP" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --gpu-memory-utilization 0.95 \
      --block-size 64 \
      --no-enable-prefix-caching \
      --async-scheduling \
      > "$SERVER_LOG" 2>&1 &
  local SERVER_PID=$!
  set +x

  wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

  run_benchmark_serving \
      --model "$MODEL" \
      --port "$PORT" \
      --backend vllm \
      --input-len "$ISL" \
      --output-len "$OSL" \
      --random-range-ratio "$RANDOM_RANGE_RATIO" \
      --num-prompts "$((CONC * 10))" \
      --max-concurrency "$CONC" \
      --result-filename "$RESULT_FILENAME" \
      --result-dir /workspace/

  stop_gpu_monitor
  cleanup_vllm

  for f in "/workspace/${RESULT_FILENAME}.json" "$SERVER_LOG" /workspace/gpu_metrics.csv; do
    [ -f "$f" ] || continue
    local base
    base=$(basename "$f")
    case "$base" in
      server.log)      mv "$f" "$OUT_BASE/${RESULT_FILENAME}.server.log" ;;
      gpu_metrics.csv) mv "$f" "$OUT_BASE/${RESULT_FILENAME}.gpu_metrics.csv" ;;
      *)               mv "$f" "$OUT_BASE/$base" ;;
    esac
  done
}

n=${#COMBOS[@]}
i=0
for combo in "${COMBOS[@]}"; do
  i=$((i+1))
  read ISL OSL TP CONC <<< "$combo"
  echo
  echo "==========================================================="
  echo "[$i/$n] ISL=$ISL OSL=$OSL TP=$TP CONC=$CONC  MAX_MODEL_LEN=$((ISL+OSL+256))"
  echo "==========================================================="
  start=$(date +%s)
  ( run_one_combo "$ISL" "$OSL" "$TP" "$CONC" ) > "$OUT_BASE/gptoss_widegraph_default_mi355x_isl${ISL}_osl${OSL}_tp${TP}_conc${CONC}.stdout" 2>&1
  rc=$?
  end=$(date +%s)
  echo "[$i/$n] exit=$rc  elapsed=$((end-start))s"
done

echo
echo "Sweep complete. Results in $OUT_BASE"
ls -la "$OUT_BASE"
