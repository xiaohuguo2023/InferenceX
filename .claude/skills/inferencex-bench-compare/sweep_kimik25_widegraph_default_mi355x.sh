#!/usr/bin/env bash
# Kimi-K2.5 (MXFP4) sweep on MI355X, widegraph-default style.
#
# Adapted from sweep_gptoss_widegraph_default_mi355x.sh for Kimi-K2.5:
#   * model:  amd/Kimi-K2.5-MXFP4 (DeepseekV3 backbone, MLA attention,
#             MXFP4 MoE weights+activations)
#   * MoE: keep AITER MOE on, but no FUSED_MOE_A16W4 (this is MXFP4 not A16W4)
#   * MLA attention: force VLLM_ATTENTION_BACKEND=TRITON_MLA per AMD recipe
#   * RMSNORM AITER=0 per AMD/vLLM Kimi-K2.5 recipe
#   * fused shared experts off (VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0)
#   * Kimi parsers: --reasoning-parser kimi_k2 --tool-call-parser kimi_k2
#   * --trust-remote-code (Kimi config has remote code)
#   * NO --compilation-config (vLLM defaults; widegraph-default style)
#   * NO --enforce-eager (we want HIP graphs for steady-state perf)
#
# Matrix matches InferenceX dashboard MI355X vLLM Kimi-K2.5 fp4 coverage
# (cfg 672 = TP4, cfg 603 = TP8) for direct comparison:
#   TP=4 and TP=8, ISL/OSL in {(1k,1k),(1k,8k),(8k,1k)}, CONC in [4,8,16,32,64].
#
# WARNING: vllm-project/vllm#36337 reports Kimi-K2.5-MXFP4 produces gibberish
# on MI350X with ROCm 7.2 + vLLM 0.17.0. Our v0.17 image is exactly that combo,
# so output text may be garbage even though latency/throughput numbers are
# meaningful. Prefer running this in a NEWER vLLM image (e.g. nightly) for
# accuracy; v0.17 only for direct dashboard apples-to-apples.
#
# Run inside a container with /data and the InferenceX checkout mounted:
#   docker run -d --rm --name xguo-kimi \
#     --device /dev/kfd --device /dev/dri --ipc host --network host \
#     -v /home/xiaohugu:/home -v /data:/data \
#     -v /home/xiaohugu/work/sweep_v017_output:/workspace \
#     vllm/vllm-openai-rocm:v0.17.0 sleep infinity
#   docker exec xguo-kimi bash /home/work/InferenceX/sweep_kimik25_widegraph_default_mi355x.sh

# NOTE: do NOT set -u. benchmark_lib.sh references several env vars
# (EVAL_ONLY, RUN_EVAL, etc.) without defaults; with -u they trip immediately.
cd /home/work/InferenceX

# Defaults for benchmark_lib helpers.
export EVAL_ONLY="${EVAL_ONLY:-false}"
export RUN_EVAL="${RUN_EVAL:-false}"

source benchmarks/benchmark_lib.sh

MODEL="${MODEL:-/data/amd/Kimi-K2.5-MXFP4}"

# --- Widegraph AITER env, Kimi-K2.5 flavor.
export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-1}"
export VLLM_ROCM_USE_AITER_MOE="${VLLM_ROCM_USE_AITER_MOE:-1}"
# Per AMD/vLLM Kimi-K2.5 recipe: RMSNORM AITER off.
export VLLM_ROCM_USE_AITER_RMSNORM="${VLLM_ROCM_USE_AITER_RMSNORM:-0}"
# MLA attention backend (DeepseekV3 backbone).
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_MLA}"
# Keep unified attention bits in sync with widegraph-default style.
export VLLM_USE_AITER_UNIFIED_ATTENTION="${VLLM_USE_AITER_UNIFIED_ATTENTION:-1}"
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION="${VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION:-1}"
export VLLM_ROCM_USE_AITER_MHA="${VLLM_ROCM_USE_AITER_MHA:-0}"
# Per Kimi-K2-Thinking-MXFP4 README: shared-experts fusion off.
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS="${VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS:-0}"
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION="${VLLM_ROCM_QUICK_REDUCE_QUANTIZATION:-INT4}"
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}"

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/home/work/.triton_cache_kimik25}"
mkdir -p "${TRITON_CACHE_DIR}"

MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
PORT="${PORT:-8891}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"

OUT_BASE="${OUT_BASE:-/workspace/sweep_kimik25_widegraph_default_$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT_BASE"
echo "Sweep results will be saved under $OUT_BASE"
echo "Model: $MODEL"
echo "Compilation config: <vLLM defaults>"

# Combos: "ISL OSL TP CONC"
# Matches dashboard MI355x vllm kimik2.5 fp4 cfgs 672 (TP=4) + 603 (TP=8).
COMBOS=(
  # ISL=1024, OSL=1024
  "1024 1024 4 4"  "1024 1024 4 8"  "1024 1024 4 16" "1024 1024 4 32" "1024 1024 4 64"
  "1024 1024 8 4"  "1024 1024 8 8"  "1024 1024 8 16" "1024 1024 8 32" "1024 1024 8 64"
  # ISL=1024, OSL=8192
  "1024 8192 4 4"  "1024 8192 4 8"  "1024 8192 4 16" "1024 8192 4 32" "1024 8192 4 64"
  "1024 8192 8 4"  "1024 8192 8 8"  "1024 8192 8 16" "1024 8192 8 32" "1024 8192 8 64"
  # ISL=8192, OSL=1024
  "8192 1024 4 4"  "8192 1024 4 8"  "8192 1024 4 16" "8192 1024 4 32" "8192 1024 4 64"
  "8192 1024 8 4"  "8192 1024 8 8"  "8192 1024 8 16" "8192 1024 8 32" "8192 1024 8 64"
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
  local RESULT_FILENAME="kimik25_widegraph_default_mi355x_isl${ISL}_osl${OSL}_tp${TP}_conc${CONC}"
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
      --gpu-memory-utilization 0.90 \
      --reasoning-parser kimi_k2 \
      --tool-call-parser kimi_k2 \
      --no-enable-prefix-caching \
      --async-scheduling \
      --trust-remote-code \
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
      --result-dir /workspace/ \
      --trust-remote-code

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

# Optional filters: comma-separated values to keep. Empty/unset = no filter.
#   TP_FILTER=8                    keep only TP=8 combos
#   CONC_FILTER=4,16               keep only CONC=4 and CONC=16
#   ISL_OSL_FILTER=8192/1024       keep only that ISL/OSL pair (slash-separated)
filter_combos() {
  local key="$1" filter="$2" idx="$3"
  if [ -z "$filter" ]; then return; fi
  local FILTERED=()
  for combo in "${COMBOS[@]}"; do
    read _ISL _OSL _TP _CONC <<< "$combo"
    local fields=("$_ISL" "$_OSL" "$_TP" "$_CONC")
    local val
    if [ "$idx" = "isl_osl" ]; then val="${_ISL}/${_OSL}"; else val="${fields[$idx]}"; fi
    case ",${filter}," in *",${val},"*) FILTERED+=("$combo") ;; esac
  done
  COMBOS=("${FILTERED[@]}")
  echo "$key=${filter} -> ${#COMBOS[@]} combos remain"
}
filter_combos TP_FILTER       "${TP_FILTER:-}"       2
filter_combos CONC_FILTER     "${CONC_FILTER:-}"     3
filter_combos ISL_OSL_FILTER  "${ISL_OSL_FILTER:-}"  isl_osl

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
  ( run_one_combo "$ISL" "$OSL" "$TP" "$CONC" ) 2>&1 | tee "$OUT_BASE/kimik25_widegraph_default_mi355x_isl${ISL}_osl${OSL}_tp${TP}_conc${CONC}.stdout"
  rc=${PIPESTATUS[0]}
  end=$(date +%s)
  echo "[$i/$n] exit=$rc  elapsed=$((end-start))s"
done

echo
echo "Sweep complete. Results in $OUT_BASE"
ls -la "$OUT_BASE"
