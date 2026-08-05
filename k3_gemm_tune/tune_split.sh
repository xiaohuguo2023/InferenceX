#!/usr/bin/env bash
# Two-phase GEMM tuning to avoid the N=896,K=7168 hipBLASLt explosion.
#
# Root cause: bf16gemm_bf16_tn_256x256 asm rejects N=896 (896%256!=0), then hipBLASLt
# autotune fans out to ~239k task groups and GPU-memory-faults all GPUs. The worker
# pool hangs forever ("Waiting for 239631 tasks") and gemm_tuner.py never retries.
#
# Phase 1: all shapes EXCEPT (N=896,K=7168) — full libtype + hipBLASLt
# Phase 2: N=896,K=7168 only — flydsl,hipblaslt,skinny,opus (no asm)
#
# Usage (on compute node):
#   ./tune_split.sh            # foreground
#   ./tune_split.sh bg         # background -> tune_split.log

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

USER_HOME="${SPUR_USER_HOME:-/home/$(id -un)}"
export HOME="$USER_HOME"
SRC="${CSV:-$USER_HOME/work/InferenceX/kimik3_bf16_tuning_gemm_v2.csv}"
TAG="${TAG:-k3-bf16-gemm-tune:gfx950}"
AITER_SRC="${AITER_SRC:-$USER_HOME/work/aiter}"
LOG="${LOG:-$HERE/tune_split.log}"

MAIN_CSV="$HERE/kimik3_bf16_tuning_main.csv"
N896_CSV="$HERE/kimik3_bf16_tuning_n896.csv"
OUT_MAIN="$HERE/kimik3_bf16_tuned_main.csv"
OUT_N896="$HERE/kimik3_bf16_tuned_n896.csv"
OUT_FINAL="$HERE/kimik3_bf16_tuned_gemm.csv"

split_csvs() {
  echo "[split] input=$SRC"
  {
    echo "M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle"
    awk -F, 'NR>1 && !($2==896 && $3==7168)' "$SRC"
  } > "$MAIN_CSV"
  {
    echo "M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle"
    awk -F, 'NR>1 && ($2==896 && $3==7168)' "$SRC"
  } > "$N896_CSV"
  echo "[split] main=$(($(wc -l < "$MAIN_CSV")-1)) shapes  n896=$(($(wc -l < "$N896_CSV")-1)) shapes"
}

docker_tune() {
  local in_csv="$1" out_csv="$2" profile="$3" log_tag="$4"
  echo "[tune:$log_tag] in=$in_csv out=$out_csv profile=$profile"
  docker run --rm \
    --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --ipc host --shm-size 16g \
    -v "$HERE:/work" \
    -v "$AITER_SRC:/opt/aiter" \
    -v "$HERE/tune.sh:/usr/local/bin/tune.sh:ro" \
    -e AITER_LIVE_MOUNT=1 \
    -e TUNE_BATCH="${TUNE_BATCH:-10}" \
    -e TUNE_SHAPE_GROUPED="${TUNE_SHAPE_GROUPED:-1}" \
    -e AITER_HIPBLASLT_FAST_MAX="${AITER_HIPBLASLT_FAST_MAX:-8192}" \
    -e TUNE_LIBTYPE_PROFILE="$profile" \
    -e INPUT_CSV="/work/$(basename "$in_csv")" \
    -e OUTPUT_CSV="/work/$(basename "$out_csv")" \
    "$TAG"
}

merge_outputs() {
  {
    echo "M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle,libtype,solidx,splitK,us,kernelName,err_ratio,tflops,bw"
    tail -n +2 "$OUT_MAIN" 2>/dev/null || true
    tail -n +2 "$OUT_N896" 2>/dev/null || true
  } > "$OUT_FINAL"
  echo "[merge] -> $OUT_FINAL ($(($(wc -l < "$OUT_FINAL")-1)) rows)"
}

run_all() {
  split_csvs
  docker_tune "$MAIN_CSV" "$OUT_MAIN" "safe" "main"
  docker_tune "$N896_CSV" "$OUT_N896" "n896" "n896"
  merge_outputs
  echo "[DONE] install: cp $OUT_FINAL ~/work/aiter/aiter/configs/model_configs/kimik3_bf16_tuned_gemm.csv"
}

if [[ "${1:-}" == "bg" ]]; then
  nohup bash "$0" >>"$LOG" 2>&1 &
  echo "[split] background PID=$!  tail -f $LOG"
else
  run_all
fi
