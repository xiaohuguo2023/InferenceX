#!/usr/bin/env bash
# Tune shard-2 "killer" large-M shapes (4096/32768 x 1024 x 4096) on a clean node.
# Uses hipblaslt+flydsl+skinny only — skips opus/asm that triggered GPU faults on m-061.
#
# Usage (compute node):
#   ./tune_s2_hard.sh          # foreground
#   ./tune_s2_hard.sh bg       # background -> tune_s2_hard.log
#
# Env: TAG AITER_SRC TUNE_BATCH (default 1 = one shape per batch)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

USER_HOME="${SPUR_USER_HOME:-/home/$(id -un)}"
export HOME="$USER_HOME"
TAG="${TAG:-k3-bf16-gemm-tune:gfx950}"
AITER_SRC="${AITER_SRC:-$USER_HOME/work/aiter}"
IN_HOST="$HERE/shards/kimik3_bf16_tuning_main_s2_hard.csv"
OUT_HOST="$HERE/shards/kimik3_bf16_tuned_main_s2_hard.csv"
LOG="${LOG:-$HERE/tune_s2_hard.log}"
# No opus/asm — large M prefill shapes faulted under full libtype on m-061.
LIBTYPE="${LIBTYPE:-flydsl,hipblaslt,skinny}"
TUNE_BATCH="${TUNE_BATCH:-1}"

prepare_aiter_src() {
  local local_aiter="/tmp/aiter_tune_s2_hard"
  echo "[hard] rsync aiter -> $local_aiter" >&2
  rm -rf "$local_aiter"
  rsync -a "${AITER_SRC}/" "$local_aiter/" --exclude .git
  echo "$local_aiter"
}

run_hard() {
  local aiter_mount
  aiter_mount="$(prepare_aiter_src)"
  echo "[hard] in=$IN_HOST out=$OUT_HOST libtype=$LIBTYPE batch=$TUNE_BATCH"
  docker run --rm \
    --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --ipc host --shm-size 16g \
    -v "$HERE:/work" \
    -v "$HERE/shards:/work/shards" \
    -v "$aiter_mount:/opt/aiter" \
    -v "$HERE/tune.sh:/usr/local/bin/tune.sh:ro" \
    -e AITER_LIVE_MOUNT=1 \
    -e TUNE_BATCH="$TUNE_BATCH" \
    -e TUNE_SHAPE_GROUPED=1 \
    -e AITER_HIPBLASLT_FAST_MAX="${AITER_HIPBLASLT_FAST_MAX:-8192}" \
    -e INPUT_CSV="/work/shards/kimik3_bf16_tuning_main_s2_hard.csv" \
    -e OUTPUT_CSV="/work/shards/kimik3_bf16_tuned_main_s2_hard.csv" \
    -e TUNE_LIBTYPE_PROFILE=safe \
    -e LIBTYPE="$LIBTYPE" \
    "$TAG"
  echo "[hard] DONE -> $OUT_HOST"
}

if [[ "${1:-}" == "bg" ]]; then
  nohup bash "$0" >>"$LOG" 2>&1 &
  echo "[hard] background PID=$!  tail -f $LOG"
else
  run_hard
fi
