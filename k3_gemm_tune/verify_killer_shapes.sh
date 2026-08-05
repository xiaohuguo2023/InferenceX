#!/usr/bin/env bash
# Validate killer shapes tune with updated safe/full profiles (no GPU fault).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
USER_HOME="${SPUR_USER_HOME:-/home/$(id -un)}"
export HOME="$USER_HOME"
TAG="${TAG:-k3-bf16-gemm-tune:gfx950}"
AITER_SRC="${AITER_SRC:-$USER_HOME/work/aiter}"
PROFILE="${1:-safe}"
OUT="/work/shards/kimik3_bf16_tuned_main_s2_hard_verify_${PROFILE}.csv"
LOG="$HERE/tune_s2_hard_verify_${PROFILE}.log"

local_aiter="/tmp/aiter_tune_hard_verify_${PROFILE}_$$"
rm -rf "$local_aiter" 2>/dev/null || sudo rm -rf "$local_aiter"
rsync -a "${AITER_SRC}/" "$local_aiter/" --exclude .git

echo "[verify] profile=$PROFILE out=$OUT" | tee "$LOG"
docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --ipc host --shm-size 16g \
  -v "$HERE:/work" \
  -v "$HERE/shards:/work/shards" \
  -v "$local_aiter:/opt/aiter" \
  -v "$HERE/tune.sh:/usr/local/bin/tune.sh:ro" \
  -e AITER_LIVE_MOUNT=1 \
  -e TUNE_BATCH=1 \
  -e TUNE_SHAPE_GROUPED=1 \
  -e TUNE_LIBTYPE_PROFILE="$PROFILE" \
  -e INPUT_CSV="/work/shards/kimik3_bf16_tuning_main_s2_hard.csv" \
  -e OUTPUT_CSV="$OUT" \
  "$TAG" \
  2>&1 | tee -a "$LOG"

echo "[verify] grep faults/opus/asm:"
grep -E "GPU Fault|Mapping Error|opus candidate|retried num|Tuning result|profile=" "$LOG" || true
wc -l "$HERE/shards/kimik3_bf16_tuned_main_s2_hard_verify_${PROFILE}.csv"
