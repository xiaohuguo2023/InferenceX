#!/usr/bin/env bash
# Run one shard of main-phase GEMM tuning (parallel multi-node).
#
# Split 481 main shapes across NUM_SHARDS nodes; each writes a disjoint CSV.
# After all shards finish, run merge_tuned_shards.sh.
#
# Usage (compute node):
#   NUM_SHARDS=4 SHARD=0 ./tune_shard.sh          # shard 0 (+ n896 on shard 0)
#   NUM_SHARDS=4 SHARD=2 ./tune_shard.sh bg
#
# From login node:
#   spur exec 33052 bash -lc 'export HOME=/home/xiaohugu; cd ~/work/InferenceX/k3_gemm_tune && NUM_SHARDS=4 SHARD=0 ./tune_shard.sh bg'
#
# Env: SHARD NUM_SHARDS TAG AITER_SRC TUNE_BATCH TUNE_SHAPE_GROUPED
#       TUNE_LIBTYPE_PROFILE=safe|n896|full  SKIP_SPLIT=1  CHECKPOINT_COMPACT=1
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

USER_HOME="${SPUR_USER_HOME:-/home/$(id -un)}"
export HOME="$USER_HOME"
SHARD="${SHARD:?set SHARD=0..NUM_SHARDS-1}"
NUM_SHARDS="${NUM_SHARDS:?set NUM_SHARDS=4 (or 8)}"
TAG="${TAG:-k3-bf16-gemm-tune:gfx950}"
AITER_SRC="${AITER_SRC:-$USER_HOME/work/aiter}"
MAIN_IN="${MAIN_IN:-$HERE/kimik3_bf16_tuning_main.csv}"
SHARD_DIR="${SHARD_DIR:-$HERE/shards}"
LOG="${LOG:-$HERE/tune_shard_s${SHARD}.log}"
TUNE_LIBTYPE_PROFILE="${TUNE_LIBTYPE_PROFILE:-safe}"

mkdir -p "$SHARD_DIR"

if [ ! -f "$MAIN_IN" ]; then
  SRC="${CSV:-$USER_HOME/work/InferenceX/kimik3_bf16_tuning_gemm_v2.csv}"
  echo "[shard] creating $MAIN_IN from $SRC"
  {
    echo "M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle"
    awk -F, 'NR>1 && !($2==896 && $3==7168)' "$SRC"
  } > "$MAIN_IN"
  {
    echo "M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle"
    awk -F, 'NR>1 && ($2==896 && $3==7168)' "$SRC"
  } > "$HERE/kimik3_bf16_tuning_n896.csv"
fi

split_main_shards() {
  if [[ "${SKIP_SPLIT:-0}" == "1" ]] && [[ -f "$IN_CSV" ]]; then
    echo "[shard] SKIP_SPLIT=1, reusing $(basename "$IN_CSV") ($(($(wc -l < "$IN_CSV") - 1)) shapes)"
    return 0
  fi
  [ -f "$MAIN_IN" ] || {
    echo "ERROR: missing $MAIN_IN — run tune_split.sh split first or copy CSV" >&2
    exit 1
  }
  local n=$(( $(wc -l < "$MAIN_IN") - 1 ))
  echo "[shard] splitting $n main shapes into $NUM_SHARDS shards -> $SHARD_DIR"
  awk -F, -v n="$NUM_SHARDS" -v d="$SHARD_DIR" '
    NR==1 { hdr=$0; for (i=0;i<n;i++) print hdr > d "/kimik3_bf16_tuning_main_s" i ".csv"; next }
    { s=(NR-2)%n; print >> d "/kimik3_bf16_tuning_main_s" s ".csv" }
  ' "$MAIN_IN"
  for i in $(seq 0 $((NUM_SHARDS - 1))); do
    local f="$SHARD_DIR/kimik3_bf16_tuning_main_s${i}.csv"
    echo "[shard] s${i}: $(($(wc -l < "$f") - 1)) shapes"
  done
}

IN_CSV="$SHARD_DIR/kimik3_bf16_tuning_main_s${SHARD}.csv"
OUT_CSV="$SHARD_DIR/kimik3_bf16_tuned_main_s${SHARD}.csv"

seed_partial() {
  # Preserve progress from a prior single-node run (first ~30 shapes usually land in s0).
  local partial="$HERE/kimik3_bf16_tuned_main.csv"
  [ -f "$partial" ] || return 0
  [ "$SHARD" = "0" ] || return 0
  [ -f "$OUT_CSV" ] && [ "$(wc -l < "$OUT_CSV")" -gt 1 ] && return 0
  echo "[shard] seeding s0 output from partial $partial"
  cp "$partial" "$OUT_CSV"
}

log_checkpoint() {
  local in_ctr="$1" out_ctr="$2" tag="$3"
  python3 "$HERE/checkpoint.py" --here "$HERE" --num-shards "$NUM_SHARDS" status 2>/dev/null | {
    echo "[checkpoint:$tag] shard=$SHARD resume via existing output CSV (skip tuned shapes)"
    grep -E "^s${SHARD} |^overall:" || true
  } || true
  if [[ "${CHECKPOINT_COMPACT:-1}" == "1" ]] && [[ -f "$OUT_CSV" ]]; then
    python3 "$HERE/checkpoint.py" --here "$HERE" compact 2>/dev/null | grep -F "$(basename "$OUT_CSV")" || true
  fi
  echo "[checkpoint:$tag] in=$in_ctr out=$out_ctr (append/resume — aiter --tuned_file skips done shapes)"
}

prepare_aiter_src() {
  # Parallel shards must not share NFS jit/build — copy aiter to node-local disk.
  if [ "$NUM_SHARDS" -le 1 ]; then
    echo "$AITER_SRC"
    return
  fi
  local local_aiter="/tmp/aiter_tune_s${SHARD}"
  echo "[shard] rsync aiter -> $local_aiter (avoid NFS JIT races)" >&2
  rm -rf "$local_aiter"
  rsync -a "${AITER_SRC}/" "$local_aiter/" --exclude .git
  echo "$local_aiter"
}

docker_tune() {
  local in_host="$1" out_host="$2" in_ctr="$3" out_ctr="$4" _libtype="$5" log_tag="$6" profile="${7:-}"
  local aiter_mount
  aiter_mount="$(prepare_aiter_src)"
  # Snapshot tune.sh: bash parses the mounted script by byte offset, so editing
  # the source while a shard runs shifts the offsets and corrupts its parse.
  local tune_snapshot="/tmp/tune_sh_s${SHARD}.sh"
  cp -f "$HERE/tune.sh" "$tune_snapshot"
  echo "[tune:$log_tag] shard=$SHARD/$NUM_SHARDS in=$in_host out=$out_host profile=${profile:-$TUNE_LIBTYPE_PROFILE} aiter=$aiter_mount"
  local -a extra_env=()
  [ -n "${HIP_VISIBLE_DEVICES:-}" ] && extra_env+=(-e "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES")
  [ -n "${LIBTYPE:-}" ] && extra_env+=(-e "LIBTYPE=$LIBTYPE")
  extra_env+=(
    -e "TUNE_LIBTYPE_PROFILE=${profile:-$TUNE_LIBTYPE_PROFILE}"
    -e "AITER_TUNE_ASM_MAX_M=${AITER_TUNE_ASM_MAX_M:-2048}"
    -e "AITER_TUNE_ASM_MAX_MN=${AITER_TUNE_ASM_MAX_MN:-4194304}"
    -e "AITER_TUNE_OPUS_MAX_M=${AITER_TUNE_OPUS_MAX_M:-2048}"
  )
  docker run --rm \
    --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --ipc host --shm-size 16g \
    -v "$HERE:/work" \
    -v "$SHARD_DIR:/work/shards" \
    -v "$aiter_mount:/opt/aiter" \
    -v "$tune_snapshot:/usr/local/bin/tune.sh:ro" \
    -e AITER_LIVE_MOUNT=1 \
    -e TUNE_BATCH="${TUNE_BATCH:-10}" \
    -e TUNE_SHAPE_GROUPED="${TUNE_SHAPE_GROUPED:-1}" \
    -e AITER_HIPBLASLT_FAST_MAX="${AITER_HIPBLASLT_FAST_MAX:-8192}" \
    -e INPUT_CSV="$in_ctr" \
    -e OUTPUT_CSV="$out_ctr" \
    "${extra_env[@]}" \
    "$TAG"
}

run_shard() {
  split_main_shards
  [ -f "$IN_CSV" ] || { echo "ERROR: missing $IN_CSV" >&2; exit 1; }
  seed_partial
  log_checkpoint "/work/shards/$(basename "$IN_CSV")" "/work/shards/$(basename "$OUT_CSV")" "main-s${SHARD}"
  docker_tune "$IN_CSV" "$OUT_CSV" \
    "/work/shards/$(basename "$IN_CSV")" "/work/shards/$(basename "$OUT_CSV")" \
    "" "main-s${SHARD}" "$TUNE_LIBTYPE_PROFILE"
  if [ "$SHARD" = "0" ]; then
    local n896_in="$HERE/kimik3_bf16_tuning_n896.csv"
    local n896_out="$HERE/kimik3_bf16_tuned_n896.csv"
    if [ -f "$n896_in" ] && [ "$(wc -l < "$n896_in")" -gt 1 ]; then
      log_checkpoint "/work/$(basename "$n896_in")" "/work/$(basename "$n896_out")" "n896"
      docker_tune "$n896_in" "$n896_out" \
        "/work/$(basename "$n896_in")" "/work/$(basename "$n896_out")" \
        "" "n896" "n896"
    fi
  fi
  echo "[shard] DONE shard=$SHARD/$NUM_SHARDS -> $OUT_CSV"
}

if [[ "${1:-}" == "bg" ]]; then
  nohup bash "$0" >>"$LOG" 2>&1 &
  echo "[shard] background PID=$!  SHARD=$SHARD/$NUM_SHARDS  tail -f $LOG"
else
  run_shard
fi
