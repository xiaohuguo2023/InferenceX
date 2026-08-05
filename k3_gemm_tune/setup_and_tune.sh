#!/usr/bin/env bash
# Build the K3 bf16 GEMM tuning image and start tuning on an MI355X compute node.
# Follows ~/work/InferenceX/skill.md §4 and k3_gemm_tune/README.md.
#
# Usage (on compute node, or via hold job attach):
#   cd ~/work/InferenceX/k3_gemm_tune
#   ./setup_and_tune.sh build          # build docker image only
#   ./setup_and_tune.sh tune           # run tuning (needs image)
#   ./setup_and_tune.sh all            # build + tune
#   ./setup_and_tune.sh tune-split       # legacy two-phase (usually unnecessary after N=896 patch)
#   ./setup_and_tune.sh tune-split bg    # background via tune_split.sh
#
# From login node, run on your hold job:
#   JOB=33052
#   spur exec "$JOB" bash -lc 'cd ~/work/InferenceX/k3_gemm_tune && ./setup_and_tune.sh all bg'
#
# Env overrides:
#   AITER_SRC=~/work/aiter
#   CSV=~/work/InferenceX/kimik3_bf16_tuning_gemm_v2.csv
#   TAG=k3-bf16-gemm-tune:gfx950
#   LIBTYPE=all
#   INPUT_CSV=/work/kimik3_bf16_tuning_gemm_v2.csv
#   OUTPUT_CSV=/work/kimik3_bf16_tuned_gemm.csv

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

USER_HOME="${SPUR_USER_HOME:-/home/$(id -un)}"
export HOME="$USER_HOME"
AITER_SRC="${AITER_SRC:-$USER_HOME/work/aiter}"
CSV="${CSV:-$USER_HOME/work/InferenceX/kimik3_bf16_tuning_gemm_v2.csv}"
TAG="${TAG:-k3-bf16-gemm-tune:gfx950}"
LIBTYPE="${LIBTYPE:-flydsl,hipblaslt,skinny}"
TUNE_LIBTYPE_PROFILE="${TUNE_LIBTYPE_PROFILE:-safe}"
INPUT_CSV="${INPUT_CSV:-/work/kimik3_bf16_tuning_gemm_v2.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-/work/kimik3_bf16_tuned_gemm.csv}"
LOG="${LOG:-$HERE/tune.log}"

require_compute_node() {
  if [[ "$(hostname)" == crs-m2m-cpu-spur-* ]]; then
    echo "ERROR: run this on a compute node (attach with: ~/spur-node attach)" >&2
    echo "Or from login node:" >&2
    echo "  spur exec <JOBID> bash -lc 'cd ~/work/InferenceX/k3_gemm_tune && ./setup_and_tune.sh $*'" >&2
    exit 1
  fi
}

require_prereqs() {
  command -v docker >/dev/null || { echo "ERROR: docker not found" >&2; exit 1; }
  [ -d "$AITER_SRC/aiter" ] || { echo "ERROR: aiter not found at $AITER_SRC" >&2; exit 1; }
  [ -f "$CSV" ] || { echo "ERROR: shape CSV not found at $CSV" >&2; exit 1; }
  docker image inspect vllm/vllm-openai-rocm:kimi-k3 >/dev/null 2>&1 || {
    echo "Pulling base image vllm/vllm-openai-rocm:kimi-k3 ..."
    docker pull vllm/vllm-openai-rocm:kimi-k3
  }
}

show_aiter_note() {
  local head
  head="$(git -C "$AITER_SRC" log --oneline -1)"
  echo "[setup] host aiter (live mount): $head"
  echo "[setup] container uses $AITER_SRC -> /opt/aiter at runtime (not the baked image copy)."
  echo "[setup] Tuned libtype/solidx must match the same aiter build used at serve time."
}

docker_run_common() {
  local image="${1:?docker image tag required}"
  docker run --rm \
    --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --ipc host --shm-size 16g \
    -v "$HERE:/work" \
    -v "$AITER_SRC:/opt/aiter" \
    -v "$HERE/tune.sh:/usr/local/bin/tune.sh:ro" \
    -e AITER_LIVE_MOUNT=1 \
    -e INPUT_CSV="$INPUT_CSV" \
    -e OUTPUT_CSV="$OUTPUT_CSV" \
    -e TUNE_LIBTYPE_PROFILE="$TUNE_LIBTYPE_PROFILE" \
    -e LIBTYPE="$LIBTYPE" \
    "$image"
}

do_build() {
  echo "[setup] building docker image $TAG"
  AITER_SRC="$AITER_SRC" CSV="$CSV" TAG="$TAG" ./build.sh
}

do_tune() {
  mkdir -p "$HERE"
  cp -f "$CSV" "$HERE/kimik3_bf16_tuning_gemm_v2.csv"

  echo "[setup] starting tuner container (live aiter mount: $AITER_SRC)"
  echo "[setup] input=$INPUT_CSV output=$OUTPUT_CSV libtype=$LIBTYPE"
  docker_run_common "$TAG"
}

do_tune_bg() {
  mkdir -p "$HERE"
  cp -f "$CSV" "$HERE/kimik3_bf16_tuning_gemm_v2.csv"
  echo "[setup] launching tuner in background -> $LOG"
  echo "[setup] live aiter mount: $AITER_SRC"
  nohup docker_run_common "$TAG" >"$LOG" 2>&1 &
  echo "[setup] PID=$!  tail -f $LOG"
}

cmd="${1:-all}"
bg="${2:-}"

require_compute_node
require_prereqs
show_aiter_note

case "$cmd" in
  build)
    do_build
    ;;
  tune)
    if [[ "$bg" == "bg" ]]; then
      do_tune_bg
    else
      do_tune
    fi
    ;;
  tune-split)
    chmod +x "$HERE/tune_split.sh"
    if [[ "${2:-}" == "bg" ]]; then
      "$HERE/tune_split.sh" bg
    else
      "$HERE/tune_split.sh"
    fi
    ;;
  all)
    do_build
    if [[ "$bg" == "bg" ]]; then
      do_tune_bg
    else
      do_tune
    fi
    ;;
  *)
    echo "Usage: $0 {build|tune|tune-split|all} [bg]" >&2
    exit 1
    ;;
esac
