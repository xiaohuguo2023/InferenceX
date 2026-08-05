#!/bin/bash
# Stage a minimal build context (aiter without .git/build caches) + the shape CSV,
# then build the tuning image. Run this on a box that has ~/work/aiter @ 00cbe979f.
set -euo pipefail
USER_HOME="${SPUR_USER_HOME:-/home/$(id -un)}"
export HOME="$USER_HOME"
AITER_SRC="${AITER_SRC:-$USER_HOME/work/aiter}"
CSV="${CSV:-$USER_HOME/work/InferenceX/k3_gemm_tune/kimik3_bf16_tuning_gemm_v2.csv}"
TAG="${TAG:-k3-bf16-gemm-tune:gfx950}"
HERE="$(cd "$(dirname "$0")" && pwd)"

[ -d "$AITER_SRC/aiter" ] || { echo "ERROR: aiter source not found at $AITER_SRC" >&2; exit 1; }
[ -f "$CSV" ] || { echo "ERROR: shape CSV not found at $CSV" >&2; exit 1; }

CTX="$(mktemp -d)"; trap 'rm -rf "$CTX"' EXIT
echo "[build] staging context in $CTX (aiter minus .git/build caches)..."
rsync -a --exclude '.git' --exclude 'aiter/jit/build' --exclude '**/__pycache__' \
      --exclude '*.log' "$AITER_SRC/" "$CTX/aiter/"
cp "$CSV" "$CTX/kimik3_bf16_tuning_gemm.csv"
cp "$HERE/Dockerfile" "$HERE/tune.sh" "$CTX/"

echo "[build] docker build -> $TAG"
docker build -t "$TAG" "$CTX"
echo "[build] done: $TAG"
echo
echo "Run on any MI355X (gfx950), output lands in \$PWD:"
echo "  docker run --rm --device /dev/kfd --device /dev/dri --group-add video \\"
echo "    --security-opt seccomp=unconfined --ipc host --shm-size 16g \\"
echo "    -v \$PWD:/work $TAG"
echo
echo "Ship to another machine:"
echo "  docker save $TAG | zstd > k3tune.tzst   # then: zstd -d | docker load"
