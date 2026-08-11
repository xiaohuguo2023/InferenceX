#!/usr/bin/env bash
# DSpark fp8-asm recipe validation (docs/kimik3_dspark_fp8asm_recipe.md).
# Runs on a compute node after allocation:
#   K3_CTR=k3-dspark-benchmark TAG=k3dsp_193 nohup bash _k3_dspark_recipe_test_node.sh > k3dspark_recipe.log 2>&1 &
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

export HOME="${SPUR_USER_HOME:-/home/$(id -un)}"
export K3_CTR="${K3_CTR:-k3-dspark-benchmark}"
TAG="${TAG:-k3dsp_$(hostname -s | grep -oE '[0-9]+$' || echo local)}"
PORT="${PORT:-8890}"
NUM_SPEC="${NUM_SPEC:-2}"
LOG="${LOG:-k3dspark_recipe_${TAG}.log}"
SERVE_LOG="/workspace/serve_k3_bench_spec${NUM_SPEC}.log"
BENCH_ROOT="/workspace/k3_dspark_fp8asm_bench"

echo "########## DSpark recipe test TAG=$TAG node=$(hostname) $(date '+%F %T') ##########"
echo "K3_CTR=$K3_CTR PORT=$PORT NUM_SPEC=$NUM_SPEC"

bash _lint_scripts.sh _k3_dspark_recipe_test_node.sh setup_benchmark.sh \
  _k3_dspark_fp8asm_apply_patches.sh _serve_k3_bench_spec.sh _bench_k3_dspark_fp8asm.sh \
  _drain_vllm_host.sh _drain_vllm_procs.sh

echo "========== host GPU drain (before container recreate) =========="
bash _drain_vllm_host.sh || true

./setup_benchmark.sh start-dspark
./setup_benchmark.sh setup-dspark
./setup_benchmark.sh verify-dspark-patches

echo "========== serve DSpark (fp8-asm, ASM_PADDING=asm) =========="
echo "========== host GPU drain (before serve) =========="
bash _drain_vllm_host.sh || true
export PORT NUM_SPEC GPU_MEM="${GPU_MEM:-0.88}" MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}" RESEED_SHIPPED_MLA="${RESEED_SHIPPED_MLA:-0}"
./setup_benchmark.sh serve-dspark

docker exec "$K3_CTR" bash -lc "
  export PORT='$PORT' SERVE_LOG='$SERVE_LOG' BENCH_ROOT='$BENCH_ROOT'
  bash /workspace/_k3_dspark_recipe_test_inner.sh
"
overall=$?
echo "########## DSpark recipe test DONE $(date '+%F %T') rc=$overall log=$LOG ##########"
exit "$overall"
