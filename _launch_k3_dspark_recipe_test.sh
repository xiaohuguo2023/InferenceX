#!/usr/bin/env bash
# Launch DSpark recipe validation on an already-allocated exclusive node.
#
#   JOB=54069 SUFFIX=193 bash _launch_k3_dspark_recipe_test.sh
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
export HOME="${SPUR_USER_HOME:-/home/$(id -un)}"

JOB="${JOB:-54069}"
SUFFIX="${SUFFIX:-193}"
TAG="${TAG:-k3dsp_${SUFFIX}}"
LOG="$HOME/work/InferenceX/k3dspark_recipe_n${SUFFIX}_$(date +%m%d_%H%M).log"

bash _lint_scripts.sh _launch_k3_dspark_recipe_test.sh _k3_dspark_recipe_test_node.sh _spur_clean_node.sh

bash _spur_clean_node.sh "$JOB" "n${SUFFIX}"

echo "=== launch DSpark recipe test TAG=$TAG on job $JOB (nightly cb8104839) ==="
spur exec "$JOB" bash -c "
  set -uo pipefail
  export HOME='$HOME'
  cd '$HERE'
    export K3_CTR=k3-dspark-benchmark TAG='$TAG' RESEED_SHIPPED_MLA='${RESEED_SHIPPED_MLA:-0}'
  nohup bash _k3_dspark_recipe_test_node.sh > '$LOG' 2>&1 &
  echo \"started pid=\$! log=$LOG\"
"
echo "Monitor: tail -f $LOG"
