#!/bin/bash
# Install the final PR code (v4: v3 + 128-token thresholds) over v1/v2/v3.
# Usage: k3_install_v4.sh JOB CONTAINER
source /home/xiaohugu/work/InferenceX_dcp/results_ixci/host_logs/k3_lib_retry.sh
JOB=$1; C=$2
retry timeout 240 srun --jobid=$JOB --overlap docker exec $C bash -c 'set -e; S=/usr/local/lib/python3.12/dist-packages; H=/inferencex_dcp/results_ixci/host_logs
L=$S/vllm/models/kimi_k3/amd/linear.py
if ! grep -q "_ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD = 128" $L; then
  for p in prk3overlap_core_v3 prk3overlap_core_v2 prk3overlap_core; do
    if patch --dry-run -s -R -p1 -d $S < $H/reverts/$p.patch >/dev/null 2>&1; then patch -s -R -p1 -d $S < $H/reverts/$p.patch; break; fi
  done
  patch -s -p1 -d $S < $H/reverts/prk3overlap_core_v4.patch
fi
cp $H/mla_overlap_gate_v4.py $S/vllm/models/kimi_k3/amd/mla.py
cd /tmp; python3 -c "import vllm.models.kimi_k3.amd.linear as l; assert l._ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD == 128; from vllm.models.kimi_k3.amd.mla import KimiK3MultiHeadLatentAttentionWrapper; print(\"v4 installed\")"' 2> >(grep -v "raw mode" >&2)
