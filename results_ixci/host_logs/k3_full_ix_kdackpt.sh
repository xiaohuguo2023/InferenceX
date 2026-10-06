#!/bin/bash
# Full conc-1 IX on the v4 overlap stack (both overlap flags on). ARM=ckpt adds the
# KDA prefill-checkpoint prototype (prkdackpt_fix.patch, feature flag on) with
# --prefix-match-unit 64; ARM=base runs the same stack without the patch.
# Usage: k3_full_ix_kdackpt.sh JOB NODE_TAG CONTAINER ARM
source /home/xiaohugu/work/InferenceX_dcp/results_ixci/host_logs/k3_lib_retry.sh
R=/home/xiaohugu/work/InferenceX_dcp/results_ixci; RC=/inferencex_dcp/results_ixci
JOB=$1; N=$2; C=$3; ARM=$4
case "$ARM" in
  ckpt) CK=1; PATCHES=$RC/host_logs/reverts/prkdackpt_fix.patch; PMU_ARG="--prefix-match-unit 64" ;;
  *) CK=0; PATCHES=""; PMU_ARG="" ;;
esac
TAG=v4on_kda${ARM}_full_ix
LOGD=ixci_repro_c1_n${N}_${TAG}_dcp1gs
PRS="57978 58633 58381 58732 58737 h12 58861 router"
dxt() { local t=$1; shift; timeout "$t" srun --jobid=$JOB --overlap docker exec "$@" 2> >(grep -v "raw mode" >&2); }
bash $R/host_logs/k3_install_v4.sh $JOB $C || { echo "V4 FAILED"; exit 1; }
# The recipe re-applies its own kda.patch at launch, so the prototype patch must
# go on after it: reverse any applied copy here and pass it through EXTRA_VLLM_PATCHES.
retry timeout 120 srun --jobid=$JOB --overlap docker exec $C bash -c 'S=/usr/local/lib/python3.12/dist-packages; H=/inferencex_dcp/results_ixci/host_logs/reverts; for P in prkdackpt_fix.patch prkdackpt_dbg_v1.patch prkdackpt_dbg.patch prkdackpt.patch; do for D in $S/vllm $S; do if patch --dry-run -s -R -p1 -d $D < $H/$P >/dev/null 2>&1; then patch -s -R -p1 -d $D < $H/$P && echo reverted-$P; fi; done; done'
retry dxt 150 $C python3 $RC/host_logs/k3_stop_serve.py
sleep 15
retry timeout 60 srun --jobid=$JOB --overlap docker exec -d \
  -e VLLM_DISABLE_SHARED_EXPERTS_STREAM=0 \
  -e VLLM_KIMI_K3_AMD_MOE_ROUTER_DOWN_PROJ_STREAM=1 \
  -e VLLM_KIMI_K3_AMD_MLA_GATE_STREAM=1 \
  -e VLLM_K3_AMD_KDA_PREFILL_CHECKPOINT=$CK \
  -e EXTRA_VLLM_PATCHES="$PATCHES" \
  -e EXTRA_VLLM_ARGS_ADD="$PMU_ARG" \
  -e AITER_MLA_DEFAULT_MAX_SPLIT=128 \
  -e ARM=dcp1gs -e PR58814=1 -e EXTRA_PRS="$PRS" -e TAG_PREFIX=$TAG \
  $C bash -lc "mkdir -p $RC/$LOGD; bash $RC/host_logs/k3_ix_c1_img.sh $N > $RC/$LOGD/launch.log 2>&1"
echo "[$(date +%T)] n$N $ARM IX started"
for _ in $(seq 1 900); do grep -aq 'IXCI_REPRO_RC' "$R/$LOGD/launch.log" 2>/dev/null && break; sleep 20; done
echo "[$(date +%T)] n$N $ARM IX finished: $(grep -a -m1 IXCI_REPRO_RC $R/$LOGD/launch.log)"
grep -aoE "prefix_match_unit.: [0-9]+" "$R/$LOGD/launch.log" | head -1
python3 $R/host_logs/ix_frITL_sum.py $R/$LOGD
