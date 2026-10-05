#!/bin/bash
# Bring one ablation node to the full 170+ stack on the latest nightly.
# Runs under the node agent, so it is already on the node and the srun shim
# makes "srun --overlap" local. Leaves the container ready; no serve.
set -uo pipefail
HOSTN=$(hostname)
NODE=${HOSTN##*-}                 # crsuse2-m2m-021 -> 021
C=k3-repro-385dce-n$NODE
IMG=vllm/vllm-openai-rocm:nightly-rocm100-1a001d584244bd21829a5e5bd4a376ea141c0ce3
BASE=/shared_nfs/xiaohugu/k3_crossover
RC=/inferencex_dcp/results_ixci
AB=$BASE/ab/n$NODE
mkdir -p "$AB" "$BASE/cache/n$NODE"
log() { echo "[$(date -u +%H:%M:%S)] n$NODE $*"; }
dx() { docker exec "$C" "$@"; }

log "setup $IMG"
IMAGE=$IMG SKIP_REAP=1 bash /home/xiaohugu/work/InferenceX_dcp/_k3_ixci_repro_setup.sh "$NODE" \
  > "$AB/setup.log" 2>&1
grep -aq SETUP_OK "$AB/setup.log" || { log "SETUP FAILED"; tail -20 "$AB/setup.log"; exit 1; }
dx bash -lc 'python3 -c "import vllm;print(\"vllm\",vllm.__version__)"'

log "tabulate"
dx bash -lc 'pip install -q tabulate' || true

log "runtime PRs (WITH4: 58633 58381 58732 58737 + 58861 59023 59245)"
dx bash -lc "WITH4=1 python3 $RC/host_logs/frozen_1a001d58/apply_runtime.py" > "$AB/patches.log" 2>&1 \
  || { log "runtime patches FAILED"; tail -20 "$AB/patches.log"; exit 1; }

log "h12"
dx bash -lc 'site=/usr/local/lib/python3.12/dist-packages
  p=/inferencex_dcp/results_ixci/host_logs/reverts/prh12.patch
  patch --dry-run --forward --batch -p1 -d "$site" < $p && patch --forward --batch -p1 -d "$site" < $p && echo APPLY_OK h12 || echo APPLY_FAIL h12'

log "MLA graph schedule (hand-rebased onto 1a001d584)"
L=/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/rocm_aiter_mla.py
dx bash -lc "md5sum $L" | tee "$AB/mla_base_md5.txt"
dx bash -lc "cp -f $L $BASE/ab/n$NODE/rocm_aiter_mla.base.py
  cp -f $BASE/gs_rebase/rocm_aiter_mla.py $L
  python3 -c 'import py_compile;py_compile.compile(\"$L\",doraise=True);print(\"gs ok\")'"

log "GEMM table armB"
dx cp -f "$BASE/gemm_armB_image_k3_6089fly.csv" \
  /usr/local/lib/python3.12/dist-packages/aiter/configs/bf16_tuned_gemm.csv
dx bash -lc 'wc -l < /usr/local/lib/python3.12/dist-packages/aiter/configs/bf16_tuned_gemm.csv'

log "snapshot the full-stack tree for fast per-arm reverts"
dx bash -lc "cd /usr/local/lib/python3.12/dist-packages && tar czf $BASE/ab/n$NODE/fullstack_vllm.tgz vllm aiter/configs/bf16_tuned_gemm.csv 2>/dev/null; ls -la $BASE/ab/n$NODE/fullstack_vllm.tgz"

echo "AB_SETUP_OK n$NODE"
