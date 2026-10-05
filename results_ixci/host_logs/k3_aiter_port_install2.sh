#!/bin/bash
# Install the aiter MLA port bundle (aiter#5695 native 12 heads + parallel
# metadata planner for multi-token decode) into this container, rebuild the
# three MLA JIT modules, and validate them with aiter's persistent MLA test.
# Run inside the container while no vLLM serve is using aiter.
set -euo pipefail
B=${BUNDLE:-/inferencex_dcp/results_ixci/host_logs/aiter_port}
T=/inferencex_dcp/results_ixci/host_logs/aiter5695/test_mla_persistent.py
P=/usr/local/lib/python3.12/dist-packages
M=$P/aiter_meta
BK=/root/aiter_preport

declare -A DST=(
  [mla.py]=$P/aiter/mla.py
  [attention.py]=$P/aiter/ops/attention.py
  [v1_2_device.cuh]=$M/csrc/kernels/mla/metadata/v1_2_device.cuh
  [reduce.cu]=$M/csrc/kernels/mla/reduce.cu
  [asm_mla.cu]=$M/csrc/py_itfs_cu/asm_mla.cu
  [v1_comm.cuh]=$M/csrc/kernels/mla/metadata/v1_comm.cuh
)

if [ ! -d "$BK" ]; then
  mkdir -p "$BK/co" "$BK/jit"
  for f in "${!DST[@]}"; do cp -a "${DST[$f]}" "$BK/$f"; done
  cp -a "$M"/hsa/gfx950/mla/mla_a8w8_qh32_qseqlen4_gqaratio32_*ps.co "$BK/co/"
  cp -a "$P"/aiter/jit/module_mla_{asm,metadata,reduce}.so "$BK/jit/" 2>/dev/null || true
fi

for f in "${!DST[@]}"; do cp "$B/$f" "${DST[$f]}"; done
cp "$B"/co/*.co "$M/hsa/gfx950/mla/"
rm -rf "$P"/aiter/jit/module_mla_{asm,metadata,reduce}.so "$P"/aiter/jit/build/module_mla_{asm,metadata,reduce}

cd /tmp
HIP_VISIBLE_DEVICES="${TEST_GPU:-0}" python3 "$T" -n 12,8 16,8 -d fp8 -kvd fp8 -c 8192 100000 -b 1 4 -ms 32 256 2>&1 \
  | grep -vE "^\[aiter\] import" | tail -n 40
ls -la "$P"/aiter/jit/module_mla_{asm,metadata,reduce}.so
