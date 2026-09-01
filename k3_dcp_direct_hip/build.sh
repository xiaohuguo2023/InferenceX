#!/usr/bin/env bash
# Build the native-HIP DCP direct a2a-lse-reduce op straight with hipcc (NO
# hipify, NO torch cpp_extension). Produces dcp_direct_a2a_lse_reduce.so, loaded
# at runtime via torch.ops.load_library(...). Run inside the DCP container
# (k3-nightly4-test); flags mirror what torch's ROCm cpp_extension emits, minus
# hipify, minus torch_python, and pinned to gfx950 only.
set -euo pipefail
cd "$(dirname "$0")"

TORCH_INC="$(python3 -c 'import torch,os;print(os.path.dirname(torch.__file__))')/include"
TORCH_LIB="$(python3 -c 'import torch,os;print(os.path.dirname(torch.__file__))')/lib"

SRC=dcp_direct_a2a_lse_reduce_hip.hip
OBJ=dcp_direct_a2a_lse_reduce.o
OUT=dcp_direct_a2a_lse_reduce.so

echo "[build] compiling $SRC (native HIP, gfx950)"
/opt/rocm/bin/hipcc \
  -DWITH_HIP -DUSE_ROCM=1 -D__HIP_PLATFORM_AMD__=1 -DHIPBLAS_V2 \
  -DTORCH_API_INCLUDE_EXTENSION_H \
  -D__HIP_NO_HALF_OPERATORS__=1 -D__HIP_NO_HALF_CONVERSIONS__=1 \
  -DHIP_ENABLE_WARP_SYNC_BUILTINS=1 \
  -isystem "$TORCH_INC" \
  -isystem "$TORCH_INC/torch/csrc/api/include" \
  -isystem "$TORCH_INC/THH" \
  -isystem /opt/rocm/include \
  -fPIC -std=c++20 --offload-arch=gfx950 -fno-gpu-rdc -O3 \
  -c "$SRC" -o "$OBJ"

echo "[build] linking $OUT"
c++ "$OBJ" -shared \
  -L"$TORCH_LIB" -lc10 -lc10_hip -ltorch_cpu -ltorch_hip -ltorch \
  -L/opt/rocm/lib -lamdhip64 \
  -o "$OUT"

echo "[build] done -> $(pwd)/$OUT"
