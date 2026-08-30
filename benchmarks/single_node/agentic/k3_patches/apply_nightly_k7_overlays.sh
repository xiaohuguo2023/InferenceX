#!/usr/bin/env bash
# Apply the K=7 stock-nightly overlays. Run INSIDE the container.
# Does NOT rebuild aiter (BOOTSTRAP=0). Image aiter stays as shipped.
#
#   docker cp benchmarks/single_node/agentic "$CTR:/opt/k3-recipe"
#   docker exec "$CTR" bash /opt/k3-recipe/k3_patches/apply_nightly_k7_overlays.sh
#
set -euo pipefail
PATCHES="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST="${DIST:-/usr/local/lib/python3.12/dist-packages}"

# Image aiter (6d4562c / 1dc464d). Override if the nightly layout moves.
export AITER_MLA="${AITER_MLA:-$DIST/aiter/mla.py}"
export SPLITK_CU="${SPLITK_CU:-$DIST/aiter_meta/csrc/py_itfs_cu/asm_gemm_a16w16.cu}"
GEMM_SRC="$PATCHES/merged_bf16_tuned_gemm.worktree.csv"
GEMM_DST="${AITER_CONFIG_GEMM_BF16:-/opt/aiter-local/aiter/configs/merged_bf16_tuned_gemm.csv}"

say() { echo "############### $* ###############"; }

[ -f "$GEMM_SRC" ] || { echo "!! missing $GEMM_SRC" >&2; exit 1; }

say "1/5 dense prefix-cache retention (unset env -> None)"
python3 "$PATCHES/patch_prefix_cache_retention_dense.py"

say "2/5 aiter get_block_n_fp8 80/96/112 + .get(..., 64)"
if [ -f "$AITER_MLA" ]; then
  python3 "$PATCHES/patch_aiter_blockn_fp8.py"
else
  echo "!! $AITER_MLA missing — skip blockn (set AITER_MLA)" >&2
  exit 1
fi

say "3/5 aiter a16w16 split-K cudagraph guard (no rebuild)"
if [ -f "$SPLITK_CU" ]; then
  python3 "$PATCHES/patch_aiter_splitk_cudagraph.py"
else
  echo "!! $SPLITK_CU missing — skip splitk (set SPLITK_CU)" >&2
  exit 1
fi

say "4/5 full-attn eagle prefix-veto (DRAM read path)"
python3 "$PATCHES/patch_offload_eagle_prefix_veto.py"

say "5/5 tuned GEMM CSV -> $GEMM_DST"
mkdir -p "$(dirname "$GEMM_DST")"
cp "$GEMM_SRC" "$GEMM_DST"
n=$(wc -l < "$GEMM_DST")
[ "$n" -eq 3027 ] || { echo "!! expected 3027 GEMM rows, got $n" >&2; exit 1; }
if grep -q ",opus,1212," "$GEMM_DST"; then
  echo "!! kernel 1212 still in CSV" >&2
  exit 1
fi
echo "  GEMM rows=$n no-1212 OK"

echo
echo "DONE — overlays on stock nightly. Serve with:"
echo "  export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm"
echo "  export AITER_CONFIG_GEMM_BF16=$GEMM_DST"
echo "  NUM_SPEC_TOKENS=7 SYNTHETIC_ACCEPT_LEN=3.84 KV_CACHE_MEMORY=47691420128 \\"
echo "    CAPTURE_SIZES=1,2,3,4,6,8,12,16,24,30,32,36,40,48,56,60,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,272,288,304,320,336,352,368,384 \\"
echo "    bash /opt/k3-recipe/kimik3_fp4_mi355x_vllm_mtp.sh"
