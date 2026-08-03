#!/bin/bash
# Kimi-K3 bf16 (a16w16) GEMM tuner entrypoint. Runs the aiter tuner over the curated
# shape list, benchmarking flydsl/asm/hipBLASLt/skinny/opus per (M,N,K) and writing the
# per-shape winner. Output lands in /work (mount a host dir there to keep it).
set -euo pipefail
export GPU_ARCHS=gfx950

IN="${INPUT_CSV:-/opt/tune_input/kimik3_bf16_tuning_gemm.csv}"
OUT="${OUTPUT_CSV:-/work/kimik3_bf16_tuned_gemm.csv}"
LIBTYPE="${LIBTYPE:-all}"

# One-time (first run): rebuild the compiled aiter core to match the source. The image
# stripped the stale committed .so at build; this GPU-side JIT build takes ~10s.
if ! python -c "from aiter.jit.module_aiter_core import MlaVersion" >/dev/null 2>&1; then
  echo "[tune] building aiter jit core (first run, ~10s)..."
  AITER_REBUILD=1 python -c "import aiter; from aiter.jit.module_aiter_core import MlaVersion; print('[tune] core ok:', list(MlaVersion.__members__))"
fi

echo "[tune] input=$IN"
echo "[tune] output=$OUT  libtype=$LIBTYPE  (+hipblaslt)"
mkdir -p "$(dirname "$OUT")"
cd /opt/aiter/csrc/gemm_a16w16
python gemm_a16w16_tune.py \
  --input_file "$IN" \
  --tuned_file "$OUT" \
  --libtype "$LIBTYPE" \
  --with-hipblaslt

echo "[tune] DONE -> $OUT"
echo "[tune] install on the serving box:"
echo "       cp $OUT <aiter>/aiter/configs/model_configs/kimik3_bf16_tuned_gemm.csv"
echo "       (or /tmp/aiter_configs/bf16_tuned_gemm.csv), then re-serve."
