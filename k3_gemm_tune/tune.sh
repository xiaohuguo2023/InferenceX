#!/bin/bash
# Kimi-K3 bf16 (a16w16) GEMM tuner entrypoint. Runs the aiter tuner over the curated
# shape list, benchmarking flydsl/asm/hipBLASLt/skinny/opus per (M,N,K) and writing the
# per-shape winner. Output lands in /work (mount a host dir there to keep it).
set -euo pipefail
export GPU_ARCHS=gfx950

IN="${INPUT_CSV:-/opt/tune_input/kimik3_bf16_tuning_gemm.csv}"
OUT="${OUTPUT_CSV:-/work/kimik3_bf16_tuned_gemm.csv}"
# safe: flydsl+hipblaslt+skinny (default — avoids opus/asm GPU faults on large-M)
# n896: +opus for MoE N=896 shapes (asm still skipped via N%tileN patch)
# full: +opus+asm with AITER_TUNE_*_MAX_M shape guards (M>2048 skips asm/opus)
TUNE_LIBTYPE_PROFILE="${TUNE_LIBTYPE_PROFILE:-safe}"
case "$TUNE_LIBTYPE_PROFILE" in
  safe) _DEFAULT_LIBTYPE="flydsl,hipblaslt,skinny" ;;
  n896) _DEFAULT_LIBTYPE="flydsl,hipblaslt,skinny,opus" ;;
  full) _DEFAULT_LIBTYPE="flydsl,hipblaslt,skinny,opus,asm" ;;
  *)    echo "ERROR: TUNE_LIBTYPE_PROFILE must be safe, n896, or full" >&2; exit 2 ;;
esac
LIBTYPE="${LIBTYPE:-$_DEFAULT_LIBTYPE}"
TUNE_BATCH="${TUNE_BATCH:-10}"
export AITER_TUNE_ASM_MAX_M="${AITER_TUNE_ASM_MAX_M:-2048}"
export AITER_TUNE_ASM_MAX_MN="${AITER_TUNE_ASM_MAX_MN:-4194304}"
export AITER_TUNE_OPUS_MAX_M="${AITER_TUNE_OPUS_MAX_M:-2048}"
# Group kernel candidates per shape on one GPU; default batch=100 explodes to 200k+ task groups.
TUNE_SHAPE_GROUPED="${TUNE_SHAPE_GROUPED:-1}"
export AITER_HIPBLASLT_FAST_MAX="${AITER_HIPBLASLT_FAST_MAX:-8192}"

if [[ "${AITER_LIVE_MOUNT:-0}" == "1" ]]; then
  echo "[tune] live aiter mount detected at /opt/aiter"
  if command -v git >/dev/null 2>&1 && git -C /opt/aiter rev-parse --short HEAD >/dev/null 2>&1; then
    echo "[tune] aiter HEAD: $(git -C /opt/aiter log --oneline -1)"
  fi
  echo "[tune] applying GEMM tune patches (N=896 asm filter, hipBLASLt cap, asm/opus shape guards, zero-task grouping)..."
  python3 /work/_patch_gemm_n896.py /opt/aiter
  python3 /work/_patch_gemm_tune_safe.py /opt/aiter
  python3 /work/_patch_mp_tuner_zerotask.py /opt/aiter
  echo "[tune] reinstalling editable aiter from mounted source..."
  cd /opt/aiter
  pip uninstall -y aiter amd-aiter >/dev/null 2>&1 || true
  rm -f aiter/jit/*.so
  rm -rf aiter/jit/build
  pip install -e . --no-build-isolation --no-deps
fi

# Rebuild the compiled aiter core to match the active source.
if ! python -c "from aiter.jit.module_aiter_core import MlaVersion" >/dev/null 2>&1; then
  echo "[tune] building aiter jit core (first run, ~10s)..."
  AITER_REBUILD=1 python -c "import aiter; from aiter.jit.module_aiter_core import MlaVersion; print('[tune] core ok:', list(MlaVersion.__members__))"
fi

echo "[tune] input=$IN"
echo "[tune] output=$OUT  profile=$TUNE_LIBTYPE_PROFILE  libtype=$LIBTYPE  (+hipblaslt)"
echo "[tune] asm/opus guards: ASM_MAX_M=$AITER_TUNE_ASM_MAX_M OPUS_MAX_M=$AITER_TUNE_OPUS_MAX_M"
mkdir -p "$(dirname "$OUT")"
cd /opt/aiter/csrc/gemm_a16w16
TUNE_ARGS=(--input_file "$IN" --tuned_file "$OUT" --libtype "$LIBTYPE" --with-hipblaslt --batch "$TUNE_BATCH")
if [[ "$TUNE_SHAPE_GROUPED" == "1" ]]; then
  TUNE_ARGS+=(--shape_grouped)
fi
echo "[tune] batch=$TUNE_BATCH shape_grouped=$TUNE_SHAPE_GROUPED hipb_fast_max=$AITER_HIPBLASLT_FAST_MAX"
# gemm_tuner.py wraps the tuner in a subprocess and retries after GPU crashes.
python gemm_tuner.py "${TUNE_ARGS[@]}"

echo "[tune] DONE -> $OUT"
echo "[tune] install on the serving box:"
echo "       cp $OUT <aiter>/aiter/configs/model_configs/kimik3_bf16_tuned_gemm.csv"
echo "       (or /tmp/aiter_configs/bf16_tuned_gemm.csv), then re-serve."
