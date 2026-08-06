#!/bin/bash
# Verify the a8w4 SiTU MoE path is actually engaged, not just requested.
#
# Both AITER_SITUV2_A8W4 and VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4 must reach the
# server: vLLM's flag drives the interleaved w13 shuffle and GateMode.INTERLEAVE,
# while aiter's SiTUv2 branch reads AITER_SITUV2_A8W4 directly. With only the
# vLLM one set, every kernel silently falls back to bf16 activations, which shows
# up as flydsl_moe1_abf16_* instead of *_afp8_*_gui_fp8.
#
# Usage: _check_situv2.sh [serve_log]   (default /workspace/serve_nightly_k3.log)
LOG="${1:-/workspace/serve_nightly_k3.log}"
[ -f "$LOG" ] || { echo "no such log: $LOG" >&2; exit 1; }

echo "=== flags in the running server's environment ==="
pid=$(pgrep -f "vllm serve" | head -1)
if [ -n "$pid" ]; then
  tr '\0' '\n' < "/proc/$pid/environ" \
    | grep -E "^(AITER_SITUV2_A8W4|VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4|AITER_BF16_FP8_MOE_BOUND)=" \
    | sort
  echo "  (server pid $pid)"
else
  echo "  no running server — reporting from the log only"
fi

echo
echo "=== MoE kernels actually selected ==="
afp8=$(grep -acoE "moe[0-9]*_afp8" "$LOG")
abf16=$(grep -acoE "moe[0-9]*_abf16" "$LOG")
gui=$(grep -ac "_gui" "$LOG")
printf "  afp8 (a8w4, wanted)      : %s\n" "$afp8"
printf "  abf16 (a16w4, fallback)  : %s\n" "$abf16"
printf "  _gui (gate-up interleave): %s\n" "$gui"

echo
if [ "$afp8" -gt 0 ] && [ "$abf16" -eq 0 ]; then
  echo "RESULT: a8w4 engaged"
elif [ "$afp8" -eq 0 ] && [ "$abf16" -gt 0 ]; then
  echo "RESULT: FALLBACK — a8w4 NOT engaged; check that AITER_SITUV2_A8W4=1 is exported" >&2
  exit 1
elif [ "$afp8" -eq 0 ] && [ "$abf16" -eq 0 ]; then
  echo "RESULT: inconclusive — no MoE kernel selection logged yet (server still starting?)"
else
  echo "RESULT: MIXED — both afp8 and abf16 present; inspect the log" >&2
  exit 1
fi
