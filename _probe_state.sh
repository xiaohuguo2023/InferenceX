#!/bin/bash
# Probe container state: aiter build, MoE policy defaults, server liveness.
echo "=== aiter staged commit ==="
if [ -d /opt/aiter-local ]; then
  git -C /opt/aiter-local rev-parse --short HEAD 2>/dev/null || echo "  (not a git checkout)"
  cat /opt/aiter-local/aiter/VERSION 2>/dev/null || true
else
  echo "  /opt/aiter-local missing"
fi

echo "=== aiter python module + version ==="
python3 -c "import aiter,os;print('path:',os.path.dirname(aiter.__file__));print('ver:',getattr(aiter,'__version__','n/a'))" 2>&1 | head -4

echo "=== MOE_DISPATCH_POLICY references in aiter ==="
grep -rn "MOE_DISPATCH_POLICY" /opt/aiter-local/aiter/ 2>/dev/null | head -8 || echo "  none found"

echo "=== server alive? ==="
pgrep -fc "vllm serve" || echo "  0 (no server)"

echo "=== current env of any server proc ==="
pid=$(pgrep -f "vllm serve" | head -1)
if [ -n "$pid" ]; then
  tr '\0' '\n' < "/proc/$pid/environ" | grep -iE "MOE_DISPATCH|SITUV2|BF16_FP8" | sort
else
  echo "  (no process to inspect)"
fi
