#!/usr/bin/env python3
"""Env-gated loader patch for the native-HIP direct DCP a2a op.

Inserts, right after `logger = init_logger(__name__)` in vLLM's dcp_utils.py, a
guarded `torch.ops.load_library(os.environ["K3_DCP_A2A_SO"])` so every worker
registers `torch.ops._C.direct_dcp_a2a_lse_reduce` before `_init_combine` runs.
Idempotent; makes a .k3bak backup. No behavior change when K3_DCP_A2A_SO unset.
"""
import sys

F = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/dcp_utils.py"
src = open(F).read()
if "K3_DCP_A2A_SO" in src:
    print("ALREADY PATCHED")
    sys.exit(0)

anchor = "logger = init_logger(__name__)\n"
patch = anchor + """
# --- K3 native-HIP direct DCP a2a op loader (env-gated, self-reverting) ---
import os as _k3_os
_k3_so = _k3_os.environ.get("K3_DCP_A2A_SO")
if _k3_so and not hasattr(torch.ops._C, "direct_dcp_a2a_lse_reduce"):
    try:
        torch.ops.load_library(_k3_so)
        logger.info("K3: loaded native-HIP direct DCP a2a op from %s", _k3_so)
    except Exception as _k3_e:
        logger.warning("K3: failed to load %s: %s", _k3_so, _k3_e)
# --- end K3 patch ---
"""
assert src.count(anchor) == 1, "anchor not unique/absent"
open(F + ".k3bak", "w").write(src)
open(F, "w").write(src.replace(anchor, patch, 1))
print("PATCHED (backup -> .k3bak)")
