#!/usr/bin/env python3
"""Reuse one FlyDSL MoE stage-1 output buffer instead of one per layer per shape.

An allocator snapshot at util 0.95 showed 652 live buffers from
_flydsl_moe_stage1_impl totalling 6.98 GiB, in only four distinct sizes between
10.50 and 10.55 MiB: the size comes from expert padding
(sorted_expert_ids * tile_m), not from token count, so a 1-token graph reserves
essentially as much as a 128-token graph. Every one of them stays live because
each capture allocates its own and the graph bakes in the pointer.

They are interchangeable. stage-1's output is consumed by stage-2 inside the
same layer and never read again, and every aiter launch targets
torch.cuda.current_stream(), so the kernels are serialized within a capture and
one buffer can serve every layer and every captured shape. The key includes the
ubatch id so overlapping microbatches never share a buffer.

Scoped deliberately narrow: only the v2-output-layout fp8 branch (the 6.70 GiB)
and the sorted-scale buffer (the 0.28 GiB). Split-K atomic-add paths and the fp4
branches are left alone, since those have zero-init expectations this change
should not reason about.

Applies to /opt/aiter-local inside the container. Not for upstream as-is: the
buffer should come from vLLM's WorkspaceManager via an injected provider rather
than a dict inside aiter.
"""

import os
import shutil
import sys

TARGET = "/opt/aiter-local/aiter/ops/flydsl/moe_kernels.py"

HELPER = '''

# PATCH(moe-scratch-reuse)
_MOE_STAGE1_SCRATCH: dict = {}


def _moe_stage1_scratch(shape, dtype, dev):
    """Return a reusable stage-1 scratch buffer for (ubatch, device, dtype, shape).

    Sized exactly as the per-call allocation it replaces, so semantics are
    unchanged apart from the buffer being shared across layers and captured
    shapes. Contents are undefined on entry, matching torch.empty.
    """
    try:
        from vllm.v1.worker.ubatching import dbo_current_ubatch_id

        ubatch_id = dbo_current_ubatch_id()
    except Exception:
        ubatch_id = 0
    key = (ubatch_id, str(dev), dtype, tuple(int(s) for s in shape))
    buf = _MOE_STAGE1_SCRATCH.get(key)
    if buf is None:
        buf = torch.empty(shape, dtype=dtype, device=dev)
        _MOE_STAGE1_SCRATCH[key] = buf
    return buf
'''

OLD_OUT = """            else:
                out = torch.empty(
                    (_sorted_rows, inter_dim), dtype=dtypes.fp8, device=dev
                )"""

NEW_OUT = """            else:
                # PATCH(moe-scratch-reuse)
                out = _moe_stage1_scratch(
                    (_sorted_rows, inter_dim), dtypes.fp8, dev
                )"""

OLD_SCALE = """    out_scale_sorted_flat = (
        torch.empty(padded_rows * padded_cols, dtype=torch.uint8, device=dev)
        if _need_sort
        else torch.empty(0, dtype=torch.uint8, device=dev)
    )"""

NEW_SCALE = """    # PATCH(moe-scratch-reuse)
    out_scale_sorted_flat = (
        _moe_stage1_scratch((padded_rows * padded_cols,), torch.uint8, dev)
        if _need_sort
        else torch.empty(0, dtype=torch.uint8, device=dev)
    )"""


def main():
    src = open(TARGET, encoding="utf-8").read()
    if "PATCH(moe-scratch-reuse)" in src:
        print("[moe] already applied")
        return 0

    for name, old in (("stage1 out", OLD_OUT), ("sorted scale", OLD_SCALE)):
        n = src.count(old)
        if n != 1:
            print("[moe] ERROR: anchor %r found %d times, expected 1" % (name, n))
            return 1

    # Insert the helper after the import block so it is defined before use.
    marker = "\nimport torch\n"
    if marker not in src:
        print("[moe] ERROR: could not find 'import torch' to anchor the helper")
        return 1
    src = src.replace(marker, marker + HELPER, 1)

    src = src.replace(OLD_OUT, NEW_OUT)
    src = src.replace(OLD_SCALE, NEW_SCALE)

    shutil.copy2(TARGET, TARGET + ".moe_bak")
    open(TARGET, "w", encoding="utf-8").write(src)
    print("[moe] applied (backup: %s.moe_bak)" % os.path.basename(TARGET))
    return 0


if __name__ == "__main__":
    sys.exit(main())
