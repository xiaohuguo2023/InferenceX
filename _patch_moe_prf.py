#!/usr/bin/env python3
"""Apply the upstream PR-F stage-1 buffer reuse to /opt/aiter-local.

This is the same optimisation _patch_moe_scratch.py prototyped, rewritten as the
form actually proposed upstream so the benchmark measures the code we intend to
ship rather than the diagnostic. Three differences matter for the numbers:

  * The cache lives in fused_moe.py and is only consulted from the internal
    stage1->stage2 wrapper, so the public flydsl_moe_stage1() keeps allocating
    per call. Standalone callers are unaffected.
  * Keyed by (device, stream, shape) instead of a vLLM ubatch id, which removes
    the aiter->vLLM import and keeps concurrent streams from sharing storage.
  * Output only. The 0.28 GiB sorted-scale buffer is left duplicated, so expect
    slightly less recovery here than the diagnostic patch showed.

moe_kernels.py gains only a shape assert on caller-provided v2-layout buffers:
the kernel takes out.view(-1), so an undersized buffer would be a silent
out-of-bounds write rather than an error.

Reverts cleanly via the .prf_bak backups that _agentic_ladder.sh rewinds.
"""

import os
import shutil
import sys

FUSED = "/opt/aiter-local/aiter/fused_moe.py"
KERNELS = "/opt/aiter-local/aiter/ops/flydsl/moe_kernels.py"

CACHE_ANCHOR = "kernel_bench_callable = None\n"

CACHE = '''
# PATCH(prf-stage1-out-reuse)
# FlyDSL v2 stage1 produces an intermediate consumed immediately by stage2.
# Reuse it across layers and CUDA graphs on the same stream instead of retaining
# one allocation per layer and captured shape. Separate stream keys preserve
# correctness for overlapping launches, while exact shape keys keep graph-baked
# pointers stable.
_FLYDSL_STAGE1_OUT_CACHE: dict[
    tuple[torch.device, int, tuple[int, int]], torch.Tensor
] = {}


def _get_flydsl_stage1_out(
    shape: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    stream = torch.cuda.current_stream(device=device).cuda_stream
    key = (device, stream, shape)
    out = _FLYDSL_STAGE1_OUT_CACHE.get(key)
    if out is None:
        out = torch.empty(shape, dtype=dtypes.fp8, device=device)
        _FLYDSL_STAGE1_OUT_CACHE[key] = out
    return out

'''

OLD_GUARD = """    if out_dtype is not None:
        parsed = {**parsed, "out_dtype": out_dtype}
"""

NEW_GUARD = '''    if out_dtype is not None:
        parsed = {**parsed, "out_dtype": out_dtype}
    # PATCH(prf-stage1-out-reuse)
    if (
        out is None
        and v2_output_layout
        and parsed["out_dtype"] == "fp8"
        and parsed.get("k_batch", 1) == 1
        # a16w4 allocates and returns its own sorted intermediate before it
        # looks at `out`, so a buffer handed to it is silently dropped.
        and not (parsed["a_dtype"] == "bf16" and parsed["b_dtype"] == "fp4")
    ):
        device = hidden_states.device
        inter_dim = w1.shape[1] // 2
        sorted_rows = max(
            sorted_token_ids.shape[0],
            sorted_expert_ids.shape[0] * parsed["tile_m"],
        )
        out = _get_flydsl_stage1_out((sorted_rows, inter_dim), device)
'''

# Anchored on the statement that follows the `if out is None:` block rather than
# on the block's own last line: the tail of that block has drifted between aiter
# revisions (the installed one has an _alloc indirection the branch does not),
# while `if _is_splitk:` has not.
OLD_ASSERT = "\n    if _is_splitk:\n"

NEW_ASSERT = '''    elif _v2_output_layout:
        # PATCH(prf-stage1-out-reuse)
        # Nothing downstream re-checks a caller-provided buffer -- the kernel
        # takes out.view(-1) -- so an undersized one is an out-of-bounds write
        # rather than an error. Callers that size their own buffer duplicate
        # the padding rule above; fail loudly if the two ever diverge.
        _expected_shape = (
            max(sorted_token_ids.shape[0], sorted_expert_ids.shape[0] * tile_m),
            inter_dim // 2 if _need_fp4 else inter_dim,
        )
        if tuple(out.shape) != _expected_shape:
            raise ValueError(
                f"stage1 out has shape {tuple(out.shape)}, "
                f"but the v2 output layout requires {_expected_shape}"
            )

    if _is_splitk:
'''

MARK = "PATCH(prf-stage1-out-reuse)"


PLAN = {
    FUSED: [
        ("stage1 out cache", CACHE_ANCHOR, CACHE_ANCHOR + CACHE),
        ("wrapper reuse guard", OLD_GUARD, NEW_GUARD),
    ],
    KERNELS: [("v2 out shape assert", OLD_ASSERT, NEW_ASSERT)],
}


def main():
    # Resolve every anchor in every file before writing any of them. A partial
    # apply is worse than no apply: it leaves an arm that is neither baseline
    # nor treatment, and the ladder would happily benchmark it.
    staged = {}
    for path, edits in PLAN.items():
        src = open(path, encoding="utf-8").read()
        if MARK in src:
            print("[prf] %s already applied" % os.path.basename(path))
            continue
        for name, old, _new in edits:
            n = src.count(old)
            if n != 1:
                print("[prf] ERROR: %s anchor %r found %d times, expected 1" % (
                    os.path.basename(path), name, n))
                return 1
        for _name, old, new in edits:
            src = src.replace(old, new, 1)
        staged[path] = src

    for path, src in staged.items():
        shutil.copy2(path, path + ".prf_bak")
        open(path, "w", encoding="utf-8").write(src)
        print("[prf] patched %s (backup: %s.prf_bak)" % (
            os.path.basename(path), os.path.basename(path)))

    # Import the patched module rather than trusting the text edit: a syntax
    # error here would otherwise surface as an engine-core death 20 minutes into
    # the serve, long after the node time is spent.
    import subprocess

    check = subprocess.run(
        [sys.executable, "-c",
         "import aiter.fused_moe as m; "
         "assert hasattr(m, '_get_flydsl_stage1_out'); "
         "print('[prf] import check ok')"],
        capture_output=True, text=True,
    )
    print(check.stdout.strip() or check.stderr.strip()[-500:])
    return check.returncode


if __name__ == "__main__":
    sys.exit(main())
