#!/usr/bin/env python3
"""Shape tracing for the direct DCP a2a combine (K3, MI355X).

Purpose: the DIAG-OFF serve dies at DSpark speculator capture with

    Memory access fault by GPU node-N on address 0x... Reason: Unknown.

which carries no Python frame -- the fault lands on whichever rank owned the
clobbered symmetric allocation, not the rank that issued the bad access. To
attribute it we need to know the exact tensor shape each call presents, and in
particular whether the *draft* path differs from the target path.

NOTE ON SEMANTICS (learned the hard way -- do not "fix" this again):
`lse_reduce` is a REDUCE-SCATTER. `partial_output` legitimately carries every
rank's heads, i.e. `world_size * heads_per_rank` (96 = 8 x 12 for K3), while the
returned tensor has only `heads_per_rank` (12). Asserting
`partial_output.shape[1] == heads_per_rank` is WRONG and false-fires on the
first perfectly valid call. The HIP op already gets this right:

    TORCH_CHECK(num_tokens > 0 && num_tokens <= max_num_tokens)
    TORCH_CHECK(total_heads % world_size == 0)
    heads_per_rank = total_heads / world_size

So this patch does NOT re-assert what C++ already covers. It only:
  1. logs the workspace envelope once at construction, and
  2. logs each DISTINCT (num_tokens, total_heads, head_dim) once per rank, so
     the draft's shape shows up in the log immediately before the fault.

Logging is once-per-distinct-shape, so it stays off the steady-state path.
Idempotent; safe to re-run.
"""

import sys

PATH = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/dcp_utils.py"

MARK = "K3 dcp shape trace"

# --- the previous (incorrect) revision of this patch, to be stripped ---------
OLD_INIT_START = "        # --- K3 diag: record the envelope"
OLD_INIT_END = "        # --- end K3 diag ---\n"
OLD_CALL_START = "        # --- K3 bounds check:"
OLD_CALL_END = "        # --- end K3 bounds check ---\n"

INIT_ANCHOR = """        self.max_num_tokens = max_num_tokens
        self.heads_per_rank = heads_per_rank
        self.head_dim = head_dim
"""

INIT_PATCH = """        self.max_num_tokens = max_num_tokens
        self.heads_per_rank = heads_per_rank
        self.head_dim = head_dim
        # --- K3 dcp shape trace (init) ---
        self._k3_seen_shapes: set = set()
        logger.info(
            "K3 dcp workspace: max_num_tokens=%d heads_per_rank=%d head_dim=%d "
            "world_size=%d num_ubatches=%d dtype=%s (expects total_heads=%d)",
            max_num_tokens,
            heads_per_rank,
            head_dim,
            self.world_size,
            num_ubatches,
            dtype,
            self.world_size * heads_per_rank,
        )
        # --- end K3 dcp shape trace (init) ---
"""

CALL_ANCHOR = """        ubatch = dbo_current_ubatch_id()
        num_tokens = partial_output.shape[0]
"""

CALL_PATCH = """        ubatch = dbo_current_ubatch_id()
        num_tokens = partial_output.shape[0]
        # --- K3 dcp shape trace (call). Reduce-scatter: partial_output carries
        # world_size*heads_per_rank heads; the result carries heads_per_rank. ---
        _k3_shape = (
            num_tokens,
            partial_output.shape[1],
            partial_output.shape[2],
            partial_output.dtype,
            partial_output.is_contiguous(),
            partial_lse.shape[0],
            partial_lse.shape[1],
            partial_lse.is_contiguous(),
        )
        if _k3_shape not in self._k3_seen_shapes:
            self._k3_seen_shapes.add(_k3_shape)
            logger.info(
                "K3 dcp shape trace: out=(%d,%d,%d) %s contig=%s | "
                "lse=(%d,%d) contig=%s | envelope max_num_tokens=%d "
                "total_heads=%d ubatch=%d",
                _k3_shape[0],
                _k3_shape[1],
                _k3_shape[2],
                _k3_shape[3],
                _k3_shape[4],
                _k3_shape[5],
                _k3_shape[6],
                _k3_shape[7],
                self.max_num_tokens,
                self.world_size * self.heads_per_rank,
                ubatch,
            )
        # --- end K3 dcp shape trace (call) ---
"""


def _strip(src: str, start: str, end: str) -> str:
    """Remove a previously inserted block, inclusive of its end marker."""
    while start in src:
        i = src.index(start)
        j = src.index(end, i) + len(end)
        src = src[:i] + src[j:]
    return src


def main() -> int:
    src = open(PATH).read()

    if MARK in src:
        print("  already patched")
        return 0

    # Remove the earlier, incorrect bounds-check revision if present.
    if OLD_INIT_START in src or OLD_CALL_START in src:
        src = _strip(src, OLD_INIT_START, OLD_INIT_END)
        src = _strip(src, OLD_CALL_START, OLD_CALL_END)
        print("  stripped previous (incorrect) bounds-check revision")

    for name, anchor, patch in (
        ("init trace", INIT_ANCHOR, INIT_PATCH),
        ("call trace", CALL_ANCHOR, CALL_PATCH),
    ):
        if src.count(anchor) != 1:
            print(f"  !! anchor for {name} matched {src.count(anchor)}x, expected 1")
            return 1
        src = src.replace(anchor, patch)
        print(f"  applied: {name}")

    open(PATH, "w").write(src)
    print(f"  wrote {PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
