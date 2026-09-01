#!/usr/bin/env python3
"""Two cp_common.py fixes for K3 DCP on MI355X: skip the NVLS multicast probe,
and release the symmetric-memory peer mesh in a SAFE ORDER at teardown.

Both target the same file, so they live in one applier. Each hunk is guarded by
its own marker and is independently idempotent.

=============================================================================
PART 2 -- ORDERED SYMM_MEM TEARDOWN  (the driver-leak fix)
=============================================================================
`DirectCPWorkspace._allocate()` exports one dma-buf per rank and imports it on
all `world_size` peers, three times (output/lse/signal) -- an 8x8x3 cross-process
BO mesh -- then pins storage+handle+views in `self._allocations` for the process
lifetime. There is no `__del__` and no free path.

At ordinary process exit those references die in ARBITRARY order, so an exporter
routinely calls release while peers still hold imports of its BO. The driver's
`amdgpu_bo_release_notify()` then loses the `dma_resv_trylock` on the ALIASED
private resv (amdgpu_object.c:1351 -- its premise "nobody else should have a
pointer to it" is false for an exported BO, because `ttm_bo_individualize_resv`
returns early at ttm_bo.c:196 when `resvp == &_resv` and never splits it). Losing
that lock means `amdgpu_amdkfd_remove_all_eviction_fences()` is SKIPPED, and the
BO strands at refcount 7 with live eviction fences.

Those unsignalled fences block the KFD delayed restore work, so the exiting rank
wedges in `cancel_delayed_work_sync` inside `kfd_process_notifier_release_internal`
-- which runs inside the GLOBAL mmu_notifier SRCU read section. Every other
process on the box doing `__mmu_notifier_release` then queues behind it in
`synchronize_srcu`. That is why jobs which never touch DCP saw ncclCommInitRank go
15s -> 279s. Measured dose-response, 8 ranks, clean exit, from a 0-orphan driver:
+16 orphans per round, 16.4s -> 36.5s -> 220.8s. ~50 orphans is enough for the
full stall, and they never self-clear.

THE FIX IS ORDERING, NOT FREEING. Every rank drops all its IMPORTS, then a
barrier guarantees no importer anywhere still references a peer's BO, and only
then do the EXPORTS go. Each release now sees a refcount nobody else holds and
takes the lock it expects.

Set VLLM_DCP_SKIP_SYMM_TEARDOWN=1 to restore the old (leaking) behaviour for A/B.

=============================================================================
PART 1 -- SKIP THE NVLS MULTICAST PROBE ON ROCm
=============================================================================

WHY
---
`_symm_mem_spans_group()` probes whether the DCP group has NVLS symmetric-memory
multicast. On ROCm that question is unanswerable-and-useless, but asking it is
not free -- it is the cause of the multi-minute distributed-group-init stalls.

Two distinct costs:

1. CONDITIONAL COLLECTIVE. `symm_mem.rendezvous()` inside the probe is a
   collective: every rank must call it, the same number of times. But the
   function has two paths that return *before* reaching it -- when
   `has_multicast_support()` is false, and when anything raises (the whole body
   is wrapped in `except Exception: return False`). Any rank-to-rank asymmetry
   in either path leaves some ranks blocked inside a rendezvous that their peers
   already skipped. That is exactly the signature we saw: group init sitting for
   minutes where a healthy box takes seconds.

2. PERMANENT LEAK. The probe tensor and its rendezvous handle are never freed,
   so each process leaks a symmetric allocation. An unclean exit leaks it at the
   driver level, so repeated killed runs degrade the box progressively -- and a
   reboot appears to "cure" it, which is what kept getting this misdiagnosed as
   box health rather than as our own code.

WHY SKIPPING IS SAFE
--------------------
Our native-HIP port deliberately has no multicast at all. From
`dcp_direct_common_hip.h`: "The CUDA-only multimem/multicast helpers are
intentionally DROPPED -- the a2a LSE-reduce op never used them (they belonged to
a separate all-reduce op)." So `_multicast_ptrs()` can only ever yield zeros,
and the gather paths raise "requires NVLS symmetric-memory multicast" if the
answer were ever True. Returning False early changes no reachable behaviour; it
only removes a collective and an allocation from the init path.

Note this probe is reachable *because of us*: setting VLLM_USE_DIRECT_DCP_A2A=1
makes `_direct_dcp_enabled()` short-circuit to True, after which
`_direct_dcp_multicast_enabled()` calls the probe unconditionally.

Idempotent; safe to re-run.
"""

import os
import sys

# The probe has moved twice: ops/dcp_utils.py -> ops/dcp.py (dev1046 nightly)
# -> ops/cp_common.py (f94666, where the CP/DCP split landed). Resolve by
# looking for the function rather than pinning a filename, so an image bump
# stops silently no-op'ing this patch and leaving the leaking probe live.
OPS = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops"
CANDIDATES = ("cp_common.py", "dcp.py", "dcp_utils.py")

MARK = "K3: skip NVLS multicast probe on ROCm"
MARK2 = "K3: ordered symm_mem teardown"

ANCHOR = '''def _symm_mem_spans_group(group: GroupCoordinator) -> bool:
    """Probe whether the group has NVLS symmetric memory."""
    if not symm_mem_available:
        return False
'''

PATCH = '''def _symm_mem_spans_group(group: GroupCoordinator) -> bool:
    """Probe whether the group has NVLS symmetric memory."""
    if not symm_mem_available:
        return False
    # --- K3: skip NVLS multicast probe on ROCm ---
    # The rendezvous below is a COLLECTIVE, but this function can return before
    # reaching it (no multicast support, or any exception -> the blanket
    # `except Exception: return False`). Rank-to-rank asymmetry there strands
    # some ranks inside a rendezvous their peers skipped, which is what stalls
    # distributed group init for minutes. The probe also leaks its tensor and
    # handle every call.
    #
    # Safe to skip: our native-HIP a2a port has no multicast kernels at all
    # (see dcp_direct_common_hip.h -- the multimem/multicast helpers were
    # intentionally dropped), so a True answer is unusable and the gather paths
    # raise on it regardless. Bail out before allocating or entering the
    # collective.
    if torch.version.hip is not None:
        return False
    # --- end K3: skip NVLS multicast probe on ROCm ---
'''

# --------------------------------------------------------------------------
# PART 2 hunks. Applied all-or-nothing under MARK2: every anchor is checked
# before anything is written, so a drifted file leaves cp_common.py untouched
# rather than half-patched.
# --------------------------------------------------------------------------

IMPORTS_ANCHOR = '''import functools
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.parallel_state import in_the_same_node_as
'''

IMPORTS_PATCH = '''import atexit
import functools
import os
import weakref
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import in_the_same_node_as
'''

REGISTRY_ANCHOR = '''class DirectCPWorkspace:
    def __init__(
'''

REGISTRY_PATCH = '''# --- K3: ordered symm_mem teardown ---
# Every live workspace, weakly held (a strong ref here would itself keep the
# peer mesh alive forever, which is the bug we are fixing).
_LIVE_CP_WORKSPACES: weakref.WeakSet = weakref.WeakSet()
_K3_ATEXIT_REGISTERED = False

# Seconds to wait on the teardown barrier before giving up. Generous: the
# ranks are unwinding an 8x8x3 BO mesh, and the KFD restore work behind it is
# not instant. If a peer has already died we would rather eat this than hang.
_K3_TEARDOWN_BARRIER_S = 120


def _k3_close_cp_workspaces() -> None:
    for workspace in list(_LIVE_CP_WORKSPACES):
        try:
            workspace.close()
        except Exception as error:  # never let teardown break exit
            logger.warning("Direct CP ordered teardown failed: %s", error)


def _k3_register_atexit() -> None:
    # Registered from __init__, NOT at import: atexit runs handlers in REVERSE
    # registration order, and this module is imported very early. Registering
    # at first-workspace-creation time puts us ahead of the process-group
    # teardown that vLLM registers at import, so dist is still usable when we
    # run.
    global _K3_ATEXIT_REGISTERED
    if not _K3_ATEXIT_REGISTERED:
        atexit.register(_k3_close_cp_workspaces)
        _K3_ATEXIT_REGISTERED = True


# --- end K3: ordered symm_mem teardown ---


class DirectCPWorkspace:
    def __init__(
'''

INIT_ANCHOR = '''        self._allocations: list[tuple[torch.Tensor, Any, list[torch.Tensor]]] = []

    def _allocate(
'''

INIT_PATCH = '''        self._allocations: list[tuple[torch.Tensor, Any, list[torch.Tensor]]] = []
        # --- K3: ordered symm_mem teardown ---
        self._closed = False
        _LIVE_CP_WORKSPACES.add(self)
        _k3_register_atexit()
        # --- end K3: ordered symm_mem teardown ---

    def _allocate(
'''

CLOSE_ANCHOR = '''    def _multicast_ptrs(self, storage: torch.Tensor) -> list[int]:
'''

CLOSE_PATCH = '''    # --- K3: ordered symm_mem teardown ---
    def close(self) -> None:
        """Release the symmetric-memory peer mesh in exporter-last order.

        `_allocate()` exports one BO per rank and imports it on every peer,
        then pins (storage, handle, views) here for the process lifetime.
        Left to interpreter teardown those refs die in arbitrary order, so an
        exporter routinely releases while peers still hold imports of its BO.
        `amdgpu_bo_release_notify()` then loses the `dma_resv_trylock` on the
        aliased private resv (amdgpu_object.c:1351 -- its premise "nobody else
        should have a pointer to it" is false for an exported BO, because
        `ttm_bo_individualize_resv` returns early at ttm_bo.c:196 when
        `resvp == &_resv` and never splits it), so
        `amdgpu_amdkfd_remove_all_eviction_fences()` is skipped and the BO
        strands at refcount 7 with live eviction fences. Those fences block
        the KFD delayed restore work, the exiting rank wedges in
        `cancel_delayed_work_sync` inside the GLOBAL mmu_notifier SRCU read
        section, and every other exiting process on the box queues behind it.

        So the fix is ORDERING, not freeing: drop all IMPORTS everywhere,
        barrier, then drop the EXPORTS. Each release then sees a refcount
        nobody else holds and takes the lock it expects.

        Idempotent. Never raises.
        """
        if self._closed:
            return
        self._closed = True
        if os.environ.get("VLLM_DCP_SKIP_SYMM_TEARDOWN", "0") == "1":
            return  # A/B escape hatch: restores the old leaking behaviour
        allocations, self._allocations = self._allocations, []
        if not allocations:
            return

        # 1. imported peer BOs
        for _storage, _handle, views in allocations:
            views.clear()
        try:
            torch.accelerator.synchronize()
        except Exception as error:
            logger.warning("Direct CP teardown sync failed: %s", error)

        # 2. no importer anywhere still references a peer BO. Best-effort: a
        #    dead peer must not turn teardown into a hang, so this is guarded
        #    and time-bounded. Without it the ordering is only local, which is
        #    still better than nothing but does not close the race.
        try:
            if dist.is_available() and dist.is_initialized():
                work = dist.barrier(group=self.group, async_op=True)
                work.wait(timedelta(seconds=_K3_TEARDOWN_BARRIER_S))
        except Exception as error:
            logger.warning("Direct CP teardown barrier skipped: %s", error)

        # 3. handles + our own exported storage
        allocations.clear()
        try:
            torch.accelerator.synchronize()
        except Exception as error:
            logger.warning("Direct CP teardown sync failed: %s", error)

    # --- end K3: ordered symm_mem teardown ---

    def _multicast_ptrs(self, storage: torch.Tensor) -> list[int]:
'''

HUNKS2 = (
    ("imports", IMPORTS_ANCHOR, IMPORTS_PATCH),
    ("workspace registry + atexit", REGISTRY_ANCHOR, REGISTRY_PATCH),
    ("__init__ registration", INIT_ANCHOR, INIT_PATCH),
    ("close()", CLOSE_ANCHOR, CLOSE_PATCH),
)


def resolve() -> str | None:
    for name in CANDIDATES:
        path = os.path.join(OPS, name)
        if os.path.exists(path) and "_symm_mem_spans_group" in open(path).read():
            return path
    return None


def main() -> int:
    path = resolve()
    if path is None:
        print(f"  !! _symm_mem_spans_group not found in {CANDIDATES} under {OPS}")
        return 1
    src = original = open(path).read()

    # Part 1 -- skip the NVLS multicast probe on ROCm.
    if MARK in src:
        print("  part 1 (skip multicast probe): already patched")
    elif src.count(ANCHOR) != 1:
        print(f"  !! part 1 anchor matched {src.count(ANCHOR)}x, expected 1")
        return 1
    else:
        src = src.replace(ANCHOR, PATCH)
        print("  part 1 (skip multicast probe): applied")

    # Part 2 -- ordered symm_mem teardown. All-or-nothing: check every anchor
    # before writing, so a drifted file is left untouched rather than half done.
    if MARK2 in src:
        print("  part 2 (ordered symm_mem teardown): already patched")
    else:
        for name, anchor, _ in HUNKS2:
            if src.count(anchor) != 1:
                print(
                    f"  !! part 2 anchor '{name}' matched "
                    f"{src.count(anchor)}x, expected 1 -- nothing written"
                )
                return 1
        for _, anchor, patch in HUNKS2:
            src = src.replace(anchor, patch)
        print("  part 2 (ordered symm_mem teardown): applied")

    if src == original:
        print(f"  no change -> {path}")
        return 0
    open(path, "w").write(src)
    print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
