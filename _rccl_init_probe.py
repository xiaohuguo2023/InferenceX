#!/usr/bin/env python3
"""Is 8-rank RCCL init itself broken here? No vLLM, no weights, no DCP.

Why this exists
---------------
A DCP serve wedged with all 8 workers parked in ``ncclCommInitRank``
(``recv`` inside librccl's TCP bootstrap) at
``initialize_model_parallel`` line 1843 -- which is the **TP** group, built
*before* the DCP group at line 1861 and long before any weight loads (VRAM was
6 GiB) or any DCP kernel. So the wedge cannot be attributed to DCP, and the
model loader is not involved either.

That leaves one question worth answering cheaply: does plain 8-rank RCCL
bootstrap work in this container at all? This reproduces exactly the sequence
vLLM performs -- world group, then a subgroup over the same ranks, then a
collective on it -- in about a minute instead of an eight-minute serve boot.

What it found (2026-08-21, run under an identical foreign 8-rank job)
--------------------------------------------------------------------
It reproduced the wedge exactly: all 8 ranks cleared ``init_process_group`` in
0.8 s, then never completed the first ``all_reduce`` -- the point where
``ncclCommInitRank`` actually builds the communicator. Same place vLLM parks.

``NCCL_DEBUG=INFO`` named the cause. RCCL enumerates **112 channels** on this
node, each with its own tree and proxy connection, and it was still advancing
(proxy connection 102) when killed -- so it is not a deadlock, it is setup work
whose cost scales with channel count and explodes when a second process set is
GPU-resident:

    channels | world all_reduce | outcome
    ---------|------------------|-------------------------------------------
    112      | not in 200 s     | hang
      8      | 187.6 s          | timed out building the 2nd communicator
      2      |  79.3 s          | PASS at 126.9 s (world + tp + dcp)

So the fix is to bound the channel count (``NCCL_MAX_NCHANNELS``), which is now
set in ``_serve_k3_dcp_test.sh``. Note this fixes *init*; the H2D weight-load
cost under contention is a separate axis, so a busy box is still a bad place to
load 519,000 tensors.

Re-run this whenever a boot parks in ``ncclCommInitRank``: it distinguishes
"channel-setup cost" from a genuine collective deadlock in about two minutes,
and it needs only ~1 GiB per GPU so it runs on a box someone else is using.

Run it under an external timeout so a wedge terminates rather than traps GPUs:

    NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,ENV \
    timeout --signal=TERM --kill-after=20s 300s \
        torchrun --standalone --nproc-per-node=8 _rccl_init_probe.py
"""

import builtins
import functools
import os
import time

import torch
import torch.distributed as dist

# Any stall must show partial output, so never block-buffer.
print = functools.partial(builtins.print, flush=True)  # noqa: A001


def stamp(rank, tag, t0):
    print(f"[rank {rank}] {tag:<34s} +{time.perf_counter() - t0:7.2f} s")


def main():
    t0 = time.perf_counter()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    stamp(rank, "set_device", t0)

    dist.init_process_group(backend="nccl")
    stamp(rank, "init_process_group (world)", t0)

    # Force the world communicator to actually materialise.
    x = torch.ones(1024, device=f"cuda:{local_rank}")
    dist.all_reduce(x)
    torch.cuda.synchronize()
    stamp(rank, "world all_reduce", t0)
    assert x[0].item() == world, x[0].item()

    # This is the step vLLM is wedged in: a *subgroup* over the same ranks.
    tp = dist.new_group(ranks=list(range(world)), backend="nccl")
    stamp(rank, "new_group(tp) created", t0)

    y = torch.ones(1024, device=f"cuda:{local_rank}")
    dist.all_reduce(y, group=tp)
    torch.cuda.synchronize()
    stamp(rank, "tp all_reduce", t0)
    assert y[0].item() == world, y[0].item()

    # And the second subgroup -- the one DCP would add.
    dcp = dist.new_group(ranks=list(range(world)), backend="nccl")
    stamp(rank, "new_group(dcp) created", t0)

    z = torch.ones(1024, device=f"cuda:{local_rank}")
    dist.all_reduce(z, group=dcp)
    torch.cuda.synchronize()
    stamp(rank, "dcp all_reduce", t0)

    dist.barrier()
    if rank == 0:
        print()
        print("PASS: 8-rank RCCL bootstrap + two same-rank subgroups are healthy.")
        print("      => the serve wedge is NOT environmental RCCL.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
