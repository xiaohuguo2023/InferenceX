#!/usr/bin/env python3
"""Bisect DCP bring-up phase by phase: TP8, no weights, no `vllm serve`.

WHY THIS EXISTS
---------------
Every DCP failure so far has looked the same from the outside -- an idle-looking
hang -- and each one cost a full server launch to reach. That conflates at least
six independent systems (group init, symmetric memory, the head fold, the AITER
MLA producer, the combine, graph capture) with two that have nothing to do with
DCP (loading 1.5 TiB of weights, and process cleanup after a wedge). The most
recent load wedge happened *before the DCP operator ever executed*, so full-server
launches are mostly re-testing the model loader.

This harness runs the DCP bring-up sequence in isolation, one phase at a time,
in under a minute, with no checkpoint I/O and no API server. Each phase runs
under an external timeout (see `_dcp_phase_bisect.sh`), and every phase boundary
samples the three things that actually distinguish "slow" from "wedged":

  * KFD ``evicted_ms``  -- is the driver evicting queues? This is the signal that
    goes with the box-wide degradation; it is per-process and per-GPU.
  * GTT used            -- userptr/DMA pressure against the 1.62 TiB budget.
  * tiny-kernel latency -- a 17 us reference on a healthy box. Single-process GPU
    work stays clean even when collectives collapse, so this separates "this GPU
    is sick" from "anything crossing ranks is sick".

WHAT IS ALREADY RULED IN / OUT
------------------------------
The op itself is cleared: `_test_a2a_syncfree.py` drives 60 back-to-back combines
with no per-call barrier and deliberate skew, and gets cos=0.999999 on all 8
ranks. But it does that over ONE process group. Production does not:
``initialize_model_parallel`` builds **tp, dcp, pcp, pp, dp, ep** -- and the ep
group is created whenever ``model_config.is_moe``, regardless of
``enable_expert_parallel``. tp and dcp additionally carry a message-queue
broadcaster (shared memory + ZMQ) and a custom all-reduce with its own IPC
buffers. So a real boot has seven overlapping communicators, each with its own
handle exchange, where the passing test had one.

That is the largest untested delta, which is why `groups` is phase 1.

PHASES (cumulative -- phase N assumes N-1 succeeded)
  groups   create tp/dcp/pcp/pp/dp/ep exactly as vLLM does, in vLLM's order,
           with the same broadcaster/all2all flags, and drive a collective on each
  symm     allocate + rendezvous the direct-DCP output/LSE/signal buffers
  combine  run the direct A2A combine on the real dcp group
  fold     build the 96 -> 32-head folded metadata
  mla      run the raw AITER cprr mla_decode_fwd producer
  graph    capture and replay the whole MLA -> combine chain
  serve    the same chain at SERVE SHAPE: the whole capture-size ladder into one
           shared graph pool, under real VRAM pressure, with a per-step layer
           loop -- the three things every phase above fakes

`fold`, `mla` and `graph` reuse the builders in `_dcp_folded_mla_standalone.py`,
which already drives that chain single-GPU against *fake* groups. The one thing
that changes here is where the groups come from: `phase_groups` publishes the
real coordinators into `parallel_state`, so `get_dcp_group()` resolves to a live
NCCL communicator instead of a stub. Everything downstream -- the fold factor,
the cprr metadata, the asm producer -- is then the production code path, with the
model loader and the API server the only things still absent.

NO GPU TRAPS. Nothing here calls `__builtin_trap()` or asserts on device. A trap
raises a queue fault, the driver releases buffer objects badly, and the KFD
evict/restore workqueue then thrashes box-wide -- a poisoned box that a
`docker restart` does not cure, because the state is in the kernel, not the
container. That mechanism is why the trap was removed from the combine kernel.
(Verified 2026-08-24: the shipped `dcp_direct_a2a_lse_reduce.so` disassembles to
zero `s_trap` instructions, so it cannot be the cause of any *future* wedge.)

SECOND MODE: LEAK ARMS (--arm)
------------------------------
The phases above ask "does the chain work". They cannot answer the other
question -- "does running it leave residue in the kernel driver" -- because that
damage is what makes the box slow *afterwards*, in unrelated processes. Since a
reboot cures it and `docker restart` does not, it lives in the driver, and the
suspect is the cross-process dma-buf mesh `symm_mem` builds on ROCm
(hipMemCreate + hipMemExportToShareableHandle -> KFD dma-buf, imported by all
seven peers, three times per workspace).

That is a differential test, not a functional one. Each arm is run twice --
exiting cleanly, and SIGKILLed mid-flight -- and the residue is differenced:

  nccl      A  NCCL init + all-reduce only .............. control
  symmonly  B  A + the symm_mem mesh, no DCP kernel ..... the dma-buf mesh
  probe     C  B + the leaking multicast probe .......... the probe leak
  dcp       D  real groups + workspace + combine ........ the production path
  dcpstall  D' as dcp, but one rank withholds its signal so peers run the
               full ~8 s spin -- the worst case to be killed in

A vs B is the decisive comparison: if a killed A leaves nothing and a killed B
leaves stale `kfd_process` nodes, the leak is symmetric memory, and no model,
MLA or server is implicated. Run each arm N times and read the residue as a
*slope*, since the claim is progressive degradation.

USAGE
    torchrun --standalone --nproc-per-node=8 _dcp_phase_bisect.py --stop-after groups
    torchrun --standalone --nproc-per-node=8 _dcp_phase_bisect.py \
        --arm symmonly --require-clean --clean-exit

Always run it under an external timeout so a wedge terminates rather than
trapping the GPUs -- `_dcp_phase_bisect.sh` does this for you, and its
`leak` mode drives the whole arm matrix with dmesg deltas.
"""

import argparse
import builtins
import functools
import glob
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist

# Any stall must show partial output, so never block-buffer.
print = functools.partial(builtins.print, flush=True)  # noqa: A001

RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "8"))

ALL_PHASES = ["groups", "symm", "combine", "fold", "mla", "graph", "serve"]

# The fold / MLA / graph phases reuse the builders in
# `_dcp_folded_mla_standalone.py`, which lives one directory up. That driver
# already proves this chain works on ONE GPU with *fake* groups; the whole point
# here is to run the same chain against the REAL overlapping communicators, so
# the only thing that changes is where the groups come from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

T0 = time.perf_counter()


def log(phase, event, extra=""):
    """Every collective gets one of these before and one after.

    Rank and epoch are both in the line because the failure we are chasing is a
    partial one: some ranks pass a collective and others do not, and the pair
    (rank, epoch) is what tells you which.
    """
    print(f"[r{RANK} +{time.perf_counter() - T0:7.2f}s] "
          f"{phase:<8s} {event:<26s} {extra}")


# ---------------------------------------------------------------- instrumentation

def kfd_evicted_ms():
    """Summed KFD ``evicted_ms`` across this process's per-GPU stat dirs.

    This is the counter that moves when the driver starts evicting queues. It
    lives under the *process* node, so it only exists once we have a KFD handle
    (i.e. after the first CUDA call), and it is zero on a healthy run.
    """
    total, found = 0, 0
    for d in glob.glob("/sys/class/kfd/kfd/proc/*/stats_*/evicted_ms"):
        try:
            with open(d) as fh:
                total += int(fh.read().strip())
            found += 1
        except (OSError, ValueError):
            continue
    return total, found


def nested_pid_ns():
    """True if our /proc cannot resolve the host pids that sysfs reports.

    sysfs is box-wide but ``/proc`` is namespaced, so inside a container with its
    own pid namespace (docker's default -- ``--pid=host`` is what turns this off)
    every KFD node belonging to any other process looks like it has no pid. That
    turns a healthy box shared with another user into a fake pile of leaks, which
    is exactly the misreading this whole investigation is trying to avoid making
    in the other direction.

    Detected by namespace inode: the kernel's *initial* pid namespace always has
    inode ``PROC_PID_INIT_INO`` = 0xEFFFFFFC. (``/proc/self/status``'s ``NSpid``
    does not work for this -- read from inside the namespace it lists only the
    pids at or below our own level, so it looks unnested from either side.)
    """
    try:
        return os.readlink("/proc/self/ns/pid") != "pid:[4026531836]"
    except OSError:
        return False


def kfd_procs():
    """(live, stale) KFD process nodes; stale is None when unadjudicable.

    ``/sys/class/kfd/kfd/proc/<pid>`` exists for as long as the kernel's
    ``kfd_process`` does. A node whose pid is gone from ``/proc`` is a
    ``kfd_process`` the driver could not free -- normally because something else
    still holds a reference to its buffer objects, which on our stack means a
    peer's dma-buf import. That is the sharpest single signal we have for the
    leak we are chasing: a *stale* count above zero is residue, full stop.

    But only if we can actually see the pids. From a nested pid namespace we
    cannot, so return ``None`` rather than a number that would be all
    false positives -- see :func:`nested_pid_ns`. Run the leak matrix from the
    host, or from a ``--pid=host`` container, to get a real answer here.

    The counts are box-wide, not per-process, because the damage is box-wide.
    """
    nodes = [os.path.basename(d) for d in glob.glob("/sys/class/kfd/kfd/proc/*")]
    nodes = [p for p in nodes if p.isdigit()]
    if nested_pid_ns():
        return len(nodes), None
    live, stale = 0, []
    for pid in nodes:
        if os.path.exists(f"/proc/{pid}"):
            live += 1
        else:
            stale.append(int(pid))
    return live, sorted(stale)


def dmabuf_fds(pid="self"):
    """Count this process's open dma-buf fds.

    Each `symm_mem` peer mapping exports/imports a dma-buf, which shows up here
    as an `anon_inode:dmabuf` link. No root needed, so it works from inside the
    rank. Counting it per rank is how we turn "symmetric memory presumably makes
    dma-bufs" into a number we can difference across arms.
    """
    n = 0
    try:
        for fd in os.listdir(f"/proc/{pid}/fd"):
            try:
                if "dmabuf" in os.readlink(f"/proc/{pid}/fd/{fd}"):
                    n += 1
            except OSError:
                continue
    except OSError:
        return -1
    return n


def dmabuf_global():
    """(count, total_bytes) of every dma-buf on the box, or (-1, -1).

    Needs debugfs mounted (`mount -t debugfs none /sys/kernel/debug` in a
    privileged exec -- see `_dcp_phase_bisect.sh`). This is the only view that
    survives the exporting process, so it is what shows a dma-buf outliving the
    rank that made it.
    """
    path = "/sys/kernel/debug/dma_buf/bufinfo"
    try:
        with open(path) as fh:
            text = fh.read()
    except OSError:
        return -1, -1
    count, total = 0, 0
    for line in text.splitlines():
        parts = line.split()
        # bufinfo rows start with the size; the header line does not.
        if len(parts) >= 2 and parts[0].isdigit():
            count += 1
            total += int(parts[0])
    return count, total


def _rss_gib():
    """This process's resident host memory. The serve phase's ballast is meant to
    be pure device memory, so RSS is the tell that it is not."""
    try:
        with open("/proc/self/status") as fh:
            for ln in fh:
                if ln.startswith("VmRSS:"):
                    return int(ln.split()[1]) / (1 << 20)
    except OSError:
        pass
    return 0.0


def gtt_used_gib():
    """Summed GTT usage over all amdgpu cards, in GiB.

    The model-loading wedge has a 1.5 TiB mmap cache pressing on a 1.62 TiB GTT
    budget, so this is the number that says whether we are near that wall. With
    no weights loaded it should stay tiny -- if it climbs here, something in DCP
    bring-up is pinning host memory.
    """
    tot = 0
    for p in glob.glob("/sys/class/drm/card*/device/mem_info_gtt_used"):
        try:
            with open(p) as fh:
                tot += int(fh.read().strip())
        except (OSError, ValueError):
            continue
    return tot / (1 << 30)


def tiny_kernel_us(n=32):
    """Median latency of a trivial kernel + sync. Healthy reference: ~17 us.

    Deliberately single-process and single-GPU: on the degraded box this stays
    clean while collectives collapse by 4 orders of magnitude, so a normal value
    here plus a bad phase time localises the problem to the cross-rank path.
    """
    x = torch.ones(16, device="cuda")
    for _ in range(4):                       # warm
        x.add_(1.0)
    torch.cuda.synchronize()
    samples = []
    for _ in range(n):
        t = time.perf_counter()
        x.add_(1.0)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t) * 1e6)
    return statistics.median(samples)


def driver_state():
    """One dict with every driver-residue counter we can read from userspace.

    Printed by the leak arms before and after the body, and by the outer script
    before and after the whole job. Everything here is box-wide except
    ``self_dmabuf``, so it stays meaningful once the rank that produced it is
    gone -- which is the entire point when the question is what a *killed* run
    leaves behind.
    """
    ev, nstat = kfd_evicted_ms()
    live, stale = kfd_procs()
    ndb, bytes_db = dmabuf_global()
    return {
        "evicted_ms": ev,
        "stat_dirs": nstat,
        "kfd_nodes": live if stale is None else live + len(stale),
        "kfd_live": live,
        # None means "cannot tell from here", not "none found".
        "kfd_stale": None if stale is None else len(stale),
        "kfd_stale_pids": stale,
        "dmabuf_n": ndb,
        "dmabuf_gib": -1.0 if bytes_db < 0 else bytes_db / (1 << 30),
        "self_dmabuf": dmabuf_fds(),
        "gtt_gib": gtt_used_gib(),
    }


def fmt_state(s):
    db = "n/a" if s["dmabuf_n"] < 0 else f"{s['dmabuf_n']}/{s['dmabuf_gib']:.2f}GiB"
    if s["kfd_stale"] is None:
        stale = "n/a(nested-pidns)"
    else:
        stale = f"{s['kfd_stale']}{s['kfd_stale_pids'] or ''}"
    return (f"evicted_ms={s['evicted_ms']} statdirs={s['stat_dirs']} "
            f"kfd_nodes={s['kfd_nodes']} kfd_stale={stale} "
            f"dmabuf={db} self_dmabuf={s['self_dmabuf']} "
            f"gtt={s['gtt_gib']:.2f}GiB")


def _d_stale(pre, post):
    """Residue delta, falling back to total node count in a nested pid ns.

    ``kfd_nodes`` is a weaker signal than ``kfd_stale`` -- another user starting
    a job also raises it -- but it is the honest one when we cannot resolve pids,
    and a *rising* node count with our own ranks gone is still evidence.
    """
    if pre["kfd_stale"] is None or post["kfd_stale"] is None:
        return f"d_nodes={post['kfd_nodes'] - pre['kfd_nodes']:+d}(no pidns)"
    return f"d_stale={post['kfd_stale'] - pre['kfd_stale']:+d}"


class Probe:
    """Snapshot the counters, then report deltas across a phase."""

    def __init__(self):
        self.s = driver_state()
        self.ev, self.nstat = self.s["evicted_ms"], self.s["stat_dirs"]
        self.gtt = self.s["gtt_gib"]
        self.t = time.perf_counter()

    def report(self, phase, with_kernel=True):
        s = driver_state()
        ev, nstat, gtt = s["evicted_ms"], s["stat_dirs"], s["gtt_gib"]
        dt = time.perf_counter() - self.t
        d_ev = ev - self.ev
        knl = f" tiny_kernel={tiny_kernel_us():.1f}us" if with_kernel else ""
        log(phase, "PHASE DONE",
            f"wall={dt:.2f}s d_evicted_ms={d_ev} (abs={ev}, {nstat} stat dirs) "
            f"gtt={gtt:.2f}GiB (d={gtt - self.gtt:+.2f}) "
            f"{_d_stale(self.s, s)} "
            f"d_dmabuf={s['dmabuf_n'] - self.s['dmabuf_n']:+d}{knl}")
        # Queue eviction starting inside a phase is the stop condition: it is
        # the transition into the box-wide degraded state, and every phase after
        # it would be measuring a different machine.
        return d_ev


# ---------------------------------------------------------------- phase: groups

def phase_groups(args):
    """Build tp/dcp/pcp/pp/dp/ep exactly as vLLM's initialize_model_parallel does.

    Ranks are computed from the same `all_ranks` tensor and unbound in the same
    order, and the same broadcaster/all2all flags are passed, so that if the
    overlap of these communicators is what wedges a real boot, it wedges here too
    -- in under a minute, with no weights behind it.

    Not called via `initialize_model_parallel` itself because that reads a live
    VllmConfig for the ep/all2all decisions; constructing the groups directly
    keeps the harness free of the model config while preserving the call
    sequence that matters.
    """
    from vllm.distributed.parallel_state import (
        get_world_group,
        init_distributed_environment,
        init_model_parallel_group,
    )

    tp, dcp = args.tp, args.dcp
    dp = pp = pcp = 1

    log("groups", "init_dist_env BEFORE")
    init_distributed_environment(world_size=WORLD, rank=RANK,
                                 local_rank=LOCAL_RANK, backend="nccl")
    log("groups", "init_dist_env AFTER", f"epoch=0 world={WORLD}")

    all_ranks = torch.arange(WORLD).reshape(-1, dp, pp, pcp, tp)
    local_rank = get_world_group().local_rank

    def ranks_of(t, width):
        return [x.tolist() for x in t.reshape(-1, width).unbind(0)]

    # (name, group_ranks, kwargs) in vLLM's construction order.
    dcp_src = all_ranks.transpose(-1, -2) if dcp > 1 else all_ranks
    specs = [
        ("tp", ranks_of(all_ranks, tp), dict(use_message_queue_broadcaster=True)),
        ("dcp", ranks_of(dcp_src, dcp), dict(use_message_queue_broadcaster=True)),
        ("pcp", ranks_of(all_ranks.transpose(3, 4), pcp), {}),
        ("pp", ranks_of(all_ranks.transpose(2, 4), pp), {}),
        ("dp", ranks_of(all_ranks.transpose(1, 4), dp), {}),
        # vLLM creates ep for ANY MoE model, not just when expert parallel is on.
        ("ep", ranks_of(all_ranks.transpose(1, 2), dp * pcp * tp),
         dict(use_all2all=args.use_all2all)),
    ]

    groups = {}
    epoch = 0
    for name, group_ranks, kw in specs:
        log("groups", f"create {name} BEFORE", f"epoch={epoch} ranks={group_ranks}")
        t = time.perf_counter()
        groups[name] = init_model_parallel_group(
            group_ranks, local_rank, "nccl", group_name=name, **kw)
        epoch += 1
        log("groups", f"create {name} AFTER",
            f"epoch={epoch} took={time.perf_counter() - t:.2f}s")

        # Creating a communicator is lazy; a collective is what forces
        # ncclCommInitRank to actually run, which is where boots have parked.
        log("groups", f"collective {name} BEFORE", f"epoch={epoch}")
        t = time.perf_counter()
        x = torch.ones(1024, device="cuda")
        dist.all_reduce(x, group=groups[name].device_group)
        torch.cuda.synchronize()
        epoch += 1
        n = groups[name].world_size
        ok = abs(x[0].item() - n) < 1e-3
        log("groups", f"collective {name} AFTER",
            f"epoch={epoch} took={time.perf_counter() - t:.2f}s "
            f"sum={x[0].item():.0f}/{n} {'OK' if ok else 'WRONG'}")
        if not ok:
            raise RuntimeError(f"{name} all_reduce produced {x[0].item()}, want {n}")

    # Publish the coordinators the way initialize_model_parallel would. This is
    # what makes the later phases meaningful: `_dcp_folded_mla_standalone.py`
    # installs *stubs* over get_dcp_group/get_tp_group so it can run one GPU with
    # no NCCL, and the whole question here is what the same chain does against
    # live communicators. Setting the module globals means the production
    # accessors resolve to the real groups and nothing has to be monkeypatched --
    # including the copy of `get_dcp_group` the aiter MLA backend binds at import.
    import vllm.distributed.parallel_state as ps

    ps._TP, ps._DCP, ps._PCP = groups["tp"], groups["dcp"], groups["pcp"]
    ps._PP, ps._DP, ps._EP = groups["pp"], groups["dp"], groups["ep"]
    log("groups", "published to parallel_state",
        f"dcp.rank_in_group={groups['dcp'].rank_in_group} "
        f"dcp.world_size={groups['dcp'].world_size}")

    return groups


# ---------------------------------------------------------------- phase: symm

def phase_symm(args, groups):
    """Allocate + rendezvous the direct-DCP output/LSE/signal buffers.

    Same workspace the passing sync-free test uses, but built on the *real* dcp
    GroupCoordinator created above rather than the world group -- so this is the
    first point where the symmetric-memory handle exchange happens alongside six
    other live communicators.
    """
    try:
        from vllm.v1.attention.ops.dcp import DirectDCPA2AWorkspace
    except ImportError:
        from vllm.v1.attention.ops.dcp_utils import DirectDCPA2AWorkspace

    so = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "dcp_direct_a2a_lse_reduce.so")
    log("symm", "load_library BEFORE", so)
    torch.ops.load_library(so)
    assert hasattr(torch.ops._C, "direct_dcp_a2a_lse_reduce"), "op not registered"
    log("symm", "load_library AFTER")

    pg = groups["dcp"].device_group
    dev = torch.device("cuda", LOCAL_RANK)
    log("symm", "workspace BEFORE",
        f"max_nt={args.max_nt} hpr={args.hpr} hdim={args.hdim}")
    t = time.perf_counter()
    ws = DirectDCPA2AWorkspace(pg, dev, args.max_nt, args.hpr, args.hdim,
                               torch.bfloat16)
    log("symm", "workspace AFTER", f"took={time.perf_counter() - t:.2f}s")
    return ws


# ---------------------------------------------------------------- phase: combine

def phase_combine(args, ws):
    """Drive the direct A2A combine on the real dcp group.

    Sync-free on purpose: no per-call barrier, no per-call synchronize. The point
    is not whether the kernel is correct -- that is settled -- but whether it
    still retires when six other communicators are live in the same process.
    """
    dev = torch.device("cuda", LOCAL_RANK)
    torch.manual_seed(1234 + RANK)
    # lse_reduce wants TOTAL heads and divides by the DCP world size itself, so
    # `hpr` (heads per rank) must be scaled up here -- passing it raw gives
    # "attention heads must divide evenly across DCP ranks" whenever hpr < WORLD.
    po = torch.randn(args.t, WORLD * args.hpr, args.hdim, device=dev,
                     dtype=torch.bfloat16)
    pl = torch.randn(args.t, WORLD * args.hpr, device=dev, dtype=torch.float32)

    log("combine", "warm BEFORE", "epoch=0")
    ws.lse_reduce(po, pl, True)
    torch.cuda.synchronize()
    log("combine", "warm AFTER", "epoch=1")

    log("combine", f"x{args.iters} sync-free BEFORE", "epoch=1")
    t = time.perf_counter()
    for _ in range(args.iters):
        ws.lse_reduce(po, pl, True)
    launch_wall = time.perf_counter() - t
    torch.cuda.synchronize()
    total = time.perf_counter() - t
    log("combine", f"x{args.iters} sync-free AFTER",
        f"epoch={1 + args.iters} launch={launch_wall / args.iters * 1e3:.3f}ms/call "
        f"total={total / args.iters * 1e3:.3f}ms/call")

    # ---- pad-128 VIEW vs .contiguous(): closes the _patch_padview_contig.py
    # question without a serve.
    #
    # Under pad-128 `forward_mqa` returns `o[:, :dcp_heads, :]` -- a NON-contiguous
    # view of a (T, 128, hdim) parent -- and that is what reaches lse_reduce as
    # `partial_output`. _patch_padview_contig.py framed the observed capture-time
    # memory fault as (a) "the combine op needs packed heads and reads a strided
    # view OOB" vs (b) "any post-kernel copy races the combine". Test (a) head-on:
    # run the SAME data both ways and compare. The padding lanes are poisoned with
    # a value nothing else can produce, so if the kernel ignored stride(0) and read
    # the buffer as packed it would pull poison into the result and the two answers
    # would diverge -- a silent-wrong-answer detector, not just a crash detector.
    H = WORLD * args.hpr
    PAD = 128 if H <= 128 else H
    POISON = 1e4
    po_pad = torch.randn(args.t, PAD, args.hdim, device=dev, dtype=torch.bfloat16)
    pl_pad = torch.randn(args.t, PAD, device=dev, dtype=torch.float32)
    po_pad[:, H:, :] = POISON
    pl_pad[:, H:] = POISON
    po_view, pl_view = po_pad[:, :H, :], pl_pad[:, :H]
    log("combine", "padview BEFORE",
        f"H={H} pad={PAD} po_stride={tuple(po_view.stride())} "
        f"contig={po_view.is_contiguous()} lse_stride={tuple(pl_view.stride())}")
    out_view = ws.lse_reduce(po_view, pl_view, True)
    out_contig = ws.lse_reduce(po_view.contiguous(), pl_view.contiguous(), True)
    torch.cuda.synchronize()
    d = (out_view.float() - out_contig.float()).abs().max().item()
    ok = bool(torch.isfinite(out_view.float()).all()) and d == 0.0
    log("combine", "padview AFTER",
        f"max|view-contig|={d:.3e} finite={bool(torch.isfinite(out_view.float()).all())} "
        f"VERDICT={'view-is-safe' if ok else 'VIEW-DIFFERS-needs-contiguous'}")

    # 0.08 ms/call is the healthy reference and 1680 ms/call is the degraded one,
    # so this is a 4-orders-of-magnitude discriminator, not a judgement call.
    per_call_ms = total / args.iters * 1e3
    if per_call_ms > 1.0:
        log("combine", "SLOW",
            f"{per_call_ms:.1f}ms/call vs 0.08ms healthy -- box is degraded, "
            "treat any later number as contaminated")
    return per_call_ms


# ------------------------------------------------------- phases: fold/mla/graph
#
# These three reuse `_dcp_folded_mla_standalone.py` wholesale. That driver is
# already the validated description of this chain -- config, kv spec, ctx layer,
# metadata, impl shim -- and duplicating it here would mean two things to keep in
# step. The ONLY substantive difference is that `install_fake_groups` is never
# called: `phase_groups` has already put the real coordinators in
# `parallel_state`, so `AiterMLAMetadataBuilder` reads a live dcp group.
#
# The vllm config is built once and left current for the rest of the run, because
# the metadata builder and `forward_mqa` both read it, and a phase boundary must
# not be a config boundary or the phases stop being comparable.

def phase_fold(args, ctx):
    """Build the 96 -> 32-head folded decode metadata against the real groups.

    The fold factor and the cprr metadata are derived from
    ``get_dcp_group().world_size``. Under the stubs that is a constant; here it
    comes from a communicator that had to be created and rendezvous'd first,
    which is the whole point of running this after `groups`.
    """
    import _dcp_folded_mla_standalone as sa
    from vllm.config import set_current_vllm_config

    if args.model is None:
        args.model = sa._snapshot(sa.DEFAULT_TARGET)
    if args.draft is None:
        args.draft = sa._snapshot(sa.DEFAULT_DRAFT)

    device = torch.device("cuda", LOCAL_RANK)

    # load_format="dummy" is the model-loading bypass: DummyModelLoader's
    # download_model is a no-op and load_weights only walks modules calling
    # initialize_dummy_weights, so zero checkpoint bytes are read. That is what
    # separates "the DCP decode path" from "the 1.5 TiB mmap that has been
    # wedging boots before DCP ever executes".
    log("fold", "build_vllm_config BEFORE", f"load_format=dummy tp={args.tp} dcp={args.dcp}")
    t = time.perf_counter()
    vllm_config = sa.build_vllm_config(args)
    log("fold", "build_vllm_config AFTER", f"took={time.perf_counter() - t:.2f}s")

    # Kept open for the remaining phases -- closed by main()'s finally.
    cm = set_current_vllm_config(vllm_config)
    cm.__enter__()
    ctx["config_cm"] = cm
    ctx["vllm_config"] = vllm_config
    ctx["device"] = device

    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAMetadataBuilder

    # f94666 added vllm/v1/worker/workspace.py; the builder now allocates through
    # it, and it is process-global state that only Worker.init_device() sets up
    # (gpu_worker.py:418). Serve-free drivers must do it themselves or the ctor
    # asserts "WorkspaceManager not initialized". Mirror the worker's arguments:
    # num_ubatches=1 (no DBO) and num_lanes=2, because DSpark's target and draft
    # cudagraphs hold workspace views concurrently (_num_workspace_lanes -> 2
    # when the V2 runner + a DSpark speculative_config are both present, which is
    # exactly our config -- see memory k3-dspark-forces-v2-model-runner).
    from vllm.v1.worker.workspace import init_workspace_manager

    _spec_cfg = vllm_config.speculative_config
    _lanes = 2 if _spec_cfg is not None and _spec_cfg.use_dspark() else 1
    log("fold", "init_workspace_manager", f"num_ubatches=1 num_lanes={_lanes}")
    init_workspace_manager(device, 1, _lanes)

    head_size = args.kv_lora_rank + args.qk_rope_head_dim
    spec = sa.build_kv_spec(vllm_config, head_size, args.non_causal)

    layer_name = "model.layers.0.self_attn.attn"
    vllm_config.compilation_config.static_forward_context[layer_name] = (
        sa.make_ctx_layer(vllm_config, args.non_causal)
    )

    log("fold", "builder ctor BEFORE")
    t = time.perf_counter()
    builder = AiterMLAMetadataBuilder(spec, [layer_name], vllm_config, device)
    log("fold", "builder ctor AFTER", f"took={time.perf_counter() - t:.2f}s")

    qlen = args.qlen or builder._mtp_decode_qlen
    log("fold", "fold params",
        f"num_heads={builder.num_heads} dcp={builder.dcp_world_size} "
        f"decode_num_heads={builder._decode_num_heads} "
        f"fold_factor={builder._dcp_fold_factor} "
        f"fold_heads={builder._dcp_fold_heads} qlen={qlen}")

    common, num_pages = sa.build_common_metadata(
        device, args.reqs, qlen, args.ctx, builder.dcp_world_size)

    log("fold", "builder.build BEFORE", f"reqs={args.reqs} qlen={qlen} ctx={args.ctx}")
    t = time.perf_counter()
    md = builder.build(0, common)
    torch.cuda.synchronize()
    log("fold", "builder.build AFTER", f"took={time.perf_counter() - t:.2f}s")

    decode = md.decode
    if decode is None:
        raise RuntimeError("builder produced no decode metadata")
    log("fold", "decode metadata",
        f"max_qo_len={decode.max_qo_len} "
        f"fold_qo_indptr={'set' if decode.fold_qo_indptr is not None else 'None'} "
        f"fold_num_reqs={decode.fold_num_reqs} "
        f"cp_world_size={decode.cp_world_size} cp_rank={decode.cp_rank} "
        f"persistent={decode.has_persistent_metadata}")
    if decode.fold_qo_indptr is None:
        log("fold", "NOTE",
            "fold path NOT selected for this shape -- forward_mqa will take the "
            "unfolded dcp branch, so `mla` is testing a different kernel")

    ctx.update(builder=builder, md=md, qlen=qlen, num_pages=num_pages,
               head_size=head_size)
    return ctx


def phase_mla(args, ctx):
    """Run the raw AITER cprr producer through the real ``forward_mqa``.

    Eager first (so a fault has a stack), then unsynchronized back-to-back
    iterations -- the async surface, where a fault surfaces at an arbitrary later
    sync rather than at the call that caused it.
    """
    import aiter

    import _dcp_folded_mla_standalone as sa

    vllm_config, device = ctx["vllm_config"], ctx["device"]
    builder, md = ctx["builder"], ctx["md"]
    head_size, num_pages, qlen = ctx["head_size"], ctx["num_pages"], ctx["qlen"]

    kv_dtype = (aiter.dtypes.fp8 if args.kv_cache_dtype.startswith("fp8")
                else vllm_config.model_config.dtype)
    dcp_eff = builder.dcp_world_size
    dcp_heads = builder.num_heads * dcp_eff
    num_tokens = args.reqs * qlen

    kv_cache = torch.randn(num_pages, 1, head_size, dtype=torch.float32,
                           device=device).to(kv_dtype)
    q = torch.randn(num_tokens, dcp_heads, head_size, dtype=torch.float32,
                    device=device).to(kv_dtype)

    impl = sa.build_impl(vllm_config, builder.num_heads, dcp_eff,
                         ctx["dcp_rank"], args.kv_cache_dtype)
    layer = sa._Layer(device)

    log("mla", "eager forward_mqa BEFORE",
        f"q={tuple(q.shape)} kv={tuple(kv_cache.shape)} epoch=0")
    t = time.perf_counter()
    out, lse = impl.forward_mqa(q, kv_cache, md, layer)
    torch.cuda.synchronize()
    finite_o = bool(torch.isfinite(out.float()).all())
    finite_l = lse is None or bool(torch.isfinite(lse.float()).all())
    log("mla", "eager forward_mqa AFTER",
        f"epoch=1 took={time.perf_counter() - t:.2f}s o={tuple(out.shape)} "
        f"lse={tuple(lse.shape) if lse is not None else None} "
        f"finite_o={finite_o} finite_lse={finite_l}")
    if not (finite_o and finite_l):
        raise RuntimeError("forward_mqa produced non-finite output")

    log("mla", f"x{args.iters} unsynced BEFORE", "epoch=1")
    t = time.perf_counter()
    for _ in range(args.iters):
        impl.forward_mqa(q, kv_cache, md, layer)
    launch_wall = time.perf_counter() - t
    torch.cuda.synchronize()
    total = time.perf_counter() - t
    log("mla", f"x{args.iters} unsynced AFTER",
        f"epoch={1 + args.iters} "
        f"launch={launch_wall / args.iters * 1e3:.3f}ms/call "
        f"total={total / args.iters * 1e3:.3f}ms/call")

    ctx.update(impl=impl, q=q, kv_cache=kv_cache, layer=layer)
    return ctx


def phase_graph(args, ctx):
    """Capture and replay the whole chain.

    Capture is the phase most likely to expose a DCP-specific problem that eager
    hides: a graph replays persistent pointers and metadata asynchronously, so
    anything that depended on host-side re-derivation per step is frozen at
    capture time and then replayed against buffers that may have moved.
    """
    impl, q, kv_cache, md, layer = (ctx["impl"], ctx["q"], ctx["kv_cache"],
                                    ctx["md"], ctx["layer"])

    for _ in range(3):                       # warm before capture
        impl.forward_mqa(q, kv_cache, md, layer)
    torch.cuda.synchronize()

    log("graph", "capture BEFORE", "epoch=0")
    t = time.perf_counter()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_out, _g_lse = impl.forward_mqa(q, kv_cache, md, layer)
    log("graph", "capture AFTER", f"epoch=1 took={time.perf_counter() - t:.2f}s")

    log("graph", f"x{args.graph_iters} replay BEFORE", "epoch=1")
    t = time.perf_counter()
    for _ in range(args.graph_iters):
        graph.replay()
    torch.cuda.synchronize()
    per = (time.perf_counter() - t) / args.graph_iters * 1e3
    g_ok = bool(torch.isfinite(g_out.float()).all())
    log("graph", f"x{args.graph_iters} replay AFTER",
        f"epoch={1 + args.graph_iters} {per:.3f}ms/replay finite={g_ok}")
    if not g_ok:
        raise RuntimeError("graph replay produced non-finite output")

    # ---- the chain that actually faulted: producer -> strided view -> combine,
    # captured, replayed, NO sync anywhere.
    #
    # Capturing forward_mqa alone (above) cannot reproduce padview_1020: the
    # reported fault was at END OF CAPTURE of the *pair*, where pad-128's
    # `o[:, :dcp_heads, :]` view is handed straight to the symm-mem a2a combine.
    # phase_combine already showed the op reads that stride correctly (bit-exact
    # vs .contiguous(), poisoned pad lanes), so if a fault survives here it is
    # ordering, not addressing -- i.e. hypothesis (b), and pad-128 buys nothing.
    ws = ctx.get("ws")
    if ws is None:
        log("graph", "chain SKIP", "no workspace in ctx (run the symm phase)")
        return ctx
    n_tok = q.shape[0]
    if n_tok > args.max_nt:
        log("graph", "chain SKIP",
            f"num_tokens={n_tok} > workspace max_nt={args.max_nt} -- rerun with "
            f"--max-nt {n_tok} to exercise the full chain")
        return ctx

    o_e, lse_e = impl.forward_mqa(q, kv_cache, md, layer)
    log("graph", "chain eager BEFORE",
        f"o={tuple(o_e.shape)} contig={o_e.is_contiguous()} "
        f"stride={tuple(o_e.stride())} lse_contig={lse_e.is_contiguous()}")
    ref = ws.lse_reduce(o_e, lse_e, True).clone()
    torch.cuda.synchronize()
    log("graph", "chain eager AFTER",
        f"combined={tuple(ref.shape)} finite={bool(torch.isfinite(ref.float()).all())}")

    log("graph", "chain capture BEFORE", "producer+combine, no sync")
    t = time.perf_counter()
    cgraph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cgraph):
        c_o, c_lse = impl.forward_mqa(q, kv_cache, md, layer)
        c_comb = ws.lse_reduce(c_o, c_lse, True)
    torch.cuda.synchronize()
    log("graph", "chain capture AFTER", f"took={time.perf_counter() - t:.2f}s")

    log("graph", f"chain x{args.graph_iters} replay BEFORE", "no sync between replays")
    t = time.perf_counter()
    for _ in range(args.graph_iters):
        cgraph.replay()
    torch.cuda.synchronize()
    c_per = (time.perf_counter() - t) / args.graph_iters * 1e3
    c_ok = bool(torch.isfinite(c_comb.float()).all())
    # Inputs are fixed, so every replay must land on the eager reference. If the
    # combine raced the producer (hypothesis (b)) the consumer would sometimes
    # read half-written o/lse and the replayed answer would drift off `ref` --
    # which is the failure a finiteness check alone would sail straight past.
    c_d = (c_comb.float() - ref.float()).abs().max().item()
    log("graph", f"chain x{args.graph_iters} replay AFTER",
        f"{c_per:.3f}ms/replay finite={c_ok} max|replay-eager|={c_d:.3e} "
        f"VERDICT={'sync-free-ok' if (c_ok and c_d == 0.0) else 'RACE-OR-DRIFT'}")
    if not c_ok:
        raise RuntimeError("captured producer+combine chain produced non-finite output")
    if c_d != 0.0:
        raise RuntimeError(
            f"captured chain drifted from the eager reference by {c_d:.3e} -- the "
            "combine is not safely ordered after the producer without a sync")
    return ctx


def phase_serve(args, ctx):
    """The chain at SERVE shape -- the three things every phase above fakes.

    `graph` clears the *mechanism*: one shape, 40 tokens, an empty card. A serve
    differs in exactly three ways, and each is a plausible home for the wedge:

      1. MANY captured shapes, not one, sharing ONE cudagraph memory pool. vLLM
         captures the whole CAPTURE_SIZES ladder back to back, so capture N runs
         against a pool that captures 1..N-1 already carved up, and every graph
         holds frozen pointers into it. This also makes all 9 graphs share the
         workspace's signal/epoch buffers -- the one piece of DCP state that is
         genuinely global across shapes, and that no single-shape test can stress.
      2. REAL VRAM PRESSURE. At GPU_MEM=0.95 a serve captures with the card
         nearly full, so the pool cannot grow and the allocator has to reuse.
      3. LAYER COUNT. A decode step is ~61 MLA+combine pairs, not one, so the
         epoch counter advances 61x per step and the signal buffers are recycled
         that many times before anyone synchronises.

    All three are reachable without a model, a scheduler or an API server. What
    is still absent: real weights, the MoE/dense path between attention layers,
    and the scheduler's shape churn. So a PASS here still does not prove the
    serve boots -- it removes the three most likely reasons it would not.
    """
    impl, layer = ctx["impl"], ctx["layer"]
    builder, qlen, device = ctx["builder"], ctx["qlen"], ctx["device"]
    ws = ctx.get("ws")
    if ws is None:
        raise RuntimeError("phase_serve needs the symm workspace -- run from `groups`")

    import _dcp_folded_mla_standalone as sa

    dcp_heads = builder.num_heads * builder.dcp_world_size
    head_size = ctx["head_size"]
    kv_dtype = ctx["kv_cache"].dtype

    # ---- (2) ballast first: capture must happen on a card that is already full,
    # because an empty card lets the pool grow instead of forcing reuse.
    free_b, total_b = torch.cuda.mem_get_info()
    gib = 1 << 30
    want_used = args.serve_vram * total_b
    used_b = total_b - free_b
    ballast = []
    need = want_used - used_b
    log("serve", "ballast BEFORE",
        f"used={used_b / gib:.1f}GiB total={total_b / gib:.1f}GiB "
        f"target={args.serve_vram:.2f} need={max(need, 0) / gib:.1f}GiB")

    # Fill in chunks, and CHECK EACH ONE LANDED IN VRAM -- trust mem_get_info, not
    # the absence of an exception. If free did not drop by roughly the chunk size
    # the memory is not where we think it is, and everything measured after that
    # point is about some other allocator.
    #
    # Measured on this box: device ballast is genuinely device-resident and fast
    # (240 GiB in ~0.1 s with flat RSS, whether alone, 8-way, under NCCL, or under
    # a symm_mem mesh). The 200 GiB of anon-rss that got rank 0 oom-killed on
    # 2026-08-25 was the zero-chunk spin below, not a GTT spill -- host memory is
    # tight here (/dev/shm holds ~1.4 TiB of weights, leaving ~1.5 TiB), so any
    # per-rank host growth multiplies by 8 and takes the box's oom-killer with it.
    # Chunk size matters: each one is a KFD buffer object, and BO creation is the
    # thing that turned out to be slow here, not the byte count. 8 GiB keeps the
    # count to ~30.
    spilled = slow = False
    t_ball = time.perf_counter()
    # `need` is a float, so the last partial chunk truncates to 0 bytes; guard the
    # loop on a whole chunk rather than on `need > 0`, or it spins forever
    # allocating nothing and subtracting nothing (cost me a run on 2026-08-25).
    min_chunk = 1 << 20
    while need >= min_chunk:
        chunk = int(min(need, args.serve_ballast_gib * gib))
        before, _ = torch.cuda.mem_get_info()
        t_chunk = time.perf_counter()
        try:
            buf = torch.empty(chunk, dtype=torch.uint8, device=device)
        except torch.OutOfMemoryError:
            break
        after, _ = torch.cuda.mem_get_info()
        dt = time.perf_counter() - t_chunk
        if before - after < chunk * 0.9:
            del buf
            spilled = True
            break
        ballast.append(buf)
        need -= chunk
        # A single allocation that takes longer than a heartbeat is itself the
        # finding: on a clean 8-rank job this whole loop is ~0.1 s, so anything
        # measurable means BO creation is serialising behind driver state.
        if dt > 0.5 or len(ballast) % 8 == 0:
            slow = slow or dt > 0.5
            log("serve", f"ballast chunk {len(ballast)}",
                f"dt={dt:.2f}s free={after / gib:.1f}GiB rss={_rss_gib():.1f}GiB "
                f"elapsed={time.perf_counter() - t_ball:.1f}s")
        # Host-side backstop. The ballast is supposed to be device memory, so RSS
        # must stay flat; if it is climbing, something is host-backed and 8 ranks
        # will take the box's oom-killer with them (measured 2026-08-25).
        if _rss_gib() > args.serve_rss_cap_gib:
            log("serve", "ballast ABORT",
                f"rss={_rss_gib():.1f}GiB exceeds cap {args.serve_rss_cap_gib}GiB "
                "-- device ballast is consuming HOST memory; stopping before the "
                "host oom-killer does it for us")
            break
    log("serve", "ballast loop",
        f"{len(ballast)} chunks in {time.perf_counter() - t_ball:.1f}s"
        + (" (SLOW: individual allocations exceeded 0.5s)" if slow else ""))

    free_b, _ = torch.cuda.mem_get_info()
    got = total_b - free_b
    log("serve", "ballast AFTER",
        f"{len(ballast)} chunks, used={got / gib:.1f}GiB ({got / total_b:.1%}) "
        f"free={free_b / gib:.1f}GiB"
        + (" SPILLED-TO-GTT: stopped short, further chunks were host-backed"
           if spilled else ""))
    if got / total_b < args.serve_vram - 0.05:
        # Say it plainly rather than letting a thin run read as a full-pressure
        # PASS. This is a real limit of the rig, not a detail.
        log("serve", "ballast SHORT",
            f"reached {got / total_b:.1%} of the {args.serve_vram:.0%} target -- "
            "capture is running under LESS memory pressure than a real serve")

    # ---- (1) the ladder. Sizes are M = (1+num_spec)*conc decode tokens, which is
    # what CAPTURE_SIZES enumerates; keep only the ones this workspace was sized
    # for and that divide evenly into whole requests.
    sizes = [int(s) for s in args.serve_sizes.split(",") if s.strip()]
    usable = [m for m in sizes if m % qlen == 0 and m <= args.max_nt]
    dropped = [m for m in sizes if m not in usable]
    if dropped:
        # Never let a bounded sweep read as full coverage.
        log("serve", "ladder DROPPED",
            f"{dropped} -- need m%qlen==0 (qlen={qlen}) and m<=max_nt "
            f"({args.max_nt}); rerun with --max-nt {max(sizes)}")
    if not usable:
        raise RuntimeError(
            f"no usable capture sizes from {sizes} at qlen={qlen}, "
            f"max_nt={args.max_nt}")
    log("serve", "ladder", f"{usable} qlen={qlen} layers={args.serve_layers}")

    # Metadata first, for every shape, because the KV cache has to be sized for
    # the BIGGEST one. `phase_mla` allocated its cache for args.reqs (8) requests,
    # but this ladder runs up to max(usable)//qlen (48), and build_common_metadata
    # hands out page ids as arange(num_reqs * (ctx+qlen)) -- so reusing that cache
    # would index ~6x past its end. The asm kernel would read whatever followed it
    # and the answer would be garbage rather than a fault, which is the worst way
    # for a test to be wrong.
    metas, max_pages = [], 0
    for m in usable:
        common, num_pages = sa.build_common_metadata(
            device, m // qlen, qlen, args.ctx, builder.dcp_world_size)
        metas.append((m, builder.build(0, common)))
        max_pages = max(max_pages, num_pages)
    kv_cache = torch.randn(max_pages, 1, head_size, dtype=torch.float32,
                           device=device).to(kv_dtype)
    log("serve", "kv_cache", f"pages={max_pages} (was {ctx['kv_cache'].shape[0]} "
                             f"for reqs={args.reqs}) dtype={kv_dtype}")

    pool = torch.cuda.graph_pool_handle()   # ONE pool, exactly like vLLM
    entries = []
    for m, md_m in metas:
        reqs = m // qlen
        q_m = torch.randn(m, dcp_heads, head_size, dtype=torch.float32,
                          device=device).to(kv_dtype)

        for _ in range(2):                  # warm this shape before capturing it
            o_w, l_w = impl.forward_mqa(q_m, kv_cache, md_m, layer)
            ws.lse_reduce(o_w, l_w, True)
        torch.cuda.synchronize()

        o_e, l_e = impl.forward_mqa(q_m, kv_cache, md_m, layer)
        ref = ws.lse_reduce(o_e, l_e, True).clone()
        torch.cuda.synchronize()
        if not torch.isfinite(ref.float()).all():
            raise RuntimeError(f"eager reference for m={m} is non-finite")

        t = time.perf_counter()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=pool):
            # (3) the layer loop, INSIDE the graph -- a decode step is this many
            # MLA+combine pairs, and they all recycle the same signal buffers.
            for _ in range(args.serve_layers):
                c_o, c_l = impl.forward_mqa(q_m, kv_cache, md_m, layer)
                c_out = ws.lse_reduce(c_o, c_l, True)
        torch.cuda.synchronize()
        free_b, _ = torch.cuda.mem_get_info()
        log("serve", f"capture m={m}",
            f"reqs={reqs} took={time.perf_counter() - t:.2f}s "
            f"free={free_b / gib:.1f}GiB")
        # Hold everything: a serve keeps every graph, its inputs and its metadata
        # alive for the process lifetime, and freeing any of it here would hand
        # the pool back memory a real serve never gets.
        entries.append(dict(m=m, g=g, out=c_out, ref=ref, q=q_m, md=md_m))

    log("serve", "capture ALL DONE",
        f"{len(entries)} graphs live in one pool, "
        f"{args.serve_layers} layers each")

    # ---- replay. Round-robin across shapes, not in ladder order: a scheduler
    # jumps between batch sizes step to step, so consecutive replays hit graphs
    # captured at different pool epochs. Every rank walks the identical order --
    # the combine is a collective, so a per-rank order would deadlock outright.
    log("serve", f"replay x{args.serve_rounds} BEFORE", "round-robin, no sync")
    t = time.perf_counter()
    for _ in range(args.serve_rounds):
        for e in entries:
            e["g"].replay()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t
    n_replay = args.serve_rounds * len(entries)
    log("serve", f"replay x{args.serve_rounds} AFTER",
        f"{n_replay} replays, {wall / n_replay * 1e3:.3f}ms/replay "
        f"({wall / n_replay / args.serve_layers * 1e3:.3f}ms/layer)")

    # Inputs never change, so every shape must still land on its own eager
    # reference. A pool-aliasing bug shows up here as one shape's output being
    # perturbed by another shape's graph -- silent, and invisible to a finiteness
    # check, which is why this compares values and not just NaNs.
    worst, bad = 0.0, []
    for e in entries:
        d = (e["out"].float() - e["ref"].float()).abs().max().item()
        worst = max(worst, d)
        if d != 0.0 or not torch.isfinite(e["out"].float()).all():
            bad.append((e["m"], d))
    log("serve", "verify AFTER",
        f"max|replay-eager|={worst:.3e} over {len(entries)} shapes "
        f"VERDICT={'serve-shape-ok' if not bad else 'DRIFT ' + repr(bad)}")
    if bad:
        raise RuntimeError(
            f"serve-shape replay drifted from the eager references: {bad} -- "
            "shapes are interfering through the shared graph pool or the "
            "workspace signal/epoch buffers")

    del ballast
    return ctx


# ---------------------------------------------------------------- leak arms

def _symm_shapes(args):
    """The exact three allocations DirectDCPA2AWorkspace makes (dcp.py:858)."""
    w = WORLD
    return [
        ((1, 2, w, args.max_nt, args.hpr, args.hdim), torch.bfloat16),
        ((1, 2, w, args.max_nt, args.hpr), torch.float32),
        ((1, 2, w), torch.int32),
    ]


def _plain_nccl_init():
    """Bare torch.distributed, deliberately WITHOUT vLLM's parallel_state.

    Arms A-C exist to isolate the symmetric-memory mesh, so they must not drag
    in the six extra communicators, the message-queue broadcaster or the custom
    all-reduce's own IPC buffers -- otherwise a residue reading cannot be
    attributed. Arm D uses the real thing; that difference is the experiment.
    """
    dev = torch.device("cuda", LOCAL_RANK)
    dist.init_process_group("nccl", device_id=dev)
    return dev


def arm_nccl(args):
    """A -- control: what residue does ANY 8-rank NCCL job leave?

    Every other arm is read as a delta against this one. Without it, a stale KFD
    node after arm B proves nothing: it could just be what SIGKILLing eight
    torch processes always does.
    """
    dev = _plain_nccl_init()
    x = torch.ones(1 << 20, device=dev)
    for i in range(args.arm_iters):
        dist.all_reduce(x)
        if i % 50 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    log("nccl", "allreduce loop done", f"iters={args.arm_iters}")


def arm_symmonly(args):
    """B -- A plus the symmetric-memory mesh, and nothing else.

    Same three allocations, same shapes, same peer-buffer views as
    DirectCPWorkspace._allocate (cp_common.py:105) -- but no DCP kernel, no
    aiter, no vLLM. On ROCm each rendezvous exports a KFD dma-buf that the other
    seven ranks import, so this arm creates 3 x 8 x 8 cross-process buffer-object
    mappings and does nothing with them.

    A vs B is the decisive comparison in this whole harness.
    """
    import torch.distributed._symmetric_memory as symm_mem

    dev = _plain_nccl_init()
    pg = dist.distributed_c10d._get_default_group()
    log("symmonly", "self_dmabuf BEFORE", str(dmabuf_fds()))
    held = []
    for i, (shape, dtype) in enumerate(_symm_shapes(args)):
        t = time.perf_counter()
        storage = symm_mem.empty(shape, device=dev, dtype=dtype)
        storage.zero_()
        torch.cuda.synchronize()
        handle = symm_mem.rendezvous(storage, pg.group_name)
        assert handle is not None, "rendezvous returned None"
        handle.barrier()
        views = [handle.get_buffer(p, list(shape), dtype, 0) for p in range(WORLD)]
        held.append((storage, handle, views))
        log("symmonly", f"alloc {i} AFTER",
            f"took={time.perf_counter() - t:.2f}s self_dmabuf={dmabuf_fds()}")
    dist.barrier()
    log("symmonly", "mesh established", f"allocs={len(held)} peers={WORLD}")
    return held


def arm_probe(args):
    """C -- B plus the multicast probe, to size its leak on its own.

    `_symm_mem_spans_group` (cp_common.py:32) allocates a probe tensor,
    rendezvouses it, and frees neither -- and has two early returns *before* the
    collective, so ranks can diverge inside it. Our
    `_patch_dcp_skip_multicast_probe.py` disables it on ROCm; this arm runs the
    probe body inline (not the wrapper) so the patch cannot mask it, and so the
    measurement is of the mechanism rather than of vLLM's gating.
    """
    import torch.distributed._symmetric_memory as symm_mem

    held = arm_symmonly(args)
    dev = torch.device("cuda", LOCAL_RANK)
    pg = dist.distributed_c10d._get_default_group()
    for i in range(args.probe_iters):
        try:
            probe = symm_mem.empty(8, dtype=torch.uint8, device=dev)
            probe.zero_()
            torch.cuda.synchronize()
            h = symm_mem.rendezvous(probe, pg.group_name)
            mc = 0 if h is None else h.multicast_ptr
        except Exception as err:                       # the real one swallows this
            mc = f"raised {type(err).__name__}"
        log("probe", f"probe {i} done", f"multicast_ptr={mc} self_dmabuf={dmabuf_fds()}")
    dist.barrier()
    return held


def arm_dcp(args, stall=False):
    """D -- the production path: real groups, real workspace, real combine.

    With ``stall=True`` the last rank skips its final call, so its peers enter
    ``wait_signals_kernel`` for an epoch that never arrives and run the full
    ~8 s backoff (kSpinLimit, dcp_direct_common_hip.h:47). That is the state we
    most want to kill a job in: a resident spinning wave is the hard case for
    CWSR queue preemption during eviction.
    """
    groups = phase_groups(args)
    ws = phase_symm(args, groups)
    dev = torch.device("cuda", LOCAL_RANK)
    torch.manual_seed(1234 + RANK)
    po = torch.randn(args.t, WORLD * args.hpr, args.hdim, device=dev,
                     dtype=torch.bfloat16)
    pl = torch.randn(args.t, WORLD * args.hpr, device=dev, dtype=torch.float32)
    withholder = WORLD - 1
    for i in range(args.arm_iters):
        last = i == args.arm_iters - 1
        if stall and last and RANK == withholder:
            log("dcpstall", "WITHHOLDING signal",
                f"rank={RANK} iter={i} -- peers will spin ~8s then time out")
            continue
        ws.lse_reduce(po, pl, True)
    torch.cuda.synchronize()
    log("dcpstall" if stall else "dcp", "combine loop done",
        f"iters={args.arm_iters} self_dmabuf={dmabuf_fds()}")
    return ws


LEAK_ARMS = {
    "nccl": lambda a: arm_nccl(a),
    "symmonly": lambda a: arm_symmonly(a),
    "probe": lambda a: arm_probe(a),
    "dcp": lambda a: arm_dcp(a, stall=False),
    "dcpstall": lambda a: arm_dcp(a, stall=True),
}


def run_arm(args):
    """Run exactly one leak arm, print driver state around it, then optionally hold.

    The arms are alternatives, not a cumulative ladder, so this does not reuse
    the --stop-after plan. ``--hold`` keeps the process alive after the body so
    the outer script can SIGKILL it mid-flight; the HOLDING line is the cue.
    """
    torch.cuda.set_device(LOCAL_RANK)
    if RANK == 0:
        print(f"[r0] arm={args.arm} PRE  {fmt_state(driver_state())}")
    t = time.perf_counter()
    held = LEAK_ARMS[args.arm](args)                 # noqa: F841 -- keep alive
    wall = time.perf_counter() - t
    dist.barrier()
    if RANK == 0:
        print(f"[r0] arm={args.arm} POST wall={wall:.2f}s "
              f"{fmt_state(driver_state())}")
    if args.hold > 0:
        log(args.arm, "HOLDING", f"{args.hold}s -- kill window open")
        time.sleep(args.hold)
    if args.clean_exit:
        dist.destroy_process_group()
        log(args.arm, "CLEAN EXIT", "process group destroyed")
    return 0


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop-after", default="groups", choices=ALL_PHASES,
                    help="run phases up to and including this one")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--dcp", type=int, default=8)
    ap.add_argument("--use-all2all", action="store_true",
                    help="pass use_all2all to the ep group (matches an all2all "
                         "MoE backend); off by default, as in our recipe")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--t", type=int, default=5, help="tokens (nspec2 verify: 1+2*2)")
    ap.add_argument("--hpr", type=int, default=12, help="heads per rank (K3: 12)")
    ap.add_argument("--hdim", type=int, default=512, help="MLA v-dim")
    ap.add_argument("--max-nt", type=int, default=16)

    # fold / mla / graph: same knobs as _dcp_folded_mla_standalone.py, so a
    # finding here reproduces there (single-GPU, fake groups) and vice versa.
    ap.add_argument("--model", default=None, help="target config dir (no weights read)")
    ap.add_argument("--draft", default=None, help="draft config dir")
    ap.add_argument("--num-spec", type=int, default=2)
    ap.add_argument("--reqs", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--qlen", type=int, default=0,
                    help="0 = production verify length (DSpark: 1 + 2*num_spec)")
    ap.add_argument("--non-causal", action="store_true",
                    help="simulate the replicated DSpark draft group (#51705)")
    ap.add_argument("--interleave", type=int, default=1,
                    help="cp_kv_cache_interleave_size")
    ap.add_argument("--kv-cache-dtype", default="fp8")
    ap.add_argument("--kv-lora-rank", type=int, default=512)
    ap.add_argument("--qk-rope-head-dim", type=int, default=64)
    ap.add_argument("--qk-nope-head-dim", type=int, default=128)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--mnbt", type=int, default=4096)
    ap.add_argument("--graph-iters", type=int, default=50)
    # --- serve phase ---------------------------------------------------------
    # The default ladder is the nspec=2 arm of the CAPTURE_SIZES the bench serves
    # with (M = 3*conc for conc 1..48); see the k3-longctx-bench skill.
    ap.add_argument("--serve-sizes", default="3,6,12,24,36,48,72,96,144",
                    help="decode token counts M to capture, comma separated")
    ap.add_argument("--serve-layers", type=int, default=61,
                    help="MLA+combine pairs per captured step (K3 layer count)")
    ap.add_argument("--serve-rounds", type=int, default=20,
                    help="round-robin passes over the whole ladder")
    ap.add_argument("--serve-ballast-gib", type=int, default=8,
                    help="ballast chunk size; each chunk is one KFD buffer object")
    ap.add_argument("--serve-rss-cap-gib", type=float, default=24.0,
                    help="abort the ballast if host RSS passes this")
    ap.add_argument("--serve-vram", type=float, default=0.85,
                    help="fraction of card to hold before capturing. The bench "
                         "serves at 0.95, but test rigs stay <=0.90 on this box.")

    # leak arms (differential driver-residue test; see run_arm)
    ap.add_argument("--arm", default=None, choices=sorted(LEAK_ARMS),
                    help="run ONE leak arm instead of the cumulative phases")
    ap.add_argument("--arm-iters", type=int, default=200)
    ap.add_argument("--probe-iters", type=int, default=4)
    ap.add_argument("--hold", type=int, default=0,
                    help="seconds to stay alive after the arm body, so the "
                         "driver can SIGKILL mid-flight")
    ap.add_argument("--clean-exit", action="store_true",
                    help="destroy_process_group before returning (the clean arm)")
    ap.add_argument("--require-clean", action="store_true",
                    help="abort unless the box is provably unpoisoned")
    ap.add_argument("--max-stat-dirs", type=int, default=8,
                    help="pre-launch stat-dir ceiling, used by the shell "
                         "harness; the in-process gate cannot use it because "
                         "our own ranks inflate the count")
    ap.add_argument("--evict-budget-ms", type=int, default=60000,
                    help="per-phase KFD queue-eviction budget. symm_mem evicts "
                         "~22-36 s per run as a matter of course (measured: the "
                         "nccl control evicts 0), so only a phase well past that "
                         "baseline stops the bisect")
    args = ap.parse_args()

    # Never measure on a box that is already degraded. The 08-21 bisect started
    # at evicted_ms=12,657,032 across 128 stat dirs and every number it produced
    # was worthless; this makes that failure loud instead of silent.
    #
    # Gate only on signals our OWN ranks cannot inflate. By the time this runs,
    # torchrun has started every rank, and a rank that has touched CUDA already
    # owns a kfd_process with one stats dir per GPU -- so `stat_dirs` counts us
    # (8 ranks x 8 GPUs = 64 on a perfectly healthy box) and gating on it aborts
    # the job on its own footprint. `evicted_ms` and `kfd_stale` stay honest: a
    # live rank is not stale, and a healthy rank evicts nothing. The strict
    # pre-launch check lives in _dcp_phase_bisect.sh, where nothing is running
    # yet and `stat_dirs` is meaningful.
    if args.require_clean:
        s = driver_state()
        why = []
        if s["evicted_ms"] > 0:
            why.append(f"evicted_ms={s['evicted_ms']}")
        if s["kfd_stale"] is None:
            # Not a hard failure -- the run is still valid, its sharpest metric
            # just is not. Say so, so nobody later reads a silent zero as clean.
            if RANK == 0:
                print(f"[r0] WARNING: nested pid namespace -- kfd_stale is "
                      f"unmeasurable here ({s['kfd_nodes']} nodes, ownership "
                      f"unknown). Re-run from the host or a --pid=host "
                      f"container for the decisive signal.")
        elif s["kfd_stale"] > 0:
            why.append(f"stale kfd_process nodes {s['kfd_stale_pids']}")
        if why:
            if RANK == 0:
                print(f"[r0] ABORT: box is not clean -- {'; '.join(why)}")
                print(f"[r0] state: {fmt_state(s)}")
            return 2
        if RANK == 0:
            print(f"[r0] gate PASSED: {fmt_state(s)}")

    if args.arm is not None:
        return run_arm(args)

    stop_idx = ALL_PHASES.index(args.stop_after)
    plan = ALL_PHASES[:stop_idx + 1]

    torch.cuda.set_device(LOCAL_RANK)
    if RANK == 0:
        ev, nstat = kfd_evicted_ms()
        print(f"[r0] baseline: evicted_ms={ev} ({nstat} stat dirs) "
              f"gtt={gtt_used_gib():.2f}GiB tiny_kernel={tiny_kernel_us():.1f}us")
        print(f"[r0] plan: {' -> '.join(plan)}")

    groups, ws = None, None
    ctx = {}
    stopped = None
    try:
        for phase in plan:
            probe = Probe()
            log(phase, "PHASE START")
            if phase == "groups":
                groups = phase_groups(args)
                ctx["dcp_rank"] = groups["dcp"].rank_in_group
            elif phase == "symm":
                ws = phase_symm(args, groups)
                ctx["ws"] = ws
            elif phase == "combine":
                phase_combine(args, ws)
            elif phase == "fold":
                phase_fold(args, ctx)
            elif phase == "mla":
                phase_mla(args, ctx)
            elif phase == "graph":
                phase_graph(args, ctx)
            elif phase == "serve":
                phase_serve(args, ctx)
            d_ev = probe.report(phase)
            # Any eviction at all used to stop the bisect. The leak matrix
            # (2026-08-25, --arm nccl vs symmonly) showed that is too strict:
            # symmetric memory evicts ~22-36 s of queue time on EVERY run, with
            # zero VM faults and no lasting damage, while the plain-NCCL control
            # evicts nothing. So eviction on the symm/dcp phases is the expected
            # cost of symm_mem, not a fault -- and stopping on it means the
            # bisect can never reach `combine` and beyond. Warn under budget,
            # stop only when a phase evicts far past the measured baseline.
            if d_ev > args.evict_budget_ms:
                log(phase, "STOP",
                    f"queue eviction {d_ev} ms exceeds budget "
                    f"{args.evict_budget_ms} ms -- this is the phase to fix")
                stopped = phase
                break
            if d_ev > 0:
                log(phase, "WARN",
                    f"queue eviction +{d_ev} ms (within the "
                    f"{args.evict_budget_ms} ms symm_mem budget) -- continuing")
    finally:
        # set_current_vllm_config is a context manager held open across phases;
        # leaving it entered on the way out would corrupt the global config for
        # anything else in this process (and mask the real error).
        cm = ctx.pop("config_cm", None)
        if cm is not None:
            cm.__exit__(None, None, None)

    dist.barrier()
    if RANK == 0:
        print()
        if stopped is None:
            print(f"PASS: phases [{' '.join(plan)}] completed on {WORLD} ranks.")
        else:
            print(f"STOP: eviction began in phase '{stopped}' on {WORLD} ranks.")
    dist.destroy_process_group()
    return 0 if stopped is None else 1


if __name__ == "__main__":
    sys.exit(main())
