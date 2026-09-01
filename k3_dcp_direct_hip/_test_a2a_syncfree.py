#!/usr/bin/env python3
"""Reproduce (or clear) the DIAG-off sync-free warmup deadlock in
direct_dcp_a2a_lse_reduce -- and A/B it against the two RCCL-mediated combines.

Run: torchrun --standalone --nproc_per_node=8 _test_a2a_syncfree.py

THE A/B (BACKEND=all)
---------------------
vLLM ships three DCP combines in v1/attention/ops/dcp.py, picked by
MLADCPManager._init_combine. They compute the same thing and are timed here on
one shape so the ordering is measured rather than assumed:

  ag_rs   cp_lse_ag_out_rs      allgather(lse) + reduce_scatter(out).  vLLM's
                                DEFAULT for K3/DeepSeek (set_dcp_defaults
                                defaults dcp_comm_backend="ag_rs"; only
                                GlmMoeDsa opts into a2a). Moves the most bytes.
  a2a     dcp_a2a_lse_reduce    triton pack -> one dist.all_to_all_single ->
                                local combine. What ATOM uses (its
                                dcp_ops.cp_lse_a2a is the same algorithm).
  direct  DirectDCPA2AWorkspace one-sided symm_mem writes + a signal/spin
                                handshake, no RCCL kernel. Same bytes as a2a.
                                On ROCm this is OFF unless forced:
                                direct_cp_enabled() falls back to
                                current_platform.is_cuda(), so it is only
                                reachable via VLLM_USE_DIRECT_DCP_A2A=1.

Only the `direct` arm allocates symmetric memory, so ag_rs/a2a cannot build the
cross-process dma-buf mesh behind the SRCU wedge. The workspace is therefore
allocated lazily and `direct` runs LAST.

  BACKEND=all|direct|a2a|ag_rs   default `direct` (historical behaviour)
  AB_BARRIER=1                   barrier between timed calls, to separate raw
                                 cost from skew absorption
  SOAK=1                         run the sync-free deadlock soak (direct only)

WHY THIS EXISTS
---------------
_test_direct_a2a.py calls dist.barrier() + torch.cuda.synchronize() around every
lse_reduce(), i.e. it is effectively DIAG-ON: every call is host-serialised, so
cross-rank skew never exceeds one epoch and the deadlock can never appear. That
test passing (cos 0.9704) says nothing about the sync-free path.

This driver is the opposite: back-to-back eager calls, NO per-call barrier, NO
per-call synchronize, plus DELIBERATE per-rank skew so ranks drift apart in
epoch the way they do during pre-capture eager warmup.

THE PROTOCOL UNDER TEST (per call, epoch E, parity p = E & 1)
  1 increment_epoch_kernel   epoch += 1
  2 dispatch_*               write my slice into every peer's slot [p][me]
  3 signal_kernel            release-store peer_signal[p*W + me] = E
  4 wait_lse_combine_kernel  for each src s: spin until signal[p*W+s] == E

Only 2 slots (parity) => tolerates < 2 epochs of skew. The stream dependency
chain is supposed to bound skew below that, because a rank cannot retire epoch E
until every peer has signalled E. This test checks that reasoning empirically.

WHAT THE FAILURE MODES LOOK LIKE
  spin/trap  -> GPU pegged ~100%, then "direct DCP A2A timeout source=..." and
                __builtin_trap() (kSpinLimit = 1e8 exhausted). Handshake bug.
  true hang  -> GPU ~0%, ranks parked in kfd_wait_on_events, no timeout print.
                That is NOT this kernel spinning; look outside the op.
  clean      -> all iterations retire and the final numeric check matches.
"""
import os
import sys
import threading
import time

import torch
import torch.distributed as dist

RANK = int(os.environ["RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD = int(os.environ["WORLD_SIZE"])

SO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  "dcp_direct_a2a_lse_reduce.so")
T = int(os.environ.get("T", "5"))          # tokens (nspec2 verify: 1+2*2)
# Sweep several T in ONE process. RCCL is lazy -- init_process_group returns
# instantly and the whole connection setup (100+ channels, thousands of P2P/IPC
# proxy conns on this 8-GPU node) happens inside the first collective, minutes
# not seconds, and this driver builds TWO communicators. Relaunching torchrun
# per point paid that twice per point; sweeping in-process pays it once.
T_LIST = [int(x) for x in os.environ.get("T_LIST", str(T)).replace(",", " ").split()]
HPR = int(os.environ.get("HPR", "12"))     # heads_per_rank (K3: 12)
HDIM = 512                                  # head_dim (K3 MLA v-dim)
MAX_NT = int(os.environ.get("MAX_NT", "16"))  # direct workspace token capacity
ITERS = int(os.environ.get("ITERS", "300"))
# how far apart to drive the ranks; 0 = lockstep, higher = more drift
SKEW = float(os.environ.get("SKEW", "1.0"))
CHECK_EVERY = int(os.environ.get("CHECK_EVERY", "50"))
LOG2E = 1.4426950408889634

BACKEND = os.environ.get("BACKEND", "direct")
AB_BARRIER = os.environ.get("AB_BARRIER", "0") == "1"
AB_ITERS = int(os.environ.get("AB_ITERS", "20"))
# direct last: it is the only arm that allocates symmetric memory.
ARM_ORDER = ["ag_rs", "a2a", "direct"]


def log(m):
    print(f"[rank{RANK}] {m}", flush=True)


class Heartbeat:
    """Print progress from a side thread so a hang is visible + attributable."""

    def __init__(self):
        self.where = "init"
        self.iters = 0
        self.stop = False
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        last = -1
        stuck = 0
        while not self.stop:
            time.sleep(10)
            if self.iters == last:
                stuck += 1
                log(f"HEARTBEAT no progress {stuck * 10}s at '{self.where}' "
                    f"iter={self.iters}")
                if stuck == 3:
                    log("STALLED -- dumping GPU state hint: if GPU util is ~0% "
                        "this is NOT the combine kernel spinning")
            else:
                stuck = 0
            last = self.iters

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop = True


def reference(partial_output, partial_lse, is_lse_base_on_e):
    """fp32 reference: gather all shards, softmax-merge my heads."""
    # Size from the input, not the module-global T: the sweep calls this once
    # per T, and a fixed T would mismatch every point after the first.
    t, total_heads = partial_lse.shape
    go = torch.empty((WORLD, t, total_heads, HDIM), dtype=partial_output.dtype,
                     device=partial_output.device)
    gl = torch.empty((WORLD, t, total_heads), dtype=torch.float32,
                     device=partial_lse.device)
    dist.all_gather_into_tensor(go, partial_output.contiguous())
    dist.all_gather_into_tensor(gl, partial_lse.contiguous())
    my = slice(RANK * HPR, (RANK + 1) * HPR)
    lse = gl[:, :, my].clone()
    lse[torch.isnan(lse) | (lse == float("inf"))] = float("-inf")
    if is_lse_base_on_e:
        lse = lse * LOG2E
    lse_max = lse.amax(dim=0, keepdim=True)
    lse_max = torch.where(torch.isinf(lse_max), torch.zeros_like(lse_max),
                          lse_max)
    w = torch.exp2(lse - lse_max)
    denom = w.sum(dim=0, keepdim=True)
    w = torch.where(denom > 0, w / denom, torch.zeros_like(w))
    out = go[:, :, my, :].float() * w.unsqueeze(-1)
    return out.sum(dim=0)


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _dcp_ops():
    # ops/dcp_utils.py was renamed to ops/dcp.py in the dev1046 nightly;
    # accept either so this test runs against both images.
    try:
        from vllm.v1.attention.ops import dcp as m
    except ImportError:
        from vllm.v1.attention.ops import dcp_utils as m
    return m


def _make_cp_group():
    """A GroupCoordinator spanning all ranks, as the DCP group would be.

    ag_rs needs .all_gather/.reduce_scatter/.rank_in_group; a2a needs only
    .world_size/.device_group. Building the real thing (rather than shimming it
    with raw dist calls) keeps the device communicator vLLM actually dispatches
    through in the measurement.
    """
    from vllm.distributed.parallel_state import init_model_parallel_group

    return init_model_parallel_group(
        [list(range(WORLD))], LOCAL_RANK, "nccl", group_name="ab_dcp"
    )


def bw_probe(dev):
    """Isolate the TRANSPORT: same bytes, four mechanisms (BW=1).

    The A/B measured whole combines, so `direct`'s linear-in-T cost was only
    *inferred* to be ~19 GB/s of peer-write bandwidth by fitting out the
    dispatch+signal+spin+lse work. This measures the movement alone. Every arm
    pushes exactly `nbytes` per rank split evenly across all WORLD peers --
    the pattern dispatch_output_lse_kernel actually uses.

    Six arms, one payload size at a time, each fully barriered off from the
    next -- the first version let them overlap, so ranks still hammering peer
    memory in arm N polluted arm N+1 and every number was junk.

      hbm      plain HBM -> plain HBM. Control. Must land near line rate
               (~3400 GB/s measured standalone on this box); if it doesn't,
               the run is invalid and nothing below means anything.
      symm_rd  symm_mem -> plain HBM. LOCAL read of symmetric memory.
      symm_wr  plain HBM -> own symm_mem. LOCAL write of symmetric memory.
               These two separate "symmetric memory is just slow to touch"
               from "crossing xGMI is slow" -- the first run couldn't, because
               its control read from the symm buffer.
      peer_st  torch.mul(src, 1.0, out=peer_view): TensorIterator elementwise
               kernel reading local HBM and STORING across xGMI. This is
               direct's mechanism -- the .cu stores uint4 to get_peer_ptr().
      peer_cp  peer_view.copy_(src): same bytes, but same-device contiguous
               copy_ lowers to hipMemcpyAsync DtoD, i.e. the COPY ENGINE
               rather than a compute kernel. If this beats peer_st by a lot,
               direct is slow because of HOW it moves bytes, not because xGMI
               is slow -- a fixable restructuring rather than a dead end.
      a2a      dist.all_to_all_single: RCCL, the mechanism a2a/ATOM use.

    peer_st/peer_cp target a single peer ((RANK+1) % WORLD, a ring) so the
    number reads as per-link bandwidth. a2a is a full all-to-all of the same
    payload, so it is if anything handicapped relative to the peer arms.
    GB/s counts payload bytes moved one way, identically for every arm.
    """
    import torch.distributed._symmetric_memory as symm_mem

    group = dist.group.WORLD
    sizes_mb = [float(x) for x in
                os.environ.get("BW_MB", "1,4,16,64").split(",")]
    n_cap = int(max(sizes_mb) * 2**20) // 2
    n_cap -= n_cap % WORLD
    nxt = (RANK + 1) % WORLD

    # The ONLY symmetric-memory allocation here; everything else is plain HBM.
    sbuf = symm_mem.empty(n_cap, dtype=torch.bfloat16, device=dev)
    hdl = symm_mem.rendezvous(sbuf, group.group_name)
    sbuf.fill_(1.0 + RANK)
    peer = hdl.get_buffer(nxt, [n_cap], torch.bfloat16)
    src = torch.ones(n_cap, dtype=torch.bfloat16, device=dev)
    dst = torch.empty(n_cap, dtype=torch.bfloat16, device=dev)
    a2a_out = torch.empty(n_cap, dtype=torch.bfloat16, device=dev)

    def timed(fn, n=20):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n):
            fn()
        e.record()
        torch.cuda.synchronize()
        ms = s.elapsed_time(e) / n
        dist.barrier()   # do NOT let the next arm start under this one's load
        return ms

    if RANK == 0:
        log(f"BW probe: bf16, GB/s payload one-way, peer arms = ring to "
            f"(rank+1)%{WORLD}")
        log(f"{'MB':>6} {'hbm':>9} {'symm_rd':>9} {'symm_wr':>9} "
            f"{'peer_st':>9} {'peer_cp':>9} {'a2a':>9}")
    for mb in sizes_mb:
        n = int(mb * 2**20) // 2
        n -= n % WORLD
        if n <= 0 or n > n_cap:
            continue
        gb = n * 2 / 1e9
        arms = [
            ("hbm", lambda: dst[:n].copy_(src[:n])),
            ("symm_rd", lambda: dst[:n].copy_(sbuf[:n])),
            ("symm_wr", lambda: sbuf[:n].copy_(src[:n])),
            ("peer_st", lambda: torch.mul(src[:n], 1.0, out=peer[:n])),
            ("peer_cp", lambda: peer[:n].copy_(src[:n])),
            ("a2a", lambda: dist.all_to_all_single(
                a2a_out[:n], src[:n], group=group)),
        ]
        r = [gb / (timed(f) / 1e3) for _, f in arms]
        if RANK == 0:
            log(f"{mb:6.0f} " + " ".join(f"{v:9.1f}" for v in r))
    dist.barrier()


def time_arm(name, fn, ref, reset=None):
    """Event-time one combine. Separates GPU time inside the kernels from
    host-side launch/return overhead: gpu_ms ~= wall_ms means the cost is
    on-GPU (handshake atomics over XGMI, or the RCCL kernel); gpu_ms << wall_ms
    means the cost is launch / control plane.

    Returns (gpu_ms, wall_ms, cos) per rank -- collective arms charge
    wait-for-peers to whichever rank arrives first, so read the SLOWEST rank's
    wall as the step cost, not the mean.
    """
    for _ in range(5):
        out = fn()
    torch.cuda.synchronize()
    dist.barrier()

    n = AB_ITERS
    evs = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(n)
    ]
    wall0 = time.time()
    for s, e in evs:
        if AB_BARRIER:
            dist.barrier()
        s.record()
        out = fn()
        e.record()
    launch_wall = time.time() - wall0
    torch.cuda.synchronize()
    total_wall = time.time() - wall0
    gpu_ms = sum(s.elapsed_time(e) for s, e in evs) / n
    wall_ms = total_wall / n * 1000

    # cp_lse_ag_out_rs is DESTRUCTIVE: correct_attn_out passes (out, out) to its
    # triton kernel, so it rewrites cp_attn_out in place. Every call after the
    # first therefore re-corrects its own previous output. Timing is unaffected
    # (data-independent), but the numeric check must run on restored inputs.
    if reset is not None:
        reset()
        out = fn()
        torch.cuda.synchronize()
    if isinstance(out, tuple):
        out = out[0]
    c = cos(out, ref)
    log(
        f"ARM {name:6s} n={n}: gpu={gpu_ms:.3f}ms/call  "
        f"launch_wall={launch_wall / n * 1000:.3f}ms/call  "
        f"total_wall={wall_ms:.3f}ms/call  cos={c:.6f}"
    )
    dist.barrier()
    return gpu_ms, wall_ms, c


def make_inputs(t, dev):
    """The partials for one T, plus the reference, computed while synchronised."""
    total_heads = WORLD * HPR
    g = torch.Generator(device=dev).manual_seed(7919 * RANK + 3)
    partial_output = torch.randn((t, total_heads, HDIM), generator=g,
                                 dtype=torch.float32, device=dev
                                 ).to(torch.bfloat16).contiguous()
    partial_lse = (torch.randn((t, total_heads), generator=g,
                               dtype=torch.float32, device=dev) * 3.0
                   ).contiguous()
    ref = reference(partial_output, partial_lse, True)
    dist.barrier()
    torch.cuda.synchronize()
    return partial_output, partial_lse, ref


def make_skew(dev):
    """Skew load: rank r does r * SKEW units of unrelated GPU work each iter, so
    the ranks drift apart in epoch instead of marching in lockstep."""
    return (int(RANK * SKEW),
            torch.randn(1024, 1024, device=dev, dtype=torch.bfloat16),
            torch.randn(1024, 1024, device=dev, dtype=torch.bfloat16))


def run_point(t, arms, mod, dev, cp_group):
    """A/B every selected combine on one shape.

    Same tensors, same reference, same timing code, so the only difference
    between the numbers is the collective. MAX_NT is deliberately NOT tied to t:
    the first sweep sized the direct workspace per point, which left its scaling
    confounded with its capacity. Fix it across the sweep instead.
    """
    partial_output, partial_lse, ref = make_inputs(t, dev)
    if RANK == 0:
        log(f"A/B arms={arms} T={t} MAX_NT={MAX_NT} HPR={HPR} HDIM={HDIM} "
            f"WORLD={WORLD} bytes/rank/call: a2a-class={t * HPR * HDIM * 2 * WORLD} "
            f"ag_rs~{WORLD}x more  AB_BARRIER={int(AB_BARRIER)}")
    ws = None
    results = {}
    for name in arms:
        # Private copies per arm: ag_rs mutates its input in place, so a
        # shared buffer would let whichever arm ran first poison the rest.
        po = partial_output.clone()
        pl = partial_lse.clone()

        def reset(po=po, pl=pl):
            po.copy_(partial_output)
            pl.copy_(partial_lse)

        if name == "direct":
            pg = dist.distributed_c10d._get_default_group()
            ws = mod.DirectDCPA2AWorkspace(
                pg, dev, MAX_NT, HPR, HDIM,
                dtype=torch.bfloat16, num_ubatches=1)
            fn = lambda po=po, pl=pl: ws.lse_reduce(po, pl, True)
        else:
            impl = getattr(mod, "dcp_a2a_lse_reduce" if name == "a2a"
                           else "cp_lse_ag_out_rs", None)
            if impl is None:
                log(f"ARM {name}: not present in this image -- skipped")
                continue
            fn = lambda impl=impl, po=po, pl=pl: impl(
                po, pl, cp_group, is_lse_base_on_e=True)
        results[name] = time_arm(name, fn, ref, reset=reset)

    if RANK == 0 and len(results) > 1:
        base = results.get("ag_rs", (None, None, None))[1]
        log(f"A/B SUMMARY T={t} (rank0; read slowest rank's wall as step cost)")
        for name, (gm, wm, cm) in results.items():
            rel = f"  {base / wm:.2f}x vs ag_rs" if base else ""
            log(f"  {name:6s} gpu={gm:.3f}ms wall={wm:.3f}ms "
                f"cos={cm:.6f}{rel}")
    return results


def main():
    torch.cuda.set_device(LOCAL_RANK)
    dev = torch.device("cuda", LOCAL_RANK)
    dist.init_process_group(backend="nccl")
    arms = ARM_ORDER if BACKEND == "all" else [BACKEND]
    assert set(arms) <= set(ARM_ORDER), f"unknown BACKEND={BACKEND}"
    mod = _dcp_ops()

    ws = None
    if "direct" in arms:
        torch.ops.load_library(SO)
        assert hasattr(torch.ops._C, "direct_dcp_a2a_lse_reduce"), "op not registered"

    # Force both communicators up FRONT, timed, before any measurement. RCCL
    # defers all connection setup to the first collective on each communicator,
    # so this is where the minutes go -- charging it to the sweep made every
    # point look like a hang and hid the A/B behind a timeout.
    t0 = time.time()
    dist.all_reduce(torch.ones(1, device=dev))
    torch.cuda.synchronize()
    if RANK == 0:
        log(f"rccl default-group up in {time.time() - t0:.1f}s")
    if os.environ.get("BW", "0") == "1":
        # Transport-only mode: no combine arms, no soak. Runs after the RCCL
        # warmup above so all_to_all_single is not charged for connection setup.
        bw_probe(dev)
        dist.barrier()
        dist.destroy_process_group()
        return 0

    cp_group = None
    if set(arms) - {"direct"}:
        t0 = time.time()
        cp_group = _make_cp_group()
        # Warm it with a collective ON THAT GROUP, not a default-group barrier:
        # creating the communicator is cheap, and RCCL builds the connections
        # lazily on its first collective. Warming the wrong group leaves that
        # cost inside whichever arm runs first (ag_rs), which is exactly how a
        # 5-minute setup got charged to the measurement.
        cp_group.all_gather(torch.ones(1, device=dev), dim=0)
        cp_group.reduce_scatter(torch.ones(WORLD, device=dev), dim=0)
        torch.cuda.synchronize()
        if RANK == 0:
            log(f"cp_group up in {time.time() - t0:.1f}s")

    if os.environ.get("EVENT_TIME", "1") == "1":
        for T_CUR in T_LIST:
            run_point(T_CUR, arms, mod, dev, cp_group)
        if RANK == 0 and len(T_LIST) > 1:
            log("sweep done")
    if "direct" not in arms or os.environ.get("SOAK", "1") != "1":
        dist.barrier()
        dist.destroy_process_group()
        return 0

    partial_output, partial_lse, ref = make_inputs(T_LIST[-1], dev)
    skew_units, sa, sb = make_skew(dev)

    if ws is None:  # EVENT_TIME=0 skipped the allocation
        ws = mod.DirectDCPA2AWorkspace(
            dist.distributed_c10d._get_default_group(), dev, MAX_NT, HPR, HDIM,
            dtype=torch.bfloat16, num_ubatches=1)

    if RANK == 0:
        log(f"sync-free drive: ITERS={ITERS} SKEW={SKEW} T={T_LIST[-1]} HPR={HPR} "
            f"WORLD={WORLD} (no per-call barrier/synchronize)")

    out = None
    t0 = time.time()
    with Heartbeat() as hb:
        for i in range(ITERS):
            hb.iters = i
            hb.where = f"iter{i}:skew"
            for _ in range(skew_units):
                sa = sa @ sb
            hb.where = f"iter{i}:lse_reduce"
            # NO barrier, NO synchronize -- this is the whole point.
            out = ws.lse_reduce(partial_output, partial_lse, True)

            if CHECK_EVERY and (i + 1) % CHECK_EVERY == 0:
                hb.where = f"iter{i}:sync"
                torch.cuda.synchronize()
                c = cos(out, ref)
                if RANK == 0:
                    log(f"iter {i + 1}/{ITERS} retired  cos={c:.6f}  "
                        f"elapsed={time.time() - t0:.1f}s")
                if c < 0.999:
                    log(f"NUMERIC FAIL at iter {i + 1}: cos={c:.6f}")
                    break
        hb.where = "final-sync"
        torch.cuda.synchronize()

    c = cos(out, ref)
    md = (out.float() - ref).abs().max().item()
    ok = c > 0.999 and md < 5e-2
    log(f"done {ITERS} sync-free iters in {time.time() - t0:.1f}s "
        f"cos={c:.6f} max|d|={md:.3e} {'OK' if ok else 'FAIL'}")

    ft = torch.tensor([0 if ok else 1], dtype=torch.int64, device=dev)
    dist.all_reduce(ft, op=dist.ReduceOp.SUM)
    if RANK == 0:
        tot = int(ft.item())
        log(f"VERDICT: {'PASS' if tot == 0 else 'FAIL'} "
            f"({tot} failing ranks of {WORLD}) -- no deadlock observed"
            if tot == 0 else f"VERDICT: FAIL ({tot} ranks)")
    dist.barrier()
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
