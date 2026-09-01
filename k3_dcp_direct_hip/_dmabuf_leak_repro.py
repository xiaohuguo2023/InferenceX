#!/usr/bin/env python3
"""
Minimal reproducer for the amdgpu dma-buf / eviction-fence leak.

THE BUG (amdgpu-6.16.6-2238411.22.04)

  A native amdgpu BO owns its resv, so bo->base.resv == &bo->base._resv, and an
  exported dma_buf->resv ALIASES that private resv.  On release:

    ttm_bo_release()                      ttm/ttm_bo.c:250
      ttm_bo_individualize_resv()         ttm/ttm_bo.c:196 -> returns 0, no lock
      amdgpu_bo_release_notify()          amdgpu/amdgpu_object.c:1332
        if (WARN_ON_ONCE(!dma_resv_trylock(&bo->base._resv)))
                return;                   <-- amdgpu_object.c:1351
        amdgpu_amdkfd_remove_all_eviction_fences(abo);   <-- SKIPPED

  A peer process inside dma_buf_detach()/unmap_attachment() holds
  dma_resv_lock(dmabuf->resv) -- the SAME lock -- so the trylock loses the race
  and the eviction fences are never removed.  Control returns to
  ttm_bo_release(), which immediately tests

    if (!dma_resv_test_signaled(&bo->base._resv, DMA_RESV_USAGE_BOOKKEEP) || ...)
            bo->deleted = true; kref_init(&bo->kref); queue_work(...); return;

  The unsignalled eviction fences make the BO look busy, so it is resurrected for
  delayed destroy -- and on that second pass bo->deleted is true, so
  release_notify() is NEVER called again.  One lost trylock == one permanently
  stranded BO.  The fences also cannot be removed earlier: the helper does
  dma_resv_assert_held(resv).

  It then gets worse in ttm_bo_delayed_delete() (ttm/ttm_bo.c), whose FIRST act is

    dma_resv_wait_timeout(&bo->base._resv, DMA_RESV_USAGE_BOOKKEEP, false,
                          MAX_SCHEDULE_TIMEOUT);

  -- an unbounded wait on the very fences release_notify() failed to remove, taken
  before the worker locks anything.  bdev->wq is alloc_workqueue("ttm",
  WQ_MEM_RECLAIM|WQ_HIGHPRI|WQ_UNBOUND, 16) PER DEVICE, so 17 strands on one GPU
  saturate that GPU's TTM workqueue with blocked workers and every later BO free on
  that device -- in every process on the box -- queues behind them.

  The block is not permanent: kfd_process.c:1184 signals the eviction fence at
  process teardown (deliberately before freeing BOs), so the workers drain once the
  leaking process finally exits.  That is why the damage tracks the lifetime of a
  long-running server and appears to "self-heal" minutes after a killed job.
  Critically, our WARN came via kfd_ioctl_free_memory_of_gpu -- a LIVE process
  freeing a BO, where the fence is still unsignalled.  That is the only branch in
  which losing the trylock does harm, and it is exactly what RCCL/symmetric-memory
  communicator teardown does.

  Observed cost: the strand rate is per-teardown, and every later peer mapping
  pays for the accumulated set.  RCCL's `connections` init bucket went 2.31s ->
  114.81 -> 204.42 -> 324.76 across successive communicators in one process, i.e.
  ncclCommInitRank 11.8s -> 195s box-wide, in jobs that never touch the GPU that
  leaked.  WARN_ON_ONCE means only the first strand is ever printed.

WHAT THIS SCRIPT DOES

  Spawns N plain processes that init an RCCL communicator and do one all-reduce --
  no vLLM, no model, no application code -- then either exit cleanly or die
  simultaneously.  It reads the kernel's own dma-buf table before and after each
  round, so the leak is reported as a SLOPE.

  --stage symm additionally builds the symmetric-memory peer mesh the way
  DirectCPWorkspace._allocate() does (symm_mem.empty -> rendezvous -> get_buffer
  for all N peers, three times, at the production DCP workspace shapes).  On ROCm
  that is the HIP VMM path, so each export becomes a KFD dma-buf and each peer
  import a cross-process BO mapping: N exporters x N importers x 3 allocations.
  This is the arm that exercises the suspected leak path; --stage rccl is the
  control that does not.

  It also watches the TTM workqueue threads across the teardown window and reports
  the peak number sitting in D (uninterruptible).  That is the direct observable for
  the blocked-worker half of the chain above, and it is the only one we have:
  WARN_ON_ONCE prints once per boot, so the kernel will not tell us twice.  Because
  the block lasts only as long as the leaking process, a single reading after the
  fact sees nothing -- it has to be polled, which is what --settle now does.

  Arms (the point of the script -- each isolates one link in the chain above):

    --exit clean   ranks barrier, destroy_process_group(), exit 0
    --exit kill    parent SIGKILLs all ranks at once, mid-flight
    --p2p off      NCCL_P2P_DISABLE=1: RCCL uses SHM, never builds the peer mesh
    --stage symm   also build the symm_mem peer mesh (see above)
    --teardown ordered   drop peer IMPORTS on every rank, barrier, then release
                   the local EXPORT -- the candidate fix.  The leak is an
                   ORDERING bug: at ordinary exit an exporter releases while
                   peers still hold imports, so amdgpu_bo_release_notify loses
                   the dma_resv_trylock on the aliased private resv and skips
                   removing the eviction fences.  Pair it with the default
                   --teardown none as the A/B:
                     --stage symm --exit clean --teardown none     -> +16/round
                     --stage symm --exit clean --teardown ordered  -> expect +0

  If `kill` leaks and `clean` does not, the race is in teardown ordering.  If
  `p2p off` leaks under neither, the peer dma-buf mesh is required, which both
  confirms the mechanism and gives a workaround for debugging on a live box.

  MEASURED 2026-08-25, clean post-reboot driver, 5 rounds each, --stage rccl:
    clean/p2p on  +0 bufs   kill/p2p off  +0 bufs   kill/p2p on  +0 bufs
  So a plain 8-rank RCCL job leaks nothing even when SIGKILLed mid-flight with
  peer IPC live.  That is what makes --stage symm the next arm rather than a
  redundant one: whatever stranded 1304 BOs is not reachable from RCCL alone.

USAGE

  Needs debugfs for the measurement (container-local mount namespace is fine):
    mount -t debugfs none /sys/kernel/debug

    python3 _dmabuf_leak_repro.py --exit kill  --rounds 5
    python3 _dmabuf_leak_repro.py --exit clean --rounds 5
    python3 _dmabuf_leak_repro.py --exit kill  --rounds 5 --p2p off

  Verifying a driver patch does not need a reboot -- stop every container holding
  /dev/kfd, `modprobe -r amdgpu`, reload, and re-run.  A fixed driver keeps the
  floor flat across rounds under `--exit kill`.
"""

import argparse
import collections
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

BUFINFO = "/sys/kernel/debug/dma_buf/bufinfo"

# The bufinfo data lines are TAB separated:
#   size    flags   mode    count   exp_name        ino
# all as fixed-width lowercase hex except exp_name.  The header line starts with
# "size" so the hex anchor is what keeps it out.
_ROW = re.compile(r"^([0-9a-f]{8})\t([0-9a-f]{8})\t([0-9a-f]{8})\t([0-9a-f]{8})\t(\S+)")


class Floor:
    """A reading of the kernel's dma-buf table."""

    def __init__(self, nbufs=0, nbytes=0, refs=None, devs=None, ok=True):
        self.nbufs, self.nbytes, self.ok = nbufs, nbytes, ok
        self.refs = refs or collections.Counter()
        self.devs = devs or collections.Counter()

    def __str__(self):
        if not self.ok:
            return "bufinfo UNREADABLE"
        top = ", ".join(f"{k}->{v}" for k, v in sorted(self.refs.items())[:4])
        return f"{self.nbufs} bufs / {self.nbytes / 2**30:.2f} GiB  refcounts[{top}]"


def read_floor(path=BUFINFO):
    try:
        txt = open(path).read()
    except OSError:
        return Floor(ok=False)
    f = Floor()
    for ln in txt.splitlines():
        m = _ROW.match(ln)
        if m:
            f.nbufs += 1
            f.nbytes += int(m.group(1), 16)
            f.refs[int(m.group(4), 16)] += 1
            continue
        d = re.match(r"^Total (\d+) devices attached", ln)
        if d:
            f.devs[int(d.group(1))] += 1
    return f


def ttm_workers():
    """(blocked_pids, total) for the TTM workqueue threads.

    A WQ_MEM_RECLAIM workqueue gets a rescuer thread named after the wq ("ttm"),
    one per amdgpu device; the unbound pool workers currently running ttm items
    render as "kworker/uNNN:M-ttm".  Both carry "ttm" in comm.  An idle worker
    shows as I; one parked in dma_resv_wait_timeout() shows as D.
    """
    out = subprocess.run(["ps", "-eo", "pid,stat,comm"],
                         capture_output=True, text=True).stdout
    total, blocked = 0, []
    for ln in out.splitlines()[1:]:
        f = ln.split(None, 2)
        if len(f) < 3 or "ttm" not in f[2]:
            continue
        total += 1
        if f[1].startswith("D"):
            blocked.append(int(f[0]))
    return blocked, total


def stack_of(pid):
    """Top kernel frames of a blocked worker, or "" if unreadable.

    Needs root and kptr_restrict permitting; without it we still get the D count,
    which is the part that matters.  With it, this is what turns "a worker is
    blocked" into "blocked in dma_resv_wait_timeout <- ttm_bo_delayed_delete".
    """
    try:
        lines = open(f"/proc/{pid}/stack").read().splitlines()
    except OSError:
        return ""
    frames = [ln.split(" ", 1)[-1].strip() for ln in lines if ln.strip()]
    return " <- ".join(frames[:4])


class TtmWatch(threading.Thread):
    """Sample blocked TTM workers for the whole round, in the background.

    It has to span the round rather than run after it: on a clean exit the frees
    happen inside destroy_process_group(), i.e. while we are still reaping, and on
    a kill they happen during teardown.  Either way the peak can pass before a
    post-hoc reading would start.  Reports the peak, never an instantaneous value.
    """

    def __init__(self, poll=0.5):
        super().__init__(daemon=True)
        self.poll, self.peak, self.real_peak = poll, 0, 0
        self.stacks, self.real_samples = {}, 0
        self._done = threading.Event()  # not _stop: Thread._stop is internal

    def run(self):
        while not self._done.is_set():
            blocked, _ = ttm_workers()
            # D alone is not enough: an idle pool worker parked in the scheduler
            # (worker_thread <- kthread <- ret_from_fork) also reads D, and on the
            # first arm those outnumbered the real ones. Classify by stack every
            # poll -- never cache per-pid, or a worker seen idle once is never
            # re-examined when it later blocks for real.
            real = 0
            for pid in blocked:
                s = stack_of(pid)
                if "ttm_bo_delayed_delete" in s or "resv_wait_timeout" in s:
                    real += 1
                    self.stacks.setdefault(pid, s)
            self.peak = max(self.peak, len(blocked))
            self.real_peak = max(self.real_peak, real)
            self.real_samples += real
            self._done.wait(self.poll)

    def stop(self):
        self._done.set()
        self.join(timeout=5)
        # real_samples * poll = worker-seconds spent in the delayed-delete wait,
        # which is the quantity that competes for the 16 slots, not the peak.
        return self.peak, self.real_peak, self.real_samples * self.poll, self.stacks


CHILD = r'''
import os, sys, time, torch, torch.distributed as dist

rank  = int(os.environ["REPRO_RANK"])
world = int(os.environ["REPRO_WORLD"])
rundir, mode = os.environ["REPRO_DIR"], os.environ["REPRO_EXIT"]
stage = os.environ["REPRO_STAGE"]
teardown = os.environ["REPRO_TEARDOWN"]

torch.cuda.set_device(rank)
dist.init_process_group(
    backend="nccl", init_method="file://" + os.path.join(rundir, "store"),
    world_size=world, rank=rank)

# One real collective, so the peer transports are actually connected and the
# dma-buf mesh exists.  Without traffic RCCL may defer connection setup.
t = torch.ones(1 << 20, device=f"cuda:{rank}")
dist.all_reduce(t)
torch.cuda.synchronize()
assert int(t[0].item()) == world, "all_reduce wrong"

allocations = []
if stage == "symm":
    from torch.distributed._symmetric_memory import empty as symm_empty
    from torch.distributed._symmetric_memory import rendezvous as symm_rendezvous

    # symm_mem rendezvous is a collective keyed by the group NAME, so it needs a
    # named group -- the default group's name is not usable here on every torch
    # build.  A fresh gloo-free nccl subgroup over all ranks gives us one.
    grp = dist.new_group(list(range(world)))
    gname = grp.group_name

    # Production DCP workspace shapes (num_ubatches=2, max_num_tokens=384,
    # heads_per_rank = 128/world, head_dim=512): output bf16, lse fp32, signal
    # int32 -- three _allocate() calls, exactly as DirectDCPA2AWorkspace does.
    U, MT, HD = 2, 384, 512
    HPR = max(1, 128 // world)
    shapes = [((U, 2, world, MT, HPR, HD), torch.bfloat16),
              ((U, 2, world, MT, HPR), torch.float32),
              ((U, 2, world), torch.int32)]

    for shape, dtype in shapes:
        storage = symm_empty(shape, device=f"cuda:{rank}", dtype=dtype)
        storage.zero_()
        torch.cuda.synchronize()
        handle = symm_rendezvous(storage, gname)
        assert handle is not None, "symm_mem rendezvous returned None"
        handle.barrier()
        # get_buffer for EVERY peer is what turns one export into world_size
        # imports; this is the line that builds the cross-process BO mesh.
        views = [handle.get_buffer(peer, list(shape), dtype, 0)
                 for peer in range(world)]
        # Pin them exactly like _allocations does -- the production code never
        # frees these, so a faithful repro must not either.
        allocations.append((storage, handle, views))
    torch.cuda.synchronize()

# Announce readiness only after the mesh is up; the parent kills on this signal.
open(os.path.join(rundir, f"ready.{rank}"), "w").close()

if mode == "clean":
    if teardown == "ordered" and allocations:
        # --- ORDERED TEARDOWN: the candidate fix, tested here before it is
        # ported into vllm/v1/attention/ops/cp_common.py ---
        #
        # The leak is an ORDERING bug, not a missing-free bug.  At ordinary
        # process exit every rank drops its own export and its 8 peer imports in
        # arbitrary order, so an exporter routinely calls release while peers
        # still hold imports of it.  amdgpu_bo_release_notify then loses the
        # dma_resv_trylock on the ALIASED private resv (amdgpu_object.c:1351),
        # skips amdgpu_amdkfd_remove_all_eviction_fences, and strands the BO at
        # refcount 7 with live eviction fences.
        #
        # So: every rank drops all IMPORTS first, then a barrier guarantees no
        # importer anywhere still references a peer's BO, and only then do the
        # exports go.  Each release now sees a refcount nobody else holds and
        # takes the lock it expects.
        for _storage, _handle, views in allocations:
            views.clear()            # 1. imported peer BOs
        torch.cuda.synchronize()
        dist.barrier()               # 2. all importers gone, everywhere
        allocations.clear()          # 3. handles + our own exported storage
        torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(0)

# --exit kill: stay alive holding the mesh until the parent SIGKILLs us.
time.sleep(600)
'''


def spawn_round(args, rundir):
    """Run one N-rank round. Returns (wall_seconds, how_it_ended)."""
    env = dict(os.environ)
    env["REPRO_WORLD"] = str(args.procs)
    env["REPRO_DIR"] = rundir
    env["REPRO_EXIT"] = args.exit
    env["REPRO_STAGE"] = args.stage
    env["REPRO_TEARDOWN"] = args.teardown
    # Keep RCCL quiet unless asked; INFO here would be ~10k lines/rank.
    env.setdefault("NCCL_DEBUG", "WARN")
    if args.p2p == "off":
        env["NCCL_P2P_DISABLE"] = "1"

    procs = []
    for r in range(args.procs):
        e = dict(env, REPRO_RANK=str(r))
        procs.append(subprocess.Popen([sys.executable, "-c", CHILD], env=e,
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.PIPE))

    t0 = time.time()
    ready, deadline = set(), t0 + args.timeout
    while len(ready) < args.procs and time.time() < deadline:
        for r in range(args.procs):
            if os.path.exists(os.path.join(rundir, f"ready.{r}")):
                ready.add(r)
        dead = [p for p in procs if p.poll() is not None and p.returncode != 0]
        if dead and args.exit == "kill":
            err = (dead[0].stderr.read() or b"").decode()[-400:]
            for p in procs:
                p.kill()
            return time.time() - t0, f"RANK DIED early rc={dead[0].returncode} {err}"
        time.sleep(0.25)

    if len(ready) < args.procs:
        for p in procs:
            p.kill()
        for p in procs:
            p.wait()
        return time.time() - t0, f"TIMEOUT: only {len(ready)}/{args.procs} reached the mesh"

    if args.exit == "kill":
        # Simultaneity is the point -- the race needs a peer inside
        # dma_buf_detach() while the exporter is in release_notify().  Issue the
        # signals back to back with nothing in between, then reap.
        for p in procs:
            try:
                os.kill(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        how = "SIGKILLed together"
    else:
        how = "clean exit"

    for p in procs:
        try:
            p.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
    return time.time() - t0, how


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--exit", choices=("kill", "clean"), default="kill")
    ap.add_argument("--p2p", choices=("on", "off"), default="on")
    ap.add_argument("--stage", choices=("rccl", "symm"), default="rccl",
                    help="rccl: RCCL init + all-reduce only (control). "
                         "symm: also build the symmetric-memory peer mesh the "
                         "way DirectCPWorkspace._allocate() does")
    ap.add_argument("--teardown", choices=("none", "ordered"), default="none",
                    help="none: drop the mesh the way production does, i.e. let "
                         "process exit release exports and imports in arbitrary "
                         "order (this is what leaks). ordered: every rank drops "
                         "its peer IMPORTS, barriers, then releases its own "
                         "EXPORT -- the candidate fix. Only meaningful with "
                         "--stage symm --exit clean")
    ap.add_argument("--settle", type=float, default=20.0,
                    help="seconds to wait after teardown before reading the floor; "
                         "KFD reclaim is asynchronous and an early read reports "
                         "buffers that are merely in flight as leaked")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--bufinfo", default=BUFINFO)
    ap.add_argument("--force", action="store_true",
                    help="run even if other KFD processes hold the GPUs")
    args = ap.parse_args()

    base = read_floor(args.bufinfo)
    if not base.ok:
        sys.exit(f"cannot read {args.bufinfo} -- "
                 "run privileged and: mount -t debugfs none /sys/kernel/debug")

    busy = subprocess.run(["rocm-smi", "--showpids"], capture_output=True, text=True)
    if "No KFD PIDs" not in busy.stdout and not args.force:
        sys.exit("other KFD processes are on the GPUs; the floor delta would be "
                 "theirs as much as ours. Wait, or pass --force.")

    blocked0, ttm_total = ttm_workers()
    if ttm_total == 0:
        # Kernel threads live in the initial PID namespace.  A container without
        # --pid=host sees none of them and this probe would silently read 0 all
        # run -- indistinguishable from "nothing ever blocked", which is exactly
        # the result we are trying to measure.  Refuse rather than mislead.
        print("WARNING: no ttm kernel threads visible -- this process is in a PID "
              "namespace. The blocked-worker probe is BLIND here; run in a "
              "--pid=host container, or watch from the host in a second shell.")
    if stack_of(os.getpid()) == "":
        print("note: /proc/<pid>/stack unreadable (needs root); D-state counts only")

    print(f"arm: procs={args.procs} exit={args.exit} p2p={args.p2p} "
          f"stage={args.stage} teardown={args.teardown} "
          f"rounds={args.rounds} settle={args.settle}s")
    print(f"round 0 (baseline)   {base}  "
          f"ttm workers={ttm_total} blocked={len(blocked0)}")

    prev, rows, ttm_peaks = base, [], []
    for i in range(1, args.rounds + 1):
        rundir = tempfile.mkdtemp(prefix="dmabuf_repro_")
        watch = TtmWatch()
        watch.start()
        try:
            wall, how = spawn_round(args, rundir)
        finally:
            shutil.rmtree(rundir, ignore_errors=True)
        time.sleep(args.settle)
        peak, real_peak, wsec, stacks = watch.stop()
        ttm_peaks.append(real_peak)
        cur = read_floor(args.bufinfo)
        d = cur.nbufs - prev.nbufs
        rows.append(d)
        print(f"round {i}  {wall:6.1f}s  {how:24s} {cur}  "
              f"delta={d:+d}  ttmD={peak} delayed_delete_peak={real_peak} "
              f"wait={wsec:.1f} worker-s")
        for pid, s in stacks.items():
            print(f"    ttm worker {pid} in delayed-delete wait: {s}")
        if "TIMEOUT" in how or "DIED" in how:
            print("  ^ round did not establish the mesh; its delta means nothing")
        prev = cur

    total = prev.nbufs - base.nbufs
    print(f"\nnet {total:+d} bufs over {args.rounds} rounds "
          f"({total / max(1, args.rounds):+.1f}/round), "
          f"{(prev.nbytes - base.nbytes) / 2**30:+.2f} GiB")
    if total > 0:
        print("LEAK: buffers survived with no owning process. Per-round deltas: "
              + " ".join(f"{d:+d}" for d in rows))
        if prev.devs:
            fan = ", ".join(f"{k} devs->{v}" for k, v in sorted(prev.devs.items()))
            print(f"attachment fan-out: {fan}")
    else:
        print("no leak in this arm.")

    print("delayed-delete blocked-worker peak per round: "
          + " ".join(str(p) for p in ttm_peaks)
          + f"  (of {ttm_total} ttm threads; saturation is 16/device)")
    if max(ttm_peaks, default=0) == 0:
        print("  no worker ever blocked -- the delayed-delete wait is NOT being hit, "
              "so a leak here would have a different mechanism than the one in the "
              "module docstring.")


if __name__ == "__main__":
    main()
