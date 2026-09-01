#!/usr/bin/env python3
"""Refuse to start a long run when the box will make it hang.

The problem this closes
-----------------------
K3 serve launches have repeatedly "hung" during weight loading -- 580 s per
shard, ETA 15 h -- with no error, which cost many debugging sessions and got
misattributed to DCP. Measured root cause, isolated with no vLLM/NCCL/DCP in the
picture (``_k3_load_probe.py``, ``_h2d_source_ab.py``):

  * H2D is fixed-cost bound, not bandwidth bound. A 1 GiB copy takes ~0.2 s
    whether the source is pageable, pinned, or a tmpfs mmap -- the same ~208 ms
    that a 4 KB all-reduce costs. Clean, this box does ~54 GB/s.
  * The K3 checkpoint is ~519,000 tensors (96 shards x ~5,400, median 1.31 MiB).
    At ~30 ms per transfer that is ~4.3 h per rank. Hence the "hang".
  * Refuted along the way: GTT is not the constraint (0.42 GB used of 1623 GB
    per GPU even under a live 8-rank job), and the loader mmaps rather than
    staging the checkpoint into anonymous RAM.

The trigger is a second GPU-resident process set: two of them make KFD
ping-pong queue evict/restore, and every transfer waits a full cycle. Kernel
launches stay fast (~10 us), which is why the box looks healthy and the failure
looks like an idle hang.

So: measure the thing that actually predicts the hang -- achieved H2D bandwidth
-- and refuse to launch when it is bad. A 20 s check instead of a 15 h wedge.

A one-shot check loses a race
-----------------------------
Measured 2026-08-21: this gate passed a clean box (53.48 GiB/s, zero holders),
the serve launched 15 s later, and a colleague's 8-rank job started 17 s after
that. All 8 of my workers then parked in ``ncclCommInitRank`` -- librccl's TCP
bootstrap ``recv`` -- from that exact second, at VRAM 6 GiB, having never loaded
a weight. Same contention, one stage earlier than the H2D case.

So the gate also supports waiting for a free box and requiring it to *stay*
free: ``--stable-for`` re-checks holders over a window before giving the
all-clear, and ``--wait`` blocks until the box frees so a launch can be queued
against a busy box instead of hand-polling for it.

None of this can eliminate the race -- a job that starts after ours always can
-- so it is paired with a post-launch watchdog in ``_serve_k3_dcp_test.sh``
that fails fast instead of hanging.

Exit codes: 0 healthy, 1 degraded (do not launch), 2 could not measure.

Usage:
    python3 _gpu_preflight.py                     # gate, ~20 s
    python3 _gpu_preflight.py --min-gibs 20       # custom threshold
    python3 _gpu_preflight.py --report-only       # never fail, just print
    python3 _gpu_preflight.py --stable-for 30     # box must stay free 30 s
    python3 _gpu_preflight.py --wait 7200         # queue behind a busy box
"""

import argparse
import builtins
import functools
import subprocess
import sys
import time

# This gate is usually redirected to a file and, in --wait mode, may sit for
# hours before it says anything. Python block-buffers a piped stdout, which
# would make a waiting gate indistinguishable from a hung one.
print = functools.partial(builtins.print, flush=True)  # noqa: A001

GIB = 2**30


def foreign_holders():
    """VRAM holders as (pid, bytes), from KFD's own accounting."""
    try:
        out = subprocess.run(
            ["rocm-smi", "--showpids"], capture_output=True, text=True, timeout=60
        ).stdout
    except Exception:
        return None
    rows = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0].isdigit():
            try:
                rows.append((int(parts[0]), int(parts[3])))
            except ValueError:
                continue
    return rows


# Monitoring tools, short probes and idle REPLs hold a GiB or so without
# meaningfully contending. Only a real job is worth refusing a launch over.
HOLDER_MIN_GIB = 2.0


def live_holders(min_gib=HOLDER_MIN_GIB):
    """(pids, total_gib) of processes holding real VRAM, or (None, 0)."""
    rows = foreign_holders()
    if rows is None:
        return None, 0.0
    live = [(p, b) for p, b in rows if b / GIB >= min_gib]
    return sorted(p for p, _ in live), sum(b for _, b in live) / GIB


def report_holders(pids, tot):
    if pids is None:
        return
    if pids:
        print(f"WARNING: {len(pids)} GPU-resident process(es) holding "
              f"{tot:.0f} GiB total.")
        print("         If they are not yours, do NOT reclaim them -- wait.")
        print(f"         pids: {pids}")
    else:
        print("KFD: no process is holding VRAM.")


def wait_for_free(deadline, poll_s=20.0):
    """Block until nothing holds VRAM. Returns True if the box freed in time."""
    announced = False
    while True:
        pids, tot = live_holders()
        if pids is not None and not pids:
            if announced:
                print("box is free.")
            return True
        if time.time() >= deadline:
            print(f"still busy after --wait expired ({len(pids or [])} holders, "
                  f"{tot:.0f} GiB). Giving up.")
            return False
        if not announced:
            report_holders(pids, tot)
            print(f"waiting for the box to free (polling every {poll_s:.0f} s; "
                  "do NOT reclaim someone else's job)...")
            announced = True
        time.sleep(poll_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-gibs", type=float, default=20.0,
                    help="minimum acceptable H2D GiB/s (clean box does ~50)")
    ap.add_argument("--size-gib", type=float, default=1.0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--stable-for", type=float, default=0.0,
                    help="seconds the box must stay free after the probe "
                         "before giving the all-clear; catches a job that "
                         "starts while we are still measuring")
    ap.add_argument("--wait", type=float, default=0.0,
                    help="if the box is busy, block up to this many seconds "
                         "for it to free instead of failing immediately")
    args = ap.parse_args()

    # In --wait mode the point is to *capture* the window, so a box that frees
    # and is immediately retaken should send us back to waiting rather than
    # give up -- otherwise a queued launch loses to the first passer-by.
    deadline = time.time() + args.wait
    attempt = 0
    while True:
        attempt += 1
        rc = gate(args, deadline)
        if rc != 1 or time.time() >= deadline:
            return rc
        print(f"re-arming the gate (attempt {attempt + 1}); "
              f"{(deadline - time.time()) / 60:.0f} min of --wait left.")
        print()


def gate(args, deadline):
    """One full check: wait for a free box, measure it, confirm it stays free."""
    if args.wait > 0 and not wait_for_free(deadline):
        return 0 if args.report_only else 1

    report_holders(*live_holders())

    try:
        import torch
    except Exception as exc:                                  # pragma: no cover
        print(f"could not import torch: {exc}")
        return 2

    try:
        dev = torch.device("cuda", args.gpu)
        torch.cuda.set_device(dev)
        free_b, total_b = torch.cuda.mem_get_info(dev)
        print(f"gpu{args.gpu}: {free_b / GIB:.1f} GiB free of "
              f"{total_b / GIB:.1f} GiB")

        n = int(args.size_gib * GIB) // 2
        host = torch.empty(n, dtype=torch.bfloat16).pin_memory()

        best = 0.0
        for i in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            g = host.to(dev)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            rate = args.size_gib / dt
            best = max(best, rate)
            if i:                     # first pass is warmup
                print(f"  H2D {args.size_gib:.1f} GiB: {dt:6.3f} s "
                      f"= {rate:6.2f} GiB/s")
            del g
            torch.cuda.empty_cache()
    except Exception as exc:
        print(f"probe failed: {exc}")
        return 2

    print()
    print(f"best H2D: {best:.2f} GiB/s (threshold {args.min_gibs:.1f})")
    if best < args.min_gibs:
        print()
        print("DEGRADED -- do not start a serve. A K3 load is ~519,000 tensor "
              "transfers;")
        print(f"at this rate that is roughly "
              f"{519_000 * (0.03 * 20.0 / max(best, 0.01)) / 3600:.1f} h per "
              f"rank, which will look like a silent hang.")
        print("Wait for the other GPU-resident job to finish, then re-run this.")
        return 0 if args.report_only else 1

    if args.stable_for > 0:
        print(f"confirming the box stays free for {args.stable_for:.0f} s ...")
        end = time.time() + args.stable_for
        while time.time() < end:
            time.sleep(min(5.0, max(0.5, end - time.time())))
            pids, tot = live_holders()
            if pids:
                print()
                print(f"ABORT: a job appeared mid-check -- {len(pids)} "
                      f"process(es), {tot:.0f} GiB, pids {pids}.")
                print("Launching now would collide with it during RCCL "
                      "bootstrap or weight load. Wait it out.")
                return 0 if args.report_only else 1

    print("HEALTHY -- safe to launch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
