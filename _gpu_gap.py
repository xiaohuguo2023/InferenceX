#!/usr/bin/env python3
"""Per-rank GPU-stream idle/gap analysis for a torch trace.
Discriminates a host launch-skew bubble (GPU idle waiting for CPU) from an
on-GPU spin (GPU busy inside cross_device_reduce). Usage: _gpu_gap.py <trace.gz>"""
import gzip, json, sys, collections

f = sys.argv[1]
with gzip.open(f, "rt") as fh:
    data = json.load(fh)
ev = data.get("traceEvents", data if isinstance(data, list) else [])

# find GPU (device) streams: torch marks them with args cat 'kernel' on device tids.
# Heuristic: ph=='X' complete events whose 'cat' is 'kernel' are GPU kernels.
kern = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel"
        and "dur" in e and "ts" in e]
if not kern:
    # fallback: events on tracks named 'stream' or with 'DeviceType' gpu
    kern = [e for e in ev if e.get("ph") == "X" and "dur" in e and "ts" in e
            and ("Gpu" in str(e.get("args", {}).get("device", "")) or
                 e.get("pid", 0) and "device" in str(e.get("args", {})))]

# group by (pid,tid) stream, pick the busiest stream (main compute stream)
by_stream = collections.defaultdict(list)
for e in kern:
    by_stream[(e["pid"], e["tid"])].append((e["ts"], e["dur"], e.get("name", "")))

# merge ALL gpu kernel intervals across streams into a device-busy timeline
# (device is idle only when NO stream has a kernel running).
iv = sorted((ts, ts + dur) for s in by_stream.values() for (ts, dur, _) in s)
if not iv:
    print("no GPU kernel events found"); sys.exit(0)
span0, span1 = iv[0][0], max(b for _, b in iv)
# union of intervals
busy = 0.0; cur_s, cur_e = iv[0]
gaps = []
for s, e in iv[1:]:
    if s > cur_e:
        gaps.append((cur_e, s - cur_e)); busy += cur_e - cur_s; cur_s, cur_e = s, e
    else:
        cur_e = max(cur_e, e)
busy += cur_e - cur_s
wall = span1 - span0
idle = wall - busy
gaps.sort(key=lambda x: -x[1])

# all-reduce total on-GPU time (sum of cross_device_reduce durations)
ar = sum(dur for s in by_stream.values() for (ts, dur, nm) in s
         if "cross_device_reduce" in nm)

print(f"file: {f.split('/')[-1]}")
print(f"  wall span        : {wall/1000:.1f} ms")
print(f"  GPU busy (union) : {busy/1000:.1f} ms  ({100*busy/wall:.1f}%)")
print(f"  GPU idle (gaps)  : {idle/1000:.1f} ms  ({100*idle/wall:.1f}%)")
print(f"  all-reduce on-GPU: {ar/1000:.1f} ms  ({100*ar/wall:.1f}% of wall)")
print(f"  # gaps > 0.2 ms  : {sum(1 for _,g in gaps if g>200)}")
print(f"  top 5 gaps (ms)  : {', '.join(f'{g/1000:.2f}' for _,g in gaps[:5])}")
