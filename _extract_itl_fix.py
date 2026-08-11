#!/usr/bin/env python3
"""Extract ITL/TTFT/throughput per concurrency from a _dspark_longctx_bench.sh ROOT.
Usage: python3 _extract_itl_fix.py <ROOT>"""
import json, glob, os, sys, re

root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/k3_dspark_longctx_bench_FIX"
rows = []
for d in sorted(glob.glob(os.path.join(root, "concurrency_*__requests_*"))):
    m = re.search(r"concurrency_(\d+)__requests_(\d+)", d)
    if not m:
        continue
    conc = int(m.group(1))
    f = os.path.join(d, "phases", "profiling", "profile_export_aiperf.json")
    if not os.path.exists(f):
        continue
    j = json.load(open(f))
    itl = j.get("inter_token_latency", {})
    ttft = j.get("time_to_first_token", {})
    rt = j.get("request_throughput", {})
    rows.append((conc, itl.get("p50"), itl.get("avg"), itl.get("min"), itl.get("max"),
                 ttft.get("p50"), rt.get("avg")))

rows.sort(key=lambda r: r[0])
print(f"{'conc':>5} | {'ITL p50':>8} | {'ITL avg':>8} | {'ITL min':>8} | {'ITL max':>8} | {'TTFT p50':>9} | {'req/s':>6}")
print("-" * 70)
for c, p50, avg, mn, mx, ttft, rt in rows:
    def f(x): return f"{x:8.2f}" if isinstance(x, (int, float)) else f"{'-':>8}"
    print(f"{c:>5} | {f(p50)} | {f(avg)} | {f(mn)} | {f(mx)} | "
          f"{ttft:9.1f}" if isinstance(ttft,(int,float)) else f"{c:>5}",
          f"| {rt:6.3f}" if isinstance(rt,(int,float)) else "")
