#!/usr/bin/env python3
"""Emit the SAME CSV columns as the non-DSpark baseline table, for a bench ROOT.
Usage: python3 _extract_full_cols.py <ROOT> <route-label>"""
import json, glob, os, re, sys

root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/k3_dspark_longctx_bench_FULLFIX"
label = sys.argv[2] if len(sys.argv) > 2 else "dspark"

def stat(j, key, s):
    x = j.get(key, {})
    return x.get(s) if isinstance(x, dict) else None

rows = []
for d in glob.glob(os.path.join(root, "concurrency_*__requests_*")):
    m = re.search(r"concurrency_(\d+)__requests_(\d+)", d)
    if not m:
        continue
    conc, reqs = int(m.group(1)), int(m.group(2))
    f = os.path.join(d, "phases", "profiling", "profile_export_aiperf.json")
    if not os.path.exists(f):
        continue
    j = json.load(open(f))
    ttft = (stat(j, "time_to_first_token", "p50"), stat(j, "time_to_first_token", "p90"),
            stat(j, "time_to_first_token", "avg"))
    itl = (stat(j, "inter_token_latency", "p50"), stat(j, "inter_token_latency", "p90"),
           stat(j, "inter_token_latency", "avg"))
    ots = stat(j, "output_token_throughput", "avg")
    tts = stat(j, "total_token_throughput", "avg")
    cache = j.get("overall_usage_prompt_cache_read_pct")
    if isinstance(cache, dict):
        cache = cache.get("avg")
    rows.append((conc, reqs, ttft, itl, ots, tts, cache))

rows.sort()
print("route,concurrency,requests,ttft_p50_ms,ttft_p90_ms,ttft_mean_ms,"
      "itl_p50_ms,itl_p90_ms,itl_mean_ms,output_tok_s,output_tok_s_gpu,total_tok_s,cache_read_pct")
for conc, reqs, ttft, itl, ots, tts, cache in rows:
    def f(x): return f"{x:.2f}" if isinstance(x, (int, float)) else "-"
    og = ots / 8 if isinstance(ots, (int, float)) else None
    print(f"{label},{conc},{reqs},{f(ttft[0])},{f(ttft[1])},{f(ttft[2])},"
          f"{f(itl[0])},{f(itl[1])},{f(itl[2])},{f(ots)},{f(og)},{f(tts)},{f(cache)}")
