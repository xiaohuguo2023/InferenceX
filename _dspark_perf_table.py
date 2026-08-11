#!/usr/bin/env python3
"""Build a DSpark perf + acceptance table from _dspark_longctx_bench.sh artifacts.

For each concurrency point dir under ROOT it reads:
  - profile_export_aiperf.json  (aiperf per-point latency/throughput summary)
  - metrics_before.txt / metrics_after.txt  (vLLM /metrics prometheus dumps)

and emits a markdown table (stdout) + perf_table.json. DSpark acceptance
(overall + per draft position) and prefix-cache hit rate are differenced from
the before/after counter snapshots. Per-GPU throughput = aggregate / TP.

  python3 _dspark_perf_table.py /workspace/k3_dspark_longctx_bench [--tp 8]
"""
import json
import os
import re
import sys

TP = 8
args = [a for a in sys.argv[1:]]
if "--tp" in args:
    i = args.index("--tp")
    TP = int(args[i + 1])
    del args[i : i + 2]
ROOT = args[0] if args else "/workspace/k3_dspark_longctx_bench"

PROM = re.compile(r'^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([-+0-9.eEnaN]+)\s*$')
POS = re.compile(r'position="?(\d+)"?')


def parse_prom(path):
    """Return {metric: float_sum} and {metric: {pos: float}} for per-pos counters."""
    scalar, perpos = {}, {}
    if not os.path.exists(path):
        return scalar, perpos
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = PROM.match(line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", m.group(3)
        try:
            v = float(val)
        except ValueError:
            continue
        if "per_pos" in name:
            pm = POS.search(labels)
            if pm:
                perpos.setdefault(name, {})
                perpos[name][int(pm.group(1))] = perpos[name].get(int(pm.group(1)), 0.0) + v
        else:
            scalar[name] = scalar.get(name, 0.0) + v
    return scalar, perpos


def first_key(d, *cands):
    for c in cands:
        if c in d:
            return d[c]
    return None


def metric_stat(aj, name, stat):
    """Pull a stat (avg/p50/p90) for a metric from an aiperf json, schema-tolerant."""
    def dig(obj):
        if isinstance(obj, dict):
            if name in obj and isinstance(obj[name], dict):
                mo = obj[name]
                for k in (stat, stat.replace("p", "p"), "value"):
                    if k in mo:
                        return mo[k]
            for v in obj.values():
                r = dig(v)
                if r is not None:
                    return r
        elif isinstance(obj, list):
            for v in obj:
                r = dig(v)
                if r is not None:
                    return r
        return None
    return dig(aj)


def load_aiperf(pt_dir):
    for root, _dirs, files in os.walk(pt_dir):
        for f in files:
            if f == "profile_export_aiperf.json":
                try:
                    return json.load(open(os.path.join(root, f)))
                except Exception as e:
                    print(f"!! {os.path.join(root,f)}: {e}", file=sys.stderr)
    return None


def conc_of(name):
    m = re.search(r"concurrency_(\d+)", name)
    return int(m.group(1)) if m else 1 << 30


rows = []
for d in sorted(os.listdir(ROOT)):
    pt = os.path.join(ROOT, d)
    if not os.path.isdir(pt) or not d.startswith("concurrency_"):
        continue
    conc = conc_of(d)
    aj = load_aiperf(pt)
    b, bpp = parse_prom(os.path.join(pt, "metrics_before.txt"))
    a, app = parse_prom(os.path.join(pt, "metrics_after.txt"))

    def dscalar(*cands):
        for c in cands:
            if c in a or c in b:
                return a.get(c, 0.0) - b.get(c, 0.0)
        return 0.0

    d_draft = dscalar("vllm:spec_decode_num_draft_tokens_total")
    d_acc = dscalar("vllm:spec_decode_num_accepted_tokens_total")
    d_drafts = dscalar("vllm:spec_decode_num_drafts_total")
    d_hits = dscalar("vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total")
    d_qry = dscalar("vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total")

    accept_pct = 100.0 * d_acc / d_draft if d_draft > 0 else float("nan")
    mean_al = (d_acc / d_drafts + 1.0) if d_drafts > 0 else float("nan")
    cache_hit = 100.0 * d_hits / d_qry if d_qry > 0 else float("nan")

    ppname = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
    perpos = {}
    if ppname in app or ppname in bpp:
        allpos = set(app.get(ppname, {})) | set(bpp.get(ppname, {}))
        for p in sorted(allpos):
            dv = app.get(ppname, {}).get(p, 0.0) - bpp.get(ppname, {}).get(p, 0.0)
            perpos[p] = 100.0 * dv / d_drafts if d_drafts > 0 else float("nan")

    row = {
        "concurrency": conc,
        "req_throughput": metric_stat(aj, "request_throughput", "avg") if aj else None,
        "req_latency_avg": metric_stat(aj, "request_latency", "avg") if aj else None,
        "ttft_p50": metric_stat(aj, "time_to_first_token", "p50") if aj else None,
        "ttft_p90": metric_stat(aj, "time_to_first_token", "p90") if aj else None,
        "itl_p50": metric_stat(aj, "inter_token_latency", "p50") if aj else None,
        "itl_p90": metric_stat(aj, "inter_token_latency", "p90") if aj else None,
        "out_tok_s": metric_stat(aj, "output_token_throughput", "avg") if aj else None,
        "isl": metric_stat(aj, "input_sequence_length", "avg") if aj else None,
        "osl": metric_stat(aj, "output_sequence_length", "avg") if aj else None,
        "accept_pct": accept_pct,
        "mean_accept_len": mean_al,
        "cache_hit_pct": cache_hit,
        "per_pos_pct": perpos,
        "d_draft": d_draft,
        "d_accepted": d_acc,
        "d_drafts": d_drafts,
    }
    rows.append(row)

rows.sort(key=lambda r: r["concurrency"])


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and x != x):
        return "-"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def per_gpu(x):
    if x is None or (isinstance(x, float) and x != x):
        return None
    return float(x) / TP


print(f"\n## Perf (per-GPU throughput = aggregate / TP={TP})\n")
print("| Conc | Req/s | ISL | OSL | TTFT P50 (ms) | TTFT P90 (ms) | ITL P50 (ms) | ITL P90 (ms) | Out tok/s/GPU | Cache Hit (%) |")
print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in rows:
    print(
        f"| {r['concurrency']} | {fmt(r['req_throughput'],3)} | {fmt(r['isl'],0)} | {fmt(r['osl'],1)} | "
        f"{fmt(r['ttft_p50'],1)} | {fmt(r['ttft_p90'],1)} | {fmt(r['itl_p50'],2)} | {fmt(r['itl_p90'],2)} | "
        f"{fmt(per_gpu(r['out_tok_s']),1)} | {fmt(r['cache_hit_pct'],2)} |"
    )

print(f"\n## DSpark acceptance (num_spec draft tokens)\n")
print("| Conc | Draft accept (%) | Mean accepted len (tok/step) | Δdrafts | Δaccepted |")
print("|---:|---:|---:|---:|---:|")
for r in rows:
    print(
        f"| {r['concurrency']} | {fmt(r['accept_pct'],2)} | {fmt(r['mean_accept_len'],3)} | "
        f"{fmt(r['d_drafts'],0)} | {fmt(r['d_accepted'],0)} |"
    )

allpos = sorted({p for r in rows for p in r["per_pos_pct"]})
if allpos:
    print(f"\n## Acceptance by draft position (%)\n")
    print("| Conc | " + " | ".join(f"Pos {p+1}" for p in allpos) + " |")
    print("|---:|" + "---:|" * len(allpos))
    for r in rows:
        cells = " | ".join(fmt(r["per_pos_pct"].get(p), 1) for p in allpos)
        print(f"| {r['concurrency']} | {cells} |")

outp = os.path.join(ROOT, "perf_table.json")
json.dump({"tp": TP, "rows": rows}, open(outp, "w"), indent=2)
print(f"\nwrote {outp}")
