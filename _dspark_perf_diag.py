#!/usr/bin/env python3
"""Latency/interactivity DIAGNOSTIC for the k3-longctx-bench sweep.

Purpose: quickly identify perf issues that hurt the agentic IX benchmark —
focused on ITL / TTFT / TPOT / per-user interactivity, NOT raw throughput.
It reads each concurrency point's *profiling* aiperf json (explicitly excluding
the warmup phase) plus the vLLM /metrics before/after snapshots, prints an
interactivity table + SLA gates, and then AUTO-FLAGS anomalies:

  - ITL non-monotonic across concurrency  -> decode bubble (e.g. conc-4/12
    PIECEWISE/eager-attention fallback: a LOWER conc slower than a HIGHER one)
  - ITL cliff between adjacent concs       -> a knee (super-linear decode cost)
  - ITL tail (p90/p50 spread)              -> stragglers / capture miss / eviction
  - TTFT cliff across concurrency          -> prefill/scheduling pressure
  - prefix-cache-hit collapse              -> KV eviction (the conc16->24 knee)
  - OSL != 350 / osl_mismatch              -> invalid point (shape didn't hold)
  - mean accepted length low               -> draft/verify degraded

  python3 _dspark_perf_diag.py /workspace/k3_dspark_longctx_bench [--tp 8]

Thresholds are deliberately conservative; tune via the CONSTANTS block. Exit
code is 0 always (diagnostic, not a gate) but the flag count is printed.
"""
import json
import os
import re
import sys

# ---- tunable thresholds -----------------------------------------------------
TP = 8
ITL_MONO_TOL = 1.05      # lower-conc ITL may exceed higher-conc ITL by at most 5%
ITL_CLIFF = 1.60         # adjacent-conc ITL_p50 jump ratio that counts as a knee
ITL_TAIL = 1.30          # p90/p50 spread that counts as a tail problem
TTFT_CLIFF = 2.00        # adjacent-conc TTFT_p50 jump ratio that counts as a cliff
CACHE_DROP_PTS = 5.0     # abs %-point prefix-cache-hit drop as conc rises
AL_MIN = 2.0             # mean accepted length floor for spec decode
CLAW_ITL = 25.0          # SLA gate (interactive "claw"): ITL p50 < 25 ms
CHAT_ITL = 66.7          # SLA gate ("chat"): ITL p50 < 66.7 ms
# -----------------------------------------------------------------------------

args = list(sys.argv[1:])
if "--tp" in args:
    i = args.index("--tp"); TP = int(args[i + 1]); del args[i:i + 2]
ROOT = args[0] if args else "/workspace/k3_dspark_longctx_bench"

PROM = re.compile(r'^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([-+0-9.eEnaN]+)\s*$')


def parse_prom(path):
    scalar = {}
    if not os.path.exists(path):
        return scalar
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = PROM.match(line)
        if not m:
            continue
        name, val = m.group(1), m.group(3)
        if "per_pos" in name:
            continue
        try:
            scalar[name] = scalar.get(name, 0.0) + float(val)
        except ValueError:
            pass
    return scalar


def find_profiling_json(pt_dir):
    """Prefer the profiling-phase json (rc=full); never the warmup one."""
    cand = os.path.join(pt_dir, "phases", "profiling", "profile_export_aiperf.json")
    if os.path.exists(cand):
        return cand
    top = os.path.join(pt_dir, "profile_export_aiperf.json")
    if os.path.exists(top):
        return top
    # last resort: any json that isn't under phases/warmup
    for root, _d, files in os.walk(pt_dir):
        if "warmup" in root:
            continue
        if "profile_export_aiperf.json" in files:
            return os.path.join(root, "profile_export_aiperf.json")
    return None


def mstat(aj, name, stat):
    """Fetch metric `name` stat (avg/p50/p90) from an aiperf json, schema-tolerant."""
    def dig(o):
        if isinstance(o, dict):
            if name in o and isinstance(o[name], dict) and any(
                    k in o[name] for k in ("avg", "p50", "p90", "unit")):
                return o[name].get(stat)
            for v in o.values():
                r = dig(v)
                if r is not None:
                    return r
        elif isinstance(o, list):
            for v in o:
                r = dig(v)
                if r is not None:
                    return r
        return None
    return dig(aj)


def conc_of(name):
    m = re.search(r"concurrency_(\d+)", name)
    return int(m.group(1)) if m else 1 << 30


rows = []
for d in sorted(os.listdir(ROOT)):
    pt = os.path.join(ROOT, d)
    if not os.path.isdir(pt) or not d.startswith("concurrency_"):
        continue
    jp = find_profiling_json(pt)
    aj = json.load(open(jp)) if jp else None
    b = parse_prom(os.path.join(pt, "metrics_before.txt"))
    a = parse_prom(os.path.join(pt, "metrics_after.txt"))

    def ds(*c):
        for k in c:
            if k in a or k in b:
                return a.get(k, 0.0) - b.get(k, 0.0)
        return 0.0

    d_acc = ds("vllm:spec_decode_num_accepted_tokens_total")
    d_drafts = ds("vllm:spec_decode_num_drafts_total")
    d_hits = ds("vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total")
    d_qry = ds("vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total")
    mean_al = (d_acc / d_drafts + 1.0) if d_drafts > 0 else float("nan")
    cache_hit = 100.0 * d_hits / d_qry if d_qry > 0 else float("nan")

    g = (lambda n, s: mstat(aj, n, s) if aj else None)
    rows.append({
        "conc": conc_of(d),
        "ttft_avg": g("time_to_first_token", "avg"),
        "ttft_p50": g("time_to_first_token", "p50"),
        "ttft_p90": g("time_to_first_token", "p90"),
        "itl_avg": g("inter_token_latency", "avg"),
        "itl_p50": g("inter_token_latency", "p50"),
        "itl_p90": g("inter_token_latency", "p90"),
        "peruser": g("output_token_throughput_per_user", "avg"),   # tok/s/user = interactivity
        "out_tps": g("output_token_throughput", "avg"),            # aggregate tok/s
        "isl": g("input_sequence_length", "avg"),
        "osl": g("output_sequence_length", "avg"),
        "osl_mm": g("osl_mismatch_diff_pct", "avg"),
        "cache_hit": cache_hit,
        "mean_al": mean_al,
    })

rows.sort(key=lambda r: r["conc"])


def f(x, nd=1):
    if x is None or (isinstance(x, float) and x != x):
        return "-"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def pg(x):
    return None if x is None else float(x) / TP


print(f"\n## Interactivity / latency (k3-longctx-bench)  ROOT={ROOT}  TP={TP}\n")
print("TPOT == ITL for steady-state decode. per-user tok/s = interactivity.\n")
print("| Conc | TTFT p50 | TTFT p90 | TTFT avg | ITL/TPOT p50 | ITL p90 | ITL avg | per-user tok/s | out tok/s/GPU | Cache% | AL | OSL |")
print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in rows:
    print(f"| {r['conc']} | {f(r['ttft_p50'])} | {f(r['ttft_p90'])} | {f(r['ttft_avg'])} | "
          f"{f(r['itl_p50'],2)} | {f(r['itl_p90'],2)} | {f(r['itl_avg'],2)} | "
          f"{f(r['peruser'],1)} | {f(pg(r['out_tps']),1)} | {f(r['cache_hit'],1)} | "
          f"{f(r['mean_al'],2)} | {f(r['osl'],0)} |")

# ---- SLA gates --------------------------------------------------------------
def max_conc_under(thr):
    ok = [r for r in rows if r["itl_p50"] is not None and r["itl_p50"] < thr]
    return max((r["conc"] for r in ok), default=None)

claw, chat = max_conc_under(CLAW_ITL), max_conc_under(CHAT_ITL)
print(f"\n## SLA gates (ITL p50)\n")
print(f"- claw  (< {CLAW_ITL} ms): max concurrency passing = **{claw if claw else 'none'}**")
print(f"- chat  (< {CHAT_ITL} ms): max concurrency passing = **{chat if chat else 'none'}**")

# ---- auto perf-issue flags --------------------------------------------------
flags = []
have = [r for r in rows if r["itl_p50"] is not None]

# 1. ITL non-monotonic across concurrency -> decode bubble
for i, r in enumerate(have):
    for hr in have[i + 1:]:                       # any strictly-higher conc
        if r["itl_p50"] > ITL_MONO_TOL * hr["itl_p50"]:
            flags.append(
                f"BUBBLE: ITL non-monotonic — conc-{r['conc']} ({f(r['itl_p50'],1)} ms) is SLOWER "
                f"than higher conc-{hr['conc']} ({f(hr['itl_p50'],1)} ms). Classic PIECEWISE/eager-"
                f"attention fallback (get_mla_metadata_v1 bubble) — check FULL-decode capture at "
                f"M=(1+N)*{r['conc']}.")
            break

# 2. ITL cliff between adjacent concs -> knee
for lo, hi in zip(have, have[1:]):
    if lo["itl_p50"] and hi["itl_p50"] / lo["itl_p50"] > ITL_CLIFF:
        flags.append(
            f"ITL KNEE: conc-{lo['conc']}->{hi['conc']} ITL p50 jumps "
            f"{f(lo['itl_p50'],1)}->{f(hi['itl_p50'],1)} ms ({f(hi['itl_p50']/lo['itl_p50'],2)}x). "
            f"Super-linear decode cost — profile this transition (profile-decode-bubble skill).")

# 3. ITL tail (p90/p50)
for r in have:
    if r["itl_p90"] and r["itl_p50"] and r["itl_p90"] / r["itl_p50"] > ITL_TAIL:
        flags.append(
            f"ITL TAIL: conc-{r['conc']} p90/p50 = {f(r['itl_p90']/r['itl_p50'],2)} "
            f"({f(r['itl_p50'],1)}/{f(r['itl_p90'],1)} ms). Stragglers / capture miss / eviction jitter.")

# 4. TTFT cliff across concurrency
tt = [r for r in rows if r["ttft_p50"] is not None]
for lo, hi in zip(tt, tt[1:]):
    if lo["ttft_p50"] and hi["ttft_p50"] / lo["ttft_p50"] > TTFT_CLIFF:
        flags.append(
            f"TTFT CLIFF: conc-{lo['conc']}->{hi['conc']} TTFT p50 jumps "
            f"{f(lo['ttft_p50'],0)}->{f(hi['ttft_p50'],0)} ms ({f(hi['ttft_p50']/lo['ttft_p50'],2)}x). "
            f"Prefill/scheduling pressure or prefix recompute.")

# 5. prefix-cache-hit collapse as conc rises
ch = [r for r in rows if r["cache_hit"] == r["cache_hit"]]  # not-NaN
for lo, hi in zip(ch, ch[1:]):
    if lo["cache_hit"] - hi["cache_hit"] > CACHE_DROP_PTS:
        flags.append(
            f"CACHE EVICTION: prefix-hit drops {f(lo['cache_hit'],1)}%->{f(hi['cache_hit'],1)}% "
            f"at conc-{lo['conc']}->{hi['conc']}. KV pressure evicting the shared prefix -> prefill "
            f"recompute -> TTFT/ITL knee. Lever: KV offload / prefix retention / pool sizing.")

# 6. OSL validity
for r in rows:
    if r["osl"] is not None and abs(r["osl"] - 350) > 1:
        flags.append(f"INVALID: conc-{r['conc']} OSL={f(r['osl'],1)} != 350 (shape didn't hold).")
    if r["osl_mm"] not in (None, 0, 0.0) and (r["osl_mm"] or 0) > 0.1:
        flags.append(f"INVALID: conc-{r['conc']} osl_mismatch={f(r['osl_mm'],2)}%.")

# 7. accepted length floor
for r in rows:
    if r["mean_al"] == r["mean_al"] and r["mean_al"] < AL_MIN:
        flags.append(f"LOW ACCEPT: conc-{r['conc']} mean AL={f(r['mean_al'],2)} < {AL_MIN} "
                     f"— draft/verify degraded (or real acceptance for this workload).")

print(f"\n## Perf-issue flags ({len(flags)})\n")
if not flags:
    print("- none — ITL monotonic, no knees/cliffs, cache stable, points valid. Clean latency curve.")
else:
    for fl in flags:
        print(f"- ⚠ {fl}")

json.dump({"tp": TP, "rows": rows, "flags": flags,
           "sla": {"claw": claw, "chat": chat}},
          open(os.path.join(ROOT, "perf_diag.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(ROOT, 'perf_diag.json')}")
