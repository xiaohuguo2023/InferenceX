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
  - ITL tail                               -> stragglers / capture miss / eviction
    NOTE: aiperf's inter_token_latency is a PER-REQUEST AVERAGE, so a point has
    only `request_count` samples (40 at conc-8), and the sweep is NOT steady
    state: requests arrive in waves of `conc` and the survivors drain at falling
    concurrency. That ramp alone yields p90/p50 ~1.2-1.4 on every healthy point,
    so the old fixed p90/p50 > 1.30 test false-positived constantly at low conc.
    Two changes: (a) the statistic is TAIL LIFT = mean(slowest decile)/p50, which
    unlike p90 does not sit on the boundary of a small straggler cluster; (b) it
    fires only when a bootstrap 95% CI LOWER BOUND clears the threshold. Measured
    separation: ramp-only points score 1.12-1.43, a real 15%-of-requests-at-2x
    straggler cluster scores 2.08. Known blind spot: with n=40 the slowest decile
    is 4 requests, so a cluster of <=2 stragglers is below resolution — raise the
    request count at that point if you need to resolve it.
  - TTFT cliff across concurrency          -> prefill/scheduling pressure
  - prefix-cache-hit collapse              -> KV eviction (the conc16->24 knee)
  - OSL != 350 / osl_mismatch              -> invalid point (shape didn't hold)
  - mean accepted length low               -> draft/verify degraded

  python3 _dspark_perf_diag.py /workspace/k3_dspark_longctx_bench [--tp 8]

Thresholds are deliberately conservative; tune via the CONSTANTS block. Exit
code is 0 always (diagnostic, not a gate) but the flag count is printed.
"""
import json
import math
import os
import random
import re
import sys

# ---- tunable thresholds -----------------------------------------------------
TP = 8
ITL_MONO_TOL = 1.05      # lower-conc ITL may exceed higher-conc ITL by at most 5%
ITL_CLIFF = 1.60         # adjacent-conc ITL_p50 jump ratio that counts as a knee
ITL_TAIL_LIFT = 1.50     # mean(slowest decile)/p50 that counts as a real tail.
                         # Ramp-only points measure 1.12-1.43; a 15%-at-2x
                         # straggler cluster measures ~2.08. See docstring NOTE.
ITL_TAIL_DECILE = 0.10   # fraction of slowest requests forming the "tail"
ITL_TAIL_BOOT = 4000     # bootstrap resamples for the tail-lift CI
ITL_TAIL_CI = 95.0       # two-sided CI; flag only if the LOWER bound > threshold
ITL_TAIL = 1.30          # FALLBACK p90/p50 test, used only when the per-request
ITL_TAIL_MIN_N = 80      # samples are missing AND n >= ITL_TAIL_MIN_N
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


def itl_samples(pt, n_profiling):
    """Per-request ITL values for the PROFILING phase only.

    profile_export.jsonl holds warmup + profiling records with no phase tag, but
    the profiling aiperf json reports how many there are, and warmup always runs
    first — so the profiling set is the last `n_profiling` by request_start_ns.
    """
    p = os.path.join(pt, "profile_export.jsonl")
    if not os.path.exists(p) or not n_profiling:
        return []
    recs = []
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            v = r["metrics"]["inter_token_latency"]
            recs.append((r["metadata"]["request_start_ns"],
                         v["value"] if isinstance(v, dict) else v))
        except (ValueError, KeyError, TypeError):
            continue
    recs.sort()
    return [v for _, v in recs][-int(n_profiling):]


def pctl(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def tail_lift(xs):
    """mean(slowest decile) / p50 -- see the ITL-tail NOTE in the docstring."""
    s = sorted(xs)
    k = max(1, round(len(s) * ITL_TAIL_DECILE))
    return (sum(s[-k:]) / k) / pctl(s, 50)


def tail_lift_ci(xs, conf=ITL_TAIL_CI, n_boot=ITL_TAIL_BOOT, seed=0):
    """Bootstrap CI for tail_lift on a small sample. Returns (obs, lo, hi)."""
    if len(xs) < 8:
        return None
    rnd = random.Random(seed)
    n = len(xs)
    boots = sorted(tail_lift([xs[rnd.randrange(n)] for _ in range(n)])
                   for _ in range(n_boot))
    a = (100.0 - conf) / 2.0
    return tail_lift(xs), pctl(boots, a), pctl(boots, 100.0 - a)


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
        "itl_n": g("inter_token_latency", "count"),
        "itl_samples": itl_samples(pt, g("inter_token_latency", "count")),
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

# 3. ITL tail (p90/p50) -- sample-size aware, see the NOTE in the module docstring.
#    aiperf ITL is a per-request average, so n == request_count (40 at conc-8) and
#    the wave-drain ramp alone yields ~1.2-1.3. Fire only when the tail survives a
#    bootstrap, or when n is large enough for the raw ratio to mean something.
for r in have:
    if not (r["itl_p90"] and r["itl_p50"]):
        continue
    n = int(r["itl_n"] or 0)
    ci = tail_lift_ci(r["itl_samples"])
    if ci is not None:
        obs, lo, hi = ci
        if lo <= ITL_TAIL_LIFT:
            continue                  # tail not resolvable from this many samples
        what = (f"tail lift = {f(obs,2)}x (95% CI [{f(lo,2)}, {f(hi,2)}], n={n}); "
                f"slowest decile vs p50 {f(r['itl_p50'],1)} ms")
    else:
        # No per-request samples on disk -> fall back to the raw p90/p50 test,
        # but only where n is large enough for it to mean anything.
        if n < ITL_TAIL_MIN_N or r["itl_p90"] / r["itl_p50"] <= ITL_TAIL:
            continue
        what = (f"p90/p50 = {f(r['itl_p90']/r['itl_p50'],2)} "
                f"({f(r['itl_p50'],1)}/{f(r['itl_p90'],1)} ms, n={n}, no raw samples)")
    flags.append(f"ITL TAIL: conc-{r['conc']} {what}. "
                 f"Stragglers / capture miss / eviction jitter.")

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
