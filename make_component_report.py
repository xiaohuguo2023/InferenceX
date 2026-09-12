"""Generate the per-component comparison section of
docs/k3_vs_b300_conc1_isl100k.md from two rank0 decode traces.

Rules and per-step extraction come from trace_components.py (tested by
test_trace_compare_k3.py) -- they are NOT duplicated here.

Usage:  python3 make_component_report.py [mi355x_trace] [b300_trace]
"""

import glob
import json
import os
import sys

from trace_components import COMPONENT_RULES, component_of, decode_kernels_per_step

DOC = "docs/k3_vs_b300_conc1_isl100k.md"
DEFAULTS = {
    "MI355X": os.path.expanduser("~/work/k3_traces/prof_isl100k/*rank0*.json*"),
    "B300": os.path.expanduser("~/work/b300/profile_20260911_102437/traces/*rank0*"),
}

RAW, STEPS = {}, {}
for i, (tag, pat) in enumerate(DEFAULTS.items()):
    path = sys.argv[i + 1] if len(sys.argv) > i + 1 else (sorted(glob.glob(pat)) or [None])[0]
    if not path:
        raise SystemExit(f"no trace for {tag} (looked for {pat})")
    per_step, n = decode_kernels_per_step(path)
    RAW[tag] = [(k, v[0], v[1]) for k, v in per_step.items()]
    STEPS[tag] = n
    print(f"{tag}: {path}  steps={n:.0f}  total={sum(v[0] for v in per_step.values()):,.0f} us/step")

comp = component_of

agg = {}
for tag, rows in RAW.items():
    for name, us, calls in rows:
        c = comp(name)
        agg.setdefault(c, {}).setdefault(tag, []).append((name, us, calls))

tot = {t: sum(r[1] for r in RAW[t]) for t in RAW}
GAP = tot["MI355X"] - tot["B300"]
def sums(c, t):
    v = agg.get(c, {}).get(t, [])
    return sum(x[1] for x in v), sum(x[2] for x in v)

order = sorted(agg, key=lambda c: -(sums(c,"MI355X")[0] - sums(c,"B300")[0]))
L = []
L.append("\n---\n")
L.append("## Component Totals and Per-Kernel Detail (KDA-clock normalised)\n")
L.append(f"> **Why the tables above this section are wrong.** They carry TWO bugs that this\n"
         f"> section does not.\n"
         f">\n"
         f"> 1. **Normalisation.** They divide by the profiler's self-reported iteration count\n"
         f">    (66 for MI355X, **492** for B300). That is right for MI355X but wrong for B300\n"
         f">    by 2.97x -- B300 emits 3 `execute_context` annotations per real decode step.\n"
         f">    This section instead uses the KDA clock: `fused_recurrent_kda` fires exactly\n"
         f">    once per KDA layer per step, so step count = its call count / 69. That gives\n"
         f">    **MI355X {STEPS['MI355X']:.0f} steps, B300 {STEPS['B300']:.0f} steps**. Only the decode\n"
         f">    `fused_recurrent_kda` may be counted -- the `chunk_*` KDA kernels are the prefill\n"
         f">    path and inflate the divisor.\n"
         f"> 2. **Prefill leak (fixed at source 2026-09-12).** `_assign_stage_from_index` charged\n"
         f">    a kernel landing in a GAP between step annotations to the NEXT step. GPU work is\n"
         f">    async, so such a kernel is the PREVIOUS step's tail; at the PREFILL->DECODE\n"
         f">    boundary the old rule donated prefill's tail to decode, inflating the MI355X\n"
         f">    decode total by 14% and dragging 207 prefill-only `chunk_*` calls in. B300 has no\n"
         f">    PREFILL steps, so the bias ran one way only.\n"
         f">\n"
         f"> Net effect: the old tables report the total ratio as 0.21x (\"we are 4.7x slower\").\n"
         f"> **The correct ratio is {tot['B300']/tot['MI355X']:.2f}x -- we are "
         f"{tot['MI355X']/tot['B300']:.2f}x slower.** Regenerate them with the fixed\n"
         f"> `trace_compare_k3.py` to make them agree.\n")
L.append(f"DECODE, rank0, ISL-matched (MI355X 99,845 / B300 ~99,757). GPU kernels only.\n")
L.append(f"- **MI355X total {tot['MI355X']:,.0f} us/step** across {sum(r[2] for r in RAW['MI355X']):,.0f} calls, {len(RAW['MI355X'])} distinct kernels")
L.append(f"- **B300 total {tot['B300']:,.0f} us/step** across {sum(r[2] for r in RAW['B300']):,.0f} calls, {len(RAW['B300'])} distinct kernels")
L.append(f"- **Gap {GAP:,.0f} us/step ({tot['MI355X']/tot['B300']:.2f}x)**\n")

L.append("| Component | MI355X us/step | calls | B300 us/step | calls | Delta | Ratio | % MI355X step | % B300 step | % of gap |")
L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for c in order:
    mu, mc = sums(c, "MI355X"); bu, bc = sums(c, "B300")
    r = f"{mu/bu:.2f}x" if bu else "n/a"
    L.append(f"| **{c}** | {mu:,.0f} | {mc:,.0f} | {bu:,.0f} | {bc:,.0f} | {mu-bu:+,.0f} | {r} | "
             f"{100*mu/tot['MI355X']:.1f}% | {100*bu/tot['B300']:.1f}% | {100*(mu-bu)/GAP:.0f}% |")
L.append(f"| **TOTAL** | **{tot['MI355X']:,.0f}** | {sum(r[2] for r in RAW['MI355X']):,.0f} | "
         f"**{tot['B300']:,.0f}** | {sum(r[2] for r in RAW['B300']):,.0f} | **{GAP:+,.0f}** | "
         f"**{tot['MI355X']/tot['B300']:.2f}x** | 100% | 100% | 100% |")

for c in order:
    mu, mc = sums(c, "MI355X"); bu, bc = sums(c, "B300")
    r = f"{mu/bu:.2f}x" if bu else "n/a"
    L.append(f"\n### {c} — MI355X {mu:,.0f} us/step vs B300 {bu:,.0f} us/step ({r}, {mu-bu:+,.0f})\n")
    L.append("| Platform | Kernel function | us/step | calls/step | us/call | % of component |")
    L.append("|---|---|---:|---:|---:|---:|")
    for tag, t in (("MI355X", mu), ("B300", bu)):
        items = sorted(agg.get(c, {}).get(tag, []), key=lambda x: -x[1])
        if not items:
            L.append(f"| {tag} | *(no kernels in this component)* | 0 | 0 | - | - |")
            continue
        for name, us, calls in items:
            nm = name.replace("|", "\\|")
            nm = nm if len(nm) <= 110 else nm[:107] + "..."
            L.append(f"| {tag} | `{nm}` | {us:,.1f} | {calls:,.1f} | {us/calls:,.2f} | {100*us/t:.1f}% |")

L.append("\n### Provenance for this section\n")
L.append("| item | value |")
L.append("|---|---|")
for k, v in [
 ("MI355X trace", "`~/work/k3_traces/prof_isl100k/dp0_pp0_tp0_dcp0_ep0_rank0.*.pt.trace.json.gz`"),
 ("B300 trace", "`~/work/b300/profile_20260911_102437/traces/dp0_pp0_tp0_dcp0_ep0_rank0.*`"),
 ("extraction", "`/dev/shm/_mla_raw.py` — reuses `trace_compare_k3.py`'s `_parse_gpu_steps` / "
                "`_build_step_index` / `_assign_stage_from_index`, so the DECODE window matches the tables above"),
 ("this section", "`/dev/shm/_mk_component_section.py`"),
 ("config", "DCP8 + DSpark K=7 + DRAM offload, TP8/EP1, ROCM_AITER_MLA fp8 asm, conc-1"),
 ("caveat", "Profiler inflation differs by platform (~+13% ours, ~+27% B300 at this ISL), so absolute "
            "us are inflated on BOTH sides. Ratios and call counts are the trustworthy part."),
 ("date", "2026-09-12"),
 ("known inference", "B300's MLA BMM1/BMM2 absorb is attributed to `nvjet_*_NNT` + `nvjet_*_TNN` "
                    "(24 calls each = 24 MLA layers, bmm transpose layouts), matching our "
                    "`batched_gemm_a16wfp4` (23 calls x2). Inferred from call count and layout, NOT confirmed."),
]:
    L.append(f"| {k} | {v} |")

txt = open(DOC).read()
marker = "## Component Totals and Per-Kernel Detail"
if marker in txt:
    txt = txt[:txt.index("\n---\n\n" + marker)]
open(DOC, "w").write(txt.rstrip() + "\n" + "\n".join(L) + "\n")
print(f"appended {len(L)} lines; components: {len(order)}")
for c in order:
    mu, _ = sums(c,"MI355X"); bu, _ = sums(c,"B300")
    print(f"  {c:<22} {mu:8,.0f} vs {bu:8,.0f}  {mu-bu:+8,.0f}")
