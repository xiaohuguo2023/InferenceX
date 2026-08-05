#!/usr/bin/env python3
"""Dashboard-style Pareto for Kimi-K3 agentic: MI355X (ours) vs B300 vs B200.
x = Interactivity (1000/mean_TPOT, tok/s/user); y = Token Throughput per GPU.
MI355X points come from the local AIPerf conc sweep (k3_sweep_c*/); B300/B200
come from the InferenceX dashboard (/tmp/k3_b300.json). Mirrors
~/work/sweep_gptoss_output/v023_final/plot_ours_pareto.py."""
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/xiaohugu/work/sweep_gptoss_output")
from plot_pareto_3way import plot_combined, _pareto_frontier

ROOT = Path("/home/xiaohugu/work/InferenceX-dspv4")
OUT = ROOT / "docs" / "kimik3_pareto"; OUT.mkdir(parents=True, exist_ok=True)
SHAPE = (0, 0, "K3 agentic (cc-traces)")   # single agentic "shape"
NGPU = 8

def g(v):  # AIPerf metric dict -> avg
    return v.get("avg") if isinstance(v, dict) else v

# ---- MI355X (ours) from the local sweeps -------------------------------------
def load_mi(dirfmt, label):
    s = {}
    for c in (1, 2, 4, 8, 16, 24):
        f = ROOT / dirfmt.format(c=c) / "aiperf_artifacts" / "profile_export_aiperf.json"
        if not f.exists():
            print(f"  [skip] {label} conc{c}: no result ({f})"); continue
        d = json.load(open(f))
        inp = g(d.get("input_token_throughput", {})) or 0.0
        out = g(d.get("output_token_throughput", {})) or 0.0
        tpot = g(d.get("inter_token_latency", {}))
        if not tpot:
            print(f"  [skip] {label} conc{c}: no TPOT"); continue
        total = inp + out
        s[(0, 0, NGPU, c)] = {"total": total, "tpot": tpot}
        print(f"  {label} conc{c:>2}: tput/gpu={total/NGPU:6.0f}  TPOT={tpot:5.1f}ms  interact={1000/tpot:5.1f}")
    return s

mi_fp8_fused = load_mi("k3_fp8asm_fused_v101_ixci_c{c}", "MI355X fp8 ASM +fused (v1.0.1)")
mi_fp8asm  = load_mi("k3_fp8asm_sweep_c{c}",    "MI355X fp8 ASM (native, 900s)")
mi_bf16asm = load_mi("k3_bf16asm2_sweep_c{c}",  "MI355X bf16 ASM (native, 900s)")

# ---- B300 / B200 from the InferenceX dashboard -------------------------------
# spec=None keeps every spec_method (back-compat); pass "none"/"mtp" to split
# B300 into its non-MTP (feature-matched to our non-MTP MI355X) and MTP arms.
def nv_series(hw, spec=None):
    s = {}
    try:
        rows = json.load(open("/tmp/k3_b300.json"))
    except FileNotFoundError:
        print("  [warn] /tmp/k3_b300.json missing — re-fetch dashboard"); return s
    for r in rows:
        if r.get("hardware") != hw:
            continue
        if spec is not None and r.get("spec_method") != spec:
            continue
        m = r["metrics"]; tpot = m.get("mean_tpot", 0) * 1000
        if tpot <= 0:
            continue
        off = m.get("offload_mode", r.get("offload_mode", ""))
        ckey = r["conc"] + (100000 if off == "on" else 0)   # unique key per on/off
        s[(0, 0, NGPU, ckey)] = {"total": m["tput_per_gpu"] * NGPU, "tpot": tpot}
    return s

series = {
    "K3 MI355X (fp8 ASM +fused, v1.0.1)": mi_fp8_fused,
    "K3 MI355X (fp8 ASM, native)": mi_fp8asm,
    "K3 MI355X (bf16 ASM, native)": mi_bf16asm,
    "K3 B300 (vLLM, non-MTP)": nv_series("b300", "none"),
    "K3 B300 (vLLM, MTP)":     nv_series("b300", "mtp"),
    "K3 B200 (dynamo-vLLM)":   nv_series("b200"),
}
styles = {
    "K3 MI355X (fp8 ASM +fused, v1.0.1)": {"color": "#e11d48", "marker": "*", "linestyle": "-"},
    "K3 MI355X (fp8 ASM, native)": {"color": "#ef4444", "marker": "o", "linestyle": ":"},
    "K3 MI355X (bf16 ASM, native)": {"color": "#a78bfa", "marker": "D", "linestyle": "-"},
    "K3 B300 (vLLM, non-MTP)": {"color": "#22c55e", "marker": "s", "linestyle": "--"},
    "K3 B300 (vLLM, MTP)":     {"color": "#15803d", "marker": "P", "linestyle": "-."},
    "K3 B200 (dynamo-vLLM)":   {"color": "#38bdf8", "marker": "^", "linestyle": "--"},
}
series = {k: v for k, v in series.items() if v}   # drop empty
plot_combined(series, out_dir=OUT, title_prefix="Kimi-K3 agentic — MI355X vs B300/B200",
              series_styles=styles, shapes=[SHAPE])

print("\n=== Pareto frontier per series (interactivity, tput/gpu) ===")
for label, data in series.items():
    pts = [(1000/d["tpot"], d["total"]/k[2], k[3]) for k, d in data.items() if d["tpot"] > 0]
    fr = _pareto_frontier(pts)
    frset = {(p[0], p[1]) for p in fr}
    print(f"\n{label}:")
    for x, y, c in sorted(pts):
        print(f"  interact={x:6.1f}  tput/gpu={y:7.0f}  conc={c%100000}{'  <==FRONTIER' if (x,y) in frset else ''}")
print("\nwrote ->", OUT)
