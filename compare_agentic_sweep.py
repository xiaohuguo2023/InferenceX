#!/usr/bin/env python3
"""Compare MI355X agentic sweep results vs B300/B200 dashboard baselines.

MI355X: k3_{TAG}_ixci_c{conc}/aiperf_artifacts/profile_export_aiperf.json
NVIDIA: JSON export from InferenceX dashboard (see fetch_nv_agentic_baselines.sh)

Usage:
  python3 compare_agentic_sweep.py [--tag fp8asm_fused] [--root /path/to/InferenceX]
  python3 compare_agentic_sweep.py --nv-json /tmp/k3_b300.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def g(v):
    return v.get("avg") if isinstance(v, dict) else v


def load_mi355x(root: Path, tag: str, conc_list: list[int]) -> dict[int, dict]:
    out = {}
    for c in conc_list:
        f = root / f"k3_{tag}_ixci_c{c}" / "aiperf_artifacts" / "profile_export_aiperf.json"
        if not f.exists():
            print(f"  [skip] MI355X conc{c}: missing {f}", file=sys.stderr)
            continue
        d = json.loads(f.read_text())
        inp = g(d.get("input_token_throughput", {})) or 0.0
        out_t = g(d.get("output_token_throughput", {})) or 0.0
        tpot = g(d.get("inter_token_latency", {}))
        ttft = g(d.get("time_to_first_token", {}))
        if not tpot:
            print(f"  [skip] MI355X conc{c}: no TPOT", file=sys.stderr)
            continue
        total = inp + out_t
        out[c] = {
            "tput_per_gpu": total / 8,
            "tpot_ms": tpot,
            "interactivity": 1000.0 / tpot,
            "ttft_ms": (ttft or 0) * 1000 if ttft and ttft < 100 else ttft,
        }
    return out


def load_nv(nv_json: Path, hw: str, offload: str | None = None) -> dict[int, dict]:
    """Load B300/B200 rows from dashboard export."""
    if not nv_json.exists():
        return {}
    rows = json.loads(nv_json.read_text())
    out = {}
    for r in rows:
        if r.get("hardware") != hw:
            continue
        off = r.get("metrics", {}).get("offload_mode", "")
        if offload is not None and off != offload:
            continue
        c = int(r["conc"])
        m = r["metrics"]
        tpot = m.get("mean_tpot", 0) * 1000
        if tpot <= 0:
            continue
        out[c] = {
            "tput_per_gpu": m["tput_per_gpu"],
            "tpot_ms": tpot,
            "interactivity": 1000.0 / tpot,
            "ttft_ms": m.get("mean_ttft", 0) * 1000,
            "offload_mode": off,
        }
    return out


def fmt_row(label: str, c: int, d: dict) -> str:
    off = f" off={d['offload_mode']}" if d.get("offload_mode") else ""
    return (
        f"| {label}{off} | {c} | {d['tput_per_gpu']:,.0f} | "
        f"{d['tpot_ms']:.1f} | {d['interactivity']:.1f} | "
        f"{d.get('ttft_ms', 0):,.0f} |"
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Compare agentic sweeps vs B300/B200")
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--tag", default="fp8asm_fused", help="Sweep output prefix (k3_{TAG}_ixci_c*)")
    p.add_argument("--nv-json", type=Path, default=Path("/tmp/k3_b300.json"))
    p.add_argument("--conc", type=int, nargs="+", default=[1, 4, 8, 16, 24])
    args = p.parse_args()

    mi = load_mi355x(args.root, args.tag, args.conc)
    b300 = load_nv(args.nv_json, "b300")
    b300_off = load_nv(args.nv_json, "b300", offload="on")
    b200 = load_nv(args.nv_json, "b200")

    if not mi:
        print("No MI355X sweep results found. Run ./setup_benchmark.sh sweep-fused first.", file=sys.stderr)
        return 1

    print(f"\n# Kimi-K3 agentic comparison — MI355X ({args.tag}) vs B300 vs B200\n")
    print("| Platform | conc | tput/gpu (tok/s) | TPOT (ms) | Interactivity | TTFT (ms) |")
    print("|---|---:|---:|---:|---:|---:|")

    for c in args.conc:
        if c in mi:
            print(fmt_row("MI355X fp8 ASM", c, mi[c]))
        for label, data in [("B300", b300), ("B300", b300_off), ("B200", b200)]:
            if c in data:
                print(fmt_row(label, c, data[c]))

    if not b300 and not b200:
        print(
            "\n_NVIDIA baselines missing._ Export from https://inferencex.com/ (Kimi-K3 agentic, "
            "B300/B200) to /tmp/k3_b300.json, or run:\n"
            "  ./fetch_nv_agentic_baselines.sh\n",
            file=sys.stderr,
        )
        return 0

    print("\n## Delta vs B300 (GPU-resident, same conc)\n")
    print("| conc | MI355X tput/gpu | B300 tput/gpu | Δ tput | MI355X TPOT | B300 TPOT | Δ TPOT |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for c in args.conc:
        if c not in mi or c not in b300:
            continue
        m, b = mi[c], b300[c]
        dt = m["tput_per_gpu"] - b["tput_per_gpu"]
        dp = m["tpot_ms"] - b["tpot_ms"]
        print(
            f"| {c} | {m['tput_per_gpu']:,.0f} | {b['tput_per_gpu']:,.0f} | "
            f"{dt:+,.0f} | {m['tpot_ms']:.1f} | {b['tpot_ms']:.1f} | {dp:+.1f} |"
        )

    pareto = args.root / "docs" / "kimik3_pareto"
    plot = args.root / "plot_kimik3_pareto.py"
    if plot.exists():
        print(f"\nPareto plot: python3 {plot} --tag {args.tag} --nv-json {args.nv_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
