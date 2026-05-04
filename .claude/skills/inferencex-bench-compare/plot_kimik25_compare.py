#!/usr/bin/env python3
"""Kimi-K2.5 fp4: MI355x (ours + IX dashboard) vs B200/B300 (IX dashboard).

Inputs
------
- IX dump JSONs at /tmp/inferencex_dump/inferencex-dump-2026-04-27/
- Our sweep dir at $SWEEP_DIR (default: latest /workspace/sweep_kimik25_widegraph_default_*)

Outputs (in this script's directory)
------------------------------------
- kimik25_mi355x_vs_b200_tp{4,8}.png   2x3 panels
- kimik25_data_table.csv               per-(isl,osl,tp,conc) numbers across all sources

Run
---
  python3 comparison_plots/plot_kimik25_compare.py [SWEEP_DIR]
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

DUMP_DIR = Path(os.environ.get(
    "IX_DUMP_DIR", "/tmp/inferencex_dump/inferencex-dump-2026-04-27"
))
OUT_DIR = Path(__file__).resolve().parent

# Configs of interest. Pick the most-recent / highest-coverage cfg per (hw, tp).
# kimik2.5 fp4 single-node:
#   MI355x vllm: 672 (TP=4, full grid), 603 (TP=8, full grid). 681 = TP=4 1k/8k rerun, fold in.
#   B200 vllm:   811 (TP=4, newer 2026-04 with 1k/1k+8k/1k), 635 (TP=4, older 2026-03 fills 1k/8k),
#                636 (TP=8, sparse — only conc=4)
#   B300 vllm:   813 (TP=4), 814 (TP=8 sparse)
SOURCES = {
    "ours (MI355x widegraph-default)": {"kind": "sweep"},
    "IX MI355x vllm":  {"kind": "ix", "cfgs": [672, 603, 681]},
    "IX B200 vllm":    {"kind": "ix", "cfgs": [811, 635, 636]},
    "IX B300 vllm":    {"kind": "ix", "cfgs": [813, 814]},
}

# Each TP gets its own figure; TPs we plot:
TPS = [4, 8]
# Columns of the 2x3 panel: (ISL, OSL)
COLS = [(1024, 1024), (8192, 1024)]
# Concurrency x-axis order
CONCS = [4, 8, 16, 32, 64, 128]


def load_ix_rows(cfgs: list[int]) -> list[dict]:
    """Load IX dump rows for the listed cfgs and (isl,osl) of interest."""
    cfg_set = set(cfgs)
    rows = []
    with open(DUMP_DIR / "configs.json") as f:
        cfg_meta = {c["id"]: c for c in json.load(f)}
    with open(DUMP_DIR / "benchmark_results.json") as f:
        for r in json.load(f):
            if r.get("error"):
                continue
            if r.get("config_id") not in cfg_set:
                continue
            if (r.get("isl"), r.get("osl")) not in COLS:
                continue
            cfg = cfg_meta[r["config_id"]]
            r["_cfg"] = cfg
            r["_tp"] = cfg["decode_tp"]
            rows.append(r)
    return rows


def latest_per_combo(rows: list[dict]) -> dict[tuple, dict]:
    """For each (isl, osl, tp, conc) keep the most recent row."""
    best: dict[tuple, dict] = {}
    for r in rows:
        key = (r["isl"], r["osl"], r["_tp"], r["conc"])
        prior = best.get(key)
        if prior is None or r["date"] > prior["date"]:
            best[key] = r
    return best


def find_latest_sweep_dir() -> Path | None:
    candidates = sorted(glob.glob("/workspace/sweep_kimik25_widegraph_default_*"))
    if not candidates:
        # Also check the v017_output drop-zone where prior gptoss sweeps landed.
        candidates = sorted(glob.glob(
            "/home/xiaohugu/work/sweep_v017_output/sweep_kimik25_widegraph_default_*"
        ))
    return Path(candidates[-1]) if candidates else None


def load_sweep(sweep_dir: Path) -> dict[tuple, dict]:
    """Load our sweep results — keyed by (isl, osl, tp, conc).

    Result JSON file pattern:
      kimik25_widegraph_default_mi355x_isl<ISL>_osl<OSL>_tp<TP>_conc<CONC>.json
    """
    out = {}
    for path in glob.glob(str(sweep_dir / "kimik25_widegraph_default_mi355x_isl*_osl*_tp*_conc*.json")):
        name = Path(path).stem
        # Parse isl/osl/tp/conc from filename
        try:
            parts = dict(p.split("isl") if "isl" in p else
                         p.split("osl") if "osl" in p else
                         p.split("tp")  if "tp"  in p else
                         p.split("conc") if "conc" in p else (None,None)
                         for p in name.split("_") if any(k in p for k in ("isl","osl","tp","conc")))
        except ValueError:
            continue
        # Easier: regex
        import re
        m = re.search(r"isl(\d+)_osl(\d+)_tp(\d+)_conc(\d+)", name)
        if not m:
            continue
        isl, osl, tp, conc = (int(x) for x in m.groups())
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception:
            continue
        out[(isl, osl, tp, conc)] = d
    return out


def metric_from_ix(row: dict, metric: str) -> float | None:
    m = row["metrics"]
    tp = row["_tp"]
    if metric == "tput":
        v = m.get("tput_per_gpu")
        return None if v is None else v * tp
    if metric == "tput_per_gpu":
        return m.get("tput_per_gpu")
    if metric == "ttft":
        v = m.get("mean_ttft")
        return None if v is None else v * 1000.0  # s -> ms
    if metric == "tpot":
        v = m.get("mean_tpot")
        return None if v is None else v * 1000.0
    if metric == "e2el":
        return m.get("mean_e2el")  # already seconds
    raise ValueError(metric)


def metric_from_sweep(row: dict, metric: str) -> float | None:
    if metric == "tput":
        return row.get("total_token_throughput")
    if metric == "tput_per_gpu":
        # sweep JSON has total throughput; we need TP from the row file naming, but
        # callers pass the row dict not the key — easier: derive via num_workers from
        # tensor parallel size encoded in the script CLI; we instead let the caller
        # divide by TP itself. Return total throughput here; pareto helper divides.
        return row.get("total_token_throughput")
    if metric == "ttft":
        return row.get("mean_ttft_ms")
    if metric == "tpot":
        return row.get("mean_tpot_ms")
    if metric == "e2el":
        v = row.get("mean_e2el_ms")
        return None if v is None else v / 1000.0  # ms -> s to match IX units
    raise ValueError(metric)


def plot_tp(tp: int, sweep: dict, ix_data: dict[str, dict], out_path: Path):
    fig, axes = plt.subplots(3, len(COLS), figsize=(11, 11), sharex=False)
    fig.suptitle(f"Kimi-K2.5 fp4 — TP={tp}: ours vs InferenceX dashboard", fontsize=14)

    metric_rows = [
        ("tput", "Total throughput (tok/s)"),
        ("ttft", "Mean TTFT (ms)"),
        ("tpot", "Mean TPOT (ms)"),
    ]
    style = {
        "ours (MI355x widegraph-default)": dict(color="tab:red",    marker="o", lw=2),
        "IX MI355x vllm":                  dict(color="tab:orange", marker="s", lw=1.5, ls="--"),
        "IX B200 vllm":                    dict(color="tab:green",  marker="^", lw=1.5),
        "IX B300 vllm":                    dict(color="tab:blue",   marker="d", lw=1.5),
    }

    for col_idx, (isl, osl) in enumerate(COLS):
        for row_idx, (metric, ylabel) in enumerate(metric_rows):
            ax = axes[row_idx][col_idx]
            for series, opts in style.items():
                if SOURCES[series]["kind"] == "sweep":
                    by_conc = {c: metric_from_sweep(sweep[(isl, osl, tp, c)], metric)
                               for c in CONCS if (isl, osl, tp, c) in sweep
                               and metric_from_sweep(sweep[(isl, osl, tp, c)], metric) is not None}
                else:
                    src_rows = ix_data[series]
                    by_conc = {c: metric_from_ix(src_rows[(isl, osl, tp, c)], metric)
                               for c in CONCS if (isl, osl, tp, c) in src_rows
                               and metric_from_ix(src_rows[(isl, osl, tp, c)], metric) is not None}
                if not by_conc:
                    continue
                xs = sorted(by_conc)
                ys = [by_conc[c] for c in xs]
                ax.plot(xs, ys, label=series, **opts)
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            if row_idx == 0:
                ax.set_title(f"ISL={isl} OSL={osl}")
            if row_idx == len(metric_rows) - 1:
                ax.set_xlabel("Concurrency")
            ax.set_xscale("log", base=2)
            ax.set_xticks([4, 8, 16, 32, 64, 128])
            ax.set_xticklabels(["4", "8", "16", "32", "64", "128"])
            ax.grid(True, alpha=0.3)
    # One legend on top.
    handles, labels = [], []
    for ax_row in axes:
        for ax in ax_row:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_pareto(tp: int, isl: int, osl: int, sweep: dict, ix_data: dict[str, dict],
                x_metric: str, out_path: Path):
    """Pareto: throughput-per-GPU (Y, log) vs interactivity-or-e2el (X, log).

    x_metric:
      'interactivity' -> X = 1000 / mean_TPOT_ms (tok/s/user)
      'e2el'          -> X = mean_e2el (s)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    style = {
        "ours (MI355x widegraph-default)": dict(color="tab:red",    marker="o", lw=2),
        "IX MI355x vllm":                  dict(color="tab:orange", marker="s", lw=1.5, ls="--"),
        "IX B200 vllm":                    dict(color="tab:green",  marker="^", lw=1.5),
        "IX B300 vllm":                    dict(color="tab:blue",   marker="d", lw=1.5),
    }

    for series, opts in style.items():
        pts = []  # list of (x, y, conc)
        for c in CONCS:
            if SOURCES[series]["kind"] == "sweep":
                row = sweep.get((isl, osl, tp, c))
                if not row:
                    continue
                tput_total = metric_from_sweep(row, "tput")
                tpot_ms = metric_from_sweep(row, "tpot")
                e2el_s  = metric_from_sweep(row, "e2el")
            else:
                row = ix_data[series].get((isl, osl, tp, c))
                if not row:
                    continue
                tput_total = metric_from_ix(row, "tput")
                tpot_ms = metric_from_ix(row, "tpot")
                e2el_s  = metric_from_ix(row, "e2el")
            if tput_total is None or (x_metric == "interactivity" and not tpot_ms) \
                    or (x_metric == "e2el" and e2el_s is None):
                continue
            y = tput_total / tp  # per-GPU throughput
            x = (1000.0 / tpot_ms) if x_metric == "interactivity" else e2el_s
            pts.append((x, y, c))
        if not pts:
            continue
        pts.sort(key=lambda p: p[0] if x_metric == "e2el" else -p[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, label=series, **opts)
        for x, y, c in pts:
            ax.annotate(f"c={c}", (x, y), fontsize=7,
                        xytext=(4, 4), textcoords="offset points",
                        color=opts["color"])

    ax.set_xscale("log")
    ax.set_yscale("log")
    if x_metric == "interactivity":
        ax.set_xlabel("Interactivity (tok/s/user)  =  1 / mean TPOT")
        subtitle = "throughput/GPU vs interactivity"
    else:
        ax.set_xlabel("End-to-end latency (s)  =  mean_e2el")
        subtitle = "throughput/GPU vs end-to-end latency"
    ax.set_ylabel("Token throughput per GPU (tok/s/GPU)  —  input+output combined")
    ax.set_title(f"Kimi-K2.5 fp4 — ISL={isl}, OSL={osl}, TP={tp}\nPareto: {subtitle} (CONC sweep along each curve)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9, title="Series")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def write_csv(sweep: dict, ix_data: dict[str, dict], out_path: Path):
    cols = ["isl", "osl", "tp", "conc", "metric"] + list(SOURCES.keys())
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for (isl, osl) in COLS:
            for tp in TPS:
                for conc in CONCS:
                    for metric in ("tput", "ttft", "tpot"):
                        row = [isl, osl, tp, conc, metric]
                        for src in SOURCES:
                            if SOURCES[src]["kind"] == "sweep":
                                v = sweep.get((isl, osl, tp, conc))
                                row.append(round(metric_from_sweep(v, metric), 3) if v else "")
                            else:
                                v = ix_data[src].get((isl, osl, tp, conc))
                                row.append(round(metric_from_ix(v, metric), 3) if v else "")
                        w.writerow(row)
    print(f"wrote {out_path}")


def main():
    sweep_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else find_latest_sweep_dir()
    if sweep_dir and sweep_dir.is_dir():
        print(f"Loading sweep from {sweep_dir}")
        sweep = load_sweep(sweep_dir)
        print(f"  {len(sweep)} sweep result files loaded")
    else:
        print("(no sweep dir found — plots will skip the 'ours' series)")
        sweep = {}

    ix_data: dict[str, dict] = {}
    for name, info in SOURCES.items():
        if info["kind"] != "ix":
            continue
        rows = load_ix_rows(info["cfgs"])
        ix_data[name] = latest_per_combo(rows)
        print(f"  {name}: {len(ix_data[name])} unique combos")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for tp in TPS:
        plot_tp(tp, sweep, ix_data, OUT_DIR / f"kimik25_mi355x_vs_b200_tp{tp}.png")
        for (isl, osl) in COLS:
            tag = f"{isl//1024}k{osl//1024}k"
            plot_pareto(tp, isl, osl, sweep, ix_data, "interactivity",
                        OUT_DIR / f"kimik25_pareto_{tag}_tp{tp}.png")
            plot_pareto(tp, isl, osl, sweep, ix_data, "e2el",
                        OUT_DIR / f"kimik25_pareto_e2el_{tag}_tp{tp}.png")
    write_csv(sweep, ix_data, OUT_DIR / "kimik25_data_table.csv")


if __name__ == "__main__":
    main()
