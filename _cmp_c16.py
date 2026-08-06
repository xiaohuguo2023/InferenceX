#!/usr/bin/env python3
"""Extract and compare aiperf summary metrics across runs.

Usage: _cmp_c16.py RUN1.log RUN2.log ...

Parses the summary tables aiperf writes at the end of a profiling phase, so it
reports aiperf's own metric computation rather than recomputing from the raw
export. Only runs with matching concurrency, duration and dataset are
comparable; the per-user metrics in particular are sensitive to how many
requests were actually running concurrently, which the engine log reports
separately as "Running: N reqs".
"""

import re
import sys
from pathlib import Path

ROWS = [
    ("Time to First Token (ms)", "ttft"),
    ("Inter Token Latency (ms)", "itl"),
    ("Request Latency (ms)", "e2e"),
    ("Input Sequence Length (tokens)", "isl"),
    ("Output Sequence Length (tokens)", "osl"),
    ("Output Token Throughput (tokens/sec)", "out_tps"),
    ("Input Token Throughput (tokens/sec)", "in_tps"),
    ("Request Count (requests)", "reqs"),
    ("Theoretical Prefix Cache Hit (%)", "theo_hit"),
    ("Overall Usage Prompt Cache Read %", "cache_read"),
    ("Tokens In Flight (tokens)", "inflight"),
    ("Total Usage Prompt Tokens (tokens)", "tot_prompt"),
    ("Total Usage Prompt Cache Read Tokens", "tot_cache_rd"),
]

NUM = r"([0-9,\.]+|N/A)"
# aiperf renders the summary tables with box-drawing separators, not pipes.
SEP = r"\s*[|\u2502]"


def parse(log: Path) -> dict:
    text = log.read_text(errors="ignore")
    out: dict = {}
    for label, key in ROWS:
        m = re.search(re.escape(label) + SEP + (r"\s*" + NUM + SEP) * 6, text)
        if not m:
            continue

        def f(s: str):
            s = s.replace(",", "")
            return None if s == "N/A" else float(s)

        out[key] = {
            "avg": f(m.group(1)), "min": f(m.group(2)), "max": f(m.group(3)),
            "p99": f(m.group(4)), "p90": f(m.group(5)), "p50": f(m.group(6)),
        }
    m = re.search(
        r"Phase profiling \(profiling\) complete \| completed=(\d+), cancelled=(\d+), "
        r"errors=(\d+)[^|]*\| sessions[^|]*\| elapsed=([0-9\.]+)s",
        text,
    )
    if m:
        out["_run"] = {
            "completed": int(m.group(1)), "cancelled": int(m.group(2)),
            "errors": int(m.group(3)), "elapsed": float(m.group(4)),
        }
    return out


def cell(d, key, stat="p50", scale=1.0, fmt="{:,.1f}") -> str:
    v = d.get(key, {}).get(stat)
    return "-" if v is None else fmt.format(v / scale)


def main(paths: list[str]) -> None:
    runs = []
    for p in paths:
        log = Path(p)
        if not log.exists():
            print(f"missing: {log}", file=sys.stderr)
            continue
        d = parse(log)
        if not d:
            print(f"no summary table: {log}", file=sys.stderr)
            continue
        runs.append((log.stem.replace("k3_", "").replace("_ixci_c16", ""), d))

    if not runs:
        sys.exit("no parseable runs")

    width = max(16, max(len(n) for n, _ in runs) + 2)
    hdr = f"{'run':<34}" + "".join(f"{n:>{width}}" for n, _ in runs)
    print(hdr)
    print("-" * len(hdr))

    def runline(label, field, fmt="{}"):
        print(f"{label:<34}" + "".join(
            f"{fmt.format(d['_run'][field]) if d.get('_run') else '-':>{width}}"
            for _, d in runs))

    def line(label, key, stat, scale=1.0, fmt="{:,.1f}"):
        print(f"{label:<34}" + "".join(
            f"{cell(d, key, stat, scale, fmt):>{width}}" for _, d in runs))

    runline("requests completed", "completed")
    runline("errors", "errors")
    runline("profiling elapsed (s)", "elapsed", "{:,.0f}")
    print()
    line("TTFT p50 (s)", "ttft", "p50", 1000)
    line("TTFT avg (s)", "ttft", "avg", 1000)
    line("TTFT p90 (s)", "ttft", "p90", 1000)
    line("ITL p50 (ms)", "itl", "p50")
    line("ITL avg (ms)", "itl", "avg")
    line("ITL p90 (ms)", "itl", "p90")
    line("E2E latency p50 (s)", "e2e", "p50", 1000)
    print()
    line("output tok/s (aggregate)", "out_tps", "avg")
    line("input tok/s (aggregate)", "in_tps", "avg")
    print()
    line("input seq len p50", "isl", "p50")
    line("input seq len avg", "isl", "avg")
    line("output seq len avg", "osl", "avg")
    print()
    line("theoretical prefix hit (%)", "theo_hit", "avg")
    line("actual cache read (%)", "cache_read", "avg")
    line("total prompt tokens (M)", "tot_prompt", "avg", 1e6, "{:,.2f}")
    line("cache-read tokens (M)", "tot_cache_rd", "avg", 1e6, "{:,.2f}")
    line("tokens in flight p90 (M)", "inflight", "p90", 1e6, "{:,.2f}")
    line("tokens in flight max (M)", "inflight", "max", 1e6, "{:,.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
