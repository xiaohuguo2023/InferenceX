#!/usr/bin/env python3
"""Extract AITER BF16 GEMM misses from per-concurrency vLLM server logs."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


SHAPE_RE = re.compile(
    r"shape is M:(?P<M>\d+), N:(?P<N>\d+), K:(?P<K>\d+) "
    r"dtype='(?P<dtype>[^']+)' otype='(?P<outdtype>[^']+)' "
    r"bias=(?P<bias>True|False), scaleAB=(?P<scaleAB>True|False), "
    r"bpreshuffle=(?P<bpreshuffle>True|False), not found tuned config"
)
CONC_RE = re.compile(r"_c(\d+)\.log$")
KEY_COLUMNS = (
    "M",
    "N",
    "K",
    "bias",
    "dtype",
    "outdtype",
    "scaleAB",
    "bpreshuffle",
)


def shape_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(row[column]) for column in KEY_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--pattern",
        default="serve_fp8asm_ms64_ixci_cold_tuned_mbt4k_c*.log",
    )
    parser.add_argument(
        "--tuned-csv",
        type=Path,
        default=Path(__file__).with_name("kimik3_bf16_tuned_gemm.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("kimik3_bf16_uncovered_shapes.csv"),
    )
    args = parser.parse_args()

    with args.tuned_csv.open(newline="") as handle:
        tuned_keys = {shape_key(row) for row in csv.DictReader(handle)}

    misses: dict[tuple[str, ...], dict[str, object]] = {}
    counts: defaultdict[tuple[str, ...], int] = defaultdict(int)
    concurrencies: defaultdict[tuple[str, ...], set[int]] = defaultdict(set)
    logs = sorted(args.root.glob(args.pattern))

    for log in logs:
        conc_match = CONC_RE.search(log.name)
        conc = int(conc_match.group(1)) if conc_match else -1
        with log.open(errors="replace") as handle:
            for line in handle:
                match = SHAPE_RE.search(line)
                if not match:
                    continue
                row = match.groupdict()
                key = shape_key(row)
                misses[key] = row
                counts[key] += 1
                concurrencies[key].add(conc)

    rows = []
    for key, row in misses.items():
        rows.append(
            {
                **row,
                "occurrences": counts[key],
                "concurrencies": ",".join(
                    str(value) for value in sorted(concurrencies[key]) if value >= 0
                ),
                "in_k3_final_csv": key in tuned_keys,
            }
        )
    rows.sort(key=lambda row: (int(row["M"]), int(row["N"]), int(row["K"])))

    fieldnames = [*KEY_COLUMNS, "occurrences", "concurrencies", "in_k3_final_csv"]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    covered = sum(bool(row["in_k3_final_csv"]) for row in rows)
    print(
        f"logs={len(logs)} unique_misses={len(rows)} "
        f"already_in_k3_csv={covered} output={args.output}"
    )


if __name__ == "__main__":
    main()
