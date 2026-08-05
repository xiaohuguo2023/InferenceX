#!/usr/bin/env python3
"""Checkpoint helpers for K3 GEMM shard tuning (stdlib only)."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SHAPE_KEY = ["M", "N", "K", "bias", "dtype", "outdtype", "scaleAB", "bpreshuffle"]


def _boolish(v: str) -> str:
    return str(v).strip().lower()


def _row_key(row: dict) -> tuple:
    return tuple(_boolish(row.get(k, "")) for k in SHAPE_KEY)


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _shape_set(rows: list[dict]) -> set[tuple]:
    out: set[tuple] = set()
    for row in rows:
        out.add(_row_key(row))
    return out


def compact_csv(path: Path, dry_run: bool = False) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    rows = _read_rows(path)
    before = len(rows)
    if before == 0:
        return 0, 0
    seen: dict[tuple, dict] = {}
    fieldnames = list(rows[0].keys())
    for row in rows:
        seen[_row_key(row)] = row
    deduped = list(seen.values())
    after = len(deduped)
    if not dry_run and after < before:
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(deduped)
    return before, after


def shard_status(shard_dir: Path, shard: int) -> dict:
    in_path = shard_dir / f"kimik3_bf16_tuning_main_s{shard}.csv"
    out_path = shard_dir / f"kimik3_bf16_tuned_main_s{shard}.csv"
    in_shapes = _shape_set(_read_rows(in_path))
    out_shapes = _shape_set(_read_rows(out_path))
    done = in_shapes & out_shapes
    remain = in_shapes - out_shapes
    return {
        "shard": shard,
        "output": out_path,
        "total": len(in_shapes),
        "done": len(done),
        "remain": len(remain),
        "output_rows": len(_read_rows(out_path)),
    }


def _extra_status(name: str, in_path: Path, out_path: Path) -> tuple[str, int, int, int, int]:
    in_shapes = _shape_set(_read_rows(in_path))
    out_shapes = _shape_set(_read_rows(out_path))
    done = len(in_shapes & out_shapes)
    total = len(in_shapes)
    return name, done, total - done, total, len(_read_rows(out_path))


def print_status(here: Path, num_shards: int) -> int:
    shard_dir = here / "shards"
    total = done = remain = 0
    print(f"{'shard':>5}  {'done':>5}  {'left':>5}  {'total':>5}  {'csv_rows':>8}  output")
    for s in range(num_shards):
        st = shard_status(shard_dir, s)
        total += st["total"]
        done += st["done"]
        remain += st["remain"]
        out_path = st["output"]
        exists = "yes" if out_path.exists() else "no"
        print(
            f"s{s:>4}  {st['done']:>5}  {st['remain']:>5}  {st['total']:>5}  "
            f"{st['output_rows']:>8}  {out_path.name} ({exists})"
        )
    hard_in = shard_dir / "kimik3_bf16_tuning_main_s2_hard.csv"
    hard_out = shard_dir / "kimik3_bf16_tuned_main_s2_hard.csv"
    if hard_in.exists():
        name, hd, hl, ht, rows = _extra_status("hard", hard_in, hard_out)
        print(f"{name:>5}  {hd:>5}  {hl:>5}  {ht:>5}  {rows:>8}  {hard_out.name}")
        total += ht
        done += hd
        remain += hl
    n896_in = here / "kimik3_bf16_tuning_n896.csv"
    n896_out = here / "kimik3_bf16_tuned_n896.csv"
    if n896_in.exists():
        name, nd, nl, nt, rows = _extra_status("n896", n896_in, n896_out)
        print(f"{name:>5}  {nd:>5}  {nl:>5}  {nt:>5}  {rows:>8}  {n896_out.name}")
        total += nt
        done += nd
        remain += nl
    pct = 100.0 * done / total if total else 0.0
    print(f"\noverall: {done}/{total} shapes tuned ({pct:.1f}%), {remain} remaining")
    print("restart: SKIP_SPLIT=1 TUNE_LIBTYPE_PROFILE=safe NUM_SHARDS=4 SHARD=N ./tune_shard.sh bg")
    return 0 if remain == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="K3 GEMM tuning checkpoint status/compact")
    ap.add_argument("--here", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--num-shards", type=int, default=4)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    cp = sub.add_parser("compact")
    cp.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    here: Path = args.here
    if args.cmd == "status":
        return print_status(here, args.num_shards)
    changed = 0
    shard_dir = here / "shards"
    for path in sorted(shard_dir.glob("kimik3_bf16_tuned_*.csv")):
        before, after = compact_csv(path, dry_run=args.dry_run)
        if before != after:
            changed += 1
            tag = "would compact" if args.dry_run else "compacted"
            print(f"{tag}: {path.name} {before} -> {after} rows")
    for name in ("kimik3_bf16_tuned_n896.csv",):
        path = here / name
        if path.exists():
            before, after = compact_csv(path, dry_run=args.dry_run)
            if before != after:
                changed += 1
                tag = "would compact" if args.dry_run else "compacted"
                print(f"{tag}: {path.name} {before} -> {after} rows")
    if changed == 0:
        print("all checkpoint CSVs already deduped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
