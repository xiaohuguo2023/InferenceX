#!/usr/bin/env bash
# Merge parallel shard outputs into final tuned CSV.
#
# Usage:
#   NUM_SHARDS=4 ./merge_tuned_shards.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

NUM_SHARDS="${NUM_SHARDS:?set NUM_SHARDS}"
SHARD_DIR="${SHARD_DIR:-$HERE/shards}"
OUT_FINAL="${OUT_FINAL:-$HERE/kimik3_bf16_tuned_gemm.csv}"

HDR="gfx,cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle,libtype,solidx,splitK,us,kernelName,err_ratio,tflops,bw"

append_csv() {
  local f="$1"
  [ -f "$f" ] || return 0
  tail -n +2 "$f"
}

{
  echo "$HDR"
  use_shards=0
  for i in $(seq 0 $((NUM_SHARDS - 1))); do
    [ -f "$SHARD_DIR/kimik3_bf16_tuned_main_s${i}.csv" ] && use_shards=1
  done
  if [ "$use_shards" = 1 ]; then
    for i in $(seq 0 $((NUM_SHARDS - 1))); do
      append_csv "$SHARD_DIR/kimik3_bf16_tuned_main_s${i}.csv"
    done
    # Large-M shard-2 outliers tuned separately (flydsl/hipblaslt/skinny only)
    append_csv "$SHARD_DIR/kimik3_bf16_tuned_main_s2_hard.csv"
  else
    append_csv "$HERE/kimik3_bf16_tuned_main.csv"
  fi
  append_csv "$HERE/kimik3_bf16_tuned_n896.csv"
} > "$OUT_FINAL"

python3 - <<PY
import csv
from pathlib import Path

p = Path("$OUT_FINAL")
key = ["M", "N", "K", "bias", "dtype", "outdtype", "scaleAB", "bpreshuffle"]
with p.open(newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

deduped = {}
for row in rows:
    deduped[tuple(row[column] for column in key)] = row

with p.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(deduped.values())

print(f"[merge] {p}: {len(rows)} -> {len(deduped)} rows (deduped)")
PY

echo "[merge] install:"
echo "  cp $OUT_FINAL ~/work/aiter/aiter/configs/model_configs/kimik3_bf16_tuned_gemm.csv"
