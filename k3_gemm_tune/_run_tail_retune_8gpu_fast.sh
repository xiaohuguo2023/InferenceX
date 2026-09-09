#!/usr/bin/env bash
# _run_tail_retune_8gpu_fast.sh — finish the large-N small-M GEMM retune tail with
# the FAST backends only (asm,opus,skinny,hipblaslt), sharded across 8 GPUs.
#
# WHY no flydsl here: py-spy on the flydsl-inclusive run proved the 100%-CPU / 0%-GPU
# stall is flydsl JIT-COMPILING every candidate tiling from scratch
# (flydsl/compiler/jit_function.py::_compile_impl via aiter/ops/flydsl/gemm_kernels.py
# ::launcher). For large-N decode shapes the tiling space is huge -> >3 min pure-Python
# compile PER shape -> 0 rows in 180s. That is the same blowup that timed out PASS1;
# sharding parallelizes it 6-ways but each shard is still compile-bound & intractable.
# asm/opus/skinny/hipblaslt pick from prebuilt kernels/solidx (no per-candidate JIT) ->
# seconds/shape. flydsl at the decode-critical M is measured separately & cheaply by
# _run_tail_flydsl_m72.sh (bounded to the M values that actually matter), so we still
# MEASURE flydsl instead of assuming it loses -- we just don't pay the full-space compile.
#
# IMPORTANT: let each shard EXIT NORMALLY (SIGTERM at worst); a SIGKILL'd tuner child
# zombie-pins its GPU ctx under the container sleep-infinity init.
set -uo pipefail

WORK=/workspace/_retune
# Parameterized (defaults reproduce the original large-N tail run exactly):
IN="${IN:-$WORK/in_tail_largeN.csv}"
OUT="${OUT:-$WORK/tuned_tail_largeN_fast.csv}"
MMAX="${MMAX:-256}"                       # only tune rows with M<=MMAX
LIBS="${LIBS:-asm,opus,skinny}"           # fast backends; hipblaslt added below
AT=/opt/aiter-local/csrc/gemm_a16w16
[ -f "$IN" ] || { echo "!! missing $IN"; exit 1; }
HDR="$(head -1 "$IN")"
export AITER_HIPBLASLT_FAST_MAX="${AITER_HIPBLASLT_FAST_MAX:-8192}"

# GPU -> "N,K" groups: auto-derive the distinct (N,K) present in $IN and round-robin
# them onto the 8 GPUs (fast backends make balance moot; 1+ group/GPU stops a bad
# shape serializing the rest). Works for ANY input, not just the fixed tail file.
declare -A ASSIGN=([0]="" [1]="" [2]="" [3]="" [4]="" [5]="" [6]="" [7]="")
i=0
while IFS= read -r nk; do
  g=$((i % 8))
  ASSIGN[$g]="${ASSIGN[$g]:+${ASSIGN[$g]} }$nk"
  i=$((i+1))
done < <(awk -F, 'NR>1{print $2","$3}' "$IN" | sort -u)
echo "== auto-derived $i (N,K) groups from $IN over 8 GPUs (MMAX=$MMAX LIBS=$LIBS) =="

cd "$AT"
mkdir -p "$WORK/shards"
pids=()
for g in 0 1 2 3 4 5 6 7; do
  groups="${ASSIGN[$g]}"
  shard_in="$WORK/shards/in_fast_g${g}.csv"
  shard_out="$WORK/shards/out_fast_g${g}.csv"
  rm -f "$shard_out"
  [ -z "$groups" ] && { echo "   GPU$g: no groups, skip"; continue; }
  echo "$HDR" > "$shard_in"
  for nk in $groups; do
    N="${nk%,*}"; K="${nk#*,}"
    awk -F, -v N="$N" -v K="$K" -v mm="$MMAX" 'NR>1 && $1<=mm && $2==N && $3==K' "$IN" >> "$shard_in"
  done
  rows=$(($(wc -l < "$shard_in")-1))
  [ "$rows" -le 0 ] && { echo "   GPU$g: 0 rows after MMAX filter, skip"; continue; }
  echo ">> GPU$g: groups=[$groups] rows=$rows -> $shard_out"
  HIP_VISIBLE_DEVICES="$g" timeout -k 60 3600 python gemm_tuner.py \
    --input_file "$shard_in" --tuned_file "$shard_out" \
    --libtype "$LIBS" --with-hipblaslt --batch 10 --shape_grouped \
    > "$WORK/shards/log_fast_g${g}.out" 2>&1 &
  pids+=($!)
done

echo "== launched ${#pids[@]} fast shard tuners: ${pids[*]} =="
fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "  shard GPU$i OK"; else echo "  shard GPU$i EXIT=$?"; fail=1; fi
done

echo "===== merge -> $OUT $(date +%H:%M:%S) ====="
MERGED="$OUT"
first=1
for g in 0 1 2 3 4 5 6 7; do
  o="$WORK/shards/out_fast_g${g}.csv"
  [ -f "$o" ] || { echo "  !! missing $o"; continue; }
  if [ "$first" = 1 ]; then cat "$o" > "$MERGED"; first=0; else tail -n +2 "$o" >> "$MERGED"; fi
done
echo "== merged rows: $(($(wc -l < "$MERGED")-1)) =="
echo "== winners by libtype =="; awk -F, 'NR>1{print $11}' "$MERGED" | sort | uniq -c
echo "== M=72 winners =="; awk -F, 'NR>1 && $3==72{print $4"x"$5"  "$11" splitK="$13" us="$14}' "$MERGED"
echo "TAILFAST_DONE $(date +%H:%M:%S)  fail=$fail"
