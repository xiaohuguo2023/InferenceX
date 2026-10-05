#!/bin/bash
# Conc-1 IX proxy v2 against a warm serve: fixed-ISL requests spread over the
# IX workload's ISL distribution, weighted by that distribution.
#
# The v1 proxy (100k/150k x 256) sat exactly where our gains are largest and
# missed the long-context half of the workload (43% of IX requests are >= 256k,
# where gains shrink and the gap to B300 grows). Per-ISL results are combined
# into an IX estimate: a weighted mixture whose p50/p90 use the IX workload's
# ISL histogram (bins <96k / 96-160k / 160-256k / >=256k).
#   WORKDIR=... bash k3_proxy2_bench.sh
set -uo pipefail
WORKDIR="${WORKDIR:?}"
ISLS="${ISLS:-88000}"
WEIGHTS="${WEIGHTS:-0.22 0.23 0.11 0.43}"
OSL="${OSL:-512}"
REQ_COUNT="${REQ_COUNT:-6}"
SEED="${SEED:-7}"
MODEL_PATH="${MODEL_PATH:-/shared_nfs/hyperloom/models/Kimi-K3}"
PORT="${PORT:-8888}"

AIPERF=""
for c in /workspace/.aiperf_rocm100_py312/bin/aiperf /workspace/.aiperf_*/bin/aiperf /opt/.aiperf_*/bin/aiperf; do
  [ -x "$c" ] || continue
  "$c" --version >/dev/null 2>&1 && { AIPERF="$c"; break; }
done
[ -n "$AIPERF" ] || { echo "!! no working aiperf"; exit 1; }
curl -sf -m5 "http://localhost:$PORT/health" > /dev/null || { echo "!! serve not healthy"; exit 1; }
mkdir -p "$WORKDIR"

for ISL in $ISLS; do
  out="$WORKDIR/isl${ISL}"
  env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE "$AIPERF" profile \
    --model moonshotai/Kimi-K3 --tokenizer "$MODEL_PATH" --tokenizer-trust-remote-code \
    --url "http://127.0.0.1:$PORT" --api-key EMPTY --endpoint-type chat --streaming --use-server-token-count \
    --prompt-input-tokens-mean "$ISL" --prompt-input-tokens-stddev 0 \
    --output-tokens-mean "$OSL" --output-tokens-stddev 0 \
    --extra-inputs ignore_eos:true --extra-inputs "min_tokens:$OSL" --extra-inputs "max_tokens:$OSL" \
    --concurrency ${PCONC:-1} --request-count "$REQ_COUNT" --warmup-request-count 1 \
    --random-seed "$SEED" --no-gpu-telemetry --output-artifact-dir "$out" \
    > "$out.log" 2>&1 || echo "!! aiperf failed for ISL=$ISL"
done

python3 - "$WORKDIR" "$ISLS" "$WEIGHTS" << 'PY'
import json, sys
wd, isls, weights = sys.argv[1], sys.argv[2].split(), [float(w) for w in sys.argv[3].split()]
assert len(isls) == len(weights), "one weight per ISL"


def pct(xs, p):
    ys = sorted(xs)
    return ys[min(len(ys) - 1, int(round(p / 100 * (len(ys) - 1))))]


def wpct(pairs, p):
    pairs = sorted(pairs)
    tot = sum(w for _, w in pairs)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= p / 100 * tot:
            return v
    return pairs[-1][0]


mix = []
for isl, w in zip(isls, weights):
    fr = []
    try:
        for line in open(f"{wd}/isl{isl}/profile_export.jsonl"):
            r = json.loads(line)
            if r["metadata"].get("benchmark_phase") != "profiling" or r.get("error"):
                continue
            m = r["metrics"]
            fr.append(m["full_decode_duration"]["value"] / m["output_sequence_length"]["value"])
    except FileNotFoundError:
        print(f"ISL {isl}: no results")
        continue
    mix += [(x, w / len(fr)) for x in fr]
    print(f"ISL {int(isl)//1000:>4d}k  w={w:.2f}  n={len(fr):2d}  frITL p50={pct(fr,50):.3f}  "
          f"p90={pct(fr,90):.3f} ms  min={min(fr):.3f}  max={max(fr):.3f}")
if mix:
    p50, p90 = wpct(mix, 50), wpct(mix, 90)
    print(f"IXEST    weighted frITL p50={p50:.3f}  p90={p90:.3f} ms  est intvty p50={1000/p50:.2f}  p90={1000/p90:.2f}")
PY
