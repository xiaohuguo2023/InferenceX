#!/usr/bin/env bash
# Container-side DSpark recipe checks (§6–§7). Called by _k3_dspark_recipe_test_node.sh.
set -uo pipefail
PORT="${PORT:-8890}"
SERVE_LOG="${SERVE_LOG:-/workspace/serve_k3_bench_spec2.log}"
BENCH_ROOT="${BENCH_ROOT:-/workspace/k3_dspark_fp8asm_bench}"
rc=0

echo "========== recipe health =========="
curl -sf "http://localhost:${PORT}/health" && echo " health OK" || { echo "!! health FAIL"; rc=1; }

echo "========== recipe coherence smoke =========="
curl -sf "http://localhost:${PORT}/v1/chat/completions" -H 'content-type: application/json' -d "{
  \"model\":\"Kimi-K3\",
  \"messages\":[{\"role\":\"user\",\"content\":\"In one sentence, what is speculative decoding?\"}],
  \"max_tokens\":64}" | python3 -c "
import sys, json
r = json.load(sys.stdin)
t = r['choices'][0]['message'].get('content') or r['choices'][0]['message'].get('reasoning_content') or ''
print(t[:200] if t else '!! empty response')
sys.exit(0 if t.strip() else 1)
" || rc=1

echo "========== recipe cudagraph capture (no eager) =========="
if grep -qiE 'enforce.eager|eager mode' "$SERVE_LOG" 2>/dev/null; then
  echo "!! eager detected in serve log"; rc=1
fi
grep -iE 'Capturing|cudagraph|PIECEWISE|FULL' "$SERVE_LOG" 2>/dev/null | tail -8 || true

echo "========== recipe acceptance length =========="
grep -iE 'acceptance|accepted|draft.accept' "$SERVE_LOG" 2>/dev/null | tail -10 || echo "(no acceptance lines yet)"

echo "========== recipe aiperf sweep =========="
export PORT
bash /workspace/_bench_k3_dspark_fp8asm.sh || rc=1

if [ -d "$BENCH_ROOT" ]; then
  echo "========== aiperf summary =========="
  python3 <<PY
import glob, json, re
for p in sorted(glob.glob("${BENCH_ROOT}/concurrency_*__requests_*/profile_export_aiperf.json")):
    d = json.load(open(p))
    c = re.search(r"concurrency_(\\d+)__requests", p)
    c = c.group(1) if c else "?"
    def g(k): v=d.get(k,{}); return v.get("avg") if isinstance(v,dict) else None
    ttft, itl, out = g("time_to_first_token"), g("inter_token_latency"), g("output_token_throughput")
    print(f"  c{c}: TTFT={ttft:.0f}ms ITL={itl:.1f}ms out={out:.0f} tok/s")
PY
fi

exit "$rc"
