#!/bin/bash
# Fixed temp=0 acceptance workload for the DSpark non-causal verify comparison.
# Sends N diverse chat generations (temp=0, max_tokens=350) concurrently, then
# reads the cumulative vllm:spec_decode_* counters and prints mean accept length.
# Run against a FRESH serve (counters ~0) so the cumulative read == this workload.
#   PORT=8890 bash _dspark_accept_workload.sh
set -uo pipefail
PORT="${PORT:-8890}"
MODEL="${MODEL:-Kimi-K3}"
MAXTOK="${MAXTOK:-350}"

prompts=(
"Explain gradient descent in three sentences."
"Write a short paragraph about why the sky is blue."
"Summarize the plot of Romeo and Juliet in five sentences."
"Describe how a hash table works and its average time complexity."
"Explain the difference between TCP and UDP."
"Write a Python function that returns the nth Fibonacci number, with a brief comment."
"Explain what a transformer neural network is to a beginner."
"Describe the water cycle step by step."
"Explain the concept of recursion using a simple example."
"What causes inflation in an economy? Answer in a short paragraph."
"Explain how vaccines train the immune system."
"Describe the process of photosynthesis in plants."
)

echo "workload: ${#prompts[@]} prompts x2 = $(( ${#prompts[@]} * 2 )) gens, temp=0, max_tokens=$MAXTOK, port=$PORT"
tmp=$(mktemp -d)
n=0
for rep in 1 2; do
  for p in "${prompts[@]}"; do
    n=$((n+1))
    body=$(python3 -c "import json,sys; print(json.dumps({'model':'$MODEL','messages':[{'role':'user','content':sys.argv[1]}],'max_tokens':$MAXTOK,'temperature':0}))" "$p")
    curl -s -m 300 "http://localhost:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' -d "$body" -o "$tmp/g_$n.json" &
  done
done
wait
ok=0; toks=0
for f in "$tmp"/g_*.json; do
  t=$(python3 -c "import json;d=json.load(open('$f'));print(d['usage']['completion_tokens'])" 2>/dev/null) && { ok=$((ok+1)); toks=$((toks+t)); }
done
echo "completed OK: $ok/$n gens ; total output tokens: $toks"

# scrape cumulative spec_decode counters
m=$(curl -s "http://localhost:$PORT/metrics")
get(){ echo "$m" | grep -E "vllm:$1\{" | grep -v '#' | awk '{print $2}'; }
drafts=$(get spec_decode_num_drafts_total)
dtoks=$(get spec_decode_num_draft_tokens_total)
acc=$(get spec_decode_num_accepted_tokens_total)
echo "drafts=$drafts draft_tokens=$dtoks accepted=$acc"
python3 -c "
d=float('$drafts'); dt=float('$dtoks'); a=float('$acc')
if d>0:
    mal=a/d+1
    pertok=a/dt if dt>0 else 0
    print(f'MEAN ACCEPT LEN = {mal:.3f}  (per-token accept = {pertok*100:.1f}%)  [N per draft = {dt/d:.2f}]')
else:
    print('no drafts recorded')
"
rm -rf "$tmp"
