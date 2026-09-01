#!/bin/bash
# GSM8K validation for K3 (per K3_Attention_Benchmark_Instructions.md):
#   1319 questions, num_fewshot=5, concurrency=64, against the served endpoint.
# Uses lm-eval-harness local-chat-completions (chat template applied server-side,
# which K3 as a THINKING model needs). We report flexible-extract accuracy because
# K3 emits <think>...</think> before the answer, which trips strict-match / the
# default gsm8k stop-tokens; hence large max_tokens too.
#
#   LIMIT=20 bash _gsm8k_k3.sh   # quick smoke before the full run
set -uo pipefail
PORT="${PORT:-8888}"
CONC="${CONC:-64}"
NSHOT="${NSHOT:-5}"
MAXTOK="${MAXTOK:-3072}"
LIMIT="${LIMIT:-}"        # empty => full 1319
OUT="${OUT:-/workspace/gsm8k_k3_baseline}"
MODEL_NAME="${MODEL_NAME:-Kimi-K3}"

curl -sf -m5 "http://localhost:$PORT/health" >/dev/null 2>&1 || { echo "!! server not up on :$PORT"; exit 1; }
mkdir -p "$OUT"
LIM_ARG=(); [ -n "$LIMIT" ] && LIM_ARG=(--limit "$LIMIT")

echo "== GSM8K nshot=$NSHOT conc=$CONC limit=${LIMIT:-full(1319)} maxtok=$MAXTOK =="
lm_eval --model local-chat-completions \
  --model_args "model=$MODEL_NAME,base_url=http://127.0.0.1:$PORT/v1/chat/completions,num_concurrent=$CONC,max_retries=3,tokenized_requests=False,timeout=1800" \
  --tasks gsm8k --num_fewshot "$NSHOT" \
  --gen_kwargs "max_tokens=$MAXTOK,temperature=0" \
  --apply_chat_template \
  "${LIM_ARG[@]}" \
  --output_path "$OUT" 2>&1 | tee "$OUT/gsm8k_run.log"
echo "== results dir: $OUT =="
