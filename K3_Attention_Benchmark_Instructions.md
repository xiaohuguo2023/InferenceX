# K3 Attention Benchmark Instructions

## Objective

The goal of this benchmark is to compare different K3 attention implementations and speculative decoding configurations.

## Reference Server Command

### Environment Variables

```bash
export VLLM_ROCM_USE_AITER=1
export SAFETENSORS_FAST_GPU=1
export AITER_SITUV2_A8W4=1
export AITER_BF16_FP8_MOE_BOUND=0
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
```

### Launch Command

```bash
MODEL_PATH=/data/Kimi-K3

vllm serve "$MODEL_PATH" \
  --served-model-name Kimi-K3 \
  --trust-remote-code \
  --moe-backend auto \
  --tensor-parallel-size 8 \
  --load-format auto \
  --max-model-len 1048576 \
  --gpu-memory-utilization 0.93 \
  --mm-encoder-tp-mode data \
  --max-num-batched-tokens 16384 \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k3 \
  --reasoning-parser kimi_k3 \
  --max-num-seqs 128 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["+fused_rms_norm_gated"]}'
```

---

## Validation Requirement

- Full GSM8K 1319 questions
- nshot=5
- concurrency=64

Results that fail validation are considered invalid.

---

## DSpark Configurations

### num_speculative_tokens=2

```bash
--speculative-config '{"model":"Inferact/Kimi-K3-DSpark","num_speculative_tokens":2,"method":"dspark","attention_backend":"TRITON_MLA","draft_sample_method":"probabilistic","rejection_sample_method":"block"}'
```

### num_speculative_tokens=7

```bash
--speculative-config '{"model":"Inferact/Kimi-K3-DSpark","num_speculative_tokens":7,"method":"dspark","attention_backend":"TRITON_MLA","draft_sample_method":"probabilistic","rejection_sample_method":"block"}'
```

---

## Benchmark Command

```bash
aiperf profile \
  --model /model_weights \
  --tokenizer /model_weights \
  --tokenizer-trust-remote-code \
  --url http://127.0.0.1:8000 \
  --api-key EMPTY \
  --endpoint-type chat \
  --streaming \
  --use-server-token-count \
  --num-prefix-prompts 1 \
  --prompt-prefix-length 63911 \
  --synthetic-input-tokens-mean 4089 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 350 \
  --output-tokens-stddev 0 \
  --extra-inputs ignore_eos:true \
  --extra-inputs min_tokens:350 \
  --extra-inputs max_tokens:350 \
  --warmup-request-count 16 \
  --sweep-type zip \
  --concurrency 48,32,24,16,12,8,4,2,1 \
  --request-count 240,160,120,80,60,40,20,10,5
```

# What to report
- Full GSM8K verification results
- Cache hit rate
- TTFT P50, P90 and Mean
- ITL P50, P90 and Mean
- Total throughput (tok/s/GPU)
- Average acceptance length (for spec-decoding only)