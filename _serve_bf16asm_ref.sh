#!/bin/bash
set -x
# reference server flags + our bf16-ASM decode (ROCM_AITER_MLA -> mla_dec_stage1_bf16 asm)
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export SAFETENSORS_FAST_GPU=1
export AITER_SITUV2_A8W4=1
export AITER_BF16_FP8_MOE_BOUND=0
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export GPU_ARCHS=gfx950
export HF_HUB_CACHE=/dev/shm/hf-cache
export HF_HOME=/dev/shm/hf-cache
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900
MODEL_PATH=/dev/shm/hf-cache/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569
setsid nohup vllm serve "$MODEL_PATH" --served-model-name moonshotai/Kimi-K3 \
  --host 0.0.0.0 --port 8888 --tensor-parallel-size 8 --async-scheduling \
  --distributed-executor-backend mp --gpu-memory-utilization 0.95 \
  --max-num-seqs 128 --max-num-batched-tokens 4096 \
  --trust-remote-code --load-format auto --moe-backend auto \
  --kv-cache-dtype auto --attention-backend ROCM_AITER_MLA \
  --mm-encoder-tp-mode data \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --enable-prefix-caching --no-disable-hybrid-kv-cache-manager \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 --enable-auto-tool-choice \
  --disable-uvicorn-access-log \
  > /workspace/serve_bf16asm_ref.log 2>&1 &
echo "serve PID $!"
