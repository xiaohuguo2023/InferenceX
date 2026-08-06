#!/bin/bash
# Reproduce the upstream K3 serve recipe on vllm/vllm-openai-rocm nightly
# cb8104839 (vLLM 0.26.1rc1.dev306). Differs from _serve_fp8_ms64.sh in the
# knobs that the upstream recipe sets: max-num-seqs 128, mnbt 16384,
# gpu-memory-utilization 0.93, --moe-backend aiter.
#
set -x
export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1 VLLM_USE_BREAKABLE_CUDAGRAPH=0
# Both flags are required, despite PR #50582 presenting the vLLM one as
# sufficient. vLLM's flag drives the interleaved w13 shuffle and GateMode.
# INTERLEAVE; the substitute it sets for AITER, AITER_BF16_FP8_MOE_BOUND=0, is
# dead code for K3 because aiter's per_1x32 dispatch tests
# `activation == ActivationType.Situv2` first and that branch reads
# AITER_SITUV2_A8W4 directly (aiter/fused_moe.py). Without it every MoE kernel
# silently drops to bf16 activations (flydsl_moe1_abf16_*) instead of the tuned
# a8w4 fp8 kernels.
export AITER_SITUV2_A8W4=1
export GPU_ARCHS=gfx950 HF_HUB_CACHE=/dev/shm/hf-cache HF_HOME=/dev/shm/hf-cache
export VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900
# NOTE: PYTORCH_HIP_ALLOC_CONF=expandable_segments:True is NOT usable here. aiter's
# custom all-reduce calls hipIpcGetMemHandle on its buffer, which fails with
# "invalid argument" on virtual-memory-backed expandable segments, and every rank
# dies during init.
#
# This nightly reserves ~10.7 GiB/GPU for the CUDA graph pool (the pre-nightly
# build reported 0.22 GiB for the identical 19 PIECEWISE + 11 FULL graphs).
# gpu_worker.py profiles that pool via profile_cudagraph_memory() but then
# discards the estimate unless VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS is set,
# so available_kv_cache_memory_bytes is computed without subtracting it and the
# instance overshoots the requested utilization by the pool size (measured:
# 53.26 GiB of KV allocated where its own log said 42.43 GiB would fit). With
# the flag on, the requested utilization is honest and can be raised again.
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${ESTIMATE_CUDAGRAPHS:-1}"

MERGED_GEMM_CSV=/opt/aiter-local/aiter/configs/merged_bf16_tuned_gemm.csv
if [ -z "${AITER_CONFIG_GEMM_BF16:-}" ] && [ -f "$MERGED_GEMM_CSV" ]; then
  export AITER_CONFIG_GEMM_BF16="$MERGED_GEMM_CSV"
fi

SHM_MODEL=/dev/shm/hf-cache/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569
NFS_MODEL="${MODEL_SRC:-/shared_nfs/models/Kimi-K3}"
if [ -f "${MODEL_PATH:-}/config.json" ]; then
  :
elif [ -f "$SHM_MODEL/config.json" ] && [ -f "$SHM_MODEL/preprocessor_config.json" ]; then
  MODEL_PATH="$SHM_MODEL"
elif [ -f "$NFS_MODEL/config.json" ]; then
  MODEL_PATH="$NFS_MODEL"
else
  echo "ERROR: Kimi-K3 weights not found (checked $SHM_MODEL and $NFS_MODEL)" >&2
  exit 1
fi

COMPILE_CFG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["+fused_rms_norm_gated"]}'
[ "${FUSED_RMS_NORM_GATED:-1}" = "0" ] && COMPILE_CFG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}'
# CUDAGRAPH_MODE / CAPTURE_SIZES exist to measure the graph pool's cost. The
# 19 PIECEWISE sizes are token counts (all reachable at mnbt 4096); only the 11
# FULL decode graphs are clamped to max_num_seqs, so trimming the list is a
# test of per-shape workspace cost, not of dead graphs.
if [ -n "${CUDAGRAPH_MODE:-}" ] || [ -n "${CAPTURE_SIZES:-}" ]; then
  COMPILE_CFG=$(CFG="$COMPILE_CFG" python3 -c '
import json, os
cfg = json.loads(os.environ["CFG"])
mode = os.environ.get("CUDAGRAPH_MODE")
if mode:
    cfg["cudagraph_mode"] = mode
sizes = os.environ.get("CAPTURE_SIZES")
if sizes:
    cfg["cudagraph_capture_sizes"] = [int(s) for s in sizes.replace(" ", "").split(",") if s]
print(json.dumps(cfg))
')
fi

# Defaults are the values the c16 agentic replay survives on this nightly, not
# the upstream recipe's (16384 / 128 / 0.93), all three of which were measured to
# abort with HSA_STATUS_ERROR_OUT_OF_RESOURCES while the KV cache was ~90% empty:
#   max_num_seqs   K3's hybrid layout forces attention block_size to 1536, and the
#                  MLA chunked-prefill workspace floor is max_num_seqs*block_size,
#                  overriding the intended 64k cap. 128 -> 196,608 tokens of
#                  context up-projection buffers; 64 -> 98,304.
#   gpu-mem-util   0.93 leaves ~6 GiB/GPU, which the a16w4 MoE survives but the
#                  a8w4 one does not (it peaked at 2292 of 2304 GiB and died).
#                  0.88 was found empirically and is really ~0.917 effective,
#                  because the CUDA graph pool escapes the KV sizing (see the
#                  VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS note above).
# max-num-batched-tokens is not the lever here (16384 and 4096 both aborted), but
# 4096 stays as the validated cap from _serve_fp8_ms64.sh.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
SERVE_LOG="${SERVE_LOG:-/workspace/serve_nightly_k3.log}"

# KV_CACHE_MEMORY (bytes) pins the KV allocation instead of deriving it from the
# utilization fraction. Needed on this nightly because the ~10.3 GiB/GPU graph
# pool escapes the KV sizing, so a requested 0.95 measured 0.986 effective; the
# server's own startup line reports the value that actually fits.
KV_CACHE_ARGS=()
[ -n "${KV_CACHE_MEMORY:-}" ] && KV_CACHE_ARGS=(--kv-cache-memory "$KV_CACHE_MEMORY")

setsid nohup vllm serve "$MODEL_PATH" --served-model-name moonshotai/Kimi-K3 \
  --host 0.0.0.0 --port 8888 --tensor-parallel-size 8 --async-scheduling \
  --distributed-executor-backend mp --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-num-seqs "$MAX_NUM_SEQS" --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  "${KV_CACHE_ARGS[@]}" \
  --trust-remote-code --load-format auto --moe-backend aiter \
  --kv-cache-dtype fp8 --attention-backend ROCM_AITER_MLA --mm-encoder-tp-mode data \
  --compilation-config "$COMPILE_CFG" \
  --enable-prefix-caching --no-disable-hybrid-kv-cache-manager \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 --enable-auto-tool-choice \
  --enable-prompt-tokens-details \
  --disable-uvicorn-access-log > "$SERVE_LOG" 2>&1 &
echo "serve PID $! (log: $SERVE_LOG)"
