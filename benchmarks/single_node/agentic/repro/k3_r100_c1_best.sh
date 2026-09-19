#!/usr/bin/env bash
# Best measured conc-1 IX agentic arm on the ROCm 10 image.
#
#   iv p90 142.7   tok/s/gpu 1558.9   (210 reqs, OSL 1749, ISL 213799, 3602 s)
#
# vs the untuned baseline on the same image (136.4 / 1526.5) and the ROCm 7.2.3
# reference abroute_asm_c1 (132.4 / 1478.1). Noise floor across four stored runs
# is ~0.5% on iv p90 and ~0.8% on throughput, so treat <1% as no change.
#
# Image: vllm/vllm-openai-rocm:nightly-rocm100-dee37d89115db4c94a820a79a78a7828e141c910
#
# Container must have: /dev/shm bind-mounted (weights), the repo at /workspace,
# and /root/.cache/huggingface/datasets populated (transplant it from a working
# container; aiperf reads the corpus from there, not from HF_HUB_CACHE).
#
# PREFLIGHT, do not skip. Six boots were lost to these:
#   - box free and zero zombies      (sleep-infinity init never reaps)
#   - whole patch set applies        (dry-run, reverse-apply means already applied)
#   - tokenizer resolves by HF id    (vLLM takes a path and never notices a stale ref)
#   - dataset loads offline
set -eo pipefail

# --- shape: DCP8, K=7, synthetic AL 3.84, LMCache. CONC=1 makes the recipe
# --- derive SPEC_NUM_TOKENS=7 / SYNTHETIC_ACCEPT_LEN=3.84 / retention 1536.
export MODEL=moonshotai/Kimi-K3
export MODEL_PATH=/dev/shm/hf-cache/models--moonshotai--Kimi-K3/snapshots/f831ab66814297da540d832a5235f8e904f29d06
export MODEL_PREFIX=kimik3          # selects the 062126 corpus; unset silently picks another
export TP=8 EP_SIZE=1 DCP_SIZE=8 CONC=1
export KV_OFFLOADING=dram
export KV_OFFLOAD_BACKEND=lmcache
export LMCACHE_VERSION=0.5.6.dev40+rocm7.2   # only +rocm7.2 wheels exist; they work on ROCm 10
export TOTAL_CPU_DRAM_GB=803
export DURATION=3600                # CI duration. 900 s is not comparable to anything.
export EVAL_ONLY=false

# --- HF: HF_HOME must NOT equal HF_HUB_CACHE. datasets resolves
# --- HF_DATASETS_CACHE from HF_HOME and lands somewhere nonexistent.
export HF_HOME=/dev/shm/hf-cache
export HF_HUB_CACHE=/dev/shm/hf-cache
export HF_DATASETS_CACHE=/root/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export AIPERF_UV_CACHE_DIR=/dev/shm/aiperf-uv-cache

# --- the two levers this arm adds over the untuned baseline ---
# multistream on (K3_AMD_GATE_MULTI_STREAM): measured neutral at conc-1 on the
# 7.2.3 image, kept here because the arm was measured with it on.
export K3_AMD_GATE_MULTI_STREAM=1
# inductor graph partition unlocks fuse_rope_kvcache_cat_mla (the veto in
# compilation.py only lifts when partitioning is on) and is what lets the
# compiled DSpark draft from #56664 actually engage.
export COMPILATION_EXTRA_JSON=',"use_inductor_graph_partition":true,"pass_config":{"fuse_rope_kvcache_cat_mla":true}'

# --- GEMM: no tuned CSV. Both shipped CSVs carry solidx/kernel ids from other
# --- aiter builds and fault this image (hipBLASLt INVALID_VALUE at the profile
# --- run; opus "Kernel id 205 not found" at speculator capture). A header-only
# --- file defeats the recipe's ${VAR:-default}, which would restore the 7.2.3 CSV.
if [ ! -s /dev/shm/_empty_tuned_gemm.csv ]; then
    head -1 "$(dirname "$0")/../../../../patches/k3-dcp8/aiter/merged_bf16_tuned_gemm_rocm10_retuned.csv" \
        > /dev/shm/_empty_tuned_gemm.csv
fi
export AITER_CONFIG_GEMM_BF16=/dev/shm/_empty_tuned_gemm.csv

export RESULT_DIR=${RESULT_DIR:-/workspace/results_ixci/r100_c1_best}
export RESULT_FILENAME=$(basename "$RESULT_DIR")
mkdir -p "$RESULT_DIR"

echo "[launch] $(date) conc=$CONC dcp=$DCP_SIZE dur=$DURATION ms=$K3_AMD_GATE_MULTI_STREAM"
exec bash "$(dirname "$0")/../kimik3_fp4_mi355x_mtp.sh"
