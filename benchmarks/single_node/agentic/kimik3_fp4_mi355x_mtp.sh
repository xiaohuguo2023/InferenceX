#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for Kimi-K3 MXFP4 on MI355X / MI350X (gfx950)
# using vLLM.
#
# The server command is the AMD reference `vllm serve` for this model, i.e. the
# upstream vLLM recipe's amd block (vllm-project/recipes,
# https://recipes.vllm.ai/moonshotai/Kimi-K3) as run in practice:
#
#   --trust-remote-code --moe-backend auto --tensor-parallel-size 8
#   --load-format auto --gpu-memory-utilization 0.95 --mm-encoder-tp-mode data
#   --max-num-seqs 128 --max-num-batched-tokens 4096 --enable-auto-tool-choice
#   --tool-call-parser kimi_k3 --reasoning-parser kimi_k3
#
# with env VLLM_ROCM_USE_AITER=1 SAFETENSORS_FAST_GPU=1 AITER_SITUV2_A8W4=1
# AITER_BF16_FP8_MOE_BOUND=0 VLLM_USE_BREAKABLE_CUDAGRAPH=0.
#
# K3 is a 2.8T-parameter natively-multimodal MoE (896 routed experts, 16/token
# plus shared) on Kimi Delta Attention, gated MLA and Attention Residuals, with
# a 1M-token native context.
#
# TP=8 ONLY. The MXFP4 checkpoint is 1.561 TB decimal (1.420 TiB, 96
# safetensors), ~195 GB/GPU across 8 GPUs of the 288 GB part; TP=4 would need
# ~390 GB/GPU and cannot load. Upstream strategy_min_gpus agrees (single_node_tp
# and multi_node_tep both 8, DEP 16+), which is why there is no DP-attention arm.
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION,
#   EP_SIZE
#
# Perf-search knobs. Each defaults to the reference command's value, so an
# otherwise-unset run reproduces the reference exactly:
#   GPU_MEM_UTIL             0.95   (reference)
#   MAX_NUM_BATCHED_TOKENS   8192   (default)
#   AITER_A8W4               1      (reference; 0 = aiter a16w4 MoE path)
#   LANGUAGE_MODEL_ONLY      true   
#   KV_CACHE_DTYPE           fp8    (default for every arm; =auto for a bf16 A/B)
#   KV_BLOCK_SIZE            unset  (unset -> vLLM sizes the page; 128 under fp8)
#   MAX_MODEL_LEN            1M     
#   SPEC_DECODE              true   (this is the _mtp DSpark recipe; =false for a no-spec A/B)
#   SPEC_NUM_TOKENS          2      (DSpark draft length; validated by the _mtp config)

source "$(dirname "$0")/../../benchmark_lib.sh"

wait_for_amd_gpu_clean

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

if [ "$TP" -ne 8 ]; then
    echo "Error: Kimi-K3 MXFP4 is a 1.56 TB checkpoint and only fits at TP=8 on" >&2
    echo "       288 GB gfx950 parts (~195 GB/GPU). Got TP=$TP." >&2
    exit 1
fi

# ROCR/HIP visibility for vLLM 0.14+
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

# `hf download` creates the target dir if missing and is itself idempotent. The
# 1.56 TB checkpoint is normally pre-staged, so these calls are a no-op there.
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
VLLM_PKG="$(python3 -c 'import os, vllm; print(os.path.dirname(vllm.__file__))')"

# ---- Source patches ---------------------------------------------------------
# This recipe ships source patches against the pinned nightly image; applying
# them here keeps a CI run and a manual repro on the same code.
#   patches/k3-perf/  decode-glue kernel-launch elimination (applies to the
#                     non-DCP path too, so it is unconditional)
#   patches/k3-dcp8/  DCP8 + DSpark on the fp8 asm cprr verify route. Also
#                     applied unconditionally: every hunk is gated on
#                     decode_context_parallel_size > 1, so the DCP1 arms run
#                     byte-identical code to the DCP8 arms and the two are
#                     comparable.
# Each apply is idempotent -- a *successful* reverse dry-run means it is in.
apply_vllm_patch() {
    local p="$1"
    if [ ! -f "$p" ]; then echo "!! missing patch: $p" >&2; return 1; fi
    if patch --dry-run -R -p1 -d "$VLLM_PKG" <"$p" >/dev/null 2>&1; then
        echo "patch already applied: ${p#"$REPO_ROOT"/}"
        return 0
    fi
    patch -p1 -d "$VLLM_PKG" <"$p"
}
for _p in envs utils kda linear; do
    apply_vllm_patch "$REPO_ROOT/patches/k3-perf/vllm/$_p.patch"
done
for _p in scheduler config cp_common speculator rocm_aiter_mla speculative_draft_dcp; do
    apply_vllm_patch "$REPO_ROOT/patches/k3-dcp8/vllm/$_p.patch"
done

# aiter's tuned bf16 GEMM config. Without it every tuned-shape lookup misses,
# the unquantized-GEMM dispatch in patches/k3-perf/vllm/utils.patch loses the
# crossover it was justified against, and the agentic launcher has been seen to
# HSA-fault outright. The _rocm10_retuned variant is the one tuned on this
# image's ROCm; merged_bf16_tuned_gemm.csv is kept for the older pin.
export AITER_CONFIG_GEMM_BF16="${AITER_CONFIG_GEMM_BF16:-$REPO_ROOT/patches/k3-dcp8/aiter/merged_bf16_tuned_gemm_rocm10_retuned.csv}"

# ---- DSpark draft causality -------------------------------------------------
# The published draft checkpoint ships without `dflash_config`, and
# qwen3_dflash._dflash_layer_causal then falls through all three of its
# resolution steps and reports every layer non-causal. That routes the draft off
# the fp8 asm MLA path ("Selected backend ROCM_AITER_MLA is not valid ...
# non-causal attention not supported") and, under DCP, is refused outright. The
# non-causal draft is also measurably worse: +44% ITL and 3.2x the p90 tail at
# concurrency 24, for +1.3% acceptance.
#
# This is a checkpoint edit, not a vLLM patch, so it does not survive a
# re-download -- hence running it on every launch, after the download.
hf download Inferact/Kimi-K3-DSpark >/dev/null
python3 - <<'PY'
import glob, json, os

REL = "models--Inferact--Kimi-K3-DSpark/snapshots/*/config.json"
roots = [
    os.environ.get("HF_HUB_CACHE"),
    os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME") else None,
    os.path.expanduser("~/.cache/huggingface/hub"),
    "/dev/shm/hf-cache",
]
found = 0
for root in filter(None, dict.fromkeys(roots)):
    for cfg in glob.glob(os.path.join(root, REL)):
        found += 1
        c = json.load(open(cfg))
        if (c.get("dflash_config") or {}).get("causal") is True:
            print("draft already causal:", cfg)
            continue
        c["dflash_config"] = {"causal": True}
        # config.json is a symlink into the content-addressed blob store;
        # writing through it would corrupt the blob for every snapshot.
        if os.path.islink(cfg):
            os.unlink(cfg)
        json.dump(c, open(cfg, "w"), indent=2)
        print("forced draft causal:", cfg)
if not found:
    raise SystemExit("!! DSpark draft not staged -- cannot force causality")
PY

rocm-smi || true
amd-smi || true

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

# ---- Reference env block ----------------------------------------------------
export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm
export VLLM_ROCM_USE_AITER=1
export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1
export AITER_SITUV2_A8W4=1
export AITER_BF16_FP8_MOE_BOUND=0
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export AITER_QUICK_REDUCE_QUANTIZATION=INT4

# Workaround for MEC FW <177 RCCL memory reclaim issue (shared with the other
# gfx950 recipes in this tree).
mec_version=$(rocm-smi --showfw 2>/dev/null | grep MEC | head -n 1 | awk '{print $NF}')
if [[ "$mec_version" == "" || ${mec_version:-0} -lt 177 ]]; then
    export HSA_NO_SCRATCH_RECLAIM=1
fi

# 2.8T of weights off a shared/NFS mount takes far longer than the default.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"

# Long agentic turns against a 1M context: keep the client from timing out
# mid-request while the server is prefill-bound.
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

SERVER_PID=""
LMCACHE_PID=""

cleanup_agentic_services() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$SERVER_PID" "vLLM server" 60
    stop_background_process_tree "$LMCACHE_PID" "LMCache server"
    exit "$exit_code"
}
trap cleanup_agentic_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ---- KV offload -------------------------------------------------------------
# TOTAL_CPU_DRAM_GB is the aggregate host-DRAM budget the matrix generator
# derives from dram-utilization and the runner's available-cpu-dram-mib, capped
# at the 3,095,781 MiB (3 TB decimal) agentic limit. Per
# benchmarks/single_node/agentic/README.md it must be consumed as given and
# never replaced with a model-specific constant.
OFFLOAD_ARGS=()

if agentic_kv_offload_enabled; then
case "${KV_OFFLOAD_BACKEND:-}" in
  vllm-simple)
    require_agentic_kv_offload_backend "$KV_OFFLOAD_BACKEND"
    CPU_BYTES_PER_RANK=$(( TOTAL_CPU_DRAM_GB * 1000 * 1000 * 1000 / TP ))
    # Identical prefixes must hash to identical block keys across ranks.
    export PYTHONHASHSEED=42
    SIMPLE_LAZY_OFFLOAD="${SIMPLE_LAZY_OFFLOAD:-false}"
    OFFLOAD_ARGS=(
        --kv-transfer-config
        "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use_per_rank\":$CPU_BYTES_PER_RANK,\"lazy_offload\":$SIMPLE_LAZY_OFFLOAD}}"
    )
    echo "SimpleCPUOffloadConnector: ${CPU_BYTES_PER_RANK} B/rank x ${TP} ranks, lazy_offload=$SIMPLE_LAZY_OFFLOAD"
    ;;
      lmcache)
    require_agentic_kv_offload_backend "$KV_OFFLOAD_BACKEND"

    # Keep the image's tested torch/ROCm stack and install only LMCache's
    # missing runtime dependencies, same as the MiniMax-M3 lmcache arm.
    # The nightly-rocm asset index is not immutable: it carries only the
    # current nightly, so a dev pin stops resolving as soon as LMCache cuts
    # the next one. dev60 is already gone ("No matching distribution found");
    # dev89 is what the index serves today and exposes the same server CLI
    # (--l1-size-gb / --chunk-size / --separate-object-groups /
    # --supported-transfer-mode). Re-pin whenever this stops resolving.
    LMCACHE_VERSION="0.5.5.dev89+rocm7.2"
    LMCACHE_ROCM_INDEX="https://github.com/LMCache/LMCache/releases/expanded_assets/nightly-rocm"
    agentic_pip_install --quiet --no-cache-dir --no-deps \
        "sortedcontainers==2.4.0" \
        "opentelemetry-exporter-prometheus==0.61b0" \
        "cupy-rocm-7-0==14.1.1" \
        "lmcache==${LMCACHE_VERSION}" --find-links "$LMCACHE_ROCM_INDEX"

    # LMCache 0.5.5's transfer-channel layer eagerly imports the Mooncake
    # backend (mooncake_te_impl.py -> `from mooncake.engine import
    # TransferEngine`), whose native .so resolves all of its DT_NEEDED libs at
    # import. The vLLM ROCm image ships none of them, so the import sanity
    # check below (and the LMCache server) would otherwise fail with
    # "ImportError: lib*.so: cannot open shared object file" (first libglog,
    # then libjsoncpp, ...). Provision Mooncake's full runtime lib set from the
    # distro before importing. apt-get install is idempotent, so run it
    # whenever any of the libs is still missing rather than gating on one.
    LMCACHE_NATIVE_LIBS=(libglog.so.0 libjsoncpp.so.25 libibverbs.so.1 librdmacm.so.1 libnuma.so.1)
    for lib in "${LMCACHE_NATIVE_LIBS[@]}"; do
        if ! ldconfig -p | grep -q "$lib"; then
            apt-get update
            apt-get install -y \
                libgoogle-glog0v5 libjsoncpp25 libibverbs1 librdmacm1 libnuma1
            break
        fi
    done
    python3 -c \
        "import cupy; import lmcache.integration.vllm.lmcache_mp_connector; import opentelemetry.exporter.prometheus" \
        >/dev/null

    # One MP server for the node, per the Kimi-K3 recipe
    # (docs.lmcache.ai/recipes/kimi_k3.html), with --chunk-size sized for
    # THIS stack rather than the recipe's CUDA-path 768: the connector
    # requires the chunk to be a multiple of every engine KV group's
    # tokens_per_block, and the hybrid KDA/MLA layout here registers
    # attention groups at 1536 ("Setting attention block size to 1536",
    # run 31644990546) plus a KDA state group at 3072 (run 31645828378),
    # so 3072 is the minimum valid chunk. The multi-group layout also
    # requires one object group per sliding-window size:
    # --separate-object-groups.
    #
    # Decode context parallelism scales the attention group's block size by
    # DCP_SIZE (1536 -> 12288 at DCP8), and the connector then rejects the
    # 3072 chunk: "LMCache chunk size 3072 must be a multiple of 12288 (the
    # vLLM block size scaled by decode_context_parallel_size)". So under DCP
    # the chunk is the DCP-scaled attention block, rounded up to stay a
    # multiple of the KDA group's 3072.
    LMCACHE_CHUNK_SIZE=3072
    if [ "${DCP_SIZE:-1}" -gt 1 ]; then
        LMCACHE_CHUNK_SIZE=$(( 1536 * ${DCP_SIZE:-1} ))
        if [ $(( LMCACHE_CHUNK_SIZE % 3072 )) -ne 0 ]; then
            LMCACHE_CHUNK_SIZE=$(( LMCACHE_CHUNK_SIZE * 2 ))
        fi
    fi
    # DCP shards decode KV across the TP ranks, so the GPU transfer pool needs
    # one worker per rank; with a single worker all eight shards of every chunk
    # queue behind one transfer. A non-DCP arm has one shard and needs one.
    LMCACHE_MAX_GPU_WORKERS=1
    if [ "${DCP_SIZE:-1}" -gt 1 ]; then
        LMCACHE_MAX_GPU_WORKERS="$DCP_SIZE"
    fi
    LMCACHE_PORT=6555
    LMCACHE_HTTP_PORT=8090
    LMCACHE_LOG="$RESULT_DIR/lmcache_server.log"

    LMCACHE_L1_SIZE_GB="$TOTAL_CPU_DRAM_GB"

    LMCACHE_CMD=(
        lmcache server
        --host 127.0.0.1
        --port "$LMCACHE_PORT"
        --http-host 127.0.0.1
        --http-port "$LMCACHE_HTTP_PORT"
        --l1-size-gb "$LMCACHE_L1_SIZE_GB"
        --l1-init-size-gb 10
        --chunk-size "$LMCACHE_CHUNK_SIZE"
        --separate-object-groups
        --enable-extra-logging
        --extra-logging-interval 30
        --max-cpu-workers 8
        --max-gpu-workers "$LMCACHE_MAX_GPU_WORKERS"
        --eviction-policy LRU
        --supported-transfer-mode lmcache_driven
        --shm-name ""
    )
    append_command "$RESULT_DIR/lmcache_command.txt" "${LMCACHE_CMD[@]}"
    "${LMCACHE_CMD[@]}" > "$LMCACHE_LOG" 2>&1 &
    LMCACHE_PID=$!
    wait_for_ready \
        --endpoint "http://127.0.0.1:${LMCACHE_HTTP_PORT}/healthcheck" \
        --log "$LMCACHE_LOG" \
        --pid "$LMCACHE_PID" \
        --sleep-interval 1 \
        --timeout 600

    # 100k-330k-token agentic prefixes make single retrieves large; use the
    # same MQ timeout headroom as the MiniMax-M3 arm.
    OFFLOAD_ARGS=(
        --kv-transfer-config
        "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_connector_module_path\":\"lmcache.integration.vllm.lmcache_mp_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.port\":$LMCACHE_PORT,\"lmcache.mp.mq_timeout\":6000.0}}"
    )
    ;;
    *)
    echo "Error: unsupported KV_OFFLOAD_BACKEND='$KV_OFFLOAD_BACKEND' (expected vllm-simple or lmcache)" >&2
    exit 1
    ;;
esac
fi

# ---- LLM server  ------------------------------------------------------------

# ---- Parallelism ------------------------------------------------------------
EP_ARGS=()
if [ "$EP_SIZE" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
fi

# ---- Speculative / Util------------------------------------------------------
# Draft depth follows the committed golden curve, at the same split points the
# ATOM companion recipe and the B300 recipe use, so a point here has a
# like-for-like counterpart there: 7 draft tokens -> AL 3.84 at concurrency
# 1-4, 3 draft tokens -> AL 3.00 from concurrency 8 up. Comparing across draft
# depths is not meaningful, so do not move these without moving the companions.
#
# PREFIX_CACHE_RETENTION_INTERVAL is the KDA/Mamba checkpoint spacing. vLLM's
# default is 0 (one checkpoint per prefix), which quantises every prefix-cache
# hit to the whole prefix; spacing checkpoints one per block recovers 2 pp of
# hit rate and ~12% of p90 interactivity at concurrency 1. The legal values
# differ by arm because the validator compares against scheduler_block_size:
#   conc 1-4 (DCP1)   -> 1536, the attention/KDA block size. Dense.
#   conc 8+  (DCP8)   -> 12288, since DCP8 scales the attention group by 8.
#                        1536 also works there, but only on a vLLM carrying
#                        the fix that validates against cache_hit_alignment_
#                        tokens rather than scheduler_block_size.
# Dense costs ~4.5x the block pool per cached token, so the conc 8+ value must
# be re-swept against the working set before it is lowered.
case "$CONC" in
    # Latency floor: everything GPU-resident, deepest published draft.
    1|2|4)
        SYNTHETIC_ACCEPT_LEN=3.84
        SPEC_NUM_TOKENS=7
        GPU_MEM_UTIL=0.9
        MAX_NUM_BATCHED_TOKENS=16384
        PREFIX_CACHE_RETENTION_INTERVAL=1536
        ;;
    # Decode is KV-bandwidth-bound over 100k+ token contexts from here up, so
    # configs/amd-master.yaml pairs these with dcp-size 8 and the DRAM tier.
    8|10|12|14)
        SYNTHETIC_ACCEPT_LEN=3.00
        SPEC_NUM_TOKENS=3
        GPU_MEM_UTIL=0.9
        MAX_NUM_BATCHED_TOKENS=8192
        PREFIX_CACHE_RETENTION_INTERVAL=12288
        ;;
    *)
        # No draft: past concurrency ~16 the batch already saturates decode, so
        # a draft only spends bandwidth verifying tokens the batch would have
        # produced anyway. 0.9/8192 rather than the older 0.85/4096 -- with the
        # draft gone the freed memory is better spent on KV, and the larger
        # prefill batch is what these concurrencies are bound by.
        SPEC_NUM_TOKENS=0
        GPU_MEM_UTIL=0.9
        MAX_NUM_BATCHED_TOKENS=8192
        PREFIX_CACHE_RETENTION_INTERVAL=12288
        ;;
esac
PREFIX_CACHE_RETENTION_INTERVAL="${PREFIX_CACHE_RETENTION_INTERVAL_OVERRIDE:-$PREFIX_CACHE_RETENTION_INTERVAL}"
MAX_NUM_BATCHED_TOKENS="${MNBT_OVERRIDE:-$MAX_NUM_BATCHED_TOKENS}"

# The DSpark draft runs its own MLA attention, separate from the target's.
# Under decode context parallelism the draft's KV is sharded across the CP
# ranks too, and TritonMLAMetadataBuilder refuses that path outright --
# "TritonMLAMetadataBuilder does not support non-causal draft MLA attention
# for DSpark with decode context parallelism" -- so the draft has to use the
# same DCP-capable backend as the target. There is no Triton fallback here:
# a DCP arm with a draft needs an image carrying the fp8 cprr assembly kernel,
# which is what patches/k3-dcp8/aiter/ supplies.
#
# ROCM_AITER_MLA everywhere, DCP or not. It needs the draft forced causal (see
# the checkpoint edit above); with that in place it beats the non-causal Triton
# draft by 44% ITL at concurrency 24, and it is the path the GEMM tuning and the
# asm cprr verify route were built for.
DRAFT_ATTN_BACKEND="${DCP_ATTN_BACKEND:-ROCM_AITER_MLA}"

SPEC_ARGS=()
if [ "$SPEC_NUM_TOKENS" -gt 0 ]; then
if [ "${EVAL_ONLY:-false}" = "true" ]; then
    SPEC_ARGS=(
        --speculative-config
        "{\"model\":\"Inferact/Kimi-K3-DSpark\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS,\"method\":\"dspark\",\"attention_backend\":\"$DRAFT_ATTN_BACKEND\",\"kv_cache_dtype\":\"fp8\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\": \"block\"}"
    )
else
    SPEC_ARGS=(
        --speculative-config
        "{\"model\":\"Inferact/Kimi-K3-DSpark\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS,\"method\":\"dspark\",\"attention_backend\":\"$DRAFT_ATTN_BACKEND\",\"kv_cache_dtype\":\"fp8\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": $SYNTHETIC_ACCEPT_LEN}"
    )
    fi
fi

# ---- HIP graph ------------------------------------------------------------
MAX_NUM_SEQS=$((2 * CONC))
MAX_CUDAGRAPH_CAPTURE_SIZE=$((MAX_NUM_SEQS * (1 + SPEC_NUM_TOKENS)))
CUDAGRAPH_CAPTURE_SIZES="$(seq -s, 2 "$MAX_CUDAGRAPH_CAPTURE_SIZE")"
# FULL_DECODE_ONLY, not FULL_AND_PIECEWISE. On the pinned image the decode step
# is already 100% FULL-captured, and PIECEWISE only adds a second set of graphs
# to capture and the boot time to capture them.
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"
# At mode 3 every custom op not listed here defaults off and is decomposed for
# inductor, so this list is the only thing that decides which fused kernels run.
# +situ_and_mul: K3's hidden_act is "situ" on all 92 shared-expert layers.
# Decomposed, inductor upcasts to fp32 with temporaries; the shipped C++ op is
# 43% faster at the concurrency-1 shape and is already below a bare clone.
CUSTOM_OPS_JSON="${CUSTOM_OPS_JSON:-\"+fused_rms_norm_gated\",\"+situ_and_mul\"}"
COMPILATION_CONFIG_ARGS=(--compilation-config "{\"mode\":3,\"cudagraph_mode\":\"$CUDAGRAPH_MODE\",\"max_cudagraph_capture_size\":$MAX_CUDAGRAPH_CAPTURE_SIZE,\"custom_ops\":[$CUSTOM_OPS_JSON],\"cudagraph_capture_sizes\":[$CUDAGRAPH_CAPTURE_SIZES]}")

echo "Starting vllm server..."
export PYTHONNOUSERSITE=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1200}"


# ---- DCP       ------------------------------------------------------------
# DCP shards decode KV across the TP ranks, so it must divide TP.
# Default 1, matching benchmark-tmpl.yml's dcp-size default and every other
# recipe here. The workflow always exports DCP_SIZE, so this only affects
# manual invocations -- where defaulting to 8 silently gave a DCP run to
# someone who asked for none.
DCP_SIZE="${DCP_SIZE:-1}"
if [ $((TP % DCP_SIZE)) -ne 0 ]; then
    echo "Error: TP='$TP' must be divisible by DCP_SIZE='$DCP_SIZE'" >&2
    exit 1
fi
CP_ARGS=()
ATTN_BE_ARGS=()
if [ "$DCP_SIZE" -gt 1 ]; then
    CP_ARGS+=(--decode-context-parallel-size "$DCP_SIZE" --dcp-comm-backend a2a)
    # ROCM_AITER_MLA is the tuned path; its DCP route needs the fp8 cprr
    # assembly kernel, which is not in any released aiter wheel -- it comes
    # from patches/k3-dcp8/aiter/. Override to TRITON_MLA only on an image
    # without it, and expect to lose the tuning.
    ATTN_BE_ARGS+=(--attention-backend "${DCP_ATTN_BACKEND:-ROCM_AITER_MLA}")
    # With any KV connector configured, vLLM realigns cp_kv_cache_interleave_size
    # from 1 up to the local block size, on the assumption that the connector is
    # PD disaggregation and the producer's KV layout has to match the consumer's.
    # A same-process offload connector saves and restores each DCP rank's own
    # shard symmetrically, so it needs no such realignment -- and interleave != 1
    # turns off *both* DCP verify routes, after which DSpark refuses to build:
    # "does not support causal multi-token MLA attention for DSpark with decode
    # context parallelism". kv_role is "kv_both" for offload and for a combined
    # PD node alike, so vLLM cannot tell them apart on its own.
    export VLLM_DCP_KEEP_INTERLEAVE=1
fi
export VLLM_USE_DIRECT_DCP_A2A=0
export VLLM_USE_DIRECT_DCP_Q_GATHER=0
export VLLM_USE_DIRECT_DCP_KV_GATHER=0

{ set +x; } 2>/dev/null
VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --moe-backend auto
    --tensor-parallel-size "$TP"
    "${EP_ARGS[@]}"
    --load-format fastsafetensors
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --language-model-only
    --max-num-seqs "$MAX_NUM_SEQS"
    --enable-auto-tool-choice
    --tool-call-parser kimi_k3
    --reasoning-parser kimi_k3
    --max-model-len 1048576
    --enable-prefix-caching
    --prefix-cache-retention-interval "$PREFIX_CACHE_RETENTION_INTERVAL"
    --kv-cache-dtype "fp8"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --attention-config '{"mla_prefill_backend":"ROCM_AITER_FA"}'
    "${ATTN_BE_ARGS[@]}"
    "${COMPILATION_CONFIG_ARGS[@]}"
    "${SPEC_ARGS[@]}"
    "${OFFLOAD_ARGS[@]}"
    "${CP_ARGS[@]}"
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
