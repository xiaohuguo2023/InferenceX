#!/usr/bin/env bash
set -eo pipefail
set -x

# Agentic trace-replay benchmark for Kimi-K3 on AMD MI355X (gfx950) via vLLM,
# WITH DSpark speculative decoding (native fp8 KV on the asm MLA path).
#
# Delta over the non-spec launcher (kimik3_fp4_mi355x_vllm.sh):
#   1. --speculative-config pointing at the DSpark draft, method=dspark, with the
#      draft FORCED CAUSAL (dflash_config.causal=true). Forcing causal takes the
#      draft off the non-causal MLA path (the cudagraph-OOB source on this ROCm
#      build) and onto the fp8 asm path like the target; the target verify still
#      catches any mispredictions, so accuracy is preserved.
#   2. VLLM_ROCM_AITER_MLA_ASM_PADDING=asm — pads K3's 12 heads/rank to 16 and
#      folds the qlen>1 spec verify onto the asm path.
#   3. dspark-safe memory defaults (the wide qlen=1+2*num_spec verify block needs
#      more headroom than plain decode): gpu-mem 0.88, max-num-seqs 16.
#
# PREREQUISITE — in-container patches (see docs/kimik3_dspark_fp8asm_recipe.md and
# _k3_dspark_fp8asm_apply_patches.sh). On top of a working baseline K3 + the
# SHIPPED recipe rocm_aiter_mla.py (main + #51011 + #51040 + #51606), DSpark
# fp8-asm needs 5 deltas already applied to the image/container:
#   - aiter get_block_n_fp8 key 80 (16*5 verify width)
#   - recipe _mtp_decode_qlen = 2*num_spec+1 for method=dspark
#   - recipe persistent-metadata gate: num_heads>=16 OR asm OR max_qo_len>1
#   - KDA stride fix (Fangzhou-Ai/vllm PR #27)
#   - the draft config forced causal (this script also enforces it, idempotently)
# Validated 2026-08-10: clean FULL_AND_PIECEWISE capture (no eager),
# mean acceptance length 2.39.
#
# Required env vars: MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION
# Optional: MAX_MODEL_LEN, MODEL_PATH, NUM_SPEC (default 2), DRAFT_MODEL,
#           DRAFT_MODEL_PATH, GPU_MEMORY_UTILIZATION, MAX_NUM_SEQS.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION

if [ "$TP" -ne 8 ]; then
    echo "Error: Kimi-K3 on MI355X requires TP=8 (~1.5 TB checkpoint), got TP='$TP'" >&2
    exit 1
fi

# Resolve target weights: MODEL_PATH (pre-staged) else HF cache.
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi
    export MODEL_PATH="$MODEL"
fi
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"; fi

# ---- DSpark draft: resolve + force causal -----------------------------------
NUM_SPEC="${NUM_SPEC:-2}"
DRAFT_MODEL="${DRAFT_MODEL:-Inferact/Kimi-K3-DSpark}"
if [[ -n "${DRAFT_MODEL_PATH:-}" ]]; then
    if [[ ! -d "$DRAFT_MODEL_PATH" || -z "$(ls -A "$DRAFT_MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$DRAFT_MODEL" --local-dir "$DRAFT_MODEL_PATH"
    fi
elif [[ "$DRAFT_MODEL" == /* ]]; then
    DRAFT_MODEL_PATH="$DRAFT_MODEL"
else
    # Prefer a pre-staged HF snapshot; else download into the shared cache.
    _dsnap="$(ls -d "${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"/models--Inferact--Kimi-K3-DSpark/snapshots/*/ 2>/dev/null | head -1)"
    if [[ -n "$_dsnap" && -f "${_dsnap}config.json" ]]; then
        DRAFT_MODEL_PATH="${_dsnap%/}"
    else
        hf download "$DRAFT_MODEL" >/dev/null
        _dsnap="$(ls -d "${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"/models--Inferact--Kimi-K3-DSpark/snapshots/*/ 2>/dev/null | head -1)"
        DRAFT_MODEL_PATH="${_dsnap%/}"
    fi
fi
[ -n "$DRAFT_MODEL_PATH" ] && [ -f "$DRAFT_MODEL_PATH/config.json" ] || {
    echo "Error: could not resolve DSpark draft config.json (DRAFT_MODEL_PATH='$DRAFT_MODEL_PATH')" >&2; exit 1; }

# Force the draft causal (idempotent). Non-causal MLA is the cudagraph-OOB source
# on this ROCm build; causal drafting keeps the draft on the fp8 asm path.
python3 - "$DRAFT_MODEL_PATH/config.json" <<'PY'
import json, sys
f = sys.argv[1]
c = json.load(open(f))
d = c.setdefault("dflash_config", {})
if d.get("causal") is True:
    print(f"draft already forced causal: {f}")
else:
    d["causal"] = True
    json.dump(c, open(f, "w"), indent=2)
    print(f"forced draft causal: {f}")
PY
echo "DRAFT_MODEL_PATH=$DRAFT_MODEL_PATH  NUM_SPEC=$NUM_SPEC"

# ---- MI355X day-0 serving environment (AITER) -------------------------------
export VLLM_ROCM_USE_AITER=1
export GPU_ARCHS=gfx950
export AITER_SITUV2_A8W4=1
export VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1
export AITER_BF16_FP8_MOE_BOUND=0
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export SAFETENSORS_FAST_GPU=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
# DSpark fp8-asm: fold the qlen>1 spec verify onto the asm path (pad 12->16).
export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm

# Pin the tuned/patched BF16 GEMM catalog (mirrors _serve_k3_bench_spec.sh:116-119).
# Without this, aiter re-merges configs/model_configs/*bf16_tuned_gemm*.csv into
# /tmp/aiter_configs/bf16_tuned_gemm.csv and reads THAT — i.e. the FlyDSL->torch
# decode reroute (docs/kimik3_conc24_regression_allreduce.md) is silently NOT in
# effect on the agentic path. Point at the patched merged catalog explicitly.
# (Perf/routing fix for decode; it does NOT by itself address the prefill HSA
# fault — see docs/kimik3_hsa_fault_agentic_launcher.md.)
MERGED_GEMM_CSV="${AITER_MERGED_GEMM_CSV:-/opt/aiter-local/aiter/configs/merged_bf16_tuned_gemm.csv}"
if [ -z "${AITER_CONFIG_GEMM_BF16:-}" ] && [ -f "$MERGED_GEMM_CSV" ]; then
    export AITER_CONFIG_GEMM_BF16="$MERGED_GEMM_CSV"
    echo "AITER_CONFIG_GEMM_BF16=$AITER_CONFIG_GEMM_BF16"
else
    echo "WARN: AITER_CONFIG_GEMM_BF16 unset and $MERGED_GEMM_CSV missing — decode GEMM reroute NOT active" >&2
fi

# ---- Resolve traces + install AIPerf (isolated venv) ------------------------
resolve_trace_source
install_agentic_deps

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

# ---- KV offloading ----------------------------------------------------------
OFFLOAD_ARGS=()
case "${KV_OFFLOAD_BACKEND:-}" in
    "")
        require_agentic_kv_offload_none
        ;;
    vllm-simple)
        require_agentic_kv_offload_backend vllm-simple
        CPU_BYTES_PER_RANK=$(( TOTAL_CPU_DRAM_GB * 1000 * 1000 * 1000 / TP ))
        export PYTHONHASHSEED=42
        OFFLOAD_ARGS=(
            --kv-transfer-config
            "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use_per_rank\":${CPU_BYTES_PER_RANK},\"lazy_offload\":false}}"
        )
        ;;
    *)
        echo "Error: unsupported KV_OFFLOAD_BACKEND='$KV_OFFLOAD_BACKEND'" >&2
        exit 1
        ;;
esac

# Aligned to the mandated single-box DSpark config (_serve_k3_bench_spec.sh):
# gpu_mem 0.95, max_num_seqs 64, MNBT 16384. Validated with the split-K
# cudagraph-safety fix + KV pin; do NOT diverge without re-validating. All three
# stay env-overridable. NOTE: with KV-offload (SimpleCPUOffloadConnector) + full
# ~131K context, 0.95 leaves less physical headroom than the single-box KV-pin
# setup — watch server.log for OOM/hipMalloc if the offload path is enabled here
# (see docs/kimik3_hsa_fault_agentic_launcher.md).
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"

# --- fp8 KV on K3 (dense MLA) via the ASM persistent MLA path -----------------
KVDTYPE_ARGS=(
    --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}"
    --attention-backend "${ATTENTION_BACKEND:-ROCM_AITER_MLA}"
)

export VLLM_ROCM_USE_AITER_MOE=1

# Fused gated-RMSNorm custom op on K3's KDA path (capture-verified on gfx950).
COMPILE_CFG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["+fused_rms_norm_gated"]}'
if [ "${FUSED_RMS_NORM_GATED:-1}" = "0" ]; then
    COMPILE_CFG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}'
fi

# DSpark spec config. Forced-causal draft runs on ROCM_AITER_MLA + fp8 asm like
# the target; #51011 routes the 12-head fp8 verify to the asm q-row-fold.
DRAFT_BACKEND="${DRAFT_BACKEND:-ROCM_AITER_MLA}"
SPEC_CONFIG=$(printf '{"model":"%s","num_speculative_tokens":%s,"method":"dspark","attention_backend":"%s","draft_sample_method":"probabilistic","rejection_sample_method":"block"}' \
    "$DRAFT_MODEL_PATH" "$NUM_SPEC" "$DRAFT_BACKEND")

echo "Starting vllm server (MI355X/AITER, Kimi-K3 DSpark fp8-asm)..."
VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0 --port "$PORT"
    --tensor-parallel-size "$TP"
    --async-scheduling
    --distributed-executor-backend mp
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --trust-remote-code
    --load-format auto
    --moe-backend auto
    --mm-encoder-tp-mode data
    "${KVDTYPE_ARGS[@]}"
    --compilation-config "$COMPILE_CFG"
    --speculative-config "$SPEC_CONFIG"
    --enable-prefix-caching
    --no-disable-hybrid-kv-cache-manager
    --reasoning-parser kimi_k3
    --tool-call-parser kimi_k3
    --enable-auto-tool-choice
    --disable-uvicorn-access-log
    "${OFFLOAD_ARGS[@]}"
)
if [ -n "${MAX_MODEL_LEN:-}" ]; then VLLM_CMD+=(--max-model-len "$MAX_MODEL_LEN"); fi
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"; printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

build_replay_cmd "$RESULT_DIR"
run_agentic_replay_and_write_outputs "$RESULT_DIR"

# cleanup: free the GPU (orphaned TP workers otherwise hold VRAM)
[[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null || true
pkill -9 -f "/usr/local/bin/vll[m]" 2>/dev/null || true
pkill -9 -f "EngineCore" 2>/dev/null || true
pkill -9 -f "multiprocessing.spawn" 2>/dev/null || true
for _ in $(seq 1 30); do pgrep -f "EngineCore|multiprocessing.spawn" >/dev/null 2>&1 || break; sleep 2; done
set +x
