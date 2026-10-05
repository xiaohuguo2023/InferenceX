#!/usr/bin/env bash
# One IX-CI / k3-agentic-repro agentic point. Runs INSIDE the container.
#
# Topology:
#   c1        DCP8 + LMCache 803 GB   (repro script's measured c1)
#   c2, c4    DCP1, no offload          (repro + amd-master.yaml)
#   8-24      DCP8 + LMCache 803 GB    (amd-master 8-14; 16/24 use same DRAM block)
#   32,48-70  DCP8 + LMCache 1028 GB  (amd-master 32/56; extras 48/64/70)
#   40        DCP8 + native DRAM 512 GB (LMCache IPC races DCP KV gather)
#
# Serve flags, patches, aiperf scenario come from kimik3_fp4_mi355x_mtp.sh.
set -euo pipefail

CONC="${1:?concurrency required}"
NODE_TAG="${2:?node tag}"
REPO_ROOT="${REPO_ROOT:-/inferencex_dcp}"
TAG_SUFFIX="${TAG_SUFFIX:-}"
TAG="ixci_repro_c${CONC}_n${NODE_TAG}${TAG_SUFFIX:+_${TAG_SUFFIX}}"

# Queue uses this to know the container is already running a point.
mkdir -p /tmp/ixci_busy
echo "$TAG $$" > "/tmp/ixci_busy/${NODE_TAG}"
trap 'rm -f /tmp/ixci_busy/${NODE_TAG}' EXIT

# Shared hub cache refs/main (f831) is missing encoding_k3.py and is not
# writable. AIPerf --tokenizer must be an HF repo id. Stage a complete
# tokenizer snapshot at the a590 revision plus a DSpark symlink.
export HF_ROOT="${HF_ROOT:-$REPO_ROOT/.hf-k3-tok}"
mkdir -p "$HF_ROOT"
_tok_hash=a590ce090cb049c93a33dfe8c208ec652aa20503
_tok_snap="$HF_ROOT/models--moonshotai--Kimi-K3/snapshots/$_tok_hash"
_tok_src="/shared_nfs/hyperloom/models/Kimi-K3"
if [[ ! -f "$_tok_snap/encoding_k3.py" ]]; then
  mkdir -p "$_tok_snap" "$HF_ROOT/models--moonshotai--Kimi-K3/refs"
  for f in encoding_k3.py tokenization_kimi.py tokenizer_config.json tiktoken.model \
           config.json configuration_kimi_k3.py generation_config.json; do
    cp -f "$_tok_src/$f" "$_tok_snap/"
  done
  printf '%s' "$_tok_hash" > "$HF_ROOT/models--moonshotai--Kimi-K3/refs/main"
fi
ln -sfn /shared_nfs/hf-hub-cache/models--Inferact--Kimi-K3-DSpark \
  "$HF_ROOT/models--Inferact--Kimi-K3-DSpark"
ln -sfn /shared_nfs/hf-hub-cache/datasets--semianalysisai--cc-traces-weka-062126 \
  "$HF_ROOT/datasets--semianalysisai--cc-traces-weka-062126"
export HF_HUB_CACHE="$HF_ROOT"
export HF_HOME="$HF_ROOT"
# Offline breaks `hf download` of the Weka traces. Tokenizer files are already
# in this cache; leave Hub reachable for dataset/revision checks.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
_hf_stub=/tmp/k3-hf-stub
mkdir -p "$_hf_stub"
cat > "$_hf_stub/hf" <<'EOF'
#!/bin/bash
if [[ " $* " == *"Inferact/Kimi-K3-DSpark"* ]]; then
  snap=$(ls -d /shared_nfs/hf-hub-cache/models--Inferact--Kimi-K3-DSpark/snapshots/*/ 2>/dev/null | head -1)
  if [[ -n "$snap" && -f "${snap%/}/config.json" ]]; then
    echo "hf stub: DSpark already staged at ${snap%/}"
    exit 0
  fi
fi
if [[ " $* " == *"moonshotai/Kimi-K3"* ]]; then
  if [[ -f /shared_nfs/hyperloom/models/Kimi-K3/config.json ]]; then
    echo "hf stub: Kimi-K3 already staged"
    exit 0
  fi
fi
exec "$(command -v hf.real 2>/dev/null || true)" "$@" 2>/dev/null || \
  exec python3 -m huggingface_hub.cli.hf "$@"
EOF
chmod +x "$_hf_stub/hf"
if command -v hf >/dev/null && [[ "$(command -v hf)" != "$_hf_stub/hf" ]]; then
  ln -sfn "$(command -v hf)" "$_hf_stub/hf.real" 2>/dev/null || true
fi
export PATH="$_hf_stub:$PATH"

export MODEL=moonshotai/Kimi-K3
export MODEL_PATH="${MODEL_PATH:-/shared_nfs/hyperloom/models/Kimi-K3}"
# Hub cache refs/main -> f831 is missing encoding_k3.py (not writable here).
# Pass the local tree via --tokenizer. Do NOT export AIPERF_TOKENIZER: that
# env name is Environment.TOKENIZER (a settings object) and aiperf rejects a
# path with "error parsing value for field TOKENIZER".
# Repo id, not a path. Snapshot is the writable HF_HUB_CACHE above.
export IXCI_REPLAY_TOKENIZER="${IXCI_REPLAY_TOKENIZER:-moonshotai/Kimi-K3}"
unset AIPERF_TOKENIZER || true
export MODEL_PREFIX=kimik3
export TP=8
export EP_SIZE=1
export CONC
export DURATION="${DURATION:-3600}"
export EVAL_ONLY="${EVAL_ONLY:-false}"
export RESULT_FILENAME="$TAG"
export RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/results_ixci/$TAG}"
export AIPERF_UV_CACHE_DIR="${AIPERF_UV_CACHE_DIR:-/dev/shm/aiperf-uv-cache}"
# The DSpark draft is pre-staged on shared storage, but vLLM still resolves its
# repo id against the Hub at startup, and from a shared cluster IP that gets a
# 429 -- aborting the launch after the whole weight load, for files that were
# already local. Go offline only when the snapshot is actually in the cache, so
# a genuinely missing checkpoint still downloads instead of failing obscurely.
# K3_ALLOW_HF_NET=1 forces the Hub back on.
_hf_hub="${HF_HUB_CACHE:-${HF_HOME:+$HF_HOME/hub}}"
if [[ "${K3_ALLOW_HF_NET:-0}" != 1 ]] \
   && compgen -G "${_hf_hub:-$HOME/.cache/huggingface/hub}/models--Inferact--Kimi-K3-DSpark/snapshots/*/config.json" >/dev/null; then
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
fi
unset _hf_hub
# Default empty. EXTRA_VLLM_ARGS is appended LAST, and argparse keeps the last
# --load-format, so defaulting it to "auto" silently overrode the recipe's
# --load-format fastsafetensors on every run -- vLLM even logs "Found duplicate
# keys --load-format". On a node whose page cache is cold that turned a ~2 minute
# weight load into ~60 minutes. Set it explicitly if a run really wants auto.
export EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
# The pinned aiter tree is prebuilt against the ROCm 7.2.3 image's toolchain and
# its jit modules need GLIBCXX_3.4.31. The rocm100 images only ship 3.4.30, so
# overlaying it there kills every worker at import:
#   ImportError: ... version `GLIBCXX_3.4.31' not found
#     (required by /opt/aiter-k3/aiter/jit/module_aiter_core.so)
# Check the ABI rather than the image tag, so this self-heals on either line.
# When it does not fit we fall back to the image's own dist-packages aiter,
# which is what the published rocm100 runs use.
aiter_overlay_usable() {
  local so need have newest
  so=$(ls /opt/aiter-k3/aiter/jit/*.so 2>/dev/null | head -1)
  [[ -n "$so" ]] || return 0
  need=$(grep -ao 'GLIBCXX_3\.4\.[0-9]*' "$so" 2>/dev/null | sort -V | tail -1)
  have=$(grep -ao 'GLIBCXX_3\.4\.[0-9]*' /lib/x86_64-linux-gnu/libstdc++.so.6 \
         2>/dev/null | sort -V | tail -1)
  [[ -n "$need" && -n "$have" ]] || return 0
  newest=$(printf '%s\n%s\n' "$need" "$have" | sort -V | tail -1)
  if [[ "$newest" != "$have" ]]; then
    echo "aiter overlay needs $need but image libstdc++ provides $have;" \
         "using the image's own aiter" >&2
    return 1
  fi
  return 0
}

# Copied aiter has flydsl-0.3.2 buffer_ops (0003). Image flydsl 0.3.0 dies in
# DSpark profile_run. The old flydsl-0.3.2.tgz overlay was incomplete (no
# flydsl.libs/libmlir_apfloat_wrappers*.so.24) and aborted every worker.
_FLYDSL_WHEEL="${FLYDSL_WHEEL:-/shared_nfs/xiaohugu/wheels/flydsl-0.3.2-cp312-cp312-manylinux_2_27_x86_64.whl}"
if ! python3 - <<'PY'
import importlib.metadata as m, pathlib, sys
if m.version("flydsl") != "0.3.2":
    sys.exit(1)
libs = pathlib.Path("/usr/local/lib/python3.12/dist-packages/flydsl.libs")
sys.exit(0 if any(libs.glob("libmlir_apfloat_wrappers*.so.24*")) else 1)
PY
then
  echo "flydsl incomplete -> pip install --force-reinstall $_FLYDSL_WHEEL"
  pip install --force-reinstall --no-deps --no-cache-dir "$_FLYDSL_WHEEL"
fi
if [[ ! -d /opt/aiter-k3/aiter && -f /shared_nfs/xiaohugu/aiter-k3.tgz ]]; then
  echo "extract /opt/aiter-k3"
  tar -C /opt -xzf /shared_nfs/xiaohugu/aiter-k3.tgz
fi

# Single decision, taken once the tree is on disk: take the pinned aiter and the
# tuned-GEMM rows that were measured against it, or take neither. Tuned kernel
# indices are aiter-build-specific, so pairing our pinned CSV with the image's
# own aiter would be worse than the config the image shipped with -- and the
# image's own config is what the published rocm100 runs use.
# AITER_OVERLAY=0 keeps the image's own aiter even when the overlay would load:
# the overlay tarball predates newer vLLM nightlies (mla_decode_fwd has no
# `causal` kwarg there).
if [[ "${AITER_OVERLAY:-1}" == 1 && -d /opt/aiter-k3/aiter ]] && aiter_overlay_usable; then
  export PYTHONPATH="/opt/aiter-k3${PYTHONPATH:+:$PYTHONPATH}"
  export AITER_CONFIG_GEMM_BF16="${AITER_CONFIG_GEMM_BF16:-$REPO_ROOT/patches/k3-dcp8/aiter/merged_bf16_tuned_gemm.csv}"
  # The pinned tarball predates fused_kda_decode. Installing that module into
  # the image site-packages is hidden once this directory is prepended, and
  # the workers then die at the first KDA forward with ModuleNotFoundError.
  _fused_src="$REPO_ROOT/.aiter_pr5709/aiter"
  if [[ -f "$_fused_src/ops/triton/gated_delta_net/fused_kda_decode.py" ]]; then
    install -D "$_fused_src/ops/triton/gated_delta_net/fused_kda_decode.py" \
      /opt/aiter-k3/aiter/ops/triton/gated_delta_net/fused_kda_decode.py
    install -D "$_fused_src/ops/triton/_triton_kernels/gated_delta_rule/decode/fused_conv_recurrent_norm.py" \
      /opt/aiter-k3/aiter/ops/triton/_triton_kernels/gated_delta_rule/decode/fused_conv_recurrent_norm.py
    echo "installed fused KDA into /opt/aiter-k3"
    python3 -c 'from aiter.ops.triton.gated_delta_net.fused_kda_decode import fused_kda_decode; print("overlay fused_kda_decode ok")'
  fi
else
  # Set this explicitly, otherwise the recipe's own default re-pins the 7.2.3
  # rows onto an aiter build that never saw them.
  _img_gemm_csv=/usr/local/lib/python3.12/dist-packages/aiter/configs/bf16_tuned_gemm.csv
  if [[ -f "$_img_gemm_csv" ]]; then
    echo "using the image's own tuned-GEMM rows: $_img_gemm_csv"
    export AITER_CONFIG_GEMM_BF16="$_img_gemm_csv"
  fi
fi

# nightly-rocm index rotates; as of 2026-09-12 the pin is dev134
# (dev118 404s: "No matching distribution found").
export LMCACHE_VERSION="${LMCACHE_VERSION:-0.5.6.dev54+rocm7.2}"
export PORT="${PORT:-8888}"

case "$CONC" in
  1)
    export DCP_SIZE=8
    export KV_OFFLOADING=dram
    export KV_OFFLOAD_BACKEND=lmcache
    export TOTAL_CPU_DRAM_GB=803
    ;;
  2|4)
    export DCP_SIZE=1
    export KV_OFFLOADING=none
    unset KV_OFFLOAD_BACKEND || true
    export TOTAL_CPU_DRAM_GB=803
    ;;
  8|10|12|14|16|24)
    export DCP_SIZE=8
    export KV_OFFLOADING=dram
    export KV_OFFLOAD_BACKEND=lmcache
    export TOTAL_CPU_DRAM_GB=803
    ;;
  40)
    # LMCache's ROCm IPC path can wedge the DCP prefill all-gather under the
    # c40 cache-pressure replay. Native offload avoids that communicator/IPC
    # progress race. 1028 GB starves the HSA runtime of host memory, so cap the
    # native arena at the validated 512 GB.
    export DCP_SIZE=8
    export KV_OFFLOADING=dram
    export KV_OFFLOAD_BACKEND=vllm-simple
    export TOTAL_CPU_DRAM_GB=512
    ;;
  32|48|56|64|70)
    export DCP_SIZE=8
    export KV_OFFLOADING=dram
    export KV_OFFLOAD_BACKEND=lmcache
    export TOTAL_CPU_DRAM_GB=1028
    ;;
  *)
    echo "ERROR: CONC=$CONC is not in the IX CI / 1-70 ladder" >&2
    exit 2
    ;;
esac

# Allow isolated comparison arms to match the upstream dashboard topology
# without changing the main DCP8 ladder defaults above.
export DCP_SIZE="${DCP_SIZE_OVERRIDE:-$DCP_SIZE}"
if [[ -n "${KV_OFFLOADING_OVERRIDE:-}" ]]; then
  export KV_OFFLOADING="$KV_OFFLOADING_OVERRIDE"
  if [[ "$KV_OFFLOADING" == "none" ]]; then
    unset KV_OFFLOAD_BACKEND || true
  elif [[ -n "${KV_OFFLOAD_BACKEND_OVERRIDE:-}" ]]; then
    export KV_OFFLOAD_BACKEND="$KV_OFFLOAD_BACKEND_OVERRIDE"
  fi
fi
export TOTAL_CPU_DRAM_GB="${TOTAL_CPU_DRAM_GB_OVERRIDE:-$TOTAL_CPU_DRAM_GB}"
# process_agentic_result rejects a dram run that names a backend but omits this
# JSON. Without it the 1-hour replay finishes and the script still exits 1.
if [[ "${KV_OFFLOADING}" == "none" ]]; then
  unset KV_OFFLOAD_BACKEND_METADATA || true
elif [[ -n "${KV_OFFLOAD_BACKEND:-}" ]]; then
  if [[ "$KV_OFFLOAD_BACKEND" == "lmcache" ]]; then
    export KV_OFFLOAD_BACKEND_METADATA="$(printf '{"name":"lmcache","version":"%s"}' "$LMCACHE_VERSION")"
  else
    export KV_OFFLOAD_BACKEND_METADATA="$(printf '{"name":"%s"}' "$KV_OFFLOAD_BACKEND")"
  fi
fi

mkdir -p "$RESULT_DIR"
echo "=== $TAG DCP=$DCP_SIZE offload=$KV_OFFLOADING dram=${TOTAL_CPU_DRAM_GB:-n/a} ==="
set +e
bash "$REPO_ROOT/benchmarks/single_node/agentic/kimik3_fp4_mi355x_mtp.sh"
rc=$?
echo "IXCI_REPRO_RC=$rc tag=$TAG"
# Hold the busy file until HBM has actually drained so the queue cannot
# start the next conc on a node still at VRAM 89% (NCCL init then dies).
# CLIENT_ONLY leaves the engine up, so a drain wait would just burn the
# remaining allocation watching a healthy server.
if [ "${CLIENT_ONLY:-false}" != "true" ]; then
  for i in $(seq 1 60); do
    vram_max=$(rocm-smi --showmemuse 2>/dev/null \
      | grep -oE 'GPU Memory Allocated \(VRAM%\): [0-9]+' \
      | awk '{if ($NF > m) m = $NF} END {print m+0}')
    if [ "${vram_max:-0}" -le 10 ]; then
      echo "GPUs drained (vram%max=$vram_max after teardown)"
      break
    fi
    echo "waiting for teardown GPU drain: vram%max=$vram_max"
    sleep 10
  done
fi
exit $rc
