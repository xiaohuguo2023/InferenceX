#!/bin/bash
# Conc-1 IX agentic point on vllm/vllm-openai-rocm:nightly-29468dde with the
# image's own aiter and PIECEWISE target graphs. FULL decode graphs plus DSpark
# spec decode memory-fault this nightly once KV block ids climb (bisected on
# pure vLLM, no patches), so FULL_DECODE_ONLY cannot finish a run here.
#   ARM=best  graph schedule + 1-stage latent tail + Aiter PR fused KDA
#   ARM=base  none of the three
set -uo pipefail
NODE="${1:?node tag}"
ARM="${ARM:?best or base}"
REPO=/inferencex_dcp
# Weights are cached in the container; anonymous Hub file listings get 429s.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Upstream vLLM PRs applied to the image before the recipe patches.
# vllm-project/vllm#58814 refreshes the DSpark context KV cache pointers after
# the KV cache is re-bound; without it, 29468dde FULL graphs memory-fault once
# KV block ids climb.
_prs="${EXTRA_PRS:-}"
[[ "${PR58814:-0}" == 1 ]] && _prs="58814 $_prs"
for _n in $_prs; do
  _pr=$REPO/results_ixci/host_logs/reverts/pr$_n.patch
  _site=/usr/local/lib/python3.12/dist-packages
  if patch --dry-run -R -p1 -d "$_site" < "$_pr" > /dev/null 2>&1; then
    echo "vllm#$_n already applied"
  else
    patch -p1 -d "$_site" < "$_pr" || { echo "!! vllm#$_n does not apply"; exit 1; }
    echo "applied vllm#$_n"
  fi
done

SKIP="linear"
if [[ "$ARM" == dcp1 || "$ARM" == dcp1gs ]]; then
  # CI conc-1 topology (DCP1, no KV offload) plus fused KDA and the 1-stage
  # latent tail. dcp1gs keeps the MLA graph schedule, which also takes the
  # eager get_mla_metadata launches off the DCP1 decode path.
  export DCP_SIZE_OVERRIDE="${DCP_SIZE_OVERRIDE:-1}"
  export KV_OFFLOADING_OVERRIDE="${KV_OFFLOADING_OVERRIDE:-none}"
  [[ "$ARM" == dcp1 ]] && SKIP="$SKIP rocm_aiter_mla_graph_schedule"
fi
if [[ "$ARM" == best || "$ARM" == dcp1 || "$ARM" == dcp1gs ]]; then
  SITE="$(python3 -c 'import os,aiter; print(os.path.dirname(aiter.__file__))')"
  # .aiter_pr5709 is the Sep-27 snapshot (c0241e91b), which ignores the stride of
  # conv_state_indices; vLLM passes spec_state_indices[:, 0], so at batch > 1 the
  # spec kernel reads and writes other sequences' conv state. .aiter_pr5709_head
  # carries the PR's later stride and NULL-slot fixes plus the gfx950 tile config
  # without which the spec path is skipped.
  _pr5709="$REPO/${AITER_PR5709_DIR:-.aiter_pr5709}"
  install -D "$_pr5709/aiter/ops/triton/gated_delta_net/fused_kda_decode.py" \
    "$SITE/ops/triton/gated_delta_net/fused_kda_decode.py"
  install -D "$_pr5709/aiter/ops/triton/_triton_kernels/gated_delta_rule/decode/fused_conv_recurrent_norm.py" \
    "$SITE/ops/triton/_triton_kernels/gated_delta_rule/decode/fused_conv_recurrent_norm.py"
  _cfg=aiter/ops/triton/configs/gfx950/triton/attention/fused_kda_decode/DEFAULT.json
  [[ -f "$_pr5709/$_cfg" ]] && install -D "$_pr5709/$_cfg" "$SITE/${_cfg#aiter/}"
  python3 -c 'from aiter.ops.triton.gated_delta_net.fused_kda_decode import fused_kda_decode; print("aiter fused_kda_decode ok")'
  echo "aiter PR #5709 KDA decode from ${AITER_PR5709_DIR:-.aiter_pr5709}"
  export AITER_KDA_FUSE="${AITER_KDA_FUSE:-spec}"
  python3 "$REPO/results_ixci/host_logs/k3_isl100k_ar1s_patch.py" || exit 1
  export AITER_AR_1STAGE=1
else
  SKIP="$SKIP kda_triton_spec_decode rocm_aiter_mla_graph_schedule"
fi

export TAG_SUFFIX="${TAG_PREFIX:-nightly}_${ARM}"
export SKIP_VLLM_PATCHES="$SKIP ${SKIP_EXTRA:-}"
export AITER_OVERLAY=0
export CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"
export K3_AMD_GATE_MULTI_STREAM=1
export COMPILATION_EXTRA_JSON=',"use_inductor_graph_partition":true,"pass_config":{"fuse_rope_kvcache_cat_mla":true}'
[[ -v COMPILATION_EXTRA_JSON_OVERRIDE ]] && export COMPILATION_EXTRA_JSON="$COMPILATION_EXTRA_JSON_OVERRIDE"
# The nightly-rocm index keeps only the newest dev wheel (dev107 as of 2026-09-27).
export LMCACHE_VERSION="${LMCACHE_VERSION:-0.5.6.dev107+rocm7.2}"
export EXTRA_VLLM_ARGS="--load-format ${LOAD_FORMAT:-fastsafetensors} ${EXTRA_VLLM_ARGS_ADD:-}"
export DURATION=3600
export ENABLE_DSPARK_TORCH_COMPILE=0

exec bash "$REPO/_k3_ixci_repro_run.sh" "${IX_CONC:-1}" "$NODE"
