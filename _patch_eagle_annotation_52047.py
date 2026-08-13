#!/usr/bin/env python3
# PATCH: port vLLM PR #52047 "[Bugfix][AMD] Annotate draft KV cache groups on the
# hybrid grouping path" onto our pinned image (0.26.1rc1.dev306+gcb8104839).
#
# WHY: K3 (24 MLA full-attn + 69 Mamba KDA) + DSpark MTP draft + CPU KV-offload.
# The DSpark hybrid spec lands on the GENERAL multi-group path of
# get_kv_cache_groups(), which -- unlike the deepseek_v4 branch -- never annotates
# which KV group is the draft (EAGLE/MTP) group. With no group flagged, the offload
# scheduler's flag-all fallback fires
#   (offloading/scheduler.py: if use_eagle and not eagle_groups:
#                                 eagle_groups = set(range(len(groups))))
# and wrongly treats the Mamba groups as draft groups. That applies the eagle
# trailing-chunk exclusion + consecutive-chunk requirement to Mamba (mamba_cache_mode
# =align), suppressing external prefix-cache offload READS -> prefill recompute
# explodes -> the c16 TTFT cliff.
#
# FIX (exactly PR #52047): annotate ONLY the true draft group via its
# non_causal_multi_token_decode marker (set on the DSpark draft MLA layer,
# models/kimi_k3/nvidia/dspark_mla.py -> MLAAttentionSpec.non_causal_multi_token_decode
# =True), and warn loudly if spec is on but no group could be identified (the
# condition that triggers the dangerous flag-all fallback).
#
# Idempotent: guards on the inserted function name; takes a one-time .bak; verifies
# with py_compile. No GPU needed.

import py_compile

F = "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py"

src = open(F).read()

MARKER = "_annotate_eagle_groups_from_draft_spec"
if MARKER in src:
    print("PATCH(eagle-annotation-52047): already applied; skipping")
    raise SystemExit(0)

# --- sanity: the anchors PR #52047 relies on must exist on this image ---
assert "def get_kv_cache_groups(" in src, "get_kv_cache_groups not found"
assert "def generate_scheduler_kv_cache_config(" in src, \
    "generate_scheduler_kv_cache_config (fn-insert anchor) not found"
assert "MambaSpec" in src, "MambaSpec must be imported for the warn helper"
assert "logger = init_logger(__name__)" in src, "module logger not found"

# The unique tail of the general multi-group path in get_kv_cache_groups().
CALL_ANCHOR = (
    "            groups.append(KVCacheGroupSpec([name], aligned))\n"
    "\n"
    "    return groups\n"
)
assert src.count(CALL_ANCHOR) == 1, (
    f"expected exactly 1 'return groups' tail anchor, found {src.count(CALL_ANCHOR)}"
)

# The module-level function that immediately follows get_kv_cache_groups().
FN_ANCHOR = "def generate_scheduler_kv_cache_config(\n"
assert src.count(FN_ANCHOR) == 1, (
    f"expected exactly 1 generate_scheduler_kv_cache_config def, found {src.count(FN_ANCHOR)}"
)

# --- 1) the two helper functions (verbatim from PR #52047) ---
HELPERS = '''\
def _annotate_eagle_groups_from_draft_spec(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> None:
    """PATCH(#52047): flag the draft (EAGLE/MTP) KV group on the hybrid path.

    The general multi-group path does not know which group belongs to the draft
    model. The draft attention layer marks its spec with
    ``non_causal_multi_token_decode=True`` (propagated through
    MLAAttentionSpec.merge), so use that marker to annotate only the real draft
    group -- avoiding the offload scheduler's flag-all fallback that would
    otherwise treat Mamba groups as draft groups.
    """
    spec_config = vllm_config.speculative_config
    if spec_config is None or not spec_config.use_eagle():
        return
    for group in kv_cache_groups:
        if getattr(group.kv_cache_spec, "non_causal_multi_token_decode", False):
            group.is_eagle_group = True


def _warn_if_unannotated_eagle_mamba(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> None:
    """PATCH(#52047): warn when spec is on but no group was identified as draft.

    This is exactly the condition that triggers the offload scheduler's
    flag-all fallback, which would wrongly treat Mamba groups as draft groups.
    """
    spec_config = vllm_config.speculative_config
    if spec_config is None or not spec_config.use_eagle():
        return
    if any(getattr(g, "is_eagle_group", False) for g in kv_cache_groups):
        return
    mamba_groups = [
        idx
        for idx, group in enumerate(kv_cache_groups)
        if isinstance(group.kv_cache_spec, MambaSpec)
    ]
    if not mamba_groups:
        return
    logger.warning(
        "Speculative decoding (method=%s) is enabled but no KV cache group "
        "could be identified as the draft model's, so every group -- including "
        "Mamba groups %s -- may be treated as a draft group by the KV-offload "
        "scheduler. External prefix-cache reads may be suppressed.",
        spec_config.method,
        mamba_groups,
    )


'''

src = src.replace(FN_ANCHOR, HELPERS + FN_ANCHOR, 1)

# --- 2) the two calls before the final `return groups` ---
CALL_REPLACEMENT = (
    "            groups.append(KVCacheGroupSpec([name], aligned))\n"
    "\n"
    "    # PATCH(#52047): annotate the draft KV group on the hybrid path so the\n"
    "    # KV-offload scheduler does not flag Mamba groups as draft groups.\n"
    "    _annotate_eagle_groups_from_draft_spec(vllm_config, groups)\n"
    "    _warn_if_unannotated_eagle_mamba(vllm_config, groups)\n"
    "    return groups\n"
)
src = src.replace(CALL_ANCHOR, CALL_REPLACEMENT, 1)

# --- write with one-time backup, then verify it compiles ---
import os
bak = F + ".bak_52047"
if not os.path.exists(bak):
    with open(bak, "w") as b:
        b.write(open(F).read())

open(F, "w").write(src)
py_compile.compile(F, doraise=True)
print("PATCH(eagle-annotation-52047): applied "
      "(_annotate_eagle_groups_from_draft_spec + _warn_if_unannotated_eagle_mamba "
      "+ 2 calls before final `return groups`); py_compile OK")
