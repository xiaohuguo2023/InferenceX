"""Component-level decode breakdown for Kimi-K3 traces (MI355X and B300).

Why this exists separately from ``trace_compare_k3.py``:

* ``trace_compare_k3.categorize_kernel`` buckets by *backend* ("GEMM",
  "Communication", "Other"), which puts MLA attention, MoE routing and glue all
  in one "Other" pile. This module buckets by *model component* instead, which
  is what a MI355X-vs-B300 gap analysis needs.
* It fixes a stage-split bug in ``trace_compare_k3`` (see ``stage_of``).

Both classifiers are regex/substring lists with no ground truth, so
``test_trace_compare_k3.py`` pins them against real kernel names AND against
call-count invariants derived from the model config -- those invariants are what
catch a mis-normalisation or a prefill leak, which name-matching alone cannot.
"""

from __future__ import annotations

import bisect
import collections
import gzip
import json

# --- Kimi-K3 structural constants (moonshotai/Kimi-K3 config.json) -----------
NUM_HIDDEN_LAYERS = 93
KDA_LAYERS = 69           # len(linear_attn_config.kda_layers)
MLA_LAYERS = 24           # len(linear_attn_config.full_attn_layers)
FIRST_K_DENSE = 1         # first_k_dense_replace
MOE_LAYERS = NUM_HIDDEN_LAYERS - FIRST_K_DENSE      # 92
ATTN_RES_BLOCK_SIZE = 12
DRAFT_LAYERS = 5          # Inferact/Kimi-K3-DSpark num_hidden_layers
# attn_res is called twice per layer (post-attention + post-MLP), model.py:1032/1065
ATTN_RES_CALLS = 2 * NUM_HIDDEN_LAYERS              # 186

# --- Component rules. ORDERED: first match wins. Keys MUST be lowercase. -----
COMPONENT_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("Communication", (
        "allreduce", "all_reduce", "reduce_scatter", "allgather", "all_gather",
        "nccl", "rccl", "msccl", "cross_device_reduce", "multimem", "lamport",
        "oneshot", "twoshot", "a2a", "alltoall", "all2all", "quickreduce",
        "barrier", "increment_epoch", "dcp_",
        # B300 direct-DCP combine path, same anonymous namespace as the q-gather
        "lse_combine", "dispatch_output_lse", "signal_kernel",
    )),
    ("MLA attention", (
        "mla", "attn_res", "tokenspeed", "fmha", "flash_fwd", "paged_attn", "_mha",
        # BMM1/BMM2 W_UK/W_UV absorb -- ours is mla_attention.py:958/1305
        "batched_gemm_a16wfp4",
        # B300's equivalent: torch.bmm -> nvjet, 24 calls = MLA layers.
        # INFERRED from call count + transpose layout, not confirmed.
        "nvjet_sm103_tst_64x8_64x16_4x1_v_bz_nnt",
        "nvjet_sm103_tst_16x64_64x16_4x1_v_bz_tnn",
    )),
    ("KDA", ("kda", "gated_delta", "gated-delta", "conv1d")),
    ("MoE routing/sort", (
        "moe_sort", "moe sort", "sorting", "grouped_topk", "moe_align",
        "routing", "finalize", "moe_aux", "topk",
        "mxfp8_quantize",   # B300's MoE activation quantiser; ours fuses it into the sort
    )),
    ("MoE expert GEMM", (
        "opus_moe", "gemm2_a4w4", "wfp4", "fused_moe", "moe_stage", "mxfp4",
        "mxe2m1", "mxe4m3",   # B300 bmm_*_MxE2m1MxE4m3_* = mxfp4 expert GEMMs
    )),
    ("Dense GEMM", (
        "cijk", "hgemm", "gemm", "cublas", "cutlass", "flatmm", "splitk",
        "nvjet", "tensile",
    )),
    ("Norm/quant", ("rmsnorm", "layernorm", "layer_norm", "norm", "quant")),
    ("Sampling", ("sampl", "argmax", "multinomial", "softmax_warp")),
    # Split out of the old Glue catch-all so the comparison is readable. Each of
    # these is real model work present on BOTH platforms, not analyser residue.
    ("Activation/gating", (
        "situ_and_mul", "act_and_mul", "silu", "swiglu",
        # the MLA output gate: attn_out * g_proj(h).sigmoid(). B300 fuses it
        # (triton_poi_fused_mul_sigmoid); we run a separate sigmoid then a mul.
        "sigmoid",
    )),
    ("Memory/copy", (
        "copybuffer", "fillbuffer", "memcpy", "memset", "catarraybatchedcopy",
        "direct_copy", "vectorized_gather",
    )),
    ("Spec-decode glue", (
        "dflash", "rejection", "spec_decode_metadata", "aligned_state_indices",
        "expand_page_indices", "logits_stats", "gumbel", "resample",
    )),
]
CATCHALL = "Glue/elementwise/misc"

# Kernels that only ever run in PREFILL. Any of these inside a DECODE window
# means the stage split leaked -- see test_no_prefill_kernels_in_decode.
PREFILL_ONLY_SUBSTRINGS = ("chunk_kda", "chunk_gated_delta")


def component_of(kernel_name: str) -> str:
    """Map a raw GPU kernel name to a model component."""
    low = kernel_name.lower()
    for name, keys in COMPONENT_RULES:
        if any(k in low for k in keys):
            return name
    return CATCHALL


def stage_of(ts: float, intervals, starts) -> str | None:
    """PREFILL/DECODE for a GPU kernel at timestamp ``ts``.

    Differs from ``trace_compare_k3._assign_stage_from_index`` in the gap case.
    That function assigns a kernel falling BETWEEN two step annotations to the
    NEXT step. For GPU work that is backwards: a kernel starting after step i's
    annotation ended is step i's async TAIL. At the PREFILL->DECODE boundary the
    shipped rule donates prefill's tail to decode -- measured at +14% on the
    MI355X decode total, and it dragged one full layer-pass of prefill-only
    chunk_* KDA kernels into the decode window. B300 is unaffected because its
    trace contains no PREFILL steps at all.
    """
    if not intervals:
        return None
    i = bisect.bisect_right(starts, ts) - 1
    if i >= 0 and intervals[i][0] <= ts <= intervals[i][1]:
        return intervals[i][2]
    return intervals[i][2] if i >= 0 else intervals[0][2]


def decode_kernels_per_step(trace_path: str) -> tuple[dict[str, tuple[float, float]], float]:
    """Aggregate DECODE GPU kernels by raw name, normalised per decode step.

    The step count comes from the KDA clock: ``fused_recurrent_kda`` fires
    exactly ``KDA_LAYERS`` times per decode step. The profiler's own iteration
    count is NOT reliable (B300 reports 492 for 164 real steps). The clock must
    match ONLY ``fused_recurrent_kda``; the ``chunk_*`` KDA kernels are the
    prefill path and including them inflates the divisor.

    Returns ({kernel_name: (us_per_step, calls_per_step)}, num_steps).
    """
    import trace_compare_k3 as T

    opener = gzip.open if trace_path.endswith(".gz") else open
    with opener(trace_path, "rt") as fh:
        events = json.load(fh)["traceEvents"]

    steps = T._parse_gpu_steps(events) or T._parse_cpu_steps(events)
    intervals, _ = T._build_step_index(steps)
    starts = [iv[0] for iv in intervals]

    agg: dict[str, list[float]] = collections.defaultdict(lambda: [0.0, 0.0])
    for e in events:
        if e.get("cat") != "kernel":
            continue
        dur = e.get("dur")
        if not dur or dur <= 0:
            continue
        if stage_of(e["ts"], intervals, starts) != "DECODE":
            continue
        slot = agg[e["name"]]
        slot[0] += dur
        slot[1] += 1

    clock = sum(v[1] for k, v in agg.items() if "fused_recurrent_kda" in k)
    if clock <= 0:
        raise ValueError(f"no fused_recurrent_kda in DECODE window of {trace_path}")
    n_steps = clock / KDA_LAYERS
    return {k: (v[0] / n_steps, v[1] / n_steps) for k, v in agg.items()}, n_steps
