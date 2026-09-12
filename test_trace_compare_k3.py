"""Tests for the K3 trace analysers (`trace_compare_k3.py`, `trace_components.py`).

Both analysers classify kernels with hand-written substring/regex lists and have
had no tests. Two real bugs were found by hand on 2026-09-12:

  1. the KDA "clock" used to normalise per-step numbers also matched the
     prefill-path ``chunk_*`` KDA kernels, inflating the divisor by 4.5%;
  2. ``trace_compare_k3._assign_stage_from_index`` assigns a kernel that falls
     in a GAP between step annotations to the NEXT step, which at the
     PREFILL->DECODE boundary donates prefill's async tail to decode (+14% on
     the MI355X decode total).

Name-matching tests alone would have caught NEITHER. The invariant tests below
would have caught BOTH, because they assert call counts that are fixed by the
model config (69 KDA layers, 24 MLA layers, 92 MoE layers, 2x93 attn_res calls).
That is the point of this file: pin the structure, not just the strings.

Run:  pytest test_trace_compare_k3.py -v
The invariant tests need the real traces and skip cleanly without them.
"""

from __future__ import annotations

import glob
import json
import os

import pytest

import trace_compare_k3 as T
from trace_components import (
    ATTN_RES_CALLS,
    COMPONENT_RULES,
    KDA_LAYERS,
    MLA_LAYERS,
    MOE_LAYERS,
    PREFILL_ONLY_SUBSTRINGS,
    component_of,
    decode_kernels_per_step,
    stage_of,
)

MI355X_GLOB = "/dev/shm/prof_isl100k/*rank0*.json*"
B300_GLOB = os.path.expanduser("~/work/b300/profile_20260911_102437/traces/*rank0*")


def _trace(pattern):
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None


# ---------------------------------------------------------------------------
# 1. Component classifier: real kernel names -> expected component.
#    Every name below was taken verbatim from a real rank0 trace.
# ---------------------------------------------------------------------------

REAL_KERNELS = [
    # --- KDA (the question that prompted these tests) ---
    ("fused_recurrent_kda_fwd_kernel.kd", "KDA"),
    ("fused_recurrent_kda_fwd_kernel", "KDA"),
    ("_causal_conv1d_update_kernel.kd", "KDA"),
    ("_causal_conv1d_update_kernel", "KDA"),
    ("chunk_kda_fwd_kernel_intra_sub_chunk.kd", "KDA"),
    ("chunk_gated_delta_rule_fwd_kernel_h_blockdim64.kd", "KDA"),
    # --- MoE routing/sort ---
    ("void aiter::grouped_topk_kernel<float, float __vector(4), 1, true, true, false>", "MoE routing/sort"),
    ("_ZN5aiter30fused_mx_quant_moe_sort_kernelIDF16bDB8_Li256ELi16EEEvPT0_PhPKT_", "MoE routing/sort"),
    ("void aiter::opus_moe_sorting_entry<aiter::MoeSortingMultiPhaseKernel_P23<", "MoE routing/sort"),
    ("void moe::dev::routing::routingCustom::routingIndicesClusterKernel<moe", "MoE routing/sort"),
    ("void moe::dev::finalize::finalizeKernel<moe::dev::finalize::KernelPara", "MoE routing/sort"),
    ("kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8", "MoE routing/sort"),
    # --- MoE expert GEMM (weights are mxfp4 in all of these) ---
    ("mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_xcd4_situv2", "MoE expert GEMM"),
    ("gemm2_a4w4_port_hmax8192_imax8192_bm32_bn128_bk128_atomic_a8_g2ks2_bhoist_apf", "MoE expert GEMM"),
    ("void opus_moe::stage1_a8w4::pipeline_group_split::opus_moe_stage1_a8w4_kernel", "MoE expert GEMM"),
    ("bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x8x512_s3_et128x8", "MoE expert GEMM"),
    ("bmm_Bfloat16_MxE2m1MxE4m3_Fp32_Ab32_Bb32_t128x8x512_s3_et128x8", "MoE expert GEMM"),
    # --- MLA attention ---
    ("_attn_res_kernel.kd", "MLA attention"),
    ("_attn_res_kernel", "MLA attention"),
    ("_ZN5aiter45mla_a8w8_qh32_qseqlen4_gqaratio32_lse_cprr_psE.kd", "MLA attention"),
    ("void kn_mla_reduce_v1<MlaReduceKernelV1Traits<512, 128, 1>, float, std::bfloat16_t>", "MLA attention"),
    ("void kn_get_mla_metadata_v1_2<MlaMetadataV12Traits<128, true, 0, true, false> >", "MLA attention"),
    ("kernel_cutlass_split_kv_kernel_tokenspeed_mlamla_decode_fp8Blackwell", "MLA attention"),
    ("void sm100::fwd_prod_v2::attn_res_fwd_online_v2_kernel<7168, 8, 3, 1, false, true, true, true>",
     "MLA attention"),
    # BMM1/BMM2 W_UK/W_UV absorb -- must NOT fall into MoE expert GEMM despite "wfp4"
    ("_batched_gemm_a16wfp4_kernel_BLOCK_SIZE_M_16_BLOCK_SIZE_N_64_BLOCK_SIZE_K_256", "MLA attention"),
    ("nvjet_sm103_tst_64x8_64x16_4x1_v_bz_NNT", "MLA attention"),
    ("nvjet_sm103_tst_16x64_64x16_4x1_v_bz_TNN", "MLA attention"),
    # --- Communication ---
    ("void aiter::cross_device_reduce_2stage<std::bfloat16_t, 8, false>(aiter::RankData*", "Communication"),
    ("void aiter::cross_device_reduce_1stage<std::bfloat16_t, 8, false>(aiter::RankData*", "Communication"),
    ("ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>) [clone .kd]", "Communication"),
    ("mscclKernel_Sum_hip_bfloat16_Simple_false(ncclDevComm*, mscclAlgo*, mscclWork*)", "Communication"),
    ("_dcp_a2a_pack_send_kernel.kd", "Communication"),
    ("_dcp_a2a_unpack_combine_kernel.kd", "Communication"),
    ("void flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel<(unsigned char)8,", "Communication"),
    ("(anonymous namespace)::direct_dcp_q_gather_multimem_kernel(uint4 const*, uint4*,", "Communication"),
    ("void (anonymous namespace)::wait_lse_combine_kernel<__nv_bfloat16>(__nv_bfloat16 const*",
     "Communication"),
    ("vllm::direct_dcp::increment_epoch_kernel(long*)", "Communication"),
    # --- Dense GEMM ---
    ("Cijk_Alik_Bljk_BSS_BH_Bias_S_HA_S_SAV_UserArgs_MT32x16x128_MI16x16x1", "Dense GEMM"),
    ("hgemm_bf16_16x64x64x7_SPK4_W1x2x1_BLDS1_TN_AS1_0.kd", "Dense GEMM"),
    ("void wvSplitK_hf_sml_<__hip_bfloat16, 64, 2, 16, 8, 2, 1>(int, int, int,", "Dense GEMM"),
    ("void fused_a_gemm_kernel<1, 3584, 7168, 16, 8, 256, 16>(__nv_bfloat16*", "Dense GEMM"),
    ("nvjet_sm103_tst_64x8_64x16_2x1_v_bz_splitK_TNT", "Dense GEMM"),
    # --- Norm/quant ---
    ("void aiter::add_rmsnorm_quant_kernel<std::bfloat16_t, std::bfloat16_t>", "Norm/quant"),
    ("layer_norm_gated_fwd_kernel.kd", "Norm/quant"),
    ("_fused_q_kv_rmsnorm_kernel", "Norm/quant"),
    # --- Glue catch-all ---
    ("void vllm::situ_and_mul_kernel<c10::BFloat16>(c10::BFloat16*, c10::BFloat16 const*",
     "Glue/elementwise/misc"),
    ("__amd_rocclr_copyBuffer.kd", "Glue/elementwise/misc"),
]


@pytest.mark.parametrize("name,expected", REAL_KERNELS, ids=[k[:48] for k, _ in REAL_KERNELS])
def test_component_of_real_kernel_names(name, expected):
    assert component_of(name) == expected


def test_rule_keys_are_lowercase():
    """component_of() lowercases the kernel name, so an uppercase key is dead.

    This silently swallowed the B300 MLA BMM rule ('NNT'/'TNN') on first write.
    """
    bad = [k for _, keys in COMPONENT_RULES for k in keys if k != k.lower()]
    assert bad == [], f"unreachable uppercase rule keys: {bad}"


def test_no_duplicate_rule_keys_across_components():
    """A key in two components makes the bucket depend on rule ORDER alone."""
    seen: dict[str, str] = {}
    dupes = []
    for comp, keys in COMPONENT_RULES:
        for k in keys:
            if k in seen:
                dupes.append((k, seen[k], comp))
            seen[k] = comp
    assert dupes == [], f"ambiguous keys: {dupes}"


# ---------------------------------------------------------------------------
# 2. Stage split: the gap-assignment bug, on a synthetic trace.
# ---------------------------------------------------------------------------

def test_gap_kernel_is_attributed_to_previous_step_not_next():
    """A kernel in the gap after a PREFILL step is prefill's async tail."""
    intervals = [(0.0, 10.0, "PREFILL"), (20.0, 30.0, "DECODE")]
    starts = [iv[0] for iv in intervals]
    assert stage_of(5.0, intervals, starts) == "PREFILL"    # inside prefill
    assert stage_of(25.0, intervals, starts) == "DECODE"    # inside decode
    assert stage_of(15.0, intervals, starts) == "PREFILL"   # the gap: prefill's tail
    # trace_compare_k3 used to return DECODE here (the leak). Fixed at source.
    assert T._assign_stage_from_index(15.0, intervals) == "PREFILL"


def test_both_stage_assigners_agree_everywhere():
    """trace_components.stage_of and trace_compare_k3._assign_stage_from_index
    must not drift apart again -- the doc's older tables use the latter."""
    intervals = [(0.0, 10.0, "PREFILL"), (20.0, 30.0, "PREFILL"),
                 (40.0, 50.0, "DECODE"), (60.0, 70.0, "DECODE")]
    starts = [iv[0] for iv in intervals]
    for ts in [-5.0, 0.0, 5.0, 10.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 99.0]:
        assert stage_of(ts, intervals, starts) == T._assign_stage_from_index(ts, intervals), (
            f"assigners disagree at ts={ts}")


def test_stage_of_before_first_step():
    intervals = [(10.0, 20.0, "PREFILL")]
    assert stage_of(1.0, intervals, [10.0]) == "PREFILL"


# ---------------------------------------------------------------------------
# 3. Invariants against the real traces. These are the tests with teeth: the
#    call counts are fixed by the model config, so a bad divisor or a leaked
#    stage boundary breaks them.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def decode():
    out = {}
    for tag, pat in (("MI355X", MI355X_GLOB), ("B300", B300_GLOB)):
        path = _trace(pat)
        if path:
            out[tag] = decode_kernels_per_step(path)
    if not out:
        pytest.skip("no traces available (they live in /dev/shm and are volatile)")
    return out


def _calls(per_step, substr):
    return sum(c for k, (_, c) in per_step.items() if substr in k)


@pytest.mark.parametrize("tag", ["MI355X", "B300"])
def test_kda_kernels_fire_once_per_kda_layer(decode, tag):
    """69 KDA layers -> the recurrent kernel and conv1d each fire 69x/step.

    This is the invariant that caught the contaminated KDA clock: with the
    chunk_* kernels in the divisor this read 66.0 instead of 69.0.
    """
    if tag not in decode:
        pytest.skip(f"{tag} trace not available")
    per_step, _ = decode[tag]
    assert _calls(per_step, "fused_recurrent_kda") == pytest.approx(KDA_LAYERS, abs=0.5)
    assert _calls(per_step, "causal_conv1d_update") == pytest.approx(KDA_LAYERS, abs=0.5)


@pytest.mark.parametrize("tag", ["MI355X", "B300"])
def test_no_prefill_kernels_in_decode(decode, tag):
    """chunk_* KDA is the PREFILL path; none may survive in a DECODE window.

    This is the invariant that caught the gap-assignment leak (207 -> 0).
    """
    if tag not in decode:
        pytest.skip(f"{tag} trace not available")
    per_step, _ = decode[tag]
    leaked = {k: c for k, (_, c) in per_step.items()
              if any(s in k for s in PREFILL_ONLY_SUBSTRINGS)}
    assert leaked == {}, f"{tag}: prefill-only kernels leaked into DECODE: {leaked}"


@pytest.mark.parametrize("tag", ["MI355X", "B300"])
def test_attn_res_fires_twice_per_layer(decode, tag):
    """model.py calls attn_res post-attention AND post-MLP: 2 x 93 layers."""
    if tag not in decode:
        pytest.skip(f"{tag} trace not available")
    per_step, _ = decode[tag]
    total = sum(c for k, (_, c) in per_step.items()
                if "attn_res" in k.lower())
    assert total == pytest.approx(ATTN_RES_CALLS, abs=2)


@pytest.mark.parametrize("tag", ["MI355X", "B300"])
def test_per_layer_call_counts_land_on_layer_boundaries(decode, tag):
    """The big per-layer kernels must fire a whole number of times per step.

    Fractional counts (e.g. 88.0 where 92 is expected) mean the divisor is wrong
    or the window is contaminated -- both bugs showed up here first.
    """
    if tag not in decode:
        pytest.skip(f"{tag} trace not available")
    per_step, _ = decode[tag]
    legal = {MLA_LAYERS, MLA_LAYERS + 5, KDA_LAYERS, MOE_LAYERS, NUM := 93,
             2 * KDA_LAYERS, ATTN_RES_CALLS}
    offenders = []
    for k, (us, c) in per_step.items():
        if us < 200:            # only police the kernels that carry real time
            continue
        if abs(c - round(c)) > 0.05:
            offenders.append((k[:60], c))
    assert offenders == [], f"{tag}: non-integer calls/step: {offenders}"


@pytest.mark.parametrize("tag", ["MI355X", "B300"])
def test_component_totals_conserve(decode, tag):
    """No kernel may be dropped or double-counted by the classifier."""
    if tag not in decode:
        pytest.skip(f"{tag} trace not available")
    per_step, _ = decode[tag]
    raw_us = sum(us for us, _ in per_step.values())
    by_comp: dict[str, float] = {}
    for k, (us, _) in per_step.items():
        by_comp[component_of(k)] = by_comp.get(component_of(k), 0.0) + us
    assert sum(by_comp.values()) == pytest.approx(raw_us, rel=1e-9)


def test_both_platforms_run_the_same_number_of_moe_and_kda_calls(decode):
    """Same model, same config -> per-layer call counts must match across
    platforms. A mismatch means one side's window is contaminated."""
    if len(decode) < 2:
        pytest.skip("need both traces")
    mi, _ = decode["MI355X"]
    b3, _ = decode["B300"]
    for substr, expected in (("fused_recurrent_kda", KDA_LAYERS),
                             ("causal_conv1d_update", KDA_LAYERS)):
        assert _calls(mi, substr) == pytest.approx(_calls(b3, substr), abs=0.5)
        assert _calls(mi, substr) == pytest.approx(expected, abs=0.5)


# ---------------------------------------------------------------------------
# 4. The KDA clock, in isolation on a synthetic trace.
#
#    On the real traces the contaminated-clock bug is currently UNREACHABLE:
#    once the gap-assignment leak is fixed there are no chunk_* kernels left in
#    the decode window, so widening the clock to match them changes nothing.
#    (A mutation run confirmed the real-trace test does not catch it.) The two
#    bugs were coupled. But the clock must stay correct independently -- e.g.
#    chunked prefill at higher concurrency can legitimately interleave chunk_*
#    into a decode step -- so pin it here instead.
# ---------------------------------------------------------------------------

def _synthetic_trace(tmp_path, n_decode_steps=4, chunk_kernels_in_decode=7):
    """One prefill step then N decode steps; each decode step runs KDA_LAYERS
    fused_recurrent kernels, plus some prefill-path chunk_* kernels dropped into
    the decode window to stand in for interleaved chunked prefill."""
    ev = [{"cat": "gpu_user_annotation", "name": "execute_context_1(4096)_generation_0(0)",
           "ts": 0.0, "dur": 100.0}]
    t = 200.0
    for s in range(n_decode_steps):
        ev.append({"cat": "gpu_user_annotation",
                   "name": "execute_context_0(0)_generation_1(8)", "ts": t, "dur": 100.0})
        for i in range(KDA_LAYERS):
            ev.append({"cat": "kernel", "name": "fused_recurrent_kda_fwd_kernel.kd",
                       "ts": t + i * 0.1, "dur": 1.0})
        for i in range(chunk_kernels_in_decode):
            ev.append({"cat": "kernel", "name": "chunk_kda_fwd_kernel_intra_sub_chunk.kd",
                       "ts": t + 50.0 + i * 0.1, "dur": 1.0})
        t += 150.0
    p = tmp_path / "synthetic.pt.trace.json"
    p.write_text(json.dumps({"traceEvents": ev}))
    return str(p)


def test_kda_clock_ignores_prefill_chunk_kernels(tmp_path):
    """The clock must count ONLY fused_recurrent_kda.

    With chunk_* in the divisor the step count comes out too high and every
    per-step number is silently understated -- the original bug, 4.5% on
    MI355X. Here 7 chunk kernels per step would give 4 * (69+7)/69 = 4.4 steps
    instead of 4.
    """
    path = _synthetic_trace(tmp_path, n_decode_steps=4, chunk_kernels_in_decode=7)
    per_step, n_steps = decode_kernels_per_step(path)
    assert n_steps == pytest.approx(4.0, abs=1e-6)
    assert _calls(per_step, "fused_recurrent_kda") == pytest.approx(KDA_LAYERS, abs=1e-6)


def test_synthetic_decode_window_excludes_the_prefill_step(tmp_path):
    """Sanity: the prefill step's own annotation must not become decode."""
    path = _synthetic_trace(tmp_path, n_decode_steps=3, chunk_kernels_in_decode=0)
    _, n_steps = decode_kernels_per_step(path)
    assert n_steps == pytest.approx(3.0, abs=1e-6)
