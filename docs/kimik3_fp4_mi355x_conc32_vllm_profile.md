# Kimi-K3 (FP4) on MI355X — vLLM conc32 8k/1k function breakdown + library per function

## Run
- **Model:** `moonshotai/Kimi-K3` (2.8T MoE, native checkpoint served text-only), image `vllm/vllm-openai-rocm:kimi-k3`.
- **Serving:** vLLM, **TP8**, `--moe-backend auto`, `--reasoning-parser kimi_k3`, `--language-model-only`,
  `--kv-cache-dtype` default, no prefix caching; env `VLLM_ROCM_USE_AITER=1`, `AITER_SITUV2_A8W4=1`,
  `AITER_BF16_FP8_MOE_BOUND=0`, `VLLM_USE_BREAKABLE_CUDAGRAPH=0`.
- **Workload:** ISL=8192, OSL=64*, conc=32 (`RANDOM_RANGE_RATIO=0.8`). Trace = rank0 torch profiler
  (`dp0_pp0_tp0…rank0`), one of 8 symmetric TP worker traces.
- **Totals (rank0):** 548,419 GPU kernels, **24,032 ms** total device time in the profiled window.

> \*The production benchmark is 8k/1k (OSL=1024). The **profile** uses OSL=64 because the full-length
> torch trace at conc32×8k crashes a worker during trace serialization (`Executor failed` in
> `/stop_profile`). The per-token decode kernel mix repeats each step, so OSL=64 captures the same kernel
> identities. **Caveat:** with 8192 prefill tokens vs 64 decode tokens, this window is **prefill-dominated** —
> Communication and dense GEMM (both prefill-heavy) are weighted higher than they'd be in steady-state decode.
> A decode-only breakdown needs a separate segmented pass.

## Component breakdown (share of GPU kernel time)

| % | ms | Component | Library / kernel family |
|---:|---:|---|---|
| **34.3** | 8244 | **Communication** (TP all-reduce) | **AITER custom-AR** `cross_device_reduce_2stage` (C++/HIP asm) |
| **22.1** | 5300 | **Dense / MLA-proj GEMM (bf16)** | **hipBLASLt (Tensile)** `Cijk_Alik_Bljk_BBS_BH…` |
| **16.1** | 3869 | **MoE expert GEMM (FP8×FP4)** | **AITER flydsl** `mfma_moe1_silu_mul_afp8_wfp4` / `mfma_moe2_afp8_wfp4_bf16_cshuffle` |
| **7.8** | 1861 | **MLA attention** | AITER `_mla_gluon` (Gluon/Triton) + `aiter::fmha_fwd_hd192_hd128` (asm) + `merge_attn_states` + `_attn_res` (Triton) |
| **6.2** | 1482 | Elementwise / other | PyTorch native (`at::native::vectorized_elementwise`, `reduce`) |
| **5.0+** | 1211 | **KDA linear attention** | **Triton (fla)** `chunk_gated_delta_rule`, `chunk_kda_*`, `chunk_gla_*`, `causal_conv1d`, `kda_gate_*`, `recompute_w_u`, `fused_recurrent_kda_*` |
| 2.1 | 503 | MoE combine/reduce | `moe_reduction_kernel_plain_bf16_topk16` (Triton/CUDA) |
| 1.6 | 385 | Norm+Quant fusion | **AITER** `add_rmsnorm_quant_kernel` |
| 1.6 | 384 | MoE routing/sort | **AITER** `grouped_topk` + `opus_moe_sorting` (ck_tile) + `mxfp4_moe_sort` / `fused_mx_quant_moe_sort` |
| 0.4 | 90 | Elementwise fusion | Triton (inductor) `triton_poi_fused__to_copy_mul_sigmoid_slice_tanh` |
| 0.3 | 66 | Memory copy | ROCr/HIP `__amd_rocclr_copyBuffer` |
| 0.2 | 55 | Quantization | AITER `dynamic_per_group_scaled_quant` |

(A ~2.4% tail — `wvSplitK*`/`hgemm_bf16` small-M dense GEMV (AITER), `l2norm_fwd`/`_gather_initial_states`
(KDA), `merge_attn_states` (MLA), `mxfp4_moe_sort` (AITER) — folds into Dense-GEMM / KDA / MLA / MoE-sort
above.)

## Library rollup

| Library | % | Where used |
|---|---:|---|
| **AITER** (custom-AR + flydsl MoE + FMHA/Gluon MLA + routing/sort + norm/quant) | **~58** | comm, MoE experts, MLA, routing, norm, quant |
| **hipBLASLt (Tensile)** `Cijk_` | **22.1** | dense / MLA-projection GEMM (**bf16, not preshuffled**) |
| **Triton** (fla for KDA + `_attn_res` + inductor fusion) | **~9.5** | KDA linear attention, attn residual, elementwise fusion |
| **PyTorch native** (HIP) | 6.2 | elementwise / reduce |
| ROCr/HIP | 0.3 | memcpy |

## Top individual kernels (rank0)

| % | count | kernel | library | component |
|---:|---:|---|---|---|
| 34.06 | 29212 | `aiter::cross_device_reduce_2stage<bf16,8>` | AITER custom-AR | Communication |
| 9.78 | 20720 | `Cijk_…MT224x256x64…` | hipBLASLt | Dense GEMM |
| 9.04 | 5336 | `mfma_moe1_silu_mul_afp8_wfp4_fp8_t64x128x256…` | AITER flydsl | MoE expert GEMM |
| 5.58 | 5336 | `mfma_moe2_afp8_wfp4_bf16_cshuffle_t64x256x128…` | AITER flydsl | MoE expert GEMM |
| 5.22 | 3936 | `Cijk_…MT256x208x64…` | hipBLASLt | Dense GEMM |
| 4.11 | 21576 | `_attn_res_kernel` | Triton | MLA attention |
| 2.65 | 7961 | `Cijk_…MT192x256x64…` | hipBLASLt | Dense GEMM |
| 2.56 | 2736 | `_mla_gluon` | AITER Gluon (Triton) | MLA attention |
| 2.13 | 4002 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | Triton (fla) | KDA linear attn |
| 2.09 | 6992 | `moe_reduction_kernel_plain_bf16_topk16_md3584` | Triton/CUDA | MoE combine |
| 1.12 | 21692 | `aiter::add_rmsnorm_quant_kernel<bf16,256,32>` | AITER | Norm+Quant |
| 0.79 | 10672 | `aiter::grouped_topk_kernel` | AITER | MoE routing |
| 0.71 | 12006 | `_causal_conv1d_fwd_kernel` | Triton (fla) | KDA linear attn |
| 0.58 | 4002 | `chunk_kda_fwd_kernel_intra_sub_chunk` | Triton (fla) | KDA linear attn |
| 0.53 | 1344 | `aiter::fmha_fwd_hd192_hd128_bf16_group` | AITER FMHA (asm) | MLA prefill attn |
| 0.28 | 5336 | `aiter::opus_moe_sorting MoeSortingMultiPhase` | AITER ck_tile | MoE sort |
| 0.23 | 5336 | `aiter::dynamic_per_group_scaled_quant<bf16,fp8>` | AITER | Quantization |

## Architecture observed from the kernels (Kimi-K3 = KDA/MLA hybrid)
- **Hybrid attention**, confirmed at the kernel level:
  - **MLA layers** — `_mla_gluon` (decode, Gluon/Triton), `aiter::fmha_fwd_hd192_hd128` (prefill flash, head dims 192/128), `merge_attn_states`, `_attn_res`.
  - **KDA (Kimi Delta Attention) linear-attention layers** — gated-delta-rule / GLA / causal-conv1d / recurrent-decode Triton kernels (`chunk_gated_delta_rule`, `chunk_kda_*`, `chunk_gla_*`, `_causal_conv1d`, `kda_gate_cumsum`, `recompute_w_u`, `fused_recurrent_kda_packed_decode`). These carry a recurrent state, not paged KV — matching the InferenceX note that DRAM KV-offload only handles the MLA layers.
- **MoE:** FP8×FP4 experts via **AITER flydsl** (`afp8_wfp4`), grouped-topk routing (topk16 in `moe_reduction`), `opus_moe_sorting` / `mxfp4_moe_sort`.
- **Dense/MLA-proj GEMM:** **bf16 via hipBLASLt Tensile (`Cijk_`)** — *not* weight-preshuffled and *not* fp4/fp8.
- **Communication:** **AITER custom all-reduce only** (`cross_device_reduce_2stage`) — no RCCL fallback (unlike Atom's DSV4 decode path).

## Optimization observations
1. **Dense GEMM is bf16 + hipBLASLt + untuned (22%).** The server logged **8,512 "not found tuned config …
   using torch/default" fallbacks**. This is the DSV4 Target-#7 pattern: a **tuned + weight-preshuffled**
   dense path (CK `b_preshuffle` / AITER `fp8gemm…BpreShuffle`) should cut a large chunk of the 22%.
2. **Communication is 34% here — but prefill-inflated.** At 8k prefill × conc32 the activation all-reduces
   dominate; steady-state decode share will be much lower. Already on the good (AITER custom-AR) path, no RCCL.
   Re-measure on a decode-segmented trace before treating comm as the top decode target.
3. **MoE experts (16%) already on the fast AITER flydsl `afp8_wfp4`** path — same best-of-breed as DSV4
   vLLM/SGLang.
4. **KDA linear attention (~5–6%)** is all Triton (fla-style) — a K3-specific surface not present in DSV4;
   a candidate for AITER/CK fusion or retuning if it grows in the decode-dominated regime.

## Caveats
- Prefill-dominated window (OSL=64) — see the run note. Comm and dense GEMM shares are upper bounds vs decode.
- Per-function ms are rank0 device time over the profiled window (not anchored to a TPOT figure, since this
  is an 8k/64 profile, not the 8k/1024 serving point). Use the shares, not absolute ms, for cross-run compare.
- Single TP rank (ranks are symmetric); comm time is per-rank.
