# vLLM vs SGLang — DeepSeek-V4 decode, function-by-function (conc=64, MI355X)

Same model/HW/concurrency; each backend's GPU kernels attached to the same algorithm functions and compared by share of decode GPU-kernel time.

| | vLLM | SGLang |
|---|---|---|
| Source trace | runtime decode (conc256-cap graph, kernels visible) | decode **graph-capture** (`--enable-profile-cuda-graph`) |
| GPU-kernel time (rank0) | 43710 ms | 107 ms |

> **Caveat:** vLLM = steady-state runtime decode; SGLang = graph-*capture* pass (eager, over bs 1–64) with capture-time artifacts in *Other* and understated *Communication* (per-rank capture has no runtime all-reduce cadence). Compare the **function identities + relative structure**, not absolute %s 1:1.

## By algorithm function (category)

| Function | vLLM % | SGLang % |
|---|---|---|
| GEMM (dense/linear) | 20.7% | 42.7% |
| GEMM (MoE experts) | 20.3% | 12.9% |
| Attention (MLA / DSA) | 18.5% | 18.1% |
| Memory/elementwise | 16.6% | 6.1% |
| MoE routing | 8.8% | 7.4% |
| Communication | 7.4% | 0.6% |
| Quantization | 3.4% | 4.3% |
| Normalization | 2.0% | 2.4% |
| RoPE | 1.2% | 3.4% |
| Other | 1.1% | 1.0% |
| Activation | 0.0% | 1.1% |

## GEMM library (who runs the matmuls)

| GEMM library | vLLM % | SGLang % |
|---|---|---|
| AITER MoE expert GEMM (ASM, FP8xFP4) | 20.3% | 12.9% |
| Composable Kernel (ck gemm_xdl, ABscale) | 11.2% | 0.0% |
| hipBLASLt (Tensile) | 5.5% | 10.9% |
| AITER a8w8 block-scale GEMM (CK) | 2.3% | 0.0% |
| wvSplitK (skinny/thin-M GEMM) | 0.6% | 0.0% |
| AITER fp8 blockscale GEMM (BpreShuffle) | 0.0% | 15.9% |
| CK blockscale GEMM (b_preshuffle) | 0.0% | 8.4% |
| Triton GEMM | 0.0% | 0.0% |
| **total matmul** | 40.0% | 48.1% |

## Per-function detail (kernel each backend uses)

### GEMM (dense/linear)  (vLLM 20.7% | SGLang 42.7%)

| Sub-function | vLLM % | vLLM kernel | SGLang % | SGLang kernel |
|---|---|---|---|---|
| CK AB-scale FP8 GEMM | 11.2% | `void ck::kernel_gemm_xdl_cshuffle_v3<ck::GridwiseGem…` | – | – |
| hipBLASLt (Cijk) | 5.5% | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16…` | 10.9% | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x1…` |
| AITER a8w8 blockscale | 2.3% | `_gemm_a8w8_blockscale_kernel_GROUP_K_128_GROUP_N_128…` | – | – |
| bf16 hgemm (AITER flydsl) | 1.1% | `hgemm_bf16_32x64x128x5_SPK8_W2x2x1_BLDS1_TN_AS1_0` | 7.5% | `hgemm_bf16_16x64x128x5_SPK8_W1x2x1_BLDS1_TN_AS1_0` |
| wvSplitK (skinny GEMM) | 0.6% | `void wvSplitKrc_<__hip_bfloat16, 64, 16, 4, 8, 1, 64…` | – | – |
| AITER fp8 blockscale (preshuffle) | – | – | 15.9% | `aiter::fp8gemm_bf16_blockscale_BpreShuffle_32x128` |
| CK blockscale (preshuffle) | – | – | 8.4% | `void ck::kernel_gemm_xdl_cshuffle_v3_multi_d_blocksc…` |
| Triton GEMM (aiter) | 0.0% | `_gemm_a16_w16_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_32…` | – | – |

### GEMM (MoE experts)  (vLLM 20.3% | SGLang 12.9%)

| Sub-function | vLLM % | vLLM kernel | SGLang % | SGLang kernel |
|---|---|---|---|---|
| MoE2 down  afp8_wfp4 | 11.0% | `mfma_moe2_afp8_wfp4_bf16_cshuffle_t32x256x256_vscale…` | 5.9% | `mfma_moe2_afp8_wfp4_bf16_cshuffle_t32x256x256_vscale…` |
| MoE1 gate/up afp8_wfp4 | 9.4% | `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8…` | 7.0% | `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8…` |

### Attention (MLA / DSA)  (vLLM 18.5% | SGLang 18.1%)

| Sub-function | vLLM % | vLLM kernel | SGLang % | SGLang kernel |
|---|---|---|---|---|
| DSA decode partial | 5.2% | `_sparse_attn_decode_partial_kernel` | – | – |
| DSA generic | 3.8% | `_fused_kv_compress_norm_rope_insert_sparse_attn` | – | – |
| AITER MLA mhc_pre_fuse | 1.8% | `_ZN5aiter23mhc_pre_big_fuse_kernelIDF16bLi320ELi4ELi…` | 3.5% | `void aiter::mhc_pre_big_fuse_kernel<bool _Accum, int…` |
| AITER MLA mhc_pre_gemm | 1.7% | `_ZN5aiter26mhc_pre_gemm_sqrsum_kernelIDF16bLi256ELi6…` | 3.4% | `_ZN5aiter26mhc_pre_gemm_sqrsum_kernelIDF16bLi4ELi64E…` |
| AITER MLA mhc_post | 1.7% | `_ZN5aiter15mhc_post_kernelIDF16bLi256ELi4ELi1024ELb0…` | 2.2% | `_ZN5aiter15mhc_post_kernelIDF16bLi4ELi4ELi1024ELb0EE…` |
| DSA save partial state | 1.0% | `_save_partial_states_kernel` | – | – |
| DSA decode reduce | 0.9% | `_sparse_attn_decode_reduce_kernel` | – | – |
| DSA prefill ragged | 0.7% | `_sparse_attn_prefill_ragged_kernel` | – | – |
| DSV4 QNorm+Rope+KV fuse | 0.7% | `void vllm::deepseek_v4_fused_ops::fusedDeepseekV4QNo…` | – | – |
| KV compress+norm+rope | 0.5% | `_fused_kv_compress_norm_rope_insert_indexer_attn` | – | – |

### Memory/elementwise  (vLLM 16.6% | SGLang 6.1%)

| Sub-function | vLLM % | vLLM kernel | SGLang % | SGLang kernel |
|---|---|---|---|---|
| Elementwise | 14.3% | `void at::native::elementwise_kernel_manual_unroll<12…` | 4.1% | `void at::native::elementwise_kernel_manual_unroll<12…` |
| Fill | 1.6% | `void at::native::vectorized_elementwise_kernel<4, at…` | 2.0% | `void at::native::vectorized_elementwise_kernel<8, at…` |
| Copy buffer | 0.7% | `__amd_rocclr_copyBuffer` | – | – |
| Memcpy | – | – | 0.0% | `memcpy_triton_kernel` |

### MoE routing  (vLLM 8.8% | SGLang 7.4%)

| Sub-function | vLLM % | vLLM kernel | SGLang % | SGLang kernel |
|---|---|---|---|---|
| TopK per-row (decode) | 4.2% | `void vllm::topKPerRowDecode<1024, true, false, true>` | – | – |
| Gating softplus+sqrt | 1.6% | `void vllm::moe::topkGatingSoftplusSqrt<6, 384, 4, 8,…` | – | – |
| MoE sorting | 1.3% | `void aiter::opus_moe_sorting_entry<aiter::MoeSorting…` | 3.1% | `void aiter::opus_moe_sorting_entry<aiter::MoeSorting…` |
| MX-quant + MoE sort | 1.1% | `_ZN5aiter30fused_mx_quant_moe_sort_kernelIDF16bDB8_L…` | 2.2% | `_ZN5aiter30fused_mx_quant_moe_sort_kernelIDF16bDB8_L…` |
| Global topk / lens | 0.6% | `_pack_global_topk_ragged_kernel` | – | – |
| Gating softplus (aiter) | – | – | 2.1% | `void aiter::topk_softplus_kernel_opt<hip_bfloat16, h…` |

### Communication  (vLLM 7.4% | SGLang 0.6%)

| Sub-function | vLLM % | vLLM kernel | SGLang % | SGLang kernel |
|---|---|---|---|---|
| CustomAR 2-stage | 5.6% | `void vllm::cross_device_reduce_2stage<__hip_bfloat16…` | – | – |
| NCCL/RCCL | 1.5% | `ncclDevKernel_Generic_1` | 0.6% | `ncclDevKernel_Generic_1` |
| CustomAR 1-stage | 0.2% | `void vllm::cross_device_reduce_1stage<__hip_bfloat16…` | – | – |

### Quantization  (vLLM 3.4% | SGLang 4.3%)

| Sub-function | vLLM % | vLLM kernel | SGLang % | SGLang kernel |
|---|---|---|---|---|
| AITER per-group quant | 2.8% | `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF1…` | 3.2% | `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF1…` |
| quant_fp8 / scaled | 0.6% | `void per_token_group_quant_8bit_kernel<c10::BFloat16…` | – | – |
| AITER dynamic quant | – | – | 1.1% | `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256E…` |

### Normalization  (vLLM 2.0% | SGLang 2.4%)

| Sub-function | vLLM % | vLLM kernel | SGLang % | SGLang kernel |
|---|---|---|---|---|
| CK-tile RMSNorm | 1.4% | `_ZN7ck_tile6kentryILi1ENS_12Rmsnorm2dFwdINS_27Rmsnor…` | – | – |
| SGL fused RMS+fp8 quant | 0.7% | `_fused_q_kv_rmsnorm_kernel` | 2.4% | `_fused_rms_fp8_group_quant_kernel` |

### RoPE  (vLLM 1.2% | SGLang 3.4%)

| Sub-function | vLLM % | vLLM kernel | SGLang % | SGLang kernel |
|---|---|---|---|---|
| RoPE | 1.2% | `_inverse_rope_gptj_kernel` | 3.4% | `_fused_qk_norm_rope_store_kernel` |

### Other  (vLLM 1.1% | SGLang 1.0%)

| Sub-function | vLLM % | vLLM kernel | SGLang % | SGLang kernel |
|---|---|---|---|---|
| void rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel<r | 0.4% | `void rocprim::ROCPRIM_400200_NS::detail::trampoline_…` | 0.1% | `void rocprim::ROCPRIM_400200_NS::detail::trampoline_…` |
| moe_reduction_kernel_plain_bf16_topk6_md7168 | 0.2% | `moe_reduction_kernel_plain_bf16_topk6_md7168` | 0.4% | `moe_reduction_kernel_plain_bf16_topk6_md7168` |
| _pack_dense_prefix_to_ragged_kernel | 0.1% | `_pack_dense_prefix_to_ragged_kernel` | – | – |
| _dequantize_and_gather_k_kernel | 0.1% | `_dequantize_and_gather_k_kernel` | – | – |
| void at::native::reduce_kernel<512, 1, at::native::ReduceOp< | 0.1% | `void at::native::reduce_kernel<512, 1, at::native::R…` | 0.0% | `void at::native::reduce_kernel<512, 1, at::native::R…` |
| _compute_slot_mapping_kernel | 0.1% | `_compute_slot_mapping_kernel` | – | – |
| _combine_topk_swa_indices_kernel | 0.0% | `_combine_topk_swa_indices_kernel` | – | – |
| _gemm_a8w8_blockscale_reduce_kernel_BLOCK_SIZE_M_32_BLOCK_SI | 0.0% | `_gemm_a8w8_blockscale_reduce_kernel_BLOCK_SIZE_M_32_…` | – | – |
| _compressed_slot_mapping_kernel | 0.0% | `_compressed_slot_mapping_kernel` | – | – |
| void aiter::mxfp4_moe_sort_kernel<256, 32, 32, 32> | 0.0% | `void aiter::mxfp4_moe_sort_kernel<256, 32, 32, 32>` | – | – |

### Activation  (vLLM 0.0% | SGLang 1.1%)

| Sub-function | vLLM % | vLLM kernel | SGLang % | SGLang kernel |
|---|---|---|---|---|
| SiLU/SwiGLU | – | – | 1.1% | `_fused_clamp_silu_mul_kernel_BLOCK_SIZE_N_4096_QUANT…` |

## Key function-by-function differences

**Same-kernel functions (both backends use the identical kernel):**
- **MoE experts** — both run AITER `flydsl` `mfma_moe1_silu_mul_afp8_wfp4` (gate/up) +
  `mfma_moe2_afp8_wfp4` (down), FP8×FP4. Identical expert GEMM.
- **MLA latent compression** — both use AITER `mhc_pre_gemm_sqrsum` / `mhc_pre_big_fuse`
  / `mhc_post` (`module_mhc`).
- **Activation quant** — both use AITER `dynamic_per_group_scaled_quant`.
- **MoE sort** — both use AITER `opus_moe_sorting` + `fused_mx_quant_moe_sort`.
- **Indexer MQA logits** — both use the gluon `deepgemm_fp8_paged_mqa_logits`.

**Different kernels for the same function (the real backend divergence):**
- **Dense/MLA-proj GEMM — weight layout differs.** SGLang runs it entirely
  *weight-preshuffled*: `aiter::fp8gemm_bf16_blockscale_BpreShuffle` (15.9%) +
  `ck…b_preshuffle` (8.4%) + hipBLASLt (10.9%) + AITER-flydsl `hgemm_bf16`. vLLM runs it
  *non-preshuffled*: `ck…GridwiseGemmMultiD_ABScale` (11.2%) + hipBLASLt (5.5%) +
  `_gemm_a8w8_blockscale` (2.3%) + wvSplitK. → vLLM eats a runtime B-layout conversion
  SGLang avoids (optimization Target #7).
- **Core attention.** vLLM: DeepSeek-Sparse-Attention decode kernels
  (`_sparse_attn_decode_partial`/`_reduce`, `fusedDeepseekV4QNormRope…`). SGLang: the
  `dsv4` unified-KV **Triton** `_paged_decode_split`/`_reduce` + compressed `flash_c*_decode`
  (jit_kernel C++). Different attention machinery; MLA compression shared (mhc_*).
- **Router gating.** vLLM `vllm::moe::topkGatingSoftplusSqrt` + `topKPerRowDecode`;
  SGLang AITER `topk_softplus_kernel_opt`.
- **RoPE / norm fusion.** SGLang fuses more (`_fused_qk_norm_rope_store`,
  `_fused_rms_fp8_group_quant`); vLLM has separate `_inverse_rope_gptj`, CK-tile RMSNorm.

**Structural differences (partly real, partly trace-method):**
- **Elementwise/copy: vLLM 16.6% vs SGLang 6.1%** — vLLM issues more unfused pointwise
  ops (a real fusion gap; the SGLang side is also deflated by the capture method, but the
  gap is large enough to be real).
- **Communication: vLLM 7.4% (`cross_device_reduce_2stage`) vs SGLang 0.6%** — **not a
  real difference**: the SGLang number is a capture-trace artifact (per-rank capture omits
  the runtime all-reduce cadence). vLLM's is the true steady-state TP=8 custom all-reduce.
- **Dense-GEMM / MoE-expert % gap** (SGLang 42.7/12.9 vs vLLM 20.7/20.3) is dominated by
  trace methodology (SGLang = structural capture over bs 1–64; vLLM = steady-state runtime),
  so treat those *shares* as directional.

## Takeaways
1. The **MoE expert GEMMs and MLA compression are the same AITER kernels** in both backends
   — the FP8×FP4 MoE path is shared.
2. The decisive real difference is **dense-GEMM weight pre-shuffling** (SGLang) vs
   **non-preshuffled** (vLLM), plus **more aggressive norm/rope/quant fusion** in SGLang
   (lower elementwise overhead).
3. **Attention is implemented differently** (vLLM DSA sparse-decode vs SGLang Triton
   paged-decode + compressed flash) but costs a similar share.
4. Method caveat: vLLM = runtime decode trace, SGLang = graph-capture trace; compare kernel
   **identities** and large structural gaps, not small %-deltas.
