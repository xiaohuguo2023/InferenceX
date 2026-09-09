# vLLM vs SGLang vs Atom — DeepSeek-V4 decode (conc=64, MI355X), function-by-function

Each backend's decode GPU kernels attached to the same algorithm functions.

| trace type | vLLM | SGLang | Atom |
|---|---|---|---|
| kind | runtime decode | **graph-capture** (structural) | runtime decode (published CI) |
| GPU-kernel time | 43710 ms | 107 ms | 3772 ms |

> Caveat: vLLM & Atom are steady-state runtime traces (directly comparable); SGLang is a
> graph-*capture* pass (structural mix over bs 1–64, comm understated). Compare kernel
> **identities** + big structural gaps.

## By algorithm function (% of decode GPU time)

| Function | vLLM | SGLang | Atom |
|---|---|---|---|
| GEMM (dense/linear) | 20.7% | 42.7% | 13.7% |
| Attention (MLA / DSA) | 18.5% | 18.1% | 25.4% |
| GEMM (MoE experts) | 20.3% | 12.9% | 16.0% |
| Memory/elementwise | 16.6% | 6.1% | 20.8% |
| Communication | 7.4% | 0.6% | 15.7% |
| MoE routing | 8.9% | 7.4% | 4.0% |
| Quantization | 3.4% | 4.3% | 3.1% |
| Normalization | 2.0% | 2.4% | 0.5% |
| RoPE | 1.2% | 3.4% | 0.3% |
| Other | 1.1% | 1.0% | 0.5% |
| Activation | – | 1.1% | 0.1% |

## GEMM library (who runs the matmuls)

| GEMM library | vLLM | SGLang | Atom |
|---|---|---|---|
| AITER MoE expert GEMM (ASM, FP8xFP4) | 20.3% | 12.9% | – |
| hipBLASLt (Tensile) | 5.5% | 10.9% | 3.9% |
| CK blockscale GEMM (b_preshuffle) | – | 8.4% | 9.8% |
| CK MoE MXGEMM BPreshuffle (Atom) | – | – | 16.0% |
| AITER fp8 blockscale GEMM (BpreShuffle) | – | 15.9% | – |
| Composable Kernel (ck gemm_xdl, ABscale) | 11.2% | – | – |
| AITER a8w8 block-scale GEMM (CK) | 2.3% | – | – |
| wvSplitK (skinny/thin-M GEMM) | 0.6% | – | – |
| Triton GEMM | 0.0% | – | – |
| **total matmul** | 40.0% | 48.1% | 29.7% |

## Per-function: representative kernel each backend uses

### GEMM (dense/linear)  (vLLM 20.7% | SGLang 42.7% | Atom 13.7%)

| Sub-function | vLLM | SGLang | Atom | kernel (first backend present) |
|---|---|---|---|---|
| hipBLASLt (Cijk) | 5.5% | 10.9% | 3.9% | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_M…` |
| CK blockscale (preshuffle) | – | 8.4% | 9.8% | `void ck::kernel_gemm_xdl_cshuffle_v3_multi_d_b…` |
| AITER fp8 blockscale (preshuffle) | – | 15.9% | – | `aiter::fp8gemm_bf16_blockscale_BpreShuffle_32x…` |
| CK AB-scale FP8 GEMM | 11.2% | – | – | `void ck::kernel_gemm_xdl_cshuffle_v3<ck::Gridw…` |
| bf16 hgemm (AITER flydsl) | 1.1% | 7.5% | 0.0% | `hgemm_bf16_32x64x128x5_SPK8_W2x2x1_BLDS1_TN_AS…` |
| AITER a8w8 blockscale | 2.3% | – | – | `_gemm_a8w8_blockscale_kernel_GROUP_K_128_GROUP…` |
| wvSplitK (skinny GEMM) | 0.6% | – | – | `void wvSplitKrc_<__hip_bfloat16, 64, 16, 4, 8,…` |
| Triton GEMM (aiter) | 0.0% | – | – | `_gemm_a16_w16_kernel_BLOCK_SIZE_M_32_BLOCK_SIZ…` |

### Attention (MLA / DSA)  (vLLM 18.5% | SGLang 18.1% | Atom 25.4%)

| Sub-function | vLLM | SGLang | Atom | kernel (first backend present) |
|---|---|---|---|---|
| Atom sparse-attn (ragged) | – | – | 13.3% | `_sparse_attn_ragged_varlen_triton_kernel` |
| AITER MLA mhc_pre_gemm | 1.7% | 3.4% | 3.9% | `_ZN5aiter26mhc_pre_gemm_sqrsum_kernelIDF16bLi2…` |
| AITER MLA mhc_post | 1.7% | 2.2% | 4.7% | `_ZN5aiter15mhc_post_kernelIDF16bLi256ELi4ELi10…` |
| AITER MLA mhc_pre_fuse | 1.8% | 3.5% | 2.7% | `_ZN5aiter23mhc_pre_big_fuse_kernelIDF16bLi320E…` |
| DSA decode partial | 5.2% | – | – | `_sparse_attn_decode_partial_kernel` |
| SGL paged decode split | – | 3.9% | – | `_paged_decode_split_kernel` |
| DSA generic | 3.8% | – | – | `_fused_kv_compress_norm_rope_insert_sparse_att…` |
| SGL flash MLA (compressed) | 0.1% | 2.3% | – | `hc_head_fuse_tilelang_kernel` |

### GEMM (MoE experts)  (vLLM 20.3% | SGLang 12.9% | Atom 16.0%)

| Sub-function | vLLM | SGLang | Atom | kernel (first backend present) |
|---|---|---|---|---|
| MoE2 down  afp8_wfp4 | 11.0% | 5.9% | – | `mfma_moe2_afp8_wfp4_bf16_cshuffle_t32x256x256_…` |
| MoE1 gate/up afp8_wfp4 | 9.4% | 7.0% | – | `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_p…` |
| CK MoE MXGEMM (Atom) | – | – | 16.0% | `void ck::kernel_moe_mxgemm_2lds<ck::GridwiseMo…` |

### Memory/elementwise  (vLLM 16.6% | SGLang 6.1% | Atom 20.8%)

| Sub-function | vLLM | SGLang | Atom | kernel (first backend present) |
|---|---|---|---|---|
| Elementwise | 14.3% | 4.1% | 10.6% | `void at::native::elementwise_kernel_manual_unr…` |
| Concat (CatArray) | – | – | 9.9% | `void at::native::` |
| Fill | 1.6% | 2.0% | 0.3% | `void at::native::vectorized_elementwise_kernel…` |
| Copy buffer | 0.7% | – | – | `__amd_rocclr_copyBuffer` |
| Memcpy | – | 0.0% | – | `memcpy_triton_kernel` |

### Communication  (vLLM 7.4% | SGLang 0.6% | Atom 15.7%)

| Sub-function | vLLM | SGLang | Atom | kernel (first backend present) |
|---|---|---|---|---|
| NCCL/RCCL | 1.5% | 0.6% | 12.3% | `ncclDevKernel_Generic_1` |
| CustomAR 2-stage | 5.6% | – | 3.3% | `void vllm::cross_device_reduce_2stage<__hip_bf…` |
| CustomAR 1-stage | 0.2% | – | – | `void vllm::cross_device_reduce_1stage<__hip_bf…` |

### MoE routing  (vLLM 8.9% | SGLang 7.4% | Atom 4.0%)

| Sub-function | vLLM | SGLang | Atom | kernel (first backend present) |
|---|---|---|---|---|
| MoE sorting | 1.3% | 3.1% | 0.6% | `void aiter::opus_moe_sorting_entry<aiter::MoeS…` |
| TopK per-row (decode) | 4.2% | – | – | `void vllm::topKPerRowDecode<1024, true, false,…` |
| TopK gather (Atom) | – | – | 3.4% | `void at::native::sbtopk::gatherTopK<float, uns…` |
| MX-quant + MoE sort | 1.1% | 2.2% | – | `_ZN5aiter30fused_mx_quant_moe_sort_kernelIDF16…` |
| Gating softplus (aiter) | – | 2.1% | – | `void aiter::topk_softplus_kernel_opt<hip_bfloa…` |
| Gating softplus+sqrt | 1.6% | – | – | `void vllm::moe::topkGatingSoftplusSqrt<6, 384,…` |
| Global topk / lens | 0.6% | – | – | `_pack_global_topk_ragged_kernel` |

### Quantization  (vLLM 3.4% | SGLang 4.3% | Atom 3.1%)

| Sub-function | vLLM | SGLang | Atom | kernel (first backend present) |
|---|---|---|---|---|
| AITER per-group quant | 2.8% | 3.2% | 2.0% | `_ZN5aiter37dynamic_per_group_scaled_quant_kern…` |
| AITER dynamic quant | – | 1.1% | 1.1% | `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16b…` |
| quant_fp8 / scaled | 0.6% | – | – | `void per_token_group_quant_8bit_kernel<c10::BF…` |

### Normalization  (vLLM 2.0% | SGLang 2.4% | Atom 0.5%)

| Sub-function | vLLM | SGLang | Atom | kernel (first backend present) |
|---|---|---|---|---|
| SGL fused RMS+fp8 quant | 0.7% | 2.4% | – | `_fused_q_kv_rmsnorm_kernel` |
| CK-tile RMSNorm | 1.4% | – | – | `_ZN7ck_tile6kentryILi1ENS_12Rmsnorm2dFwdINS_27…` |
| RMSNorm | – | – | 0.5% | `_rmsnorm_nw_kernel` |

### RoPE  (vLLM 1.2% | SGLang 3.4% | Atom 0.3%)

| Sub-function | vLLM | SGLang | Atom | kernel (first backend present) |
|---|---|---|---|---|
| RoPE | 1.2% | 3.4% | 0.3% | `_inverse_rope_gptj_kernel` |

### Other  (vLLM 1.1% | SGLang 1.0% | Atom 0.5%)

| Sub-function | vLLM | SGLang | Atom | kernel (first backend present) |
|---|---|---|---|---|
| moe_reduction_kernel_plain_bf16_topk6_md7168 | 0.2% | 0.4% | – | `moe_reduction_kernel_plain_bf16_topk6_md7168` |
| void rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel<r | 0.4% | 0.1% | 0.0% | `void rocprim::ROCPRIM_400200_NS::detail::tramp…` |
|  | – | 0.4% | – | `` |
| void at::native::vectorized_gather_kernel<16, long> | 0.0% | – | 0.2% | `void at::native::vectorized_gather_kernel<16, …` |
| void at::native::reduce_kernel<512, 1, at::native::ReduceOp< | 0.1% | 0.0% | 0.0% | `void at::native::reduce_kernel<512, 1, at::nat…` |
| _pack_dense_prefix_to_ragged_kernel | 0.1% | – | – | `_pack_dense_prefix_to_ragged_kernel` |
| _dequantize_and_gather_k_kernel | 0.1% | – | – | `_dequantize_and_gather_k_kernel` |
| void aiter::radix_topk_one_block_kernel<float, int, 11, 1024 | – | – | 0.1% | `void aiter::radix_topk_one_block_kernel<float,…` |

### Activation  (vLLM – | SGLang 1.1% | Atom 0.1%)

| Sub-function | vLLM | SGLang | Atom | kernel (first backend present) |
|---|---|---|---|---|
| SiLU/SwiGLU | – | 1.1% | 0.1% | `_fused_clamp_silu_mul_kernel_BLOCK_SIZE_N_4096…` |


## Three-way takeaways (conc=64 decode)

**MoE experts — three different backends:**
- vLLM & SGLang -> **AITER `flydsl` `mfma_moe1/2 afp8_wfp4`** (hand-gen MFMA assembly, FP8xFP4).
- **Atom -> Composable Kernel `ck::kernel_moe_mxgemm_2lds` (GridwiseMoeGemmMX, B-preshuffled)** — a different MoE GEMM path (CK, not AITER ASM), 16.0% of decode.

**MLA latent compression — universal:** all three use the same **`aiter::mhc_*`** kernels
(`mhc_pre_gemm_sqrsum` / `mhc_pre_big_fuse` / `mhc_post`). The one shared hot path.

**Dense/MLA-proj GEMM — weight layout:**
- vLLM: non-preshuffled `ck...GridwiseGemmMultiD_ABScale` + a8w8 + wvSplitK.
- SGLang: `aiter::fp8gemm...BpreShuffle` + `ck...b_preshuffle` (fully preshuffled).
- Atom: `ck...b_preshuffle` + hipBLASLt (preshuffled).
- -> **SGLang and Atom both use weight-preshuffled dense GEMM; vLLM does not.**

**Attention — three implementations, similar-to-higher cost:**
- vLLM: DeepSeek-Sparse-Attention (`_sparse_attn_decode_partial/reduce`) — 18.5%.
- SGLang: `dsv4` unified-KV Triton `_paged_decode_*` + compressed `flash_c*` — 18.1%.
- Atom: `_sparse_attn_ragged_varlen_triton` + `_fused_compress_attn` — **25.4% (highest)**.

**Communication — Atom is comm-heavy:** Atom **15.7%** (`ncclDevKernel` 12.3% +
`aiter::cross_device_reduce_2stage` 3.3%) vs vLLM 7.4% (custom AR). Atom leans on RCCL,
not just custom AR. (SGLang 0.6% is a capture-trace artifact, not real.)

**Elementwise — Atom concat-heavy:** Atom **20.8%** (incl. `CatArrayBatchedCopy` ~10% —
heavy tensor concatenation) vs vLLM 16.6% vs SGLang 6.1%.

**Quant — universal:** all three use `aiter::dynamic_per_group_scaled_quant` (~3-4%).

### Bottom line
- The **FP8xFP4 MoE GEMM is where the three backends most diverge**: AITER-ASM (vLLM/SGLang)
  vs CK-MXGEMM (Atom).
- **AITER `mhc_*` MLA compression + `dynamic_per_group_scaled_quant` are common to all three.**
- **Weight-preshuffled dense GEMM** is used by SGLang & Atom, not vLLM (optimization Target #7).
- Atom's runtime profile is **communication- and concat-heavy** (RCCL 12% + CatArray 10%), a
  distinct bottleneck vs the other two.

> Trace-method caveat: vLLM & Atom are steady-state **runtime** traces (directly comparable);
> SGLang is a **graph-capture** structural trace (comm understated, %s directional).
## Serving latency & throughput (conc=64, real / non-profiled)

Best-throughput conc64 point per backend (mi355x fp4, ISL=OSL=1024), from the InferenceX
dashboard — these are the real, un-profiled serving numbers.

| Backend | tput/GPU (tok/s) | TTFT (ms) | TPOT (ms) | recipe |
|---|---|---|---|---|
| vLLM   | 456 | 280 | 33.8 | TP8, dp-attn off |
| SGLang | 781 | 1400 | 38.3 | TP8, dp-attn, ep1 |
| Atom   | 608 | 217 | 23.4 | TP8, dp-attn off, ep1 |

> SGLang wins peak throughput (781); Atom has the lowest TPOT (23.4 ms) and TTFT (217 ms).
> Multiple conc64 recipe variants exist per backend (e.g. sglang 505/652/781 tok/s at
> different TTFT/TPOT tradeoffs); the peak-throughput point is shown.

## Absolute per-function decode time (ms per output token = function% x TPOT)

Grounded in the real TPOT above. Shown for the **runtime** traces (vLLM, Atom); SGLang is
a graph-capture trace so its per-function *proportions* aren't steady-state — its column is
**directional only** (capture% x TPOT, italic).

| Function | vLLM ms | Atom ms | SGLang ms* |
|---|---|---|---|
| GEMM (dense/linear) | 7.00 | 3.21 | *16.35* |
| Attention (MLA / DSA) | 6.26 | 5.95 | *6.93* |
| GEMM (MoE experts) | 6.87 | 3.75 | *4.94* |
| Memory/elementwise | 5.62 | 4.88 | *2.34* |
| Communication | 2.50 | 3.68 | *0.23* |
| MoE routing | 3.01 | 0.94 | *2.83* |
| Quantization | 1.15 | 0.73 | *1.65* |
| Normalization | 0.68 | 0.12 | *0.92* |
| RoPE | 0.41 | 0.07 | *1.30* |
| Other | 0.37 | 0.12 | *0.38* |
| **total (≈TPOT)** | **33.8** | **23.4** | **38.3** |

> Read across a row = wall-clock each backend spends on that function per decoded token.
> E.g. MoE experts: vLLM 6.9 ms vs Atom 3.7 ms/token; Attention: Atom 6.0 ms vs vLLM 6.3 ms;
> Communication: Atom 3.7 ms vs vLLM 2.5 ms/token (Atom's RCCL-heavy AR).
