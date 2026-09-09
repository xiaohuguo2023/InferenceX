# DeepSeek-V4-Pro — Per-Stage GPU Kernel Profile (MI355X)

| Field | Value |
|---|---|
| Model | `deepseek-ai/DeepSeek-V4-Pro` (FP4 MoE + FP8 attn, FP8 KV) |
| Hardware | 8× AMD Instinct MI355X (gfx950), TP=8 |
| Engine | vLLM 0.23.1rc1.dev714+g09663abde (image `vllm/vllm-openai-rocm:nightly-09663abde...`) |
| Recipe | `dsv4_fp4_mi355x_profiling.sh` (AITER MoE, `--moe-backend aiter`, cudagraph FULL_AND_PIECEWISE) |
| Workload | ISL=1024, OSL=1024, concurrency=256, random-range-ratio=0.8 |
| Traces | 8 GPU ranks, torch profiler (record_shapes) |

## Overall prefill vs decode split

| Stage | GPU time (ms) | % of total | steps (Σranks) |
|---|---|---|---|
| PREFILL | 102668.7 | 16.3% | 256 |
| DECODE | 525304.7 | 83.7% | 8,112 |

> Aggregate GPU-kernel time summed across 8 ranks; a torch-profiler window of 5 active engine iterations at conc=32. Prefill covers the context (ctx>0) steps, decode the pure-generation (ctx==0) steps.

## PREFILL

- Aggregate GPU-kernel time (8 ranks): **102668.7 ms**
- Kernel launches: **1,619,351**
- Steps (summed across ranks): **256**

### PREFILL — kernel time by category

| Category | time (ms) | % | launches |
|---|---|---|---|
| GEMM (dense/linear) | 27361.4 | 26.7% | 142,736 |
| Attention (MLA / DSA) | 26065.8 | 25.4% | 237,032 |
| Communication | 18259.1 | 17.8% | 31,744 |
| GEMM (MoE experts) | 10641.5 | 10.4% | 31,232 |
| Other | 6851.0 | 6.7% | 283,472 |
| Memory/elementwise | 5930.6 | 5.8% | 609,495 |
| RoPE | 2950.3 | 2.9% | 23,296 |
| Quantization | 1747.5 | 1.7% | 100,888 |
| Normalization | 1475.1 | 1.4% | 47,104 |
| MoE routing | 1386.2 | 1.4% | 112,352 |

### PREFILL — GEMM libraries (37.0% of stage GPU time)

| GEMM library | time (ms) | % | launches |
|---|---|---|---|
| Composable Kernel (ck gemm_xdl, ABscale) | 18605.0 | 18.1% | 70,144 |
| AITER MoE expert GEMM (ASM, FP8xFP4) | 10641.5 | 10.4% | 31,232 |
| hipBLASLt (Tensile) | 5212.5 | 5.1% | 56,208 |
| AITER a8w8 block-scale GEMM (CK) | 3538.7 | 3.4% | 15,616 |
| wvSplitK (skinny/thin-M GEMM) | 4.2 | 0.0% | 744 |
| Triton GEMM | 1.0 | 0.0% | 24 |

### PREFILL — top AITER functions (40.2% of stage GPU time, 34 distinct)

| time (ms) | % | launches | kernel |
|---|---|---|---|
| 10280.3 | 10.0% | 43,432 | `_sparse_attn_prefill_ragged_kernel` |
| 5848.6 | 5.7% | 14,152 | `mfma_moe1_silu_mul_afp8_wfp4_fp8_t128x256x256_pm1_fp8q_sort_async_gui_v32` |
| 4976.8 | 4.8% | 28,304 | `_ZN5aiter15mhc_post_kernelIDF16bLi256ELi4ELi1024ELb1EEEvPT_S2_S2_PfS3_iiiii` |
| 4454.0 | 4.3% | 14,152 | `mfma_moe2_afp8_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_pm1_sbm128_acc0` |
| 4226.1 | 4.1% | 30,256 | `_ZN5aiter26mhc_pre_gemm_sqrsum_kernelIDF16bLi256ELi64ELi32ELi128EEEvPfS1_PT_S1_iiiiiii` |
| 3485.1 | 3.4% | 14,152 | `_gemm_a8w8_blockscale_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_128_BLOCK_SIZE_N_128_BLO…` |
| 2559.5 | 2.5% | 28,304 | `_ZN5aiter23mhc_pre_big_fuse_kernelIDF16bLi320ELi4ELi2ELi256ELb1EEEvPfS1_PT_S1_S1_S1_S1_S3_…` |
| 1585.6 | 1.5% | 15,616 | `_fused_kv_compress_norm_rope_insert_sparse_attn` |
| 1058.1 | 1.0% | 70,144 | `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb0ELi64ELb0EEEvPT0_P…` |
| 927.1 | 0.9% | 15,128 | `_sparse_attn_decode_partial_kernel` |
| 420.6 | 0.4% | 15,128 | `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi32ELb0ELi64ELb1EEEvPT0_Pf…` |
| 292.6 | 0.3% | 15,128 | `void aiter::mxfp4_moe_sort_kernel<256, 32, 32, 32>(unsigned char*, unsigned char*, int con…` |
| 234.3 | 0.2% | 15,128 | `void aiter::opus_moe_sorting_entry<aiter::MoeSortingMultiPhaseKernel_P23<aiter::MoeSorting…` |
| 215.3 | 0.2% | 1,464 | `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v32` |
| 123.7 | 0.1% | 1,464 | `mfma_moe2_afp8_wfp4_bf16_cshuffle_t32x256x256_vscale_fix3_fp4opt_v1_pm1` |

## DECODE

- Aggregate GPU-kernel time (8 ranks): **525304.7 ms**
- Kernel launches: **33,616,529**
- Steps (summed across ranks): **8,112**

### DECODE — kernel time by category

| Category | time (ms) | % | launches |
|---|---|---|---|
| Memory/elementwise | 121459.3 | 23.1% | 12,220,289 |
| Attention (MLA / DSA) | 112063.1 | 21.3% | 6,173,232 |
| GEMM (MoE experts) | 100111.8 | 19.1% | 989,664 |
| GEMM (dense/linear) | 77819.1 | 14.8% | 4,207,376 |
| Communication | 46125.4 | 8.8% | 1,005,888 |
| MoE routing | 34346.9 | 6.5% | 2,943,008 |
| Quantization | 11325.1 | 2.2% | 2,717,520 |
| Other | 9924.5 | 1.9% | 1,128,752 |
| Normalization | 7628.5 | 1.5% | 1,492,608 |
| RoPE | 4501.0 | 0.9% | 738,192 |

### DECODE — GEMM libraries (33.9% of stage GPU time)

| GEMM library | time (ms) | % | launches |
|---|---|---|---|
| AITER MoE expert GEMM (ASM, FP8xFP4) | 100111.8 | 19.1% | 989,664 |
| Composable Kernel (ck gemm_xdl, ABscale) | 40658.5 | 7.7% | 2,222,688 |
| hipBLASLt (Tensile) | 24080.9 | 4.6% | 1,448,560 |
| AITER a8w8 block-scale GEMM (CK) | 12767.9 | 2.4% | 494,832 |
| wvSplitK (skinny/thin-M GEMM) | 304.0 | 0.1% | 41,112 |
| Triton GEMM | 7.7 | 0.0% | 184 |

### DECODE — top AITER functions (45.5% of stage GPU time, 40 distinct)

| time (ms) | % | launches | kernel |
|---|---|---|---|
| 60508.3 | 11.5% | 490,928 | `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v32` |
| 50920.6 | 9.7% | 494,832 | `_sparse_attn_decode_partial_kernel` |
| 35769.6 | 6.8% | 438,224 | `mfma_moe2_afp8_wfp4_bf16_cshuffle_t16x128x256_vscale_fix3_fp4opt_v1_pm1_sbm32` |
| 24846.1 | 4.7% | 494,832 | `_fused_kv_compress_norm_rope_insert_sparse_attn` |
| 11241.8 | 2.1% | 413,336 | `_gemm_a8w8_blockscale_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_64_BLOCK_SIZE_N_128_BLOC…` |
| 9261.1 | 1.8% | 2,222,688 | `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb0ELi64ELb0EEEvPT0_P…` |
| 8761.8 | 1.7% | 876,448 | `_ZN5aiter26mhc_pre_gemm_sqrsum_kernelIDF16bLi256ELi64ELi32ELi128EEEvPfS1_PT_S1_iiiiiii` |
| 7230.1 | 1.4% | 989,664 | `_ZN5aiter23mhc_pre_big_fuse_kernelIDF16bLi320ELi4ELi2ELi256ELb0EEEvPfS1_PT_S1_S1_S1_S1_S3_…` |
| 7117.6 | 1.4% | 989,664 | `_ZN5aiter15mhc_post_kernelIDF16bLi256ELi4ELi1024ELb0EEEvPT_S2_S2_PfS3_iiiii` |
| 6276.5 | 1.2% | 494,832 | `_ZN5aiter30fused_mx_quant_moe_sort_kernelIDF16bDB8_Li256ELi32EEEvPT0_PhPKT_PKiS9_PKfiiiiii…` |
| 3727.2 | 0.7% | 53,680 | `mfma_moe2_afp8_wfp4_bf16_cshuffle_t32x256x256_vscale_fix3_fp4opt_v1_pm1_xcd4` |
| 2800.6 | 0.5% | 494,832 | `_sparse_attn_decode_reduce_kernel` |
| 2331.3 | 0.4% | 485,072 | `void aiter::opus_moe_sorting_entry<aiter::MoeSortingMultiPhaseKernel_P23<aiter::MoeSorting…` |
| 2067.6 | 0.4% | 485,072 | `void aiter::opus_moe_sorting_entry<aiter::MoeSortingMultiPhaseKernel_P0_v2<aiter::MoeSorti…` |
| 1718.9 | 0.3% | 243,360 | `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle` |

## Key observations

- **Decode-dominated window.** At ISL=OSL=1024 the profiled window is 84% decode / 16% prefill, so the steady-state kernel mix below reflects the decode path.
- **GEMM library split is stable across stages:** MoE experts run on the **AITER assembly FP8×FP4** kernels (`mfma_moe1/moe2 … afp8_wfp4`), dense/linear layers on **Composable Kernel** `ck::gemm_xdl_cshuffle_v3` (AB-scale FP8), with **hipBLASLt (Tensile)** picking up a minority of dense shapes.
- **`wvSplitK` is decode-only** (~1.8% decode vs ~0% prefill): the skinny/thin-M GEMM path is selected when M = batch is tiny (decode), and CK/hipBLASLt take over at prefill's larger M.
- **Prefill is communication-heavy** (~23% TP=8 custom all-reduce) because activation tensors are large; decode all-reduce drops to ~6%.
- **DeepSeek Sparse Attention (DSA) + MLA** is a first-class cost: `_sparse_attn_*`, `_fused_kv_compress_norm_rope_insert`, the MQA-logits indexer, and the AITER `mhc_*` MLA helpers together are ~17–23% of each stage. Prefill uses `_sparse_attn_prefill_ragged`; decode uses `_sparse_attn_decode_partial/reduce`.
- **AITER per-group FP8 quant dominates launch count** (`dynamic_per_group_scaled_quant`, ~2.2M launches in decode): every GEMM is preceded by an activation quant.
- **Elementwise/copy is the largest decode bucket by time (~22%) and by far the largest by launch count (~12M)** — a fusion opportunity if decode latency matters.
