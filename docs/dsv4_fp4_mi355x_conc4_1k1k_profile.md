# DeepSeek-V4-Pro — Per-Stage GPU Kernel Profile (MI355X)

| Field | Value |
|---|---|
| Model | `deepseek-ai/DeepSeek-V4-Pro` (FP4 MoE + FP8 attn, FP8 KV) |
| Hardware | 8× AMD Instinct MI355X (gfx950), TP=8 |
| Engine | vLLM 0.23.1rc1.dev714+g09663abde (image `vllm/vllm-openai-rocm:nightly-09663abde...`) |
| Recipe | `dsv4_fp4_mi355x_profiling.sh` (AITER MoE, `--moe-backend aiter`, cudagraph FULL_AND_PIECEWISE) |
| Workload | ISL=1024, OSL=1024, concurrency=4, random-range-ratio=0.8 |
| Traces | 8 GPU ranks, torch profiler (record_shapes) |

## Overall prefill vs decode split

| Stage | GPU time (ms) | % of total | steps (Σranks) |
|---|---|---|---|
| PREFILL | 4198.3 | 1.8% | 24 |
| DECODE | 234394.6 | 98.2% | 8,104 |

> Aggregate GPU-kernel time summed across 8 ranks; a torch-profiler window of 5 active engine iterations at conc=32. Prefill covers the context (ctx>0) steps, decode the pure-generation (ctx==0) steps.

## PREFILL

- Aggregate GPU-kernel time (8 ranks): **4198.3 ms**
- Kernel launches: **111,132**
- Steps (summed across ranks): **24**

### PREFILL — kernel time by category

| Category | time (ms) | % | launches |
|---|---|---|---|
| Communication | 2176.2 | 51.8% | 2,976 |
| GEMM (dense/linear) | 586.0 | 14.0% | 14,152 |
| Attention (MLA / DSA) | 528.4 | 12.6% | 18,024 |
| GEMM (MoE experts) | 351.9 | 8.4% | 2,928 |
| Memory/elementwise | 204.7 | 4.9% | 39,932 |
| Other | 135.8 | 3.2% | 9,736 |
| MoE routing | 79.7 | 1.9% | 7,768 |
| Quantization | 53.7 | 1.3% | 9,016 |
| RoPE | 42.8 | 1.0% | 2,184 |
| Normalization | 39.1 | 0.9% | 4,416 |

### PREFILL — GEMM libraries (22.3% of stage GPU time)

| GEMM library | time (ms) | % | launches |
|---|---|---|---|
| Composable Kernel (ck gemm_xdl, ABscale) | 361.4 | 8.6% | 6,576 |
| AITER MoE expert GEMM (ASM, FP8xFP4) | 351.9 | 8.4% | 2,928 |
| hipBLASLt (Tensile) | 150.4 | 3.6% | 5,360 |
| AITER a8w8 block-scale GEMM (CK) | 69.8 | 1.7% | 1,464 |
| wvSplitK (skinny/thin-M GEMM) | 4.5 | 0.1% | 752 |

### PREFILL — top AITER functions (24.3% of stage GPU time, 35 distinct)

| time (ms) | % | launches | kernel |
|---|---|---|---|
| 169.2 | 4.0% | 976 | `_sparse_attn_prefill_ragged_kernel` |
| 120.3 | 2.9% | 488 | `mfma_moe1_silu_mul_afp8_wfp4_fp8_t64x256x256_pm1_fp8q_sort_async_gui_v32` |
| 99.5 | 2.4% | 976 | `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v32` |
| 77.6 | 1.8% | 1,952 | `_ZN5aiter26mhc_pre_gemm_sqrsum_kernelIDF16bLi256ELi64ELi32ELi128EEEvPfS1_PT_S1_iiiiiii` |
| 75.8 | 1.8% | 488 | `mfma_moe2_afp8_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_pm1_acc0` |
| 62.4 | 1.5% | 976 | `_ZN5aiter15mhc_post_kernelIDF16bLi256ELi4ELi1024ELb1EEEvPT_S2_S2_PfS3_iiiii` |
| 59.7 | 1.4% | 1,464 | `_fused_kv_compress_norm_rope_insert_sparse_attn` |
| 56.4 | 1.3% | 976 | `mfma_moe2_afp8_wfp4_bf16_cshuffle_t32x256x256_vscale_fix3_fp4opt_v1_pm1` |
| 48.8 | 1.2% | 488 | `_gemm_a8w8_blockscale_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_128_BLOCK_SIZE_N_128_BLO…` |
| 36.7 | 0.9% | 976 | `_ZN5aiter23mhc_pre_big_fuse_kernelIDF16bLi320ELi4ELi2ELi256ELb1EEEvPfS1_PT_S1_S1_S1_S1_S3_…` |
| 36.5 | 0.9% | 6,576 | `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb0ELi64ELb0EEEvPT0_P…` |
| 26.0 | 0.6% | 1,952 | `_ZN5aiter15mhc_post_kernelIDF16bLi256ELi4ELi1024ELb0EEEvPT_S2_S2_PfS3_iiiii` |
| 21.1 | 0.5% | 976 | `_sparse_attn_decode_partial_kernel` |
| 19.0 | 0.5% | 1,952 | `_ZN5aiter23mhc_pre_big_fuse_kernelIDF16bLi320ELi4ELi2ELi256ELb0EEEvPfS1_PT_S1_S1_S1_S1_S3_…` |
| 18.6 | 0.4% | 488 | `_gemm_a8w8_blockscale_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_128_BLOCK_SIZE_N_128_BLO…` |

## DECODE

- Aggregate GPU-kernel time (8 ranks): **234394.6 ms**
- Kernel launches: **33,316,172**
- Steps (summed across ranks): **8,104**

### DECODE — kernel time by category

| Category | time (ms) | % | launches |
|---|---|---|---|
| GEMM (dense/linear) | 57830.6 | 24.7% | 4,692,216 |
| Memory/elementwise | 54039.4 | 23.1% | 12,148,516 |
| Attention (MLA / DSA) | 42902.2 | 18.3% | 6,167,144 |
| MoE routing | 29218.8 | 12.5% | 2,455,512 |
| GEMM (MoE experts) | 14698.7 | 6.3% | 988,688 |
| Quantization | 11070.4 | 4.7% | 2,714,840 |
| Communication | 10333.9 | 4.4% | 1,004,896 |
| Normalization | 6536.5 | 2.8% | 1,491,136 |
| Other | 4447.1 | 1.9% | 915,760 |
| RoPE | 3316.9 | 1.4% | 737,464 |

### DECODE — GEMM libraries (30.9% of stage GPU time)

| GEMM library | time (ms) | % | launches |
|---|---|---|---|
| Composable Kernel (ck gemm_xdl, ABscale) | 33461.2 | 14.3% | 2,220,496 |
| hipBLASLt (Tensile) | 17320.6 | 7.4% | 1,231,808 |
| AITER MoE expert GEMM (ASM, FP8xFP4) | 14698.7 | 6.3% | 988,688 |
| wvSplitK (skinny/thin-M GEMM) | 4537.1 | 1.9% | 745,568 |
| AITER a8w8 block-scale GEMM (CK) | 2511.7 | 1.1% | 494,344 |

### DECODE — top AITER functions (33.1% of stage GPU time, 24 distinct)

| time (ms) | % | launches | kernel |
|---|---|---|---|
| 10430.7 | 4.5% | 494,344 | `_sparse_attn_decode_partial_kernel` |
| 9035.3 | 3.9% | 2,220,496 | `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb0ELi64ELb0EEEvPT0_P…` |
| 8908.1 | 3.8% | 430,904 | `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_xcd4_v32` |
| 6657.3 | 2.8% | 494,344 | `_sparse_attn_decode_reduce_kernel` |
| 5223.7 | 2.2% | 494,344 | `void aiter::opus_moe_sorting_entry<aiter::MoeSortingKernel<aiter::MoeSortingProblemEx<int,…` |
| 4943.7 | 2.1% | 988,688 | `_ZN5aiter23mhc_pre_big_fuse_kernelIDF16bLi320ELi4ELi2ELi256ELb0EEEvPfS1_PT_S1_S1_S1_S1_S3_…` |
| 4511.4 | 1.9% | 988,688 | `_ZN5aiter15mhc_post_kernelIDF16bLi256ELi4ELi1024ELb0EEEvPT_S2_S2_PfS3_iiiii` |
| 4448.3 | 1.9% | 988,688 | `_ZN5aiter26mhc_pre_gemm_sqrsum_kernelIDF16bLi256ELi64ELi16ELi128EEEvPfS1_PT_S1_iiiiiii` |
| 4271.8 | 1.8% | 430,904 | `mfma_moe2_afp8_wfp4_bf16_cshuffle_t32x256x256_vscale_fix3_fp4opt_v1_pm1_xcd4` |
| 3747.4 | 1.6% | 642,824 | `void wvSplitK_hf_sml_<__hip_bfloat16, 64, 1, 16, 8, 4, 4>(int, int, int, int, int, int, __…` |
| 3663.7 | 1.6% | 494,344 | `_fused_kv_compress_norm_rope_insert_sparse_attn` |
| 2511.7 | 1.1% | 494,344 | `_gemm_a8w8_blockscale_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_8_BLOCK_SIZE_N_64_BLOCK_…` |
| 2213.8 | 0.9% | 494,344 | `_ZN5aiter30fused_mx_quant_moe_sort_kernelIDF16bDB8_Li256ELi32EEEvPT0_PhPKT_PKiS9_PKfiiiiii…` |
| 2144.3 | 0.9% | 494,344 | `_gemm_a8w8_blockscale_reduce_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_32_ACTUAL_KSPLIT_8_MAX_KS…` |
| 1471.9 | 0.6% | 243,120 | `_fused_kv_compress_norm_rope_insert_indexer_attn` |

## Key observations

- **Decode-dominated window.** At ISL=OSL=1024 the profiled window is 98% decode / 2% prefill, so the steady-state kernel mix below reflects the decode path.
- **GEMM library split is stable across stages:** MoE experts run on the **AITER assembly FP8×FP4** kernels (`mfma_moe1/moe2 … afp8_wfp4`), dense/linear layers on **Composable Kernel** `ck::gemm_xdl_cshuffle_v3` (AB-scale FP8), with **hipBLASLt (Tensile)** picking up a minority of dense shapes.
- **`wvSplitK` is decode-only** (~1.8% decode vs ~0% prefill): the skinny/thin-M GEMM path is selected when M = batch is tiny (decode), and CK/hipBLASLt take over at prefill's larger M.
- **Prefill is communication-heavy** (~23% TP=8 custom all-reduce) because activation tensors are large; decode all-reduce drops to ~6%.
- **DeepSeek Sparse Attention (DSA) + MLA** is a first-class cost: `_sparse_attn_*`, `_fused_kv_compress_norm_rope_insert`, the MQA-logits indexer, and the AITER `mhc_*` MLA helpers together are ~17–23% of each stage. Prefill uses `_sparse_attn_prefill_ragged`; decode uses `_sparse_attn_decode_partial/reduce`.
- **AITER per-group FP8 quant dominates launch count** (`dynamic_per_group_scaled_quant`, ~2.2M launches in decode): every GEMM is preceded by an activation quant.
- **Elementwise/copy is the largest decode bucket by time (~22%) and by far the largest by launch count (~12M)** — a fusion opportunity if decode latency matters.
