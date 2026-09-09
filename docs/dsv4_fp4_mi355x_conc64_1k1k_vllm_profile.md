# DeepSeek-V4-Pro — Per-Stage GPU Kernel Profile (MI355X)

| Field | Value |
|---|---|
| Model | `deepseek-ai/DeepSeek-V4-Pro` (FP4 MoE + FP8 attn, FP8 KV) |
| Hardware | 8× AMD Instinct MI355X (gfx950), TP=8 |
| Engine | vLLM 0.23.1rc1.dev714+g09663abde (image `vllm/vllm-openai-rocm:nightly-09663abde...`) |
| Recipe | `dsv4_fp4_mi355x_profiling.sh` (AITER MoE, `--moe-backend aiter`, cudagraph FULL_AND_PIECEWISE) |
| Workload | ISL=1024, OSL=1024, concurrency=64, random-range-ratio=0.8 |
| Traces | 8 GPU ranks, torch profiler (record_shapes) |

## Overall prefill vs decode split

| Stage | GPU time (ms) | % of total | steps (Σranks) |
|---|---|---|---|
| PREFILL | 26660.3 | 7.3% | 80 |
| DECODE | 339134.0 | 92.7% | 8,176 |

> Aggregate GPU-kernel time summed across 8 ranks; a torch-profiler window of 5 active engine iterations at conc=32. Prefill covers the context (ctx>0) steps, decode the pure-generation (ctx==0) steps.

## PREFILL

- Aggregate GPU-kernel time (8 ranks): **26660.3 ms**
- Kernel launches: **474,999**
- Steps (summed across ranks): **80**

### PREFILL — kernel time by category

| Category | time (ms) | % | launches |
|---|---|---|---|
| GEMM (dense/linear) | 6909.1 | 25.9% | 45,136 |
| Attention (MLA / DSA) | 6465.1 | 24.2% | 70,888 |
| Communication | 5423.6 | 20.3% | 9,920 |
| GEMM (MoE experts) | 2792.8 | 10.5% | 9,760 |
| Other | 1728.4 | 6.5% | 76,416 |
| Memory/elementwise | 1387.6 | 5.2% | 177,415 |
| RoPE | 726.3 | 2.7% | 7,280 |
| Quantization | 457.3 | 1.7% | 31,192 |
| MoE routing | 389.4 | 1.5% | 32,272 |
| Normalization | 380.7 | 1.4% | 14,720 |

### PREFILL — GEMM libraries (36.4% of stage GPU time)

| GEMM library | time (ms) | % | launches |
|---|---|---|---|
| Composable Kernel (ck gemm_xdl, ABscale) | 4678.5 | 17.5% | 21,920 |
| AITER MoE expert GEMM (ASM, FP8xFP4) | 2792.8 | 10.5% | 9,760 |
| hipBLASLt (Tensile) | 1330.8 | 5.0% | 17,568 |
| AITER a8w8 block-scale GEMM (CK) | 894.6 | 3.4% | 4,880 |
| wvSplitK (skinny/thin-M GEMM) | 4.2 | 0.0% | 744 |
| Triton GEMM | 1.0 | 0.0% | 24 |

### PREFILL — top AITER functions (39.2% of stage GPU time, 35 distinct)

| time (ms) | % | launches | kernel |
|---|---|---|---|
| 2589.9 | 9.7% | 11,224 | `_sparse_attn_prefill_ragged_kernel` |
| 1408.3 | 5.3% | 3,416 | `mfma_moe1_silu_mul_afp8_wfp4_fp8_t128x256x256_pm1_fp8q_sort_async_gui_v32` |
| 1194.3 | 4.5% | 6,832 | `_ZN5aiter15mhc_post_kernelIDF16bLi256ELi4ELi1024ELb1EEEvPT_S2_S2_PfS3_iiiii` |
| 1069.9 | 4.0% | 3,416 | `mfma_moe2_afp8_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_pm1_sbm128_acc0` |
| 1060.2 | 4.0% | 8,784 | `_ZN5aiter26mhc_pre_gemm_sqrsum_kernelIDF16bLi256ELi64ELi32ELi128EEEvPfS1_PT_S1_iiiiiii` |
| 841.9 | 3.2% | 3,416 | `_gemm_a8w8_blockscale_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_128_BLOCK_SIZE_N_128_BLO…` |
| 618.6 | 2.3% | 6,832 | `_ZN5aiter23mhc_pre_big_fuse_kernelIDF16bLi320ELi4ELi2ELi256ELb1EEEvPfS1_PT_S1_S1_S1_S1_S3_…` |
| 418.0 | 1.6% | 4,880 | `_fused_kv_compress_norm_rope_insert_sparse_attn` |
| 279.8 | 1.0% | 21,920 | `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb0ELi64ELb0EEEvPT0_P…` |
| 199.4 | 0.7% | 1,464 | `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v32` |
| 115.2 | 0.4% | 1,464 | `mfma_moe2_afp8_wfp4_bf16_cshuffle_t32x256x256_vscale_fix3_fp4opt_v1_pm1` |
| 114.1 | 0.4% | 4,392 | `_sparse_attn_decode_partial_kernel` |
| 107.0 | 0.4% | 4,392 | `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi32ELb0ELi64ELb1EEEvPT0_Pf…` |
| 75.7 | 0.3% | 4,392 | `void aiter::mxfp4_moe_sort_kernel<256, 32, 32, 32>(unsigned char*, unsigned char*, int con…` |
| 63.9 | 0.2% | 4,392 | `void aiter::opus_moe_sorting_entry<aiter::MoeSortingMultiPhaseKernel_P23<aiter::MoeSorting…` |

## DECODE

- Aggregate GPU-kernel time (8 ranks): **339134.0 ms**
- Kernel launches: **33,665,561**
- Steps (summed across ranks): **8,176**

### DECODE — kernel time by category

| Category | time (ms) | % | launches |
|---|---|---|---|
| Memory/elementwise | 72310.5 | 21.3% | 12,309,841 |
| GEMM (MoE experts) | 68261.0 | 20.1% | 997,472 |
| GEMM (dense/linear) | 61687.0 | 18.2% | 4,272,560 |
| Attention (MLA / DSA) | 57967.2 | 17.1% | 6,221,936 |
| MoE routing | 30517.7 | 9.0% | 2,948,736 |
| Communication | 20684.0 | 6.1% | 1,013,824 |
| Quantization | 11493.7 | 3.4% | 2,738,960 |
| Normalization | 6729.3 | 2.0% | 1,504,384 |
| Other | 6097.2 | 1.8% | 913,832 |
| RoPE | 3386.4 | 1.0% | 744,016 |

### DECODE — GEMM libraries (38.3% of stage GPU time)

| GEMM library | time (ms) | % | launches |
|---|---|---|---|
| AITER MoE expert GEMM (ASM, FP8xFP4) | 68261.0 | 20.1% | 997,472 |
| Composable Kernel (ck gemm_xdl, ABscale) | 34612.7 | 10.2% | 2,240,224 |
| hipBLASLt (Tensile) | 17787.0 | 5.2% | 1,250,136 |
| AITER a8w8 block-scale GEMM (CK) | 7268.0 | 2.1% | 498,736 |
| wvSplitK (skinny/thin-M GEMM) | 1991.1 | 0.6% | 282,784 |
| Triton GEMM | 28.3 | 0.0% | 680 |

### DECODE — top AITER functions (43.5% of stage GPU time, 37 distinct)

| time (ms) | % | launches | kernel |
|---|---|---|---|
| 37087.8 | 10.9% | 495,808 | `mfma_moe2_afp8_wfp4_bf16_cshuffle_t32x256x256_vscale_fix3_fp4opt_v1_pm1_xcd4` |
| 30912.9 | 9.1% | 488,000 | `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v32` |
| 18084.5 | 5.3% | 498,736 | `_sparse_attn_decode_partial_kernel` |
| 12885.4 | 3.8% | 498,736 | `_fused_kv_compress_norm_rope_insert_sparse_attn` |
| 9416.2 | 2.8% | 2,240,224 | `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb0ELi64ELb0EEEvPT0_P…` |
| 6854.6 | 2.0% | 450,424 | `_gemm_a8w8_blockscale_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_32_BLOCK_SIZE_N_16_BLOCK…` |
| 5438.4 | 1.6% | 997,472 | `_ZN5aiter23mhc_pre_big_fuse_kernelIDF16bLi320ELi4ELi2ELi256ELb0EEEvPfS1_PT_S1_S1_S1_S1_S3_…` |
| 5027.2 | 1.5% | 997,472 | `_ZN5aiter26mhc_pre_gemm_sqrsum_kernelIDF16bLi256ELi64ELi16ELi128EEEvPfS1_PT_S1_iiiiiii` |
| 4754.2 | 1.4% | 997,472 | `_ZN5aiter15mhc_post_kernelIDF16bLi256ELi4ELi1024ELb0EEEvPT_S2_S2_PfS3_iiiii` |
| 3880.6 | 1.1% | 498,736 | `_ZN5aiter30fused_mx_quant_moe_sort_kernelIDF16bDB8_Li256ELi32EEEvPT0_PhPKT_PKiS9_PKfiiiiii…` |
| 2956.5 | 0.9% | 498,736 | `_sparse_attn_decode_reduce_kernel` |
| 2116.3 | 0.6% | 471,408 | `void aiter::opus_moe_sorting_entry<aiter::MoeSortingMultiPhaseKernel_P23<aiter::MoeSorting…` |
| 1976.0 | 0.6% | 471,408 | `void aiter::opus_moe_sorting_entry<aiter::MoeSortingMultiPhaseKernel_P0_v2<aiter::MoeSorti…` |
| 1735.0 | 0.5% | 245,280 | `_fused_kv_compress_norm_rope_insert_indexer_attn` |
| 1568.3 | 0.5% | 221,520 | `void wvSplitKrc_<__hip_bfloat16, 64, 16, 4, 8, 1, 64, 4, 2, 1>(int, int, int, int, int, in…` |

## Key observations

- **Decode-dominated window.** At ISL=OSL=1024 the profiled window is 93% decode / 7% prefill, so the steady-state kernel mix below reflects the decode path.
- **GEMM library split is stable across stages:** MoE experts run on the **AITER assembly FP8×FP4** kernels (`mfma_moe1/moe2 … afp8_wfp4`), dense/linear layers on **Composable Kernel** `ck::gemm_xdl_cshuffle_v3` (AB-scale FP8), with **hipBLASLt (Tensile)** picking up a minority of dense shapes.
- **`wvSplitK` is decode-only** (~1.8% decode vs ~0% prefill): the skinny/thin-M GEMM path is selected when M = batch is tiny (decode), and CK/hipBLASLt take over at prefill's larger M.
- **Prefill is communication-heavy** (~23% TP=8 custom all-reduce) because activation tensors are large; decode all-reduce drops to ~6%.
- **DeepSeek Sparse Attention (DSA) + MLA** is a first-class cost: `_sparse_attn_*`, `_fused_kv_compress_norm_rope_insert`, the MQA-logits indexer, and the AITER `mhc_*` MLA helpers together are ~17–23% of each stage. Prefill uses `_sparse_attn_prefill_ragged`; decode uses `_sparse_attn_decode_partial/reduce`.
- **AITER per-group FP8 quant dominates launch count** (`dynamic_per_group_scaled_quant`, ~2.2M launches in decode): every GEMM is preceded by an activation quant.
- **Elementwise/copy is the largest decode bucket by time (~22%) and by far the largest by launch count (~12M)** — a fusion opportunity if decode latency matters.
