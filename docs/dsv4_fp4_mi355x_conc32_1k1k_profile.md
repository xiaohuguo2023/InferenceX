# DeepSeek-V4-Pro — Per-Stage GPU Kernel Profile (MI355X)

| Field | Value |
|---|---|
| Model | `deepseek-ai/DeepSeek-V4-Pro` (FP4 MoE + FP8 attn, FP8 KV) |
| Hardware | 8× AMD Instinct MI355X (gfx950), TP=8 |
| Engine | vLLM 0.23.1rc1.dev714+g09663abde (image `vllm/vllm-openai-rocm:nightly-09663abde...`) |
| Recipe | `dsv4_fp4_mi355x_profiling.sh` (AITER MoE, `--moe-backend aiter`, cudagraph FULL_AND_PIECEWISE) |
| Workload | ISL=1024, OSL=1024, concurrency=32, random-range-ratio=0.8 |
| Traces | 8 GPU ranks, torch profiler (record_shapes) |

## Overall prefill vs decode split

| Stage | GPU time (ms) | % of total | steps (Σranks) |
|---|---|---|---|
| PREFILL | 14036.1 | 4.6% | 48 |
| DECODE | 288354.1 | 95.4% | 8,152 |

> Aggregate GPU-kernel time summed across 8 ranks; a torch-profiler window of 5 active engine iterations at conc=32. Prefill covers the context (ctx>0) steps, decode the pure-generation (ctx==0) steps.

## PREFILL

- Aggregate GPU-kernel time (8 ranks): **14036.1 ms**
- Kernel launches: **272,064**
- Steps (summed across ranks): **48**

### PREFILL — kernel time by category

| Category | time (ms) | % | launches |
|---|---|---|---|
| GEMM (dense/linear) | 3492.3 | 24.9% | 26,840 |
| Attention (MLA / DSA) | 3263.1 | 23.2% | 41,168 |
| Communication | 3217.5 | 22.9% | 5,952 |
| GEMM (MoE experts) | 1461.2 | 10.4% | 5,856 |
| Other | 876.7 | 6.2% | 40,544 |
| Memory/elementwise | 724.8 | 5.2% | 101,296 |
| RoPE | 352.6 | 2.5% | 4,368 |
| Quantization | 236.5 | 1.7% | 18,520 |
| MoE routing | 215.6 | 1.5% | 18,688 |
| Normalization | 195.8 | 1.4% | 8,832 |

### PREFILL — GEMM libraries (35.3% of stage GPU time)

| GEMM library | time (ms) | % | launches |
|---|---|---|---|
| Composable Kernel (ck gemm_xdl, ABscale) | 2341.5 | 16.7% | 13,152 |
| AITER MoE expert GEMM (ASM, FP8xFP4) | 1461.2 | 10.4% | 5,856 |
| hipBLASLt (Tensile) | 699.9 | 5.0% | 9,984 |
| AITER a8w8 block-scale GEMM (CK) | 445.4 | 3.2% | 2,928 |
| wvSplitK (skinny/thin-M GEMM) | 4.2 | 0.0% | 744 |
| Triton GEMM | 1.4 | 0.0% | 32 |

### PREFILL — top AITER functions (38.1% of stage GPU time, 38 distinct)

| time (ms) | % | launches | kernel |
|---|---|---|---|
| 1293.6 | 9.2% | 5,856 | `_sparse_attn_prefill_ragged_kernel` |
| 603.4 | 4.3% | 1,464 | `mfma_moe1_silu_mul_afp8_wfp4_fp8_t128x256x256_pm1_fp8q_sort_async_gui_v32` |
| 601.0 | 4.3% | 3,904 | `_ZN5aiter15mhc_post_kernelIDF16bLi256ELi4ELi1024ELb1EEEvPT_S2_S2_PfS3_iiiii` |
| 526.8 | 3.8% | 4,880 | `_ZN5aiter26mhc_pre_gemm_sqrsum_kernelIDF16bLi256ELi64ELi32ELi128EEEvPfS1_PT_S1_iiiiiii` |
| 461.3 | 3.3% | 1,464 | `mfma_moe2_afp8_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_pm1_sbm128_acc0` |
| 359.6 | 2.6% | 1,464 | `_gemm_a8w8_blockscale_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_128_BLOCK_SIZE_N_128_BLO…` |
| 311.2 | 2.2% | 3,904 | `_ZN5aiter23mhc_pre_big_fuse_kernelIDF16bLi320ELi4ELi2ELi256ELb1EEEvPfS1_PT_S1_S1_S1_S1_S3_…` |
| 224.6 | 1.6% | 2,928 | `_fused_kv_compress_norm_rope_insert_sparse_attn` |
| 148.5 | 1.1% | 488 | `mfma_moe1_silu_mul_afp8_wfp4_fp8_t64x256x256_pm1_fp8q_sort_async_gui_v32` |
| 146.5 | 1.0% | 13,152 | `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb0ELi64ELb0EEEvPT0_P…` |
| 99.3 | 0.7% | 976 | `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v32` |
| 92.3 | 0.7% | 488 | `mfma_moe2_afp8_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_pm1_acc0` |
| 64.9 | 0.5% | 488 | `_gemm_a8w8_blockscale_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_128_BLOCK_SIZE_N_128_BLO…` |
| 56.4 | 0.4% | 976 | `mfma_moe2_afp8_wfp4_bf16_cshuffle_t32x256x256_vscale_fix3_fp4opt_v1_pm1` |
| 56.1 | 0.4% | 2,440 | `_sparse_attn_decode_partial_kernel` |

## DECODE

- Aggregate GPU-kernel time (8 ranks): **288354.1 ms**
- Kernel launches: **33,569,456**
- Steps (summed across ranks): **8,152**

### DECODE — kernel time by category

| Category | time (ms) | % | launches |
|---|---|---|---|
| Memory/elementwise | 62936.6 | 21.8% | 12,276,424 |
| GEMM (dense/linear) | 58977.1 | 20.5% | 4,704,720 |
| Attention (MLA / DSA) | 50176.2 | 17.4% | 6,203,672 |
| GEMM (MoE experts) | 45062.8 | 15.6% | 994,544 |
| MoE routing | 29771.4 | 10.3% | 2,931,216 |
| Communication | 17804.3 | 6.2% | 1,010,848 |
| Quantization | 11147.7 | 3.9% | 2,730,920 |
| Normalization | 6598.9 | 2.3% | 1,499,968 |
| RoPE | 3331.9 | 1.2% | 741,832 |
| Other | 2547.3 | 0.9% | 475,312 |

### DECODE — GEMM libraries (36.1% of stage GPU time)

| GEMM library | time (ms) | % | launches |
|---|---|---|---|
| AITER MoE expert GEMM (ASM, FP8xFP4) | 45062.8 | 15.6% | 994,544 |
| Composable Kernel (ck gemm_xdl, ABscale) | 33822.8 | 11.7% | 2,233,648 |
| hipBLASLt (Tensile) | 13967.2 | 4.8% | 1,239,104 |
| AITER a8w8 block-scale GEMM (CK) | 5681.2 | 2.0% | 497,272 |
| wvSplitK (skinny/thin-M GEMM) | 5165.7 | 1.8% | 726,744 |
| Triton GEMM | 340.3 | 0.1% | 7,952 |

### DECODE — top AITER functions (40.5% of stage GPU time, 35 distinct)

| time (ms) | % | launches | kernel |
|---|---|---|---|
| 23783.1 | 8.2% | 482,632 | `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v32` |
| 20915.0 | 7.3% | 492,392 | `mfma_moe2_afp8_wfp4_bf16_cshuffle_t32x256x256_vscale_fix3_fp4opt_v1_pm1_xcd4` |
| 14189.7 | 4.9% | 497,272 | `_sparse_attn_decode_partial_kernel` |
| 9139.8 | 3.2% | 2,233,648 | `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb0ELi64ELb0EEEvPT0_P…` |
| 8354.4 | 2.9% | 497,272 | `_fused_kv_compress_norm_rope_insert_sparse_attn` |
| 5463.4 | 1.9% | 461,160 | `_gemm_a8w8_blockscale_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_16_BLOCK…` |
| 5313.0 | 1.8% | 994,544 | `_ZN5aiter23mhc_pre_big_fuse_kernelIDF16bLi320ELi4ELi2ELi256ELb0EEEvPfS1_PT_S1_S1_S1_S1_S3_…` |
| 4614.4 | 1.6% | 994,544 | `_ZN5aiter26mhc_pre_gemm_sqrsum_kernelIDF16bLi256ELi64ELi16ELi128EEEvPfS1_PT_S1_iiiiiii` |
| 4544.6 | 1.6% | 497,272 | `_sparse_attn_decode_reduce_kernel` |
| 4480.6 | 1.6% | 994,544 | `_ZN5aiter15mhc_post_kernelIDF16bLi256ELi4ELi1024ELb0EEEvPT_S2_S2_PfS3_iiiii` |
| 3329.1 | 1.2% | 461,160 | `void wvSplitKrc_<__hip_bfloat16, 64, 16, 4, 8, 1, 32, 2, 1, 1>(int, int, int, int, int, in…` |
| 2831.3 | 1.0% | 497,272 | `_ZN5aiter30fused_mx_quant_moe_sort_kernelIDF16bDB8_Li256ELi32EEEvPT0_PhPKT_PKiS9_PKfiiiiii…` |
| 2084.3 | 0.7% | 461,160 | `void aiter::opus_moe_sorting_entry<aiter::MoeSortingMultiPhaseKernel_P23<aiter::MoeSorting…` |
| 1971.0 | 0.7% | 461,160 | `void aiter::opus_moe_sorting_entry<aiter::MoeSortingMultiPhaseKernel_P0_v2<aiter::MoeSorti…` |
| 1722.4 | 0.6% | 244,560 | `_fused_kv_compress_norm_rope_insert_indexer_attn` |

## Key observations

- **Decode-dominated window.** At ISL=OSL=1024 the profiled window is 95% decode / 5% prefill, so the steady-state kernel mix below reflects the decode path.
- **GEMM library split is stable across stages:** MoE experts run on the **AITER assembly FP8×FP4** kernels (`mfma_moe1/moe2 … afp8_wfp4`), dense/linear layers on **Composable Kernel** `ck::gemm_xdl_cshuffle_v3` (AB-scale FP8), with **hipBLASLt (Tensile)** picking up a minority of dense shapes.
- **`wvSplitK` is decode-only** (~1.8% decode vs ~0% prefill): the skinny/thin-M GEMM path is selected when M = batch is tiny (decode), and CK/hipBLASLt take over at prefill's larger M.
- **Prefill is communication-heavy** (~23% TP=8 custom all-reduce) because activation tensors are large; decode all-reduce drops to ~6%.
- **DeepSeek Sparse Attention (DSA) + MLA** is a first-class cost: `_sparse_attn_*`, `_fused_kv_compress_norm_rope_insert`, the MQA-logits indexer, and the AITER `mhc_*` MLA helpers together are ~17–23% of each stage. Prefill uses `_sparse_attn_prefill_ragged`; decode uses `_sparse_attn_decode_partial/reduce`.
- **AITER per-group FP8 quant dominates launch count** (`dynamic_per_group_scaled_quant`, ~2.2M launches in decode): every GEMM is preceded by an activation quant.
- **Elementwise/copy is the largest decode bucket by time (~22%) and by far the largest by launch count (~12M)** — a fusion opportunity if decode latency matters.
