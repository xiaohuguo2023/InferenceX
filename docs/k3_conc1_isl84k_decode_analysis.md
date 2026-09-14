# Trace Comparison: k3_conc1_dcp8_isl84k

## Provenance

| | |
|---|---|
| Traces | `/dev/shm/prof_dcp8_ep1/dp0_pp0_tp{0..7}_dcp{0..7}_ep{0..7}_rank{0..7}.*.pt.trace.json.gz` (8 ranks, ~26 MB each) |
| Generator | `trace_compare_k3.py` (in this repo; derived from `~/work/vllm_traces_tp_varied/trace_compare.py`) |
| CSV twin | `docs/k3_conc1_isl84k_decode_analysis.csv` — `tp_label,stage,rank,category,sub_kernel,total_time_us,call_count,avg_time_us` |

Regenerate:

```bash
python3 trace_compare_k3.py \
  --tp-dirs /dev/shm/prof_dcp8_ep1 \
  --tp-labels k3_conc1_dcp8_isl84k \
  --output-md  docs/k3_conc1_isl84k_decode_analysis.md \
  --output-csv docs/k3_conc1_isl84k_decode_analysis.csv
```

### How the traces were captured

Serve-only boot with the torch profiler armed, then one fixed-ISL request with the
profiler open around it:

```bash
# launcher: /dev/shm/_prof_dcp8_ep1.sh  (SERVE_ONLY=true, profiler via EXTRA_VLLM_ARGS)
#   TP=8 EP_SIZE=1 DCP_SIZE=8 CONC=1, KV_OFFLOADING=dram/lmcache, TOTAL_CPU_DRAM_GB=803
#   AITER_CONFIG_GEMM_BF16=patches/k3-dcp8/aiter/merged_bf16_tuned_gemm.csv
#   --profiler-config {"profiler":"torch","torch_profiler_dir":"/dev/shm/prof_dcp8_ep1",
#                      "torch_profiler_with_stack":false,"torch_profiler_record_shapes":true}
export MODEL_PATH=/dev/shm/hf-cache/models--moonshotai--Kimi-K3/snapshots/a590ce09...
curl -X POST http://127.0.0.1:8888/start_profile
python3 /dev/shm/_probe_cold.py          # ONE request, ISL 84,240 / OSL 256
curl -X POST http://127.0.0.1:8888/stop_profile
```

### Configuration measured

Best-performing conc-1 arm: **DCP8 + DSpark K=7 + LMCache DRAM, EP_SIZE=1** (pure TP8),
synthetic acceptance AL 3.84, ROCm 7.2.3 image
`vllm/vllm-openai-rocm:nightly-385dce36bcee42309924a5ece951a96db3dce7f2`.
Same config as run `results_ixci/r72_c1_ep1` = **interactivity p90 131.35**.

### Reading this report

* **DECODE numbers are PER-ITERATION (us/step)**; PREFILL numbers are window totals. Do
  not compare the two sections directly.
* Decode iterations in this window = **67** (256 output tokens / AL 3.84).
* DECODE total is **26,350 us/step**, which cross-checks against 26.32 ms/step derived
  independently from `vllm:spec_decode_num_drafts` timeslices (no profiler involved).
* **The `Other` bucket holds the real MoE expert GEMMs** (`opus_moe::stage1_a8w4`,
  `gemm2_a4w4_port_*`, `mla_a8w8_..._cprr_ps`) -- the categoriser does not class them as
  GEMM, so `GEMM` (39%) understates MoE and overstates dense linear work.
* The torch profiler inflates decode wall time and widens collective arrival gaps. Trust
  **counts and ratios**; treat absolute microseconds as relative, not absolute.


## Component breakdown (% of total)

**DECODE — per decode step (rank 0). Total 26,351 us/step over 67 iterations.**

| Category | us/step | % of step | calls/step | us/call |
|---|---:|---:|---:|---:|
| GEMM | 10,264 | **39.0%** | 935 | 10.97 |
| Other | 5,879 | **22.3%** | 707 | 8.31 |
| Communication | 3,173 | **12.0%** | 321 | 9.88 |
| MoE Routing | 2,669 | **10.1%** | 368 | 7.25 |
| KDA Linear Attn | 1,753 | **6.7%** | 138 | 12.70 |
| Attention | 1,515 | **5.7%** | 187 | 8.10 |
| Normalization | 863 | **3.3%** | 198 | 4.35 |
| Triton Fused | 94 | **0.4%** | 26 | 3.63 |
| Sampling | 83 | **0.3%** | 18 | 4.56 |
| Memory | 36 | **0.1%** | 8 | 4.51 |
| Activation | 20 | **0.1%** | 5 | 4.02 |
| **TOTAL** | **26,351** | **100.0%** | **2,912** | 9.05 |

**DECODE — top 20 sub-kernels by share of the step.**

| Sub-kernel | Category | us/step | % of step | calls/step | us/call |
|---|---|---:|---:|---:|---:|
| `hipBLASLt (Cijk)` | GEMM | 3,871 | **14.7%** | 335 | 11.55 |
| `AITER hgemm bf16` | GEMM | 3,694 | **14.0%** | 453 | 8.15 |
| `AITER flydsl MoE` | GEMM | 2,312 | **8.8%** | 92 | 25.13 |
| `CustomAR 2-stage` | Communication | 1,615 | **6.1%** | 198 | 8.15 |
| `MoE sort mxfp4` | MoE Routing | 1,588 | **6.0%** | 276 | 5.75 |
| `MLA attn residual` | Attention | 1,515 | **5.7%** | 187 | 8.10 |
| `KDA gated-delta` | KDA Linear Attn | 1,362 | **5.2%** | 69 | 19.74 |
| `gemm2_a4w4_port_hmax8192_imax8192_bm32_bn128_bk128_atomic_` | Other | 1,284 | **4.9%** | 92 | 13.96 |
| `AITER grouped_topk` | MoE Routing | 1,081 | **4.1%** | 92 | 11.75 |
| `_ZN5aiter45mla_a8w8_qh32_qseqlen4_gqaratio32_lse_cprr_psE.` | Other | 1,064 | **4.0%** | 29 | 36.61 |
| `CustomAR 1-stage` | Communication | 983 | **3.7%** | 92 | 10.68 |
| `NCCL/RCCL` | Communication | 576 | **2.2%** | 31 | 18.53 |
| `Add+RMSNorm+quant` | Normalization | 470 | **1.8%** | 105 | 4.46 |
| `void at::native::elementwise_kernel_manual_unroll<128, 8, ` | Other | 395 | **1.5%** | 83 | 4.75 |
| `KDA conv1d` | KDA Linear Attn | 391 | **1.5%** | 69 | 5.67 |
| `void vllm::situ_and_mul_kernel<c10::BFloat16>` | Other | 382 | **1.5%** | 93 | 4.11 |
| `void kn_mla_reduce_v1<MlaReduceKernelV1Traits<512, 128, 1>` | Other | 325 | **1.2%** | 24 | 13.52 |
| `mscclKernel_Sum_hip_bfloat16_Simple_false` | Other | 319 | **1.2%** | 24 | 13.30 |
| `LayerNorm` | Normalization | 294 | **1.1%** | 69 | 4.26 |
| `__amd_rocclr_copyBuffer.kd` | Other | 276 | **1.0%** | 69 | 3.98 |

**PREFILL — window totals (rank 0). Total 7,812,550 us.**

| Category | us | % of prefill |
|---|---:|---:|
| Other | 2,039,238 | **26.1%** |
| Communication | 1,854,582 | **23.7%** |
| GEMM | 1,759,640 | **22.5%** |
| Attention | 1,184,460 | **15.2%** |
| KDA Linear Attn | 564,238 | **7.2%** |
| MoE Routing | 272,501 | **3.5%** |
| Normalization | 97,749 | **1.3%** |
| Quantization | 15,512 | **0.2%** |
| Memory | 10,418 | **0.1%** |
| Triton Fused | 6,481 | **0.1%** |
| Sampling | 6,219 | **0.1%** |
| Activation | 1,512 | **0.0%** |
| **TOTAL** | **7,812,550** | **100.0%** |

## By model component (reclassified)

The `GEMM` and `Other` buckets above are kernel-family labels, not model parts:
`GEMM` includes MoE expert stage-1, and `Other` holds MoE stage-2 and the MLA asm kernel.
This table regroups the same rank-0 decode data by what the kernel actually computes.

| Component | us/step | % of step | calls/step | distinct kernels |
|---|---:|---:|---:|---:|
| Dense linear (bf16) | 8,010 | **30.4%** | 818 | 7 |
| Collectives | 3,867 | **14.7%** | 408 | 7 |
| MoE expert math | 3,831 | **14.5%** | 232 | 3 |
| MLA attention | 3,343 | **12.7%** | 282 | 7 |
| MoE routing/sort | 2,669 | **10.1%** | 368 | 2 |
| Glue / elementwise / misc | 1,932 | **7.3%** | 449 | 43 |
| KDA attention | 1,753 | **6.7%** | 138 | 2 |
| Norm / quant | 863 | **3.3%** | 198 | 3 |
| Sampling | 83 | **0.3%** | 18 | 2 |
| **TOTAL** | **26,351** | **100.0%** | **2,912** | 76 |

## Configuration

| | k3_conc1_dcp8_isl84k |
|---|---|
| World size | 8 |
| Decode iterations | 67 |

## PREFILL

### k3_conc1_dcp8_isl84k Per-Rank Breakdown (us)

| Category | Sub-kernel | R0 | R1 | R2 | R3 | R4 | R5 | R6 | R7 | Imbal% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Other** | *(total)* | 2039238.0 | 2017211.0 | 2019098.2 | 2028166.1 | 2032396.4 | 2035663.1 | 2039819.3 | 2035803.3 | 1.1% |
| | void opus_moe::stage1_a8w4::pipeline_group_split::opus_moe_stage1_a8w4_kernel... | 487565.1 | 485631.6 | 478627.1 | 484968.0 | 484771.4 | 485681.9 | 488801.3 | 484622.0 | |
| | void opus_moe_stage2_a8w4_decode_kernel_gfx950<OpusMoeStage2A8W4DecodeShape<o... | 363597.3 | 359538.2 | 356779.2 | 362155.3 | 359318.3 | 361963.6 | 362536.0 | 365217.9 | |
| | _ZN5aiter24bf16gemm_bf16_tn_256x256E.kd | 313920.5 | 308824.3 | 310137.2 | 309802.8 | 315709.6 | 316561.5 | 318957.8 | 317179.7 | |
| | void gemm_a16w16_persistent_4g_safe_kernel<opus_gemm_a16w16_persistent_traits... | 183445.5 | 180294.1 | 183701.7 | 183577.3 | 182016.9 | 180959.4 | 181866.1 | 183912.7 | |
| | void opus_moe_stage2_reduce_token_slot_route_output_kernel_gfx950<3584, 448, ... | 91969.2 | 87528.7 | 89761.4 | 91935.2 | 90226.5 | 89350.7 | 89053.8 | 88975.0 | |
| | gemm2_a4w4_port_hmax8192_imax8192_bm32_bn128_bk128_atomic_a8_g2ks2_bhoist_apf... | 90614.7 | 91137.1 | 90843.1 | 91348.5 | 91599.4 | 90034.6 | 91463.8 | 90964.6 | |
| | _ZN5aiter45mla_a8w8_qh32_qseqlen4_gqaratio32_lse_cprr_psE.kd | 73113.4 | 72347.6 | 71798.0 | 72170.1 | 72409.9 | 72444.2 | 71780.3 | 72164.8 | |
| | void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_ker... | 55891.6 | 54703.0 | 55144.7 | 54800.1 | 55017.4 | 55442.8 | 55197.2 | 54644.1 | |
| | void vllm::situ_and_mul_kernel<c10::BFloat16> | 47357.7 | 47530.2 | 46914.9 | 47444.8 | 46909.5 | 48215.9 | 48304.0 | 47464.7 | |
| | void vllm::gather_and_maybe_dequant_cache_page<__hip_bfloat16, unsigned char,  | 47537.8 | 47514.4 | 48035.0 | 47547.4 | 47725.5 | 47819.4 | 47893.6 | 47488.5 | |
| | __amd_rocclr_copyBuffer.kd | 42617.7 | 41518.5 | 44017.9 | 41470.5 | 44231.5 | 43348.9 | 42163.0 | 41935.2 | |
| | mscclKernel_Sum_hip_bfloat16_Simple_false | 21707.6 | 23140.7 | 23242.6 | 23297.7 | 23559.4 | 22698.6 | 23064.7 | 23818.2 | |
| | void kn_mla_reduce_v1<MlaReduceKernelV1Traits<512, 128, 1>, float, std::bfloa... | 22073.9 | 21658.5 | 22020.2 | 21911.1 | 21641.3 | 21546.1 | 21886.9 | 21640.2 | |
| | void at::native:: | 21434.2 | 20680.0 | 21997.0 | 20981.1 | 21441.0 | 21109.6 | 20960.1 | 20913.1 | |
| | void gqa_d192_v128_kernel<opus_gqa_d192_traits<32, 64, 8, true, true> > | 21165.4 | 20803.7 | 20782.6 | 20910.0 | 21249.8 | 21209.8 | 21344.3 | 21266.8 | |
| | void kn_get_mla_metadata_v1_2<MlaMetadataV12Traits<128, true, 0, true, false> > | 14992.8 | 15026.7 | 14975.0 | 15048.0 | 14999.0 | 15092.4 | 15071.3 | 14921.4 | |
| | void vllm::concat_and_cache_mla_kernel<__hip_bfloat16, unsigned char,  | 13230.6 | 13034.1 | 12662.1 | 12819.4 | 12813.0 | 13468.7 | 13100.5 | 13002.0 | |
| | _dcp_a2a_pack_send_kernel.kd | 12046.3 | 12211.2 | 12196.1 | 12279.5 | 12238.9 | 12164.6 | 12049.0 | 12172.8 | |
| | void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_... | 10278.2 | 10467.5 | 10274.1 | 10272.2 | 10256.2 | 10690.3 | 10433.6 | 10264.5 | |
| | void at::native::vectorized_elementwise_kernel<8, at::native::BinaryFunctor<c... | 10031.8 | 9973.5 | 9694.5 | 9986.0 | 9765.6 | 10175.8 | 9961.6 | 9774.0 | |
| | _dcp_a2a_unpack_combine_kernel.kd | 9041.3 | 9146.7 | 9414.4 | 8963.1 | 9192.5 | 9490.6 | 9322.4 | 9190.9 | |
| | void gemm_a16w16_flatmm_splitk_kernel<opus_flatmm_splitk_traits_gfx950<256, o... | 9328.1 | 9405.4 | 8619.9 | 9050.3 | 8621.7 | 8909.2 | 8635.5 | 9002.5 | |
| | void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kerne... | 7593.2 | 6974.6 | 8244.3 | 7021.0 | 8133.8 | 7378.6 | 7173.9 | 7062.3 | |
| | _ZN5aiter37bf16gemm_fp32bf16_tn_32x64_pf3_splitkE.kd | 8037.3 | 7970.1 | 8014.0 | 7955.6 | 7984.6 | 8112.3 | 7993.7 | 7992.1 | |
| | void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add... | 6646.6 | 6565.1 | 6579.4 | 6662.9 | 6618.9 | 6743.4 | 6687.6 | 6571.2 | |
| | void opus_moe_stage2_reduce_token_slot_route_output_kernel_gfx950<3072, 384, ... | 6736.5 | 6291.2 | 6703.9 | 6710.7 | 6375.3 | 6715.3 | 6469.4 | 6444.4 | |
| | mscclKernel_Sum_hip_bfloat16_LL_false | 5753.5 | 6357.7 | 5685.2 | 6148.3 | 5440.9 | 6012.1 | 6229.0 | 6253.1 | |
| | void kn_mla_reduce_v1<MlaReduceKernelV1Traits<512, 64, 4>, float, std::bfloat... | 5850.3 | 5679.4 | 6234.2 | 5686.5 | 6118.5 | 5900.0 | 5749.2 | 5741.1 | |
| | void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_ker... | 5251.5 | 5094.2 | 5339.2 | 5121.2 | 5280.2 | 5196.0 | 5028.5 | 4941.3 | |
| | void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda... | 3180.8 | 3162.0 | 3220.5 | 3156.5 | 3220.8 | 3324.8 | 3260.3 | 3172.1 | |
| | _prepare_dflash_inputs_kernel.kd | 2677.2 | 2658.8 | 2691.0 | 2668.8 | 2679.1 | 2714.8 | 2653.7 | 2680.4 | |
| | void splitk_reduce_kernel<16, 64, std::bfloat16_t, false, std::bfloat16_t, true> | 2120.2 | 2055.0 | 2098.4 | 2033.6 | 2133.7 | 2159.7 | 2085.2 | 2086.0 | |
| | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSe... | 1828.0 | 1836.7 | 1848.0 | 1802.1 | 1853.4 | 1868.0 | 1845.5 | 1819.8 | |
| | void  | 1777.2 | 1756.1 | 1794.9 | 1756.1 | 1813.3 | 1835.2 | 1789.5 | 1756.4 | |
| | void rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel<rocprim::ROCPRIM_4... | 1730.2 | 1723.2 | 1721.7 | 1742.1 | 1736.9 | 1817.4 | 1761.6 | 1727.4 | |
| | kernel.kd | 1577.4 | 1560.0 | 1561.3 | 1581.7 | 1549.8 | 1654.5 | 1654.0 | 1548.2 | |
| | void vllm::kimi_k3_fused_ops::fusedKimiK3MLADecodeQConcatKVCacheKernel<c10::B... | 1445.2 | 1442.6 | 1442.6 | 1440.2 | 1448.9 | 1534.6 | 1477.4 | 1450.6 | |
| | void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctorOnSelf_ad... | 966.9 | 944.1 | 982.3 | 977.2 | 945.7 | 994.3 | 1004.9 | 974.5 | |
| | void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<i... | 908.8 | 898.3 | 929.7 | 891.9 | 942.5 | 942.8 | 910.0 | 897.0 | |
| | void at::native::vectorized_elementwise_kernel<4, at::native:: | 916.8 | 887.7 | 923.1 | 890.9 | 936.3 | 933.3 | 908.4 | 888.0 | |
| | void at::native::vectorized_gather_kernel<16, long> | 920.7 | 923.0 | 888.1 | 871.6 | 866.3 | 887.3 | 895.0 | 885.9 | |
| | _expand_page_indices_kernel.kd | 877.2 | 876.2 | 875.6 | 877.2 | 875.8 | 913.5 | 896.0 | 878.7 | |
| | _dcp_local_seq_lens_kernel.kd | 641.9 | 648.0 | 642.3 | 637.6 | 682.2 | 665.6 | 664.2 | 673.3 | |
| | void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctorOnSe... | 618.5 | 618.3 | 630.9 | 616.9 | 617.9 | 658.4 | 631.9 | 618.7 | |
| | _rejection_kernel.kd | 643.4 | 621.9 | 658.0 | 618.0 | 640.7 | 648.0 | 603.8 | 630.9 | |
| | __amd_rocclr_fillBufferAligned.kd | 614.2 | 610.4 | 615.6 | 611.5 | 610.7 | 642.8 | 626.9 | 615.2 | |
| | _bias_kernel.kd | 566.2 | 595.1 | 613.2 | 583.6 | 632.9 | 534.0 | 543.8 | 584.9 | |
| | _compute_slot_mappings_kernel.kd | 552.7 | 526.7 | 559.6 | 549.3 | 564.3 | 553.0 | 571.8 | 547.6 | |
| | void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<int>... | 536.3 | 520.6 | 565.1 | 521.8 | 552.8 | 532.9 | 522.4 | 516.3 | |
| | precopy_mamba_align_fused_kernel.kd | 534.8 | 529.7 | 534.6 | 524.0 | 531.1 | 548.2 | 538.6 | 547.1 | |
| | void vllm::rms_norm_kernel<c10::BFloat16, 8, 3, true> | 490.2 | 493.6 | 481.8 | 485.4 | 482.3 | 491.1 | 495.6 | 488.2 | |
| | _compute_local_logits_stats_kernel.kd | 455.8 | 470.1 | 459.3 | 469.9 | 462.6 | 473.3 | 452.7 | 461.9 | |
| | _scatter_num_accepted_kernel.kd | 408.3 | 407.1 | 408.2 | 412.1 | 396.4 | 430.2 | 419.7 | 411.0 | |
| | void vllm::rotary_embedding_kernel<c10::BFloat16, float, false> | 402.4 | 407.4 | 395.9 | 412.4 | 407.5 | 411.3 | 408.4 | 404.2 | |
| | void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat... | 361.8 | 368.8 | 367.3 | 377.4 | 368.8 | 373.6 | 402.1 | 376.3 | |
| | _post_update_kernel.kd | 392.8 | 392.8 | 387.5 | 396.1 | 358.7 | 393.0 | 391.4 | 389.4 | |
| | postprocess_mamba_fused_kernel.kd | 357.9 | 351.9 | 352.8 | 351.5 | 376.9 | 377.2 | 364.4 | 358.2 | |
| | _gather_block_tables_kernel.kd | 347.0 | 368.0 | 364.6 | 356.4 | 353.8 | 357.2 | 354.0 | 358.2 | |
| | _prepare_pos_seq_lens_kernel.kd | 330.2 | 332.6 | 337.3 | 334.4 | 329.3 | 347.3 | 330.6 | 325.1 | |
| | preprocess_mamba_align_fused_kernel.kd | 341.6 | 327.0 | 335.9 | 333.3 | 344.2 | 342.3 | 337.7 | 334.0 | |
| | _apply_write_kernel.kd | 187.5 | 189.5 | 194.7 | 189.5 | 196.4 | 183.1 | 184.6 | 196.2 | |
| | _expand_idx_mapping_kernel.kd | 171.4 | 172.0 | 174.9 | 177.3 | 176.0 | 171.2 | 176.3 | 181.2 | |
| | void at::native::vectorized_elementwise_kernel<4, at::native::compare_scalar_... | 108.3 | 94.9 | 129.3 | 92.9 | 147.6 | 105.3 | 93.9 | 95.5 | |
| | _prepare_prefill_inputs_kernel.kd | 136.8 | 138.4 | 145.3 | 134.8 | 145.8 | 135.9 | 137.4 | 135.0 | |
| | _chunk_metadata_kernel.kd | 117.8 | 102.4 | 138.1 | 95.4 | 144.5 | 109.8 | 96.4 | 96.2 | |
| | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add... | 99.2 | 87.3 | 119.2 | 84.8 | 135.8 | 95.6 | 86.0 | 87.9 | |
| | _zero_kv_blocks_kernel.kd | 48.0 | 48.0 | 53.6 | 47.8 | 51.2 | 48.5 | 47.6 | 47.7 | |
| | __amd_rocclr_streamOpsWrite.kd | 17.0 | 17.1 | 17.0 | 17.1 | 16.8 | 17.6 | 17.2 | 16.9 | |
| **Communication** | *(total)* | 1854582.2 | 2026160.5 | 1744365.5 | 1981793.3 | 1728380.4 | 1889128.4 | 1923169.2 | 1931828.3 | 15.8% :warning: |
| | NCCL/RCCL | 1514817.2 | 1684070.3 | 1409051.6 | 1646968.5 | 1394703.8 | 1556694.9 | 1594500.6 | 1594655.7 | |
| | CustomAR 2-stage | 274103.8 | 281098.0 | 271381.0 | 277057.8 | 273135.1 | 272811.2 | 270950.9 | 278046.7 | |
| | CustomAR 1-stage | 65661.2 | 60992.1 | 63932.9 | 57767.0 | 60541.6 | 59622.3 | 57717.7 | 59125.9 | |
| **GEMM** | *(total)* | 1759640.6 | 1671752.8 | 1874489.3 | 1681444.3 | 1886034.3 | 1750838.1 | 1715412.4 | 1717908.4 | 12.2% :warning: |
| | hipBLASLt (Cijk) | 1321562.4 | 1230466.7 | 1433186.3 | 1237524.6 | 1443793.5 | 1309683.8 | 1271739.7 | 1274913.6 | |
| | AITER hgemm bf16 | 252129.4 | 253617.4 | 254232.7 | 256244.3 | 253890.0 | 253405.6 | 255556.4 | 255225.1 | |
| | AITER flydsl MoE | 158253.0 | 159898.1 | 159146.3 | 159794.5 | 160245.0 | 159502.8 | 160131.8 | 160024.4 | |
| | AITER wfp4 batched | 15982.0 | 16272.3 | 16146.5 | 16340.4 | 16259.7 | 16554.8 | 16431.7 | 16159.2 | |
| | rocBLAS splitK | 11713.7 | 11498.3 | 11777.6 | 11540.5 | 11846.1 | 11691.1 | 11552.9 | 11586.1 | |
| **Attention** | *(total)* | 1184459.5 | 1134214.2 | 1182780.6 | 1156716.3 | 1189485.3 | 1164365.1 | 1156570.4 | 1161050.3 | 4.7% |
| | aiter attn | 677270.5 | 642053.1 | 682564.6 | 650607.3 | 691200.4 | 667802.6 | 659092.2 | 664592.7 | |
| | MLA attn residual | 489125.9 | 474417.7 | 482081.5 | 488382.8 | 480189.5 | 478834.1 | 479646.9 | 478677.2 | |
| | MLA merge states | 18063.1 | 17743.4 | 18134.4 | 17726.1 | 18095.3 | 17728.5 | 17831.2 | 17780.3 | |
| **KDA Linear Attn** | *(total)* | 564237.5 | 557895.8 | 569395.8 | 560860.5 | 562825.3 | 560904.3 | 562467.3 | 560514.9 | 2.0% |
| | KDA gated-delta | 381057.5 | 378623.2 | 384262.3 | 380366.5 | 379778.0 | 379603.8 | 383432.5 | 381626.0 | |
| | KDA conv1d | 86393.7 | 83899.3 | 89287.4 | 84436.1 | 89240.5 | 85554.1 | 84681.5 | 85154.0 | |
| | KDA gate/state | 65621.0 | 64170.0 | 64740.1 | 64878.1 | 63360.6 | 64583.9 | 63705.9 | 63252.4 | |
| | KDA GLA | 31165.4 | 31203.3 | 31105.9 | 31179.8 | 30446.2 | 31162.5 | 30647.3 | 30482.4 | |
| **MoE Routing** | *(total)* | 272500.9 | 265258.9 | 279441.9 | 266097.3 | 280408.6 | 270743.4 | 269309.2 | 267672.0 | 5.6% |
| | MoE sort mxfp4 | 161866.3 | 155002.1 | 169089.4 | 156002.9 | 170302.0 | 160209.3 | 158684.8 | 157332.1 | |
| | AITER grouped_topk | 110634.7 | 110256.7 | 110352.5 | 110094.3 | 110106.6 | 110534.1 | 110624.4 | 110339.9 | |
| **Normalization** | *(total)* | 97748.8 | 97245.7 | 96854.1 | 97363.7 | 96895.0 | 100152.3 | 98678.1 | 96684.6 | 3.5% |
| | Add+RMSNorm+quant | 50443.9 | 50005.9 | 50507.3 | 50126.1 | 50319.7 | 51965.2 | 51061.4 | 49679.4 | |
| | LayerNorm | 36487.4 | 36489.7 | 35535.5 | 36495.5 | 35657.5 | 36979.1 | 36567.0 | 36141.1 | |
| | RMSNorm | 10817.5 | 10750.1 | 10811.4 | 10742.1 | 10917.8 | 11208.1 | 11049.7 | 10864.0 | |
| **Quantization** | *(total)* | 15511.7 | 15467.8 | 15460.2 | 15512.1 | 15414.7 | 15545.1 | 15428.5 | 15317.1 | 1.5% |
| | dynamic per-group quant | 15511.7 | 15467.8 | 15460.2 | 15512.1 | 15414.7 | 15545.1 | 15428.5 | 15317.1 | |
| **Memory** | *(total)* | 10418.3 | 10457.1 | 10161.1 | 10211.6 | 10162.6 | 10316.0 | 10192.9 | 9963.8 | 4.8% |
| | Fill | 10166.4 | 10218.1 | 9900.2 | 9975.7 | 9899.1 | 10074.1 | 9958.1 | 9731.2 | |
| | Memcpy | 251.9 | 239.0 | 261.0 | 235.9 | 263.5 | 241.9 | 234.9 | 232.6 | |
| **Triton Fused** | *(total)* | 6481.1 | 10150.5 | 10180.4 | 6504.7 | 6770.3 | 10426.4 | 10360.1 | 6376.2 | 48.2% :warning: |
| | Triton fused op | 6481.1 | 10150.5 | 10180.4 | 6504.7 | 6770.3 | 10426.4 | 10360.1 | 6376.2 | |
| **Sampling** | *(total)* | 6219.3 | 6259.3 | 6147.2 | 6143.3 | 6183.4 | 6379.5 | 6190.7 | 6124.6 | 4.1% |
| | Sampling | 3396.8 | 3396.8 | 3386.3 | 3433.6 | 3451.4 | 3574.4 | 3471.7 | 3413.7 | |
| | ArgMax/Reduce | 2822.5 | 2862.5 | 2761.0 | 2709.7 | 2732.0 | 2805.1 | 2719.0 | 2710.9 | |
| **Activation** | *(total)* | 1512.0 | 1508.5 | 1542.6 | 2873.2 | 2904.5 | 1662.9 | 1584.6 | 1555.9 | 73.7% :warning: |
| | SiLU/SwiGLU | 1512.0 | 1508.5 | 1542.6 | 2873.2 | 2904.5 | 1662.9 | 1584.6 | 1555.9 | |
| **TOTAL** | | 7812549.9 | 7813581.9 | 7809916.9 | 7813686.2 | 7817860.8 | 7816124.8 | 7809182.6 | 7810799.4 | 0.1% |

> All ranks balanced (max 0.1% above mean)
>
> **Comm/Total ratio** (bottleneck rank): 24.9%

## DECODE

### k3_conc1_dcp8_isl84k Per-Rank Breakdown (per-iter avg us)

| Model part | Category | Sub-kernel | R0 | R1 | R2 | R3 | R4 | R5 | R6 | R7 | Imbal% |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mixed | **GEMM** | *(total)* | 10264.5 | 10324.0 | 10318.8 | 10385.0 | 10345.2 | 10355.1 | 10355.7 | 10365.4 | 1.2% |
| Shared | | hipBLASLt (Cijk) | 3870.5 | 3880.1 | 3880.9 | 3895.8 | 3888.3 | 3913.9 | 3875.7 | 3890.7 | |
| Shared | | AITER hgemm bf16 | 3693.7 | 3716.6 | 3724.6 | 3759.5 | 3721.6 | 3714.4 | 3746.1 | 3747.2 | |
| MoE | | AITER flydsl MoE | 2311.6 | 2336.3 | 2323.1 | 2336.8 | 2343.0 | 2330.1 | 2339.9 | 2337.2 | |
| MoE | | AITER wfp4 batched | 235.3 | 239.1 | 237.5 | 240.5 | 239.3 | 243.5 | 241.5 | 237.9 | |
| Shared | | rocBLAS splitK | 153.4 | 152.0 | 152.7 | 152.5 | 153.1 | 153.2 | 152.3 | 152.3 | |
| Mixed | **Other** | *(total)* | 5879.1 | 5901.4 | 5879.6 | 5891.9 | 5880.7 | 5940.2 | 5931.2 | 5894.0 | 1.0% |
| MoE | | gemm2_a4w4_port_hmax8192_imax8192_bm32_bn128_bk128_atomic_a8_g2ks2_bhoist_apf... | 1283.9 | 1290.1 | 1288.1 | 1293.7 | 1294.6 | 1272.8 | 1295.3 | 1286.6 | |
| MLA | | _ZN5aiter45mla_a8w8_qh32_qseqlen4_gqaratio32_lse_cprr_psE.kd | 1064.4 | 1055.4 | 1043.6 | 1050.1 | 1053.4 | 1054.4 | 1045.4 | 1052.3 | |
| Shared | | void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_ker... | 394.7 | 397.4 | 401.1 | 396.0 | 397.2 | 405.6 | 401.7 | 395.1 | |
| MoE | | void vllm::situ_and_mul_kernel<c10::BFloat16> | 382.3 | 390.4 | 377.2 | 388.6 | 376.7 | 398.4 | 401.9 | 389.6 | |
| Communication | | mscclKernel_Sum_hip_bfloat16_Simple_false | 319.3 | 339.5 | 341.7 | 341.3 | 344.5 | 332.7 | 338.0 | 347.3 | |
| MLA | | void kn_mla_reduce_v1<MlaReduceKernelV1Traits<512, 128, 1>, float, std::bfloa... | 324.5 | 318.5 | 324.0 | 322.2 | 318.1 | 316.9 | 322.2 | 318.0 | |
| Memory | | __amd_rocclr_copyBuffer.kd | 276.0 | 273.8 | 275.1 | 275.0 | 275.5 | 287.7 | 281.8 | 276.2 | |
| MLA | | void kn_get_mla_metadata_v1_2<MlaMetadataV12Traits<128, true, 0, true, false> > | 213.2 | 213.9 | 214.3 | 214.3 | 214.4 | 215.1 | 215.1 | 213.1 | |
| MLA/DCP | | _dcp_a2a_pack_send_kernel.kd | 174.1 | 177.2 | 175.8 | 177.6 | 176.8 | 176.2 | 174.4 | 175.8 | |
| MLA/DCP | | _dcp_a2a_unpack_combine_kernel.kd | 131.0 | 132.6 | 136.4 | 129.8 | 133.0 | 137.3 | 135.0 | 132.9 | |
| Shared | | void gemm_a16w16_flatmm_splitk_kernel<opus_flatmm_splitk_traits_gfx950<256, o... | 135.0 | 136.0 | 124.2 | 130.5 | 124.4 | 128.4 | 124.8 | 130.2 | |
| MLA | | void vllm::concat_and_cache_mla_kernel<__hip_bfloat16, unsigned char,  | 129.7 | 125.5 | 123.0 | 122.3 | 124.9 | 131.2 | 128.7 | 125.8 | |
| Shared | | _ZN5aiter37bf16gemm_fp32bf16_tn_32x64_pf3_splitkE.kd | 108.4 | 107.3 | 108.1 | 107.4 | 107.8 | 109.3 | 107.8 | 107.8 | |
| Shared | | void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_... | 99.6 | 101.1 | 101.0 | 98.6 | 99.3 | 106.5 | 101.2 | 99.1 | |
| Shared | | void at::native::vectorized_elementwise_kernel<8, at::native::BinaryFunctor<c... | 101.1 | 100.1 | 97.3 | 100.8 | 97.6 | 103.1 | 101.2 | 98.2 | |
| MLA | | void kn_mla_reduce_v1<MlaReduceKernelV1Traits<512, 64, 4>, float, std::bfloat... | 77.3 | 77.2 | 78.2 | 77.3 | 77.0 | 78.0 | 77.4 | 77.3 | |
| Communication | | mscclKernel_Sum_hip_bfloat16_LL_false | 69.4 | 74.9 | 75.7 | 72.5 | 69.9 | 72.7 | 74.6 | 74.3 | |
| Runtime/Other | | void at::native:: | 61.6 | 59.4 | 59.7 | 60.1 | 59.4 | 62.0 | 60.1 | 60.0 | |
| Shared | | void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add... | 52.1 | 51.5 | 52.0 | 52.3 | 52.3 | 54.1 | 53.4 | 51.5 | |
| Runtime/Other | | void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda... | 42.3 | 42.1 | 42.3 | 42.2 | 42.5 | 44.4 | 43.4 | 42.3 | |
| Runtime/Other | | void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kerne... | 40.2 | 40.0 | 40.4 | 40.5 | 40.3 | 41.9 | 41.5 | 40.2 | |
| Spec decode | | _prepare_dflash_inputs_kernel.kd | 38.3 | 38.1 | 38.8 | 38.3 | 38.6 | 38.9 | 38.3 | 38.6 | |
| Shared | | void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_ker... | 28.7 | 28.3 | 28.4 | 28.5 | 28.6 | 29.7 | 29.1 | 28.4 | |
| Shared | | void splitk_reduce_kernel<16, 64, std::bfloat16_t, false, std::bfloat16_t, true> | 29.0 | 28.6 | 28.7 | 27.8 | 29.4 | 29.5 | 28.8 | 28.5 | |
| Shared | | void rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel<rocprim::ROCPRIM_4... | 24.7 | 24.6 | 24.6 | 24.9 | 24.9 | 26.0 | 25.1 | 24.7 | |
| Runtime/Other | | void  | 24.5 | 24.3 | 24.4 | 24.5 | 24.5 | 25.4 | 25.0 | 24.4 | |
| Shared | | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSe... | 24.6 | 24.9 | 24.4 | 24.5 | 24.5 | 25.3 | 25.1 | 24.7 | |
| Runtime/Other | | kernel.kd | 21.3 | 21.1 | 21.1 | 21.3 | 20.9 | 22.3 | 22.3 | 20.9 | |
| MLA | | void vllm::kimi_k3_fused_ops::fusedKimiK3MLADecodeQConcatKVCacheKernel<c10::B... | 19.5 | 19.3 | 19.8 | 19.5 | 19.5 | 20.5 | 20.1 | 19.5 | |
| Shared | | _ZN5aiter24bf16gemm_bf16_tn_256x256E.kd | 19.6 | 19.2 | 19.2 | 19.4 | 19.8 | 19.5 | 19.7 | 19.8 | |
| Shared | | void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctorOnSelf_ad... | 13.1 | 12.8 | 13.3 | 13.3 | 12.7 | 13.4 | 13.6 | 13.1 | |
| MLA | | _expand_page_indices_kernel.kd | 12.5 | 12.5 | 12.5 | 12.5 | 12.5 | 13.0 | 12.8 | 12.5 | |
| Shared | | void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<i... | 11.9 | 11.9 | 11.8 | 11.9 | 11.8 | 12.4 | 12.1 | 12.0 | |
| Runtime/Other | | void at::native::vectorized_elementwise_kernel<4, at::native:: | 12.0 | 11.8 | 11.8 | 11.9 | 11.8 | 12.3 | 12.1 | 11.9 | |
| Spec decode | | _rejection_kernel.kd | 9.2 | 9.1 | 9.6 | 8.8 | 9.3 | 9.5 | 9.4 | 9.0 | |
| Runtime/Other | | void at::native::vectorized_gather_kernel<16, long> | 9.2 | 9.4 | 9.1 | 9.2 | 9.1 | 9.4 | 9.5 | 9.3 | |
| MLA/DCP | | _dcp_local_seq_lens_kernel.kd | 8.5 | 8.6 | 8.5 | 8.5 | 9.1 | 8.9 | 8.8 | 9.1 | |
| Memory | | __amd_rocclr_fillBufferAligned.kd | 8.6 | 8.6 | 8.7 | 8.6 | 8.7 | 9.0 | 8.8 | 8.7 | |
| Shared | | void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctorOnSe... | 8.3 | 8.3 | 8.5 | 8.4 | 8.4 | 8.9 | 8.5 | 8.4 | |
| Sampling | | _bias_kernel.kd | 7.5 | 7.9 | 8.2 | 7.9 | 8.6 | 7.0 | 7.1 | 7.8 | |
| Spec decode | | _compute_local_logits_stats_kernel.kd | 6.7 | 6.9 | 6.9 | 7.0 | 6.6 | 7.0 | 6.5 | 6.8 | |
| Shared | | void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<int>... | 6.3 | 6.2 | 6.3 | 6.3 | 6.2 | 6.2 | 6.3 | 6.4 | |
| Spec decode | | _scatter_num_accepted_kernel.kd | 5.7 | 5.6 | 5.6 | 5.6 | 5.5 | 6.0 | 5.8 | 5.7 | |
| KV cache | | _compute_slot_mappings_kernel.kd | 5.6 | 5.2 | 5.6 | 5.6 | 5.7 | 5.5 | 5.9 | 5.6 | |
| Runtime/Input | | _post_update_kernel.kd | 5.5 | 5.5 | 5.5 | 5.6 | 5.2 | 5.6 | 5.5 | 5.6 | |
| Shared | | void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat... | 4.9 | 5.0 | 5.0 | 5.1 | 5.0 | 5.1 | 5.5 | 5.1 | |
| KDA | | postprocess_mamba_fused_kernel.kd | 4.8 | 4.7 | 4.8 | 4.7 | 4.9 | 5.0 | 4.9 | 4.8 | |
| KDA | | precopy_mamba_align_fused_kernel.kd | 4.4 | 4.4 | 4.4 | 4.5 | 4.5 | 4.7 | 4.5 | 4.6 | |
| KV cache | | _gather_block_tables_kernel.kd | 4.4 | 4.7 | 4.7 | 4.5 | 4.4 | 4.6 | 4.6 | 4.5 | |
| Runtime/Input | | _prepare_pos_seq_lens_kernel.kd | 4.3 | 4.4 | 4.4 | 4.5 | 4.3 | 4.6 | 4.4 | 4.3 | |
| Shared | | void vllm::rms_norm_kernel<c10::BFloat16, 8, 3, true> | 4.5 | 4.5 | 4.5 | 4.4 | 4.5 | 4.5 | 4.5 | 4.4 | |
| KDA | | preprocess_mamba_align_fused_kernel.kd | 4.4 | 4.3 | 4.2 | 4.4 | 4.4 | 4.5 | 4.5 | 4.4 | |
| MLA | | void vllm::rotary_embedding_kernel<c10::BFloat16, float, false> | 4.1 | 4.1 | 4.3 | 4.2 | 4.3 | 4.4 | 4.4 | 4.1 | |
| Runtime/Input | | _expand_idx_mapping_kernel.kd | 2.5 | 2.6 | 2.5 | 2.6 | 2.6 | 2.5 | 2.6 | 2.6 | |
| Communication | **Communication** | *(total)* | 3173.5 | 3024.6 | 3034.1 | 2983.1 | 3061.7 | 2852.1 | 2876.0 | 3025.7 | 10.7% :warning: |
| Communication | | CustomAR 2-stage | 1614.6 | 1616.0 | 1587.7 | 1580.2 | 1603.5 | 1504.8 | 1550.5 | 1596.4 | |
| Communication | | CustomAR 1-stage | 982.6 | 903.7 | 940.0 | 842.3 | 896.2 | 879.3 | 849.9 | 870.2 | |
| Communication | | NCCL/RCCL | 576.3 | 504.9 | 506.4 | 560.6 | 562.1 | 468.1 | 475.7 | 559.2 | |
| MoE | **MoE Routing** | *(total)* | 2669.1 | 2668.5 | 2669.6 | 2682.6 | 2670.4 | 2693.7 | 2696.3 | 2683.7 | 1.0% |
| MoE | | MoE sort mxfp4 | 1588.3 | 1586.0 | 1590.3 | 1603.2 | 1594.8 | 1607.4 | 1611.7 | 1601.3 | |
| MoE | | AITER grouped_topk | 1080.8 | 1082.5 | 1079.3 | 1079.4 | 1075.6 | 1086.3 | 1084.6 | 1082.5 | |
| KDA | **KDA Linear Attn** | *(total)* | 1753.1 | 1753.3 | 1775.7 | 1751.8 | 1753.6 | 1759.2 | 1759.1 | 1765.9 | 1.4% |
| KDA | | KDA gated-delta | 1362.1 | 1360.3 | 1379.7 | 1359.8 | 1363.5 | 1366.4 | 1370.4 | 1366.9 | |
| KDA | | KDA conv1d | 390.9 | 393.1 | 396.0 | 392.0 | 390.1 | 392.9 | 388.7 | 399.0 | |
| MLA | **Attention** | *(total)* | 1514.6 | 1524.1 | 1523.0 | 1530.2 | 1521.2 | 1519.8 | 1521.9 | 1522.9 | 1.0% |
| MLA | | MLA attn residual | 1514.6 | 1524.1 | 1523.0 | 1530.2 | 1521.2 | 1519.8 | 1521.9 | 1522.9 | |
| Shared | **Normalization** | *(total)* | 862.6 | 860.7 | 859.8 | 865.3 | 862.6 | 906.4 | 887.3 | 855.5 | 5.8% |
| Shared | | Add+RMSNorm+quant | 469.6 | 464.2 | 470.4 | 468.7 | 470.0 | 494.6 | 483.6 | 461.0 | |
| Shared | | LayerNorm | 293.8 | 295.7 | 289.0 | 296.4 | 289.4 | 304.9 | 298.0 | 291.9 | |
| Shared | | RMSNorm | 99.2 | 100.8 | 100.4 | 100.2 | 103.2 | 106.9 | 105.7 | 102.7 | |
| Shared | **Triton Fused** | *(total)* | 94.4 | 149.0 | 148.9 | 94.9 | 98.6 | 152.4 | 151.6 | 93.0 | 48.3% :warning: |
| Shared | | Triton fused op | 94.4 | 149.0 | 148.9 | 94.9 | 98.6 | 152.4 | 151.6 | 93.0 | |
| Spec decode | **Sampling** | *(total)* | 83.3 | 84.0 | 82.6 | 82.4 | 83.1 | 85.9 | 83.1 | 82.2 | 4.4% |
| Spec decode | | Sampling | 45.7 | 45.9 | 45.7 | 46.4 | 46.8 | 48.5 | 46.9 | 46.1 | |
| Spec decode | | ArgMax/Reduce | 37.6 | 38.1 | 36.9 | 36.0 | 36.3 | 37.4 | 36.2 | 36.1 | |
| MoE | **Activation** | *(total)* | 20.4 | 20.4 | 20.9 | 38.8 | 39.2 | 22.5 | 21.4 | 21.1 | 73.5% :warning: |
| MoE | | SiLU/SwiGLU | 20.4 | 20.4 | 20.9 | 38.8 | 39.2 | 22.5 | 21.4 | 21.1 | |
| Memory | **Memory** | *(total)* | 36.3 | 36.1 | 36.2 | 36.5 | 36.2 | 37.6 | 37.7 | 36.2 | 4.2% |
| Memory | | Fill | 36.3 | 36.1 | 36.2 | 36.5 | 36.2 | 37.6 | 37.7 | 36.2 | |
| All | **TOTAL** | | 26350.8 | 26346.1 | 26349.2 | 26342.4 | 26352.6 | 26324.7 | 26321.3 | 26345.6 | 0.1% |

> All ranks balanced (max 0.0% above mean)
>
> **Comm/Total ratio** (bottleneck rank): 11.9%

## Summary

No significant findings.
