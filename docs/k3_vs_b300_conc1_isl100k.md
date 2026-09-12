# Trace Comparison: mi355x_isl100k vs b300_isl100k

## Configuration

| | mi355x_isl100k | b300_isl100k |
|---|---|---|
| World size | 8 | 8 |
| Decode iterations | 66 | 492 |

## PREFILL

### mi355x_isl100k Per-Rank Breakdown (us)

| Category | Sub-kernel | R0 | R1 | R2 | R3 | R4 | R5 | R6 | R7 | Imbal% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Other** | *(total)* | 2299199.5 | 2281648.9 | 2271367.1 | 2289402.6 | 2289370.9 | 2301341.2 | 2302437.9 | 2294488.5 | 1.4% |
| | void opus_moe::stage1_a8w4::pipeline_group_split::opus_moe_stage1_a8w4_kernel... | 558432.6 | 552039.4 | 548768.7 | 552999.2 | 552485.8 | 550203.0 | 554474.1 | 551454.1 | |
| | void opus_moe_stage2_a8w4_decode_kernel_gfx950<OpusMoeStage2A8W4DecodeShape<o... | 411366.4 | 408687.9 | 406939.4 | 410641.0 | 405147.7 | 411748.1 | 406702.2 | 407798.9 | |
| | _ZN5aiter24bf16gemm_bf16_tn_256x256E.kd | 377978.3 | 374071.6 | 372394.8 | 375365.5 | 379099.3 | 381829.7 | 385778.1 | 379991.8 | |
| | void gemm_a16w16_persistent_4g_safe_kernel<opus_gemm_a16w16_persistent_traits... | 219819.9 | 216608.9 | 214904.5 | 217168.5 | 218158.3 | 217837.4 | 217200.7 | 219405.5 | |
| | void opus_moe_stage2_reduce_token_slot_route_output_kernel_gfx950<3584, 448, ... | 106709.7 | 107327.5 | 107079.1 | 107807.5 | 106672.3 | 107292.7 | 106748.5 | 106548.0 | |
| | _ZN5aiter45mla_a8w8_qh32_qseqlen4_gqaratio32_lse_cprr_psE.kd | 76547.1 | 75951.5 | 75796.3 | 75996.2 | 76250.8 | 76321.0 | 75878.7 | 75842.1 | |
| | gemm2_a4w4_port_hmax8192_imax8192_bm32_bn128_bk128_atomic_a8_g2ks2_bhoist_apf... | 68920.0 | 66129.1 | 66687.8 | 68760.2 | 68574.3 | 66615.6 | 67064.8 | 68064.4 | |
| | void vllm::gather_and_maybe_dequant_cache_page<__hip_bfloat16, unsigned char,  | 63305.2 | 62982.9 | 62533.3 | 63111.8 | 62462.3 | 63583.6 | 65323.2 | 62926.6 | |
| | void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_ker... | 62528.9 | 62713.9 | 62174.6 | 62397.8 | 62629.3 | 62725.3 | 62756.0 | 62413.5 | |
| | void vllm::situ_and_mul_kernel<c10::BFloat16> | 50872.0 | 50902.9 | 50981.6 | 50873.2 | 50893.3 | 52098.8 | 51479.9 | 51302.9 | |
| | __amd_rocclr_copyBuffer.kd | 43155.4 | 43660.8 | 42121.6 | 43461.2 | 43408.1 | 44726.8 | 43764.0 | 43709.6 | |
| | void kn_mla_reduce_v1<MlaReduceKernelV1Traits<512, 128, 1>, float, std::bfloa... | 29431.0 | 29542.1 | 28633.9 | 27663.7 | 27659.1 | 27614.0 | 27929.2 | 27995.5 | |
| | void at::native:: | 26676.0 | 26774.9 | 27267.0 | 26630.6 | 26489.8 | 26845.6 | 27456.2 | 26503.8 | |
| | mscclKernel_Sum_hip_bfloat16_Simple_false | 22617.0 | 22817.0 | 23868.2 | 24928.5 | 24969.4 | 24343.9 | 24122.9 | 24839.5 | |
| | void gqa_d192_v128_kernel<opus_gqa_d192_traits<32, 64, 8, true, true> > | 21060.7 | 20634.1 | 20681.1 | 20817.3 | 21128.8 | 21217.5 | 21233.0 | 21060.7 | |
| | void kn_get_mla_metadata_v1_2<MlaMetadataV12Traits<128, true, 0, true, false> > | 15203.8 | 15178.5 | 15234.1 | 15145.3 | 15139.8 | 15262.5 | 15289.9 | 15182.8 | |
| | void vllm::concat_and_cache_mla_kernel<__hip_bfloat16, unsigned char,  | 13508.8 | 13516.3 | 13476.7 | 13747.3 | 13441.4 | 13773.7 | 13482.4 | 13593.8 | |
| | _dcp_a2a_pack_send_kernel.kd | 11993.5 | 12083.2 | 12134.6 | 12079.0 | 12351.6 | 12390.6 | 12123.9 | 11963.1 | |
| | void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_... | 10769.5 | 10654.9 | 10848.0 | 10844.5 | 10691.2 | 10902.1 | 11021.9 | 10974.1 | |
| | void at::native::vectorized_elementwise_kernel<8, at::native::BinaryFunctor<c... | 10325.6 | 10420.4 | 10380.1 | 10491.2 | 10245.1 | 10463.3 | 10510.9 | 10560.6 | |
| | void gemm_a16w16_flatmm_splitk_kernel<opus_flatmm_splitk_traits_gfx950<256, o... | 8771.7 | 9163.3 | 9365.4 | 9395.3 | 8809.1 | 9335.9 | 8878.2 | 8843.8 | |
| | _dcp_a2a_unpack_combine_kernel.kd | 8849.4 | 8948.1 | 9158.0 | 8925.8 | 9138.8 | 9332.7 | 9054.3 | 9081.2 | |
| | void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kerne... | 9087.8 | 9178.4 | 8757.8 | 9042.3 | 9003.9 | 9118.8 | 9031.5 | 8894.9 | |
| | _ZN5aiter37bf16gemm_fp32bf16_tn_32x64_pf3_splitkE.kd | 8122.9 | 8098.1 | 8168.1 | 8077.9 | 8034.1 | 8126.5 | 8077.4 | 8164.5 | |
| | void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_ker... | 6460.4 | 6555.6 | 6262.6 | 6411.0 | 7081.6 | 7303.2 | 7370.9 | 7340.7 | |
| | void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add... | 7081.1 | 7065.4 | 7112.3 | 7097.4 | 7052.3 | 7149.9 | 7178.3 | 7201.1 | |
| | void kn_mla_reduce_v1<MlaReduceKernelV1Traits<512, 64, 4>, float, std::bfloat... | 6270.8 | 6302.6 | 6309.6 | 6449.1 | 6230.0 | 6376.2 | 6238.6 | 6363.7 | |
| | mscclKernel_Sum_hip_bfloat16_LL_false | 6009.3 | 6040.7 | 6250.4 | 5873.9 | 6056.5 | 5958.9 | 6012.9 | 6157.2 | |
| | void opus_moe_stage2_reduce_token_slot_route_output_kernel_gfx950<3072, 384, ... | 4085.5 | 4167.3 | 4091.7 | 4070.9 | 4087.1 | 4151.8 | 4068.2 | 4117.6 | |
| | void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda... | 3503.9 | 3570.9 | 3506.3 | 3515.9 | 3830.6 | 3812.2 | 3881.7 | 3885.5 | |
| | _prepare_dflash_inputs_kernel.kd | 2649.3 | 2645.0 | 2657.0 | 2643.5 | 2654.0 | 2657.5 | 2642.7 | 2651.2 | |
| | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSe... | 2100.2 | 2090.4 | 2061.5 | 2102.0 | 2332.7 | 2419.1 | 2397.4 | 2411.5 | |
| | void  | 2057.8 | 2074.1 | 2012.1 | 2018.1 | 2353.9 | 2400.4 | 2307.5 | 2408.7 | |
| | void splitk_reduce_kernel<16, 64, std::bfloat16_t, false, std::bfloat16_t, true> | 2061.2 | 2122.5 | 2089.8 | 2121.4 | 2096.8 | 2173.5 | 2168.9 | 2103.7 | |
| | void rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel<rocprim::ROCPRIM_4... | 1759.6 | 1736.1 | 1785.4 | 1723.4 | 1749.9 | 1792.0 | 1763.2 | 1746.0 | |
| | kernel.kd | 1621.8 | 1631.8 | 1667.9 | 1627.7 | 1636.1 | 1631.1 | 1667.5 | 1629.8 | |
| | void vllm::kimi_k3_fused_ops::fusedKimiK3MLADecodeQConcatKVCacheKernel<c10::B... | 1499.0 | 1502.6 | 1497.0 | 1461.7 | 1469.5 | 1542.8 | 1531.8 | 1496.2 | |
| | void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<i... | 1235.7 | 1156.9 | 1137.7 | 1164.8 | 1480.3 | 1490.7 | 1418.2 | 1445.2 | |
| | void at::native::vectorized_elementwise_kernel<4, at::native:: | 1161.7 | 1166.9 | 1140.5 | 1136.0 | 1446.6 | 1483.3 | 1471.9 | 1471.0 | |
| | void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<int>... | 782.5 | 820.2 | 760.2 | 807.9 | 1114.5 | 1131.9 | 1091.4 | 1097.5 | |
| | void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctorOnSelf_ad... | 982.3 | 968.7 | 969.2 | 965.5 | 996.9 | 1018.6 | 1029.4 | 984.1 | |
| | void at::native::vectorized_gather_kernel<16, long> | 971.0 | 971.3 | 925.8 | 936.6 | 966.8 | 962.6 | 963.6 | 950.5 | |
| | _expand_page_indices_kernel.kd | 875.6 | 892.0 | 924.1 | 878.2 | 876.6 | 896.3 | 891.8 | 887.0 | |
| | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add... | 336.2 | 392.5 | 353.5 | 409.9 | 683.1 | 737.4 | 677.1 | 718.9 | |
| | _chunk_metadata_kernel.kd | 334.7 | 431.7 | 366.4 | 399.1 | 669.8 | 707.3 | 658.9 | 710.4 | |
| | void at::native::vectorized_elementwise_kernel<4, at::native::compare_scalar_... | 350.9 | 411.1 | 360.6 | 417.3 | 637.2 | 701.2 | 625.0 | 657.1 | |
| | _dcp_local_seq_lens_kernel.kd | 689.2 | 663.9 | 679.8 | 649.7 | 689.5 | 676.8 | 687.7 | 678.7 | |
| | void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctorOnSe... | 625.5 | 643.9 | 640.2 | 624.4 | 624.7 | 642.1 | 656.3 | 652.1 | |
| | _compute_slot_mappings_kernel.kd | 640.6 | 591.0 | 610.5 | 592.0 | 606.4 | 603.3 | 618.6 | 598.9 | |
| | _rejection_kernel.kd | 615.1 | 599.6 | 612.1 | 619.0 | 623.2 | 605.2 | 613.5 | 637.3 | |
| | precopy_mamba_align_fused_kernel.kd | 634.3 | 545.8 | 550.1 | 540.7 | 582.0 | 579.0 | 589.2 | 581.4 | |
| | __amd_rocclr_fillBufferAligned.kd | 610.1 | 613.9 | 613.0 | 612.4 | 617.2 | 630.2 | 629.8 | 631.5 | |
| | _bias_kernel.kd | 555.6 | 599.8 | 553.0 | 593.8 | 591.6 | 613.2 | 548.4 | 549.0 | |
| | void vllm::rms_norm_kernel<c10::BFloat16, 8, 3, true> | 505.6 | 501.9 | 500.6 | 497.4 | 498.1 | 510.0 | 503.2 | 506.4 | |
| | _compute_local_logits_stats_kernel.kd | 440.7 | 458.5 | 440.7 | 465.1 | 448.6 | 464.2 | 440.5 | 443.7 | |
| | void vllm::rotary_embedding_kernel<c10::BFloat16, float, false> | 414.1 | 418.4 | 418.2 | 415.4 | 420.3 | 436.1 | 433.5 | 419.9 | |
| | _scatter_num_accepted_kernel.kd | 397.8 | 396.1 | 401.0 | 394.2 | 392.0 | 416.5 | 392.4 | 397.6 | |
| | preprocess_mamba_align_fused_kernel.kd | 414.8 | 380.4 | 359.1 | 364.8 | 395.7 | 388.2 | 381.3 | 379.5 | |
| | void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat... | 373.9 | 372.4 | 383.4 | 373.0 | 371.4 | 384.5 | 389.1 | 370.7 | |
| | _gather_block_tables_kernel.kd | 353.7 | 357.9 | 349.4 | 369.4 | 382.1 | 379.5 | 366.9 | 351.8 | |
| | postprocess_mamba_fused_kernel.kd | 345.6 | 356.6 | 341.0 | 353.8 | 362.3 | 348.0 | 381.4 | 374.3 | |
| | _post_update_kernel.kd | 370.3 | 357.3 | 358.2 | 349.2 | 334.4 | 378.9 | 332.5 | 320.7 | |
| | _prepare_pos_seq_lens_kernel.kd | 340.1 | 332.2 | 335.5 | 339.5 | 364.9 | 362.9 | 352.6 | 360.5 | |
| | _prepare_prefill_inputs_kernel.kd | 164.7 | 186.3 | 170.2 | 177.0 | 207.9 | 232.5 | 207.8 | 223.6 | |
| | _apply_write_kernel.kd | 197.1 | 207.1 | 198.7 | 206.9 | 208.9 | 210.6 | 203.2 | 203.7 | |
| | _expand_idx_mapping_kernel.kd | 177.5 | 173.1 | 167.3 | 173.3 | 173.9 | 184.1 | 181.0 | 162.3 | |
| | _zero_kv_blocks_kernel.kd | 66.3 | 66.8 | 64.9 | 65.8 | 64.6 | 63.7 | 65.8 | 69.8 | |
| | __amd_rocclr_streamOpsWrite.kd | 23.7 | 24.0 | 23.8 | 23.7 | 24.2 | 24.5 | 24.4 | 24.6 | |
| **Communication** | *(total)* | 2229751.2 | 2200487.7 | 2259647.7 | 2173202.8 | 2224404.0 | 2174936.3 | 2248711.7 | 2274827.2 | 4.6% |
| | NCCL/RCCL | 1891981.3 | 1866970.0 | 1940522.6 | 1910837.6 | 1888977.2 | 1856368.1 | 1919159.0 | 1952683.2 | |
| | CustomAR 2-stage | 280355.5 | 275263.5 | 259423.2 | 207201.9 | 279167.8 | 261164.7 | 271407.7 | 265868.7 | |
| | CustomAR 1-stage | 57414.3 | 58254.2 | 59701.9 | 55163.3 | 56259.0 | 57403.6 | 58145.0 | 56275.3 | |
| **GEMM** | *(total)* | 2066716.4 | 2080050.0 | 2052340.3 | 2061483.6 | 2067158.9 | 2075908.4 | 2012343.7 | 2012009.6 | 3.3% |
| | hipBLASLt (Cijk) | 1636044.0 | 1647417.6 | 1622167.4 | 1630507.9 | 1634635.2 | 1643125.2 | 1582516.8 | 1582452.1 | |
| | AITER hgemm bf16 | 252297.4 | 252927.4 | 252920.1 | 251349.0 | 253992.3 | 253447.9 | 251354.8 | 250572.1 | |
| | AITER flydsl MoE | 150108.1 | 151642.7 | 149119.3 | 151433.1 | 150449.9 | 150803.1 | 150172.7 | 150916.9 | |
| | AITER wfp4 batched | 16145.6 | 15918.3 | 16067.7 | 16005.4 | 15922.8 | 16379.4 | 16185.0 | 15946.7 | |
| | rocBLAS splitK | 12121.3 | 12144.0 | 12065.8 | 12188.2 | 12158.7 | 12152.8 | 12114.4 | 12121.8 | |
| **Attention** | *(total)* | 1476289.3 | 1474546.1 | 1468321.9 | 1466276.3 | 1480718.8 | 1496338.0 | 1477422.4 | 1474504.0 | 2.0% |
| | aiter attn | 914746.2 | 910289.2 | 906352.9 | 903021.1 | 915987.7 | 928085.2 | 910183.5 | 903190.6 | |
| | MLA attn residual | 537206.8 | 539708.6 | 537858.1 | 538558.0 | 540511.3 | 543762.1 | 542762.4 | 546775.2 | |
| | MLA merge states | 24336.3 | 24548.2 | 24110.9 | 24697.1 | 24219.9 | 24490.7 | 24476.5 | 24538.3 | |
| **KDA Linear Attn** | *(total)* | 638698.5 | 643779.2 | 632175.2 | 637562.1 | 639489.2 | 640744.9 | 642340.2 | 639512.8 | 1.8% |
| | KDA gated-delta | 428293.2 | 427143.0 | 426004.6 | 428428.8 | 427513.7 | 427119.9 | 430044.3 | 428430.5 | |
| | KDA conv1d | 98543.0 | 104727.6 | 98261.4 | 101581.4 | 101328.8 | 101937.1 | 101030.7 | 99850.7 | |
| | KDA gate/state | 75245.2 | 75311.0 | 72736.1 | 72708.9 | 74489.6 | 75462.7 | 74922.8 | 75053.9 | |
| | KDA GLA | 36617.1 | 36597.6 | 35173.0 | 34843.0 | 36157.1 | 36225.2 | 36342.4 | 36177.6 | |
| **MoE Routing** | *(total)* | 293665.7 | 292869.5 | 291748.7 | 294707.2 | 292701.4 | 294140.4 | 293337.2 | 292258.4 | 1.0% |
| | MoE sort mxfp4 | 177098.5 | 177317.4 | 176165.4 | 179160.8 | 176944.9 | 177484.1 | 177347.0 | 176653.7 | |
| | AITER grouped_topk | 116567.2 | 115552.1 | 115583.3 | 115546.5 | 115756.5 | 116656.3 | 115990.2 | 115604.7 | |
| **Normalization** | *(total)* | 102745.0 | 103407.0 | 102351.5 | 102956.9 | 101914.3 | 103922.9 | 103739.9 | 104483.2 | 2.5% |
| | Add+RMSNorm+quant | 52763.4 | 53145.1 | 53433.7 | 53058.4 | 52209.9 | 53564.1 | 53250.3 | 53854.9 | |
| | LayerNorm | 38409.5 | 38942.8 | 37566.1 | 38673.8 | 38359.8 | 38753.9 | 39181.0 | 39289.9 | |
| | RMSNorm | 11572.1 | 11319.1 | 11351.8 | 11224.7 | 11344.5 | 11604.9 | 11308.6 | 11338.3 | |
| **Quantization** | *(total)* | 17982.6 | 17900.0 | 17740.5 | 17905.5 | 18691.7 | 17918.3 | 17946.1 | 17754.7 | 5.3% |
| | dynamic per-group quant | 17982.6 | 17900.0 | 17740.5 | 17905.5 | 18691.7 | 17918.3 | 17946.1 | 17754.7 | |
| **Memory** | *(total)* | 12050.2 | 11937.6 | 11889.0 | 11660.6 | 12540.8 | 12709.6 | 12635.9 | 12551.3 | 8.6% |
| | Fill | 11694.4 | 11570.5 | 11519.9 | 11291.1 | 12046.1 | 12186.1 | 12140.6 | 12062.7 | |
| | Memcpy | 355.7 | 367.1 | 369.1 | 369.5 | 494.7 | 523.5 | 495.3 | 488.7 | |
| **Triton Fused** | *(total)* | 6488.1 | 10128.0 | 10242.6 | 6455.5 | 6711.8 | 10247.2 | 10131.4 | 6561.6 | 45.3% :warning: |
| | Triton fused op | 6488.1 | 10128.0 | 10242.6 | 6455.5 | 6711.8 | 10247.2 | 10131.4 | 6561.6 | |
| **Sampling** | *(total)* | 6339.1 | 6220.7 | 6244.0 | 6286.0 | 6245.2 | 6421.7 | 6237.6 | 6359.2 | 3.2% |
| | Sampling | 3465.5 | 3428.5 | 3425.4 | 3456.8 | 3490.2 | 3530.4 | 3535.9 | 3534.9 | |
| | ArgMax/Reduce | 2873.6 | 2792.2 | 2818.6 | 2829.2 | 2755.0 | 2891.3 | 2701.7 | 2824.2 | |
| **Activation** | *(total)* | 1542.1 | 1581.2 | 1611.8 | 2930.1 | 2922.1 | 1615.0 | 1598.4 | 1639.6 | 71.9% :warning: |
| | SiLU/SwiGLU | 1542.1 | 1581.2 | 1611.8 | 2930.1 | 2922.1 | 1615.0 | 1598.4 | 1639.6 | |
| **TOTAL** | | 9151467.7 | 9124555.9 | 9125680.1 | 9070829.2 | 9142869.1 | 9136243.8 | 9128882.3 | 9136950.0 | 0.9% |

> All ranks balanced (max 0.3% above mean)
>
> **Comm/Total ratio** (bottleneck rank): 24.6%

## DECODE

### mi355x_isl100k Per-Rank Breakdown (per-iter avg us)

| Category | Sub-kernel | R0 | R1 | R2 | R3 | R4 | R5 | R6 | R7 | Imbal% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **GEMM** | *(total)* | 10377.0 | 10413.2 | 10375.6 | 10402.0 | 10420.4 | 10403.0 | 10386.0 | 10412.7 | 0.4% |
| | hipBLASLt (Cijk) | 3902.3 | 3900.6 | 3903.3 | 3911.3 | 3907.9 | 3894.7 | 3917.7 | 3922.7 | |
| | AITER hgemm bf16 | 3726.7 | 3739.3 | 3741.4 | 3716.3 | 3757.3 | 3746.2 | 3721.5 | 3706.2 | |
| | AITER flydsl MoE | 2358.8 | 2386.9 | 2342.4 | 2386.0 | 2368.6 | 2368.7 | 2356.5 | 2396.6 | |
| | AITER wfp4 batched | 239.4 | 237.4 | 238.0 | 238.7 | 236.9 | 243.8 | 241.1 | 238.1 | |
| | rocBLAS splitK | 149.7 | 148.9 | 150.4 | 149.7 | 149.7 | 149.7 | 149.2 | 149.1 | |
| **Other** | *(total)* | 6118.7 | 6092.0 | 6116.0 | 6151.1 | 6135.0 | 6154.1 | 6129.6 | 6167.4 | 1.2% |
| | gemm2_a4w4_port_hmax8192_imax8192_bm32_bn128_bk128_atomic_a8_g2ks2_bhoist_apf... | 1383.7 | 1349.6 | 1357.4 | 1404.5 | 1385.7 | 1353.7 | 1359.1 | 1391.4 | |
| | _ZN5aiter45mla_a8w8_qh32_qseqlen4_gqaratio32_lse_cprr_psE.kd | 1119.8 | 1110.9 | 1112.6 | 1115.6 | 1115.8 | 1116.9 | 1111.5 | 1110.6 | |
| | void kn_mla_reduce_v1<MlaReduceKernelV1Traits<512, 128, 1>, float, std::bfloa... | 439.7 | 436.7 | 427.0 | 412.9 | 413.1 | 411.4 | 416.5 | 417.8 | |
| | void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_ker... | 396.8 | 396.8 | 395.8 | 399.8 | 398.5 | 399.9 | 398.9 | 399.3 | |
| | void vllm::situ_and_mul_kernel<c10::BFloat16> | 379.3 | 380.7 | 390.0 | 381.3 | 388.3 | 399.1 | 392.0 | 391.8 | |
| | mscclKernel_Sum_hip_bfloat16_Simple_false | 332.4 | 345.5 | 346.6 | 362.6 | 366.3 | 358.3 | 351.7 | 362.3 | |
| | __amd_rocclr_copyBuffer.kd | 274.0 | 274.5 | 274.4 | 272.3 | 273.7 | 280.9 | 281.1 | 279.1 | |
| | void kn_get_mla_metadata_v1_2<MlaMetadataV12Traits<128, true, 0, true, false> > | 216.8 | 216.7 | 218.5 | 217.0 | 216.5 | 218.4 | 219.1 | 216.0 | |
| | _dcp_a2a_pack_send_kernel.kd | 173.9 | 174.6 | 175.3 | 175.3 | 178.3 | 179.5 | 176.9 | 172.8 | |
| | void gemm_a16w16_flatmm_splitk_kernel<opus_flatmm_splitk_traits_gfx950<256, o... | 127.4 | 133.1 | 136.3 | 136.1 | 127.7 | 135.3 | 129.1 | 128.8 | |
| | _dcp_a2a_unpack_combine_kernel.kd | 129.0 | 129.5 | 132.3 | 129.7 | 132.5 | 135.4 | 132.0 | 132.1 | |
| | void vllm::concat_and_cache_mla_kernel<__hip_bfloat16, unsigned char,  | 123.4 | 122.9 | 125.4 | 122.4 | 124.9 | 127.5 | 125.5 | 125.1 | |
| | _ZN5aiter37bf16gemm_fp32bf16_tn_32x64_pf3_splitkE.kd | 106.4 | 105.9 | 107.3 | 106.1 | 105.3 | 106.6 | 105.9 | 107.2 | |
| | void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_... | 99.8 | 98.4 | 101.0 | 101.0 | 99.1 | 102.7 | 104.2 | 103.7 | |
| | void at::native::vectorized_elementwise_kernel<8, at::native::BinaryFunctor<c... | 100.1 | 100.7 | 98.5 | 100.8 | 97.9 | 101.3 | 102.9 | 103.5 | |
| | void kn_mla_reduce_v1<MlaReduceKernelV1Traits<512, 64, 4>, float, std::bfloat... | 77.2 | 76.0 | 76.4 | 77.0 | 76.3 | 76.1 | 76.4 | 76.9 | |
| | mscclKernel_Sum_hip_bfloat16_LL_false | 69.6 | 71.9 | 73.4 | 70.3 | 70.6 | 71.3 | 69.0 | 73.6 | |
| | void at::native:: | 58.6 | 57.0 | 56.7 | 56.1 | 56.5 | 57.9 | 57.5 | 58.2 | |
| | void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add... | 51.6 | 51.5 | 52.0 | 51.6 | 51.3 | 52.5 | 53.3 | 52.7 | |
| | void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda... | 42.1 | 42.1 | 41.9 | 41.6 | 41.6 | 42.7 | 43.1 | 43.1 | |
| | void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kerne... | 40.6 | 40.2 | 40.2 | 40.6 | 39.6 | 40.7 | 40.6 | 41.2 | |
| | _prepare_dflash_inputs_kernel.kd | 38.1 | 38.0 | 38.4 | 38.3 | 38.3 | 38.4 | 38.3 | 38.1 | |
| | void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_ker... | 28.5 | 28.3 | 28.4 | 28.4 | 28.2 | 29.0 | 29.1 | 29.3 | |
| | void splitk_reduce_kernel<16, 64, std::bfloat16_t, false, std::bfloat16_t, true> | 27.5 | 28.6 | 28.1 | 28.2 | 28.2 | 29.3 | 29.1 | 28.1 | |
| | void  | 24.6 | 24.8 | 24.4 | 24.2 | 24.3 | 25.6 | 25.0 | 25.5 | |
| | void rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel<rocprim::ROCPRIM_4... | 24.8 | 24.7 | 24.6 | 24.6 | 24.5 | 25.2 | 25.1 | 25.1 | |
| | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSe... | 24.3 | 24.3 | 24.3 | 24.7 | 24.4 | 24.9 | 24.8 | 24.8 | |
| | kernel.kd | 21.3 | 21.4 | 21.9 | 21.4 | 21.5 | 21.4 | 21.9 | 21.4 | |
| | void vllm::kimi_k3_fused_ops::fusedKimiK3MLADecodeQConcatKVCacheKernel<c10::B... | 19.6 | 19.5 | 19.7 | 19.0 | 19.4 | 20.4 | 20.0 | 19.8 | |
| | void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctorOnSelf_ad... | 13.0 | 12.7 | 12.7 | 12.6 | 13.1 | 13.4 | 13.5 | 12.9 | |
| | _expand_page_indices_kernel.kd | 12.5 | 12.5 | 12.4 | 12.4 | 12.4 | 12.8 | 12.7 | 12.7 | |
| | void at::native::vectorized_elementwise_kernel<4, at::native:: | 12.0 | 11.8 | 11.9 | 11.8 | 11.8 | 12.3 | 12.2 | 12.1 | |
| | void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<i... | 12.0 | 11.8 | 11.8 | 11.7 | 11.9 | 12.2 | 12.2 | 12.0 | |
| | _rejection_kernel.kd | 9.3 | 8.9 | 9.1 | 9.5 | 9.2 | 9.1 | 9.1 | 9.6 | |
| | void at::native::vectorized_gather_kernel<16, long> | 9.1 | 9.1 | 9.1 | 9.2 | 9.1 | 9.4 | 9.3 | 9.3 | |
| | _dcp_local_seq_lens_kernel.kd | 9.0 | 8.6 | 8.9 | 8.4 | 8.6 | 8.7 | 8.8 | 8.6 | |
| | __amd_rocclr_fillBufferAligned.kd | 8.5 | 8.6 | 8.6 | 8.6 | 8.6 | 8.8 | 8.8 | 8.9 | |
| | void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctorOnSe... | 8.3 | 8.5 | 8.4 | 8.2 | 8.2 | 8.4 | 8.6 | 8.6 | |
| | _bias_kernel.kd | 6.9 | 7.8 | 7.0 | 7.7 | 7.5 | 7.9 | 7.0 | 6.9 | |
| | _compute_local_logits_stats_kernel.kd | 6.5 | 6.8 | 6.5 | 7.0 | 6.8 | 7.0 | 6.5 | 6.7 | |
| | void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<int>... | 6.4 | 6.4 | 6.2 | 6.3 | 6.2 | 6.3 | 6.3 | 6.3 | |
| | _compute_slot_mappings_kernel.kd | 5.9 | 5.6 | 5.8 | 5.6 | 5.4 | 5.8 | 5.9 | 5.6 | |
| | _scatter_num_accepted_kernel.kd | 5.5 | 5.6 | 5.6 | 5.5 | 5.5 | 5.7 | 5.7 | 5.5 | |
| | _post_update_kernel.kd | 5.5 | 5.5 | 5.5 | 5.3 | 5.4 | 5.5 | 5.5 | 5.1 | |
| | void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat... | 4.9 | 4.9 | 5.1 | 4.9 | 4.9 | 5.0 | 5.1 | 4.9 | |
| | _gather_block_tables_kernel.kd | 4.5 | 4.4 | 4.4 | 4.5 | 4.4 | 4.5 | 4.5 | 4.5 | |
| | _prepare_pos_seq_lens_kernel.kd | 4.5 | 4.3 | 4.4 | 4.4 | 4.3 | 4.5 | 4.4 | 4.5 | |
| | preprocess_mamba_align_fused_kernel.kd | 4.4 | 4.4 | 4.4 | 4.4 | 4.3 | 4.5 | 4.4 | 4.4 | |
| | postprocess_mamba_fused_kernel.kd | 4.1 | 4.1 | 4.1 | 4.2 | 4.2 | 4.2 | 4.2 | 4.4 | |
| | void vllm::rms_norm_kernel<c10::BFloat16, 8, 3, true> | 4.3 | 4.3 | 4.3 | 4.3 | 4.3 | 4.4 | 4.4 | 4.3 | |
| | void vllm::rotary_embedding_kernel<c10::BFloat16, float, false> | 4.2 | 4.2 | 4.2 | 4.2 | 4.0 | 4.3 | 4.4 | 4.2 | |
| | precopy_mamba_align_fused_kernel.kd | 4.0 | 4.0 | 4.1 | 4.1 | 4.0 | 4.2 | 4.1 | 4.1 | |
| | _expand_idx_mapping_kernel.kd | 2.6 | 2.6 | 2.5 | 2.7 | 2.5 | 2.7 | 2.7 | 2.5 | |
| **Communication** | *(total)* | 3104.4 | 3007.3 | 3007.6 | 2949.7 | 3006.3 | 2908.8 | 2952.2 | 2920.0 | 6.6% |
| | CustomAR 2-stage | 1668.2 | 1640.8 | 1627.8 | 1589.0 | 1626.2 | 1582.3 | 1604.7 | 1582.3 | |
| | CustomAR 1-stage | 880.3 | 868.5 | 881.3 | 795.2 | 834.4 | 856.1 | 865.4 | 803.0 | |
| | NCCL/RCCL | 556.0 | 498.0 | 498.5 | 565.5 | 545.7 | 470.5 | 482.2 | 534.7 | |
| **MoE Routing** | *(total)* | 2678.6 | 2677.8 | 2693.2 | 2697.7 | 2680.1 | 2690.9 | 2695.3 | 2689.7 | 0.7% |
| | MoE sort mxfp4 | 1600.4 | 1601.1 | 1608.9 | 1616.6 | 1596.7 | 1608.2 | 1612.1 | 1609.5 | |
| | AITER grouped_topk | 1078.2 | 1076.7 | 1084.4 | 1081.1 | 1083.4 | 1082.8 | 1083.2 | 1080.2 | |
| **KDA Linear Attn** | *(total)* | 1753.1 | 1762.1 | 1760.6 | 1762.4 | 1767.2 | 1759.6 | 1769.1 | 1764.9 | 0.9% |
| | KDA gated-delta | 1359.1 | 1371.6 | 1368.5 | 1366.8 | 1373.9 | 1370.0 | 1372.5 | 1370.1 | |
| | KDA conv1d | 394.1 | 390.5 | 392.1 | 395.5 | 393.2 | 389.7 | 396.5 | 394.8 | |
| **Attention** | *(total)* | 1517.9 | 1516.5 | 1528.2 | 1522.9 | 1526.4 | 1519.0 | 1519.7 | 1524.9 | 0.8% |
| | MLA attn residual | 1517.9 | 1516.5 | 1528.2 | 1522.9 | 1526.4 | 1519.0 | 1519.7 | 1524.9 | |
| **Normalization** | *(total)* | 858.9 | 858.5 | 860.7 | 853.7 | 850.9 | 881.4 | 877.0 | 888.2 | 4.3% |
| | Add+RMSNorm+quant | 462.6 | 461.4 | 472.7 | 458.3 | 457.7 | 479.4 | 472.7 | 482.5 | |
| | LayerNorm | 295.5 | 296.8 | 286.2 | 296.9 | 291.1 | 297.5 | 302.6 | 304.1 | |
| | RMSNorm | 100.8 | 100.4 | 101.7 | 98.5 | 102.1 | 104.5 | 101.7 | 101.6 | |
| **Triton Fused** | *(total)* | 94.3 | 149.0 | 151.4 | 94.2 | 97.5 | 151.0 | 149.3 | 97.1 | 46.5% :warning: |
| | Triton fused op | 94.3 | 149.0 | 151.4 | 94.2 | 97.5 | 151.0 | 149.3 | 97.1 | |
| **Sampling** | *(total)* | 82.5 | 80.7 | 81.4 | 81.7 | 80.8 | 83.6 | 81.3 | 83.0 | 3.5% |
| | Sampling | 45.7 | 44.8 | 45.1 | 45.3 | 45.4 | 46.2 | 46.7 | 46.8 | |
| | ArgMax/Reduce | 36.8 | 35.9 | 36.3 | 36.4 | 35.4 | 37.3 | 34.6 | 36.2 | |
| **Activation** | *(total)* | 20.3 | 20.8 | 21.2 | 38.3 | 38.3 | 21.2 | 21.0 | 21.6 | 71.3% :warning: |
| | SiLU/SwiGLU | 20.3 | 20.8 | 21.2 | 38.3 | 38.3 | 21.2 | 21.0 | 21.6 | |
| **Memory** | *(total)* | 36.3 | 36.0 | 36.3 | 35.6 | 36.0 | 36.9 | 37.0 | 36.9 | 3.7% |
| | Fill | 36.3 | 36.0 | 36.3 | 35.6 | 36.0 | 36.9 | 37.0 | 36.9 | |
| **TOTAL** | | 26642.0 | 26613.7 | 26632.1 | 26589.3 | 26638.9 | 26609.6 | 26617.6 | 26606.5 | 0.2% |

> All ranks balanced (max 0.1% above mean)
>
> **Comm/Total ratio** (bottleneck rank): 11.6%

### b300_isl100k Per-Rank Breakdown (per-iter avg us)

| Category | Sub-kernel | R0 | R1 | R2 | R3 | R4 | R5 | R6 | R7 | Imbal% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Other** | *(total)* | 3757.9 | 3790.8 | 3748.3 | 3796.6 | 3764.2 | 3758.1 | 3793.2 | 3731.3 | 1.7% |
| | void fused_a_gemm_kernel<1, 3584, 7168, 16, 8, 256, 16> | 507.6 | 507.3 | 508.8 | 506.7 | 507.6 | 503.3 | 506.3 | 507.6 | |
| | nvjet_sm103_tst_64x8_64x16_2x1_v_bz_splitK_TNT | 452.1 | 451.9 | 452.4 | 449.0 | 453.4 | 450.4 | 448.7 | 452.3 | |
| | void fused_a_gemm_kernel<1, 1536, 7168, 16, 8, 256, 16> | 312.4 | 310.9 | 310.6 | 311.3 | 311.0 | 307.6 | 311.6 | 310.9 | |
| | void flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel< | 241.0 | 235.9 | 228.0 | 243.0 | 238.8 | 239.2 | 243.9 | 227.7 | |
| | kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_ll_bf16_splitkL... | 234.4 | 234.3 | 235.0 | 236.6 | 235.1 | 233.0 | 235.7 | 234.5 | |
| | kernel_cutlass_kernel_vllmmodelskimi_k3nvidiaopscute_dsllatent_moe_tailfused_... | 215.5 | 216.5 | 216.3 | 215.9 | 215.9 | 217.4 | 216.4 | 216.3 | |
| | nvjet_sm103_tst_64x8_64x16_4x1_v_bz_TNT | 211.1 | 211.4 | 211.0 | 211.0 | 210.8 | 208.1 | 210.7 | 210.9 | |
| | void fused_a_gemm_kernel<1, 7168, 768, 16, 8, 256, 3> | 158.0 | 157.7 | 160.7 | 155.9 | 160.2 | 158.7 | 156.9 | 159.5 | |
| | void  | 145.6 | 143.4 | 144.8 | 142.9 | 143.1 | 141.2 | 142.6 | 142.6 | |
| | (anonymous namespace)::direct_dcp_q_gather_multimem_kernel(uint4 const*, uint... | 118.7 | 117.2 | 117.3 | 116.6 | 117.5 | 119.6 | 115.8 | 113.8 | |
| | void flashinfer::trtllm_mnnvl_allreduce::twoshotAllreduceKernel< | 42.3 | 73.2 | 37.1 | 70.9 | 41.1 | 65.8 | 74.0 | 26.5 | |
| | void vllm::situ_and_mul_kernel<c10::BFloat16> | 73.9 | 69.7 | 69.2 | 69.4 | 69.7 | 64.0 | 68.6 | 67.1 | |
| | void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_12... | 70.4 | 71.3 | 69.9 | 72.3 | 69.5 | 69.0 | 71.4 | 70.7 | |
| | void fused_a_gemm_kernel<1, 2112, 7168, 16, 8, 256, 16> | 69.0 | 69.2 | 69.0 | 68.8 | 69.1 | 69.8 | 68.9 | 69.3 | |
| | kernel_cutlass_kernel_vllmmodelskimi_k3nvidiaopscute_dsllatent_moe_taillampor... | 68.6 | 67.6 | 65.6 | 65.6 | 67.4 | 64.5 | 66.0 | 66.6 | |
| | kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8Quantiz... | 61.0 | 61.2 | 61.1 | 60.1 | 59.0 | 62.0 | 58.9 | 62.0 | |
| | void sm100::fwd_prod_v2::attn_res_fwd_online_v2_kernel<7168, 8, 3, 1, false, ... | 45.2 | 45.6 | 45.8 | 45.5 | 45.3 | 45.3 | 45.4 | 45.7 | |
| | void fused_a_gemm_kernel<1, 2304, 1536, 16, 8, 256, 6> | 43.9 | 43.8 | 44.1 | 44.2 | 44.1 | 43.4 | 44.2 | 44.1 | |
| | void sm100::fwd_prod_v2::attn_res_fwd_online_v2_kernel<7168, 7, 3, 1, false, ... | 43.4 | 43.8 | 43.9 | 44.0 | 44.1 | 43.8 | 43.8 | 43.9 | |
| | void sm100::fwd_prod_v2::attn_res_fwd_online_v2_kernel<7168, 6, 3, 1, false, ... | 39.4 | 40.1 | 40.1 | 40.5 | 40.0 | 39.9 | 40.4 | 40.1 | |
| | void sm100::fwd_prod_v2::attn_res_fwd_online_v2_kernel<7168, 5, 3, 1, false, ... | 38.5 | 39.0 | 38.8 | 39.2 | 38.7 | 38.7 | 39.4 | 39.1 | |
| | nvjet_sm103_tst_64x8_64x16_4x1_v_bz_NNT | 37.5 | 38.4 | 38.2 | 35.1 | 38.1 | 35.7 | 34.8 | 38.6 | |
| | nvjet_sm103_tst_64x8_64x16_4x2_h_bz_TNT | 38.2 | 37.3 | 37.0 | 36.6 | 37.5 | 36.1 | 36.9 | 37.9 | |
| | void sm100::fwd_prod_v2::attn_res_fwd_online_v2_kernel<7168, 4, 3, 1, false, ... | 37.1 | 37.6 | 37.4 | 37.8 | 37.5 | 37.8 | 38.0 | 37.6 | |
| | void sm100::fwd_prod_v2::attn_res_fwd_online_v2_kernel<7168, 3, 3, 1, false, ... | 34.3 | 35.0 | 35.0 | 35.2 | 34.8 | 34.9 | 35.3 | 35.1 | |
| | void vllm::kimi_k3_fused_ops::fusedKimiK3MLADecodeQConcatKVCacheKernel<c10::B... | 34.5 | 34.2 | 34.3 | 35.1 | 34.1 | 33.7 | 35.1 | 34.4 | |
| | void sm100::fwd_prod_v2::attn_res_fwd_online_v2_kernel<7168, 9, 3, 1, false, ... | 34.1 | 34.4 | 34.4 | 34.4 | 34.6 | 34.5 | 34.5 | 34.5 | |
| | void sm100::fwd_prod_v2::attn_res_fwd_online_v2_kernel<7168, 2, 3, 1, false, ... | 32.9 | 33.0 | 33.2 | 33.3 | 33.6 | 33.4 | 33.6 | 33.1 | |
| | vllm::direct_dcp::increment_epoch_kernel(long*) | 27.8 | 29.4 | 26.7 | 30.1 | 26.8 | 27.6 | 29.2 | 29.0 | |
| | (anonymous namespace)::signal_kernel(long const*, long const*, long, long) | 27.5 | 26.1 | 27.6 | 29.5 | 26.6 | 29.9 | 29.8 | 26.3 | |
| | nvjet_sm103_tst_16x64_64x16_4x1_v_bz_TNN | 28.1 | 28.3 | 28.2 | 28.6 | 28.0 | 27.8 | 28.5 | 28.2 | |
| | nvjet_sm103_tst_128x8_64x12_2x1_v_bz_splitK_TNT | 27.4 | 27.2 | 27.2 | 27.1 | 27.1 | 27.2 | 27.2 | 27.2 | |
| | nvjet_sm103_tst_64x16_64x16_4x1_v_bz_TNT | 22.3 | 22.4 | 22.6 | 22.5 | 22.3 | 22.0 | 22.3 | 22.5 | |
| | nvjet_sm103_tst_384x8_64x4_2x1_v_bz_TNT | 22.2 | 22.4 | 22.4 | 22.5 | 22.5 | 22.1 | 22.3 | 22.3 | |
| | void flashinfer::trtllm_mnnvl_allreduce::rmsNormLamport<__nv_bfloat16,  | 20.5 | 18.9 | 19.1 | 19.0 | 19.7 | 17.7 | 18.8 | 19.3 | |
| | nvjet_sm103_tst_64x8_64x16_1x2_h_bz_splitK_TNT | 19.0 | 18.8 | 19.0 | 19.2 | 19.4 | 18.8 | 19.1 | 19.0 | |
| | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSe... | 0.6 | 16.6 | 18.0 | 16.1 | 18.4 | 15.7 | 16.6 | 16.4 | |
| | void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocas... | 17.9 | 17.2 | 17.3 | 17.4 | 17.6 | 16.8 | 17.3 | 17.6 | |
| | nvjet_sm103_tst_192x16_64x8_2x1_2cta_v_bz_splitK_TNT | 16.2 | 16.2 | 16.1 | 16.1 | 16.4 | 16.1 | 16.2 | 16.2 | |
| | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocas... | 10.9 | 11.3 | 10.1 | 12.2 | 11.3 | 11.7 | 12.1 | 10.0 | |
| | void at::native:: | 10.8 | 9.8 | 9.4 | 10.1 | 9.6 | 9.6 | 10.0 | 9.7 | |
| | _compute_local_logits_stats_kernel | 9.2 | 8.9 | 9.0 | 8.7 | 9.3 | 5.5 | 5.9 | 8.9 | |
| | nvjet_sm103_tst_8x64_64x16_4x1_v_bz_TNN | 6.6 | 6.4 | 6.7 | 6.3 | 6.4 | 6.1 | 6.5 | 6.7 | |
| | void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kerne... | 5.8 | 5.7 | 5.7 | 6.4 | 5.9 | 6.2 | 6.3 | 5.7 | |
| | nvjet_sm103_tst_64x8_64x16_2x2_h_bz_NNT | 5.8 | 5.9 | 5.9 | 6.0 | 6.0 | 5.9 | 6.0 | 6.0 | |
| | _prepare_dflash_inputs_kernel | 5.2 | 5.0 | 5.1 | 5.4 | 5.1 | 5.2 | 5.4 | 5.0 | |
| | void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda... | 4.8 | 4.6 | 4.8 | 5.2 | 4.8 | 5.0 | 5.3 | 4.8 | |
| | void vllm::concat_and_cache_mla_kernel<__nv_bfloat16, unsigned char,  | 3.4 | 3.3 | 3.3 | 3.7 | 3.3 | 3.5 | 3.8 | 3.3 | |
| | void vllm::rms_norm_kernel<c10::BFloat16, 8, 2, true> | 3.1 | 3.1 | 3.1 | 3.3 | 3.2 | 3.1 | 3.3 | 3.1 | |
| | void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add... | 3.0 | 3.0 | 3.1 | 3.1 | 3.1 | 3.0 | 3.1 | 3.0 | |
| | _rejection_kernel | 2.2 | 2.1 | 2.1 | 2.4 | 2.2 | 2.2 | 2.3 | 2.1 | |
| | _get_aligned_state_indices_kernel | 1.9 | 1.9 | 1.9 | 2.2 | 2.0 | 2.1 | 2.1 | 1.9 | |
| | _stage_spec_decode_metadata_kernel | 1.9 | 1.9 | 1.9 | 2.2 | 2.0 | 2.1 | 2.1 | 1.9 | |
| | _post_update_kernel | 1.7 | 1.7 | 1.7 | 2.1 | 1.8 | 2.1 | 2.1 | 1.7 | |
| | _gather_block_tables_kernel | 1.7 | 1.8 | 1.7 | 1.9 | 1.7 | 2.1 | 1.8 | 1.7 | |
| | _compute_slot_mappings_kernel | 1.5 | 1.5 | 1.5 | 1.5 | 1.5 | 1.5 | 1.6 | 1.5 | |
| | void at::native::vectorized_gather_kernel<16, long> | 1.4 | 1.4 | 1.3 | 1.5 | 1.4 | 1.4 | 1.4 | 1.4 | |
| | _dcp_local_seq_lens_kernel | 1.1 | 1.1 | 1.1 | 1.1 | 1.2 | 1.1 | 1.1 | 1.1 | |
| | postprocess_mamba_fused_kernel | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.9 | 1.0 | 1.0 | |
| | precopy_mamba_align_fused_kernel | 1.0 | 1.0 | 1.0 | 0.9 | 1.0 | 0.9 | 1.0 | 0.9 | |
| | void vllm::rms_norm_kernel<c10::BFloat16, 8, 3, true> | 0.9 | 0.9 | 1.0 | 1.0 | 0.9 | 0.9 | 1.0 | 1.0 | |
| | preprocess_mamba_align_fused_kernel | 0.8 | 0.7 | 0.7 | 0.8 | 0.8 | 0.7 | 0.7 | 0.7 | |
| | void vllm::rotary_embedding_kernel<c10::BFloat16, float, false> | 0.7 | 0.7 | 0.7 | 0.7 | 0.7 | 0.7 | 0.7 | 0.8 | |
| | _prepare_pos_seq_lens_kernel | 0.6 | 0.5 | 0.5 | 0.7 | 0.6 | 0.6 | 0.7 | 0.5 | |
| | _expand_idx_mapping_kernel | 0.5 | 0.5 | 0.4 | 0.6 | 0.5 | 0.6 | 0.6 | 0.4 | |
| | _scatter_num_accepted_kernel | 0.5 | 0.5 | 0.4 | 0.5 | 0.5 | 0.5 | 0.5 | 0.4 | |
| **GEMM** | *(total)* | 1267.7 | 1277.4 | 1277.6 | 1266.9 | 1261.8 | 1264.9 | 1271.0 | 1277.3 | 1.2% |
| | CUTLASS MXFP4 GEMM | 1152.8 | 1162.8 | 1162.8 | 1158.1 | 1154.3 | 1157.2 | 1155.5 | 1162.1 | |
| | cuBLASLt splitK | 114.9 | 114.6 | 114.8 | 108.8 | 107.6 | 107.7 | 115.4 | 115.2 | |
| **MoE Routing** | *(total)* | 641.7 | 646.6 | 642.8 | 647.8 | 645.4 | 644.3 | 648.0 | 644.3 | 1.0% |
| | MoE Routing (NV) | 501.6 | 506.4 | 503.2 | 506.2 | 504.8 | 503.8 | 506.0 | 505.0 | |
| | MoE Finalize (NV) | 140.0 | 140.1 | 139.6 | 141.6 | 140.7 | 140.6 | 142.0 | 139.4 | |
| **KDA Linear Attn** | *(total)* | 361.7 | 360.5 | 360.2 | 362.9 | 361.2 | 357.7 | 361.4 | 360.0 | 1.4% |
| | KDA gated-delta | 226.9 | 226.0 | 226.5 | 227.7 | 226.7 | 224.6 | 227.0 | 226.2 | |
| | KDA conv1d | 134.8 | 134.5 | 133.7 | 135.2 | 134.5 | 133.2 | 134.4 | 133.7 | |
| **Normalization** | *(total)* | 280.7 | 272.2 | 271.4 | 266.2 | 276.0 | 261.6 | 267.4 | 270.7 | 7.1% |
| | RMSNorm | 231.6 | 223.6 | 222.2 | 216.6 | 227.0 | 215.0 | 218.2 | 221.6 | |
| | LayerNorm | 49.2 | 48.6 | 49.2 | 49.6 | 49.0 | 46.6 | 49.1 | 49.1 | |
| **Attention** | *(total)* | 238.8 | 239.8 | 240.3 | 238.8 | 239.2 | 235.9 | 237.5 | 240.0 | 1.8% |
| | TokenSpeed MLA (NV) | 200.6 | 201.7 | 202.3 | 200.4 | 201.2 | 198.4 | 198.8 | 201.8 | |
| | MLA attn residual | 38.2 | 38.2 | 38.0 | 38.4 | 38.1 | 37.5 | 38.7 | 38.2 | |
| **Sampling** | *(total)* | 26.2 | 24.9 | 27.1 | 26.6 | 26.2 | 22.4 | 23.6 | 26.1 | 18.5% :warning: |
| | Sampling | 18.0 | 17.6 | 17.7 | 18.0 | 17.9 | 14.2 | 14.6 | 17.5 | |
| | ArgMax/Reduce | 8.2 | 7.3 | 9.4 | 8.6 | 8.3 | 8.2 | 9.0 | 8.6 | |
| **Memory** | *(total)* | 20.7 | 21.0 | 20.7 | 22.8 | 21.0 | 21.9 | 22.7 | 20.8 | 9.8% |
| | Fill | 11.8 | 12.1 | 11.6 | 13.9 | 12.1 | 13.0 | 13.7 | 11.7 | |
| | Memcpy | 8.8 | 8.9 | 9.1 | 8.9 | 8.8 | 8.9 | 9.0 | 9.1 | |
| **Communication** | *(total)* | 18.8 | 18.8 | 18.7 | 18.8 | 18.8 | 18.7 | 18.8 | 18.8 | 0.9% |
| | NCCL/RCCL | 18.8 | 18.8 | 18.7 | 18.8 | 18.8 | 18.7 | 18.8 | 18.8 | |
| **Triton Fused** | *(total)* | 14.3 | 13.1 | 13.3 | 15.0 | 13.7 | 13.4 | 14.7 | 14.2 | 13.9% :warning: |
| | Triton fused op | 14.3 | 13.1 | 13.3 | 15.0 | 13.7 | 13.4 | 14.7 | 14.2 | |
| **Activation** | *(total)* | 3.4 | 3.3 | 3.3 | 3.4 | 3.5 | 3.2 | 3.3 | 3.3 | 7.4% |
| | SiLU/SwiGLU | 3.4 | 3.3 | 3.3 | 3.4 | 3.5 | 3.2 | 3.3 | 3.3 | |
| **TOTAL** | | 6631.9 | 6668.2 | 6623.8 | 6665.9 | 6631.0 | 6602.2 | 6661.6 | 6606.8 | 1.0% |

> All ranks balanced (max 0.5% above mean)
>
> **Comm/Total ratio** (bottleneck rank): 0.3%

### mi355x_isl100k vs b300_isl100k Comparison (bottleneck rank)

| Category | Sub-kernel | mi355x_isl100k max (us) | b300_isl100k max (us) | mi355x_isl100k % | b300_isl100k % | Ratio | Scaling |
|---|---|---:|---:|---:|---:|---:|---:|
| **GEMM** | *(total)* | 10420.4 | 1277.6 | 38.8% | 19.1% | 0.12x | sub-linear |
| | hipBLASLt (Cijk) | 3922.7 | 0.0 | 14.6% | 0.0% | 0.00x | |
| | AITER hgemm bf16 | 3757.3 | 0.0 | 14.0% | 0.0% | 0.00x | |
| | AITER flydsl MoE | 2396.6 | 0.0 | 8.9% | 0.0% | 0.00x | |
| | AITER wfp4 batched | 243.8 | 0.0 | 0.9% | 0.0% | 0.00x | |
| | rocBLAS splitK | 150.4 | 0.0 | 0.6% | 0.0% | 0.00x | |
| | CUTLASS MXFP4 GEMM | 0.0 | 1162.8 | 0.0% | 17.4% | new | |
| | cuBLASLt splitK | 0.0 | 115.4 | 0.0% | 1.7% | new | |
| **Other** | *(total)* | 6167.4 | 3796.6 | 22.9% | 56.7% | 0.62x | sub-linear |
| | gemm2_a4w4_port_hmax8192_imax8192_bm32_bn128_bk128_atomic_a8_g2ks2_bhoist_apf... | 1404.5 | 0.0 | 5.2% | 0.0% | 0.00x | |
| | _ZN5aiter45mla_a8w8_qh32_qseqlen4_gqaratio32_lse_cprr_psE.kd | 1119.8 | 0.0 | 4.2% | 0.0% | 0.00x | |
| | void kn_mla_reduce_v1<MlaReduceKernelV1Traits<512, 128, 1>, float, std::bfloa... | 439.7 | 0.0 | 1.6% | 0.0% | 0.00x | |
| | void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_ker... | 399.9 | 0.0 | 1.5% | 0.0% | 0.00x | |
| | void vllm::situ_and_mul_kernel<c10::BFloat16> | 399.1 | 73.9 | 1.5% | 1.1% | 0.19x | |
| | mscclKernel_Sum_hip_bfloat16_Simple_false | 366.3 | 0.0 | 1.4% | 0.0% | 0.00x | |
| | __amd_rocclr_copyBuffer.kd | 281.1 | 0.0 | 1.0% | 0.0% | 0.00x | |
| | void kn_get_mla_metadata_v1_2<MlaMetadataV12Traits<128, true, 0, true, false> > | 219.1 | 0.0 | 0.8% | 0.0% | 0.00x | |
| | _dcp_a2a_pack_send_kernel.kd | 179.5 | 0.0 | 0.7% | 0.0% | 0.00x | |
| | void gemm_a16w16_flatmm_splitk_kernel<opus_flatmm_splitk_traits_gfx950<256, o... | 136.3 | 0.0 | 0.5% | 0.0% | 0.00x | |
| | _dcp_a2a_unpack_combine_kernel.kd | 135.4 | 0.0 | 0.5% | 0.0% | 0.00x | |
| | void vllm::concat_and_cache_mla_kernel<__hip_bfloat16, unsigned char,  | 127.5 | 0.0 | 0.5% | 0.0% | 0.00x | |
| | _ZN5aiter37bf16gemm_fp32bf16_tn_32x64_pf3_splitkE.kd | 107.3 | 0.0 | 0.4% | 0.0% | 0.00x | |
| | void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_... | 104.2 | 0.0 | 0.4% | 0.0% | 0.00x | |
| | void at::native::vectorized_elementwise_kernel<8, at::native::BinaryFunctor<c... | 103.5 | 0.0 | 0.4% | 0.0% | 0.00x | |
| | void kn_mla_reduce_v1<MlaReduceKernelV1Traits<512, 64, 4>, float, std::bfloat... | 77.2 | 0.0 | 0.3% | 0.0% | 0.00x | |
| | mscclKernel_Sum_hip_bfloat16_LL_false | 73.6 | 0.0 | 0.3% | 0.0% | 0.00x | |
| | void at::native:: | 58.6 | 10.8 | 0.2% | 0.2% | 0.18x | |
| | void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add... | 53.3 | 3.1 | 0.2% | 0.0% | 0.06x | |
| | void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda... | 43.1 | 0.0 | 0.2% | 0.0% | 0.00x | |
| | void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kerne... | 41.2 | 6.4 | 0.2% | 0.1% | 0.16x | |
| | _prepare_dflash_inputs_kernel.kd | 38.4 | 0.0 | 0.1% | 0.0% | 0.00x | |
| | void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_ker... | 29.3 | 0.0 | 0.1% | 0.0% | 0.00x | |
| | void splitk_reduce_kernel<16, 64, std::bfloat16_t, false, std::bfloat16_t, true> | 29.3 | 0.0 | 0.1% | 0.0% | 0.00x | |
| | void  | 25.6 | 145.6 | 0.1% | 2.2% | 5.69x | |
| | void rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel<rocprim::ROCPRIM_4... | 25.2 | 0.0 | 0.1% | 0.0% | 0.00x | |
| | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSe... | 24.9 | 18.4 | 0.1% | 0.3% | 0.74x | |
| | kernel.kd | 21.9 | 0.0 | 0.1% | 0.0% | 0.00x | |
| | void vllm::kimi_k3_fused_ops::fusedKimiK3MLADecodeQConcatKVCacheKernel<c10::B... | 20.4 | 35.1 | 0.1% | 0.5% | 1.72x | |
| | void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctorOnSelf_ad... | 13.5 | 0.0 | 0.1% | 0.0% | 0.00x | |
| | _expand_page_indices_kernel.kd | 12.8 | 0.0 | 0.0% | 0.0% | 0.00x | |
| | void at::native::vectorized_elementwise_kernel<4, at::native:: | 12.3 | 0.0 | 0.0% | 0.0% | 0.00x | |
| | void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<i... | 12.2 | 0.0 | 0.0% | 0.0% | 0.00x | |
| | _rejection_kernel.kd | 9.6 | 0.0 | 0.0% | 0.0% | 0.00x | |
| | void at::native::vectorized_gather_kernel<16, long> | 9.4 | 1.5 | 0.0% | 0.0% | 0.16x | |
| | _dcp_local_seq_lens_kernel.kd | 9.0 | 0.0 | 0.0% | 0.0% | 0.00x | |
| | __amd_rocclr_fillBufferAligned.kd | 8.9 | 0.0 | 0.0% | 0.0% | 0.00x | |
| | void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctorOnSe... | 8.6 | 0.0 | 0.0% | 0.0% | 0.00x | |
| | _bias_kernel.kd | 7.9 | 0.0 | 0.0% | 0.0% | 0.00x | |
| | _compute_local_logits_stats_kernel.kd | 7.0 | 0.0 | 0.0% | 0.0% | 0.00x | |
| | void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<int>... | 6.4 | 0.0 | 0.0% | 0.0% | 0.00x | |
| | _compute_slot_mappings_kernel.kd | 5.9 | 0.0 | 0.0% | 0.0% | 0.00x | |
| | _scatter_num_accepted_kernel.kd | 5.7 | 0.0 | 0.0% | 0.0% | 0.00x | |
| | _post_update_kernel.kd | 5.5 | 0.0 | 0.0% | 0.0% | 0.00x | |
| | void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat... | 5.1 | 0.0 | 0.0% | 0.0% | 0.00x | |
| | _gather_block_tables_kernel.kd | 4.5 | 0.0 | 0.0% | 0.0% | 0.00x | |
| | _prepare_pos_seq_lens_kernel.kd | 4.5 | 0.0 | 0.0% | 0.0% | 0.00x | |
| | preprocess_mamba_align_fused_kernel.kd | 4.5 | 0.0 | 0.0% | 0.0% | 0.00x | |
| | postprocess_mamba_fused_kernel.kd | 4.4 | 0.0 | 0.0% | 0.0% | 0.00x | |
| | void vllm::rms_norm_kernel<c10::BFloat16, 8, 3, true> | 4.4 | 1.0 | 0.0% | 0.0% | 0.22x | |
| | void vllm::rotary_embedding_kernel<c10::BFloat16, float, false> | 4.4 | 0.8 | 0.0% | 0.0% | 0.17x | |
| | precopy_mamba_align_fused_kernel.kd | 4.2 | 0.0 | 0.0% | 0.0% | 0.00x | |
| | _expand_idx_mapping_kernel.kd | 2.7 | 0.0 | 0.0% | 0.0% | 0.00x | |
| | (anonymous namespace)::signal_kernel(long const*, long const*, long, long) | 0.0 | 29.9 | 0.0% | 0.4% | new | |
| | _compute_local_logits_stats_kernel | 0.0 | 9.3 | 0.0% | 0.1% | new | |
| | void flashinfer::trtllm_mnnvl_allreduce::twoshotAllreduceKernel< | 0.0 | 74.0 | 0.0% | 1.1% | new | |
| | _compute_slot_mappings_kernel | 0.0 | 1.6 | 0.0% | 0.0% | new | |
| | nvjet_sm103_tst_64x8_64x16_2x2_h_bz_NNT | 0.0 | 6.0 | 0.0% | 0.1% | new | |
| | _stage_spec_decode_metadata_kernel | 0.0 | 2.2 | 0.0% | 0.0% | new | |
| | kernel_cutlass_kernel_vllmmodelskimi_k3nvidiaopscute_dsllatent_moe_tailfused_... | 0.0 | 217.4 | 0.0% | 3.2% | new | |
| | kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_ll_bf16_splitkL... | 0.0 | 236.6 | 0.0% | 3.5% | new | |
| | _dcp_local_seq_lens_kernel | 0.0 | 1.2 | 0.0% | 0.0% | new | |
| | nvjet_sm103_tst_64x8_64x16_2x1_v_bz_splitK_TNT | 0.0 | 453.4 | 0.0% | 6.8% | new | |
| | void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda... | 0.0 | 5.3 | 0.0% | 0.1% | new | |
| | kernel_cutlass_kernel_vllmmodelskimi_k3nvidiaopscute_dsllatent_moe_taillampor... | 0.0 | 68.6 | 0.0% | 1.0% | new | |
| | nvjet_sm103_tst_192x16_64x8_2x1_2cta_v_bz_splitK_TNT | 0.0 | 16.4 | 0.0% | 0.2% | new | |
| | void sm100::fwd_prod_v2::attn_res_fwd_online_v2_kernel<7168, 8, 3, 1, false, ... | 0.0 | 45.8 | 0.0% | 0.7% | new | |
| | void fused_a_gemm_kernel<1, 2304, 1536, 16, 8, 256, 6> | 0.0 | 44.2 | 0.0% | 0.7% | new | |
| | _gather_block_tables_kernel | 0.0 | 2.1 | 0.0% | 0.0% | new | |
| | vllm::direct_dcp::increment_epoch_kernel(long*) | 0.0 | 30.1 | 0.0% | 0.4% | new | |
| | void fused_a_gemm_kernel<1, 7168, 768, 16, 8, 256, 3> | 0.0 | 160.7 | 0.0% | 2.4% | new | |
| | void sm100::fwd_prod_v2::attn_res_fwd_online_v2_kernel<7168, 2, 3, 1, false, ... | 0.0 | 33.6 | 0.0% | 0.5% | new | |
| | nvjet_sm103_tst_128x8_64x12_2x1_v_bz_splitK_TNT | 0.0 | 27.4 | 0.0% | 0.4% | new | |
| | void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_12... | 0.0 | 72.3 | 0.0% | 1.1% | new | |
| | nvjet_sm103_tst_64x8_64x16_1x2_h_bz_splitK_TNT | 0.0 | 19.4 | 0.0% | 0.3% | new | |
| | nvjet_sm103_tst_64x16_64x16_4x1_v_bz_TNT | 0.0 | 22.6 | 0.0% | 0.3% | new | |
| | void sm100::fwd_prod_v2::attn_res_fwd_online_v2_kernel<7168, 5, 3, 1, false, ... | 0.0 | 39.4 | 0.0% | 0.6% | new | |
| | nvjet_sm103_tst_64x8_64x16_4x1_v_bz_TNT | 0.0 | 211.4 | 0.0% | 3.2% | new | |
| | void fused_a_gemm_kernel<1, 2112, 7168, 16, 8, 256, 16> | 0.0 | 69.8 | 0.0% | 1.0% | new | |
| | nvjet_sm103_tst_64x8_64x16_4x1_v_bz_NNT | 0.0 | 38.6 | 0.0% | 0.6% | new | |
| | void fused_a_gemm_kernel<1, 1536, 7168, 16, 8, 256, 16> | 0.0 | 312.4 | 0.0% | 4.7% | new | |
| | nvjet_sm103_tst_384x8_64x4_2x1_v_bz_TNT | 0.0 | 22.5 | 0.0% | 0.3% | new | |
| | void flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel< | 0.0 | 243.9 | 0.0% | 3.6% | new | |
| | postprocess_mamba_fused_kernel | 0.0 | 1.0 | 0.0% | 0.0% | new | |
| | _get_aligned_state_indices_kernel | 0.0 | 2.2 | 0.0% | 0.0% | new | |
| | void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocas... | 0.0 | 17.9 | 0.0% | 0.3% | new | |
| | nvjet_sm103_tst_8x64_64x16_4x1_v_bz_TNN | 0.0 | 6.7 | 0.0% | 0.1% | new | |
| | void flashinfer::trtllm_mnnvl_allreduce::rmsNormLamport<__nv_bfloat16,  | 0.0 | 20.5 | 0.0% | 0.3% | new | |
| | void sm100::fwd_prod_v2::attn_res_fwd_online_v2_kernel<7168, 3, 3, 1, false, ... | 0.0 | 35.3 | 0.0% | 0.5% | new | |
| | void sm100::fwd_prod_v2::attn_res_fwd_online_v2_kernel<7168, 4, 3, 1, false, ... | 0.0 | 38.0 | 0.0% | 0.6% | new | |
| | _prepare_dflash_inputs_kernel | 0.0 | 5.4 | 0.0% | 0.1% | new | |
| | nvjet_sm103_tst_16x64_64x16_4x1_v_bz_TNN | 0.0 | 28.6 | 0.0% | 0.4% | new | |
| | void vllm::concat_and_cache_mla_kernel<__nv_bfloat16, unsigned char,  | 0.0 | 3.8 | 0.0% | 0.1% | new | |
| | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocas... | 0.0 | 12.2 | 0.0% | 0.2% | new | |
| | _rejection_kernel | 0.0 | 2.4 | 0.0% | 0.0% | new | |
| | void sm100::fwd_prod_v2::attn_res_fwd_online_v2_kernel<7168, 9, 3, 1, false, ... | 0.0 | 34.6 | 0.0% | 0.5% | new | |
| | void vllm::rms_norm_kernel<c10::BFloat16, 8, 2, true> | 0.0 | 3.3 | 0.0% | 0.0% | new | |
| | void fused_a_gemm_kernel<1, 3584, 7168, 16, 8, 256, 16> | 0.0 | 508.8 | 0.0% | 7.6% | new | |
| | _post_update_kernel | 0.0 | 2.1 | 0.0% | 0.0% | new | |
| | (anonymous namespace)::direct_dcp_q_gather_multimem_kernel(uint4 const*, uint... | 0.0 | 119.6 | 0.0% | 1.8% | new | |
| | void sm100::fwd_prod_v2::attn_res_fwd_online_v2_kernel<7168, 7, 3, 1, false, ... | 0.0 | 44.1 | 0.0% | 0.7% | new | |
| | nvjet_sm103_tst_64x8_64x16_4x2_h_bz_TNT | 0.0 | 38.2 | 0.0% | 0.6% | new | |
| | kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8Quantiz... | 0.0 | 62.0 | 0.0% | 0.9% | new | |
| | void sm100::fwd_prod_v2::attn_res_fwd_online_v2_kernel<7168, 6, 3, 1, false, ... | 0.0 | 40.5 | 0.0% | 0.6% | new | |
| **Communication** | *(total)* | 3104.4 | 18.8 | 11.5% | 0.3% | 0.01x | ← better |
| | CustomAR 2-stage | 1668.2 | 0.0 | 6.2% | 0.0% | 0.00x | |
| | CustomAR 1-stage | 881.3 | 0.0 | 3.3% | 0.0% | 0.00x | |
| | NCCL/RCCL | 565.5 | 18.8 | 2.1% | 0.3% | 0.03x | |
| **MoE Routing** | *(total)* | 2697.7 | 648.0 | 10.0% | 9.7% | 0.24x | sub-linear |
| | MoE sort mxfp4 | 1616.6 | 0.0 | 6.0% | 0.0% | 0.00x | |
| | AITER grouped_topk | 1084.4 | 0.0 | 4.0% | 0.0% | 0.00x | |
| | MoE Finalize (NV) | 0.0 | 142.0 | 0.0% | 2.1% | new | |
| | MoE Routing (NV) | 0.0 | 506.4 | 0.0% | 7.6% | new | |
| **KDA Linear Attn** | *(total)* | 1769.1 | 362.9 | 6.6% | 5.4% | 0.21x | sub-linear |
| | KDA gated-delta | 1373.9 | 227.7 | 5.1% | 3.4% | 0.17x | |
| | KDA conv1d | 396.5 | 135.2 | 1.5% | 2.0% | 0.34x | |
| **Attention** | *(total)* | 1528.2 | 240.3 | 5.7% | 3.6% | 0.16x | sub-linear |
| | MLA attn residual | 1528.2 | 38.7 | 5.7% | 0.6% | 0.03x | |
| | TokenSpeed MLA (NV) | 0.0 | 202.3 | 0.0% | 3.0% | new | |
| **Normalization** | *(total)* | 888.2 | 280.7 | 3.3% | 4.2% | 0.32x | sub-linear |
| | Add+RMSNorm+quant | 482.5 | 0.0 | 1.8% | 0.0% | 0.00x | |
| | LayerNorm | 304.1 | 49.6 | 1.1% | 0.7% | 0.16x | |
| | RMSNorm | 104.5 | 231.6 | 0.4% | 3.5% | 2.22x | |
| **Triton Fused** | *(total)* | 151.4 | 15.0 | 0.6% | 0.2% | 0.10x | sub-linear |
| | Triton fused op | 151.4 | 15.0 | 0.6% | 0.2% | 0.10x | |
| **Sampling** | *(total)* | 83.6 | 27.1 | 0.3% | 0.4% | 0.32x | sub-linear |
| | Sampling | 46.8 | 18.0 | 0.2% | 0.3% | 0.39x | |
| | ArgMax/Reduce | 37.3 | 9.4 | 0.1% | 0.1% | 0.25x | |
| **Activation** | *(total)* | 38.3 | 3.5 | 0.1% | 0.1% | 0.09x | sub-linear |
| | SiLU/SwiGLU | 38.3 | 3.5 | 0.1% | 0.1% | 0.09x | |
| **Memory** | *(total)* | 37.0 | 22.8 | 0.1% | 0.3% | 0.62x | sub-linear |
| | Fill | 37.0 | 13.9 | 0.1% | 0.2% | 0.38x | |
| | Memcpy | 0.0 | 9.1 | 0.0% | 0.1% | new | |
| **TOTAL** | | 26885.7 | 6693.4 | 100% | 100% | 0.25x | |

> mi355x_isl100k comm/total: **11.6%**
> b300_isl100k comm/total: **0.3%**

## Summary

1. **mi355x_isl100k PREFILL**: Activation has 72% rank imbalance (straggler: R3)
2. **mi355x_isl100k PREFILL**: Triton Fused has 45% rank imbalance (straggler: R5)
3. **PREFILL**: Comm time ratio (b300_isl100k/mi355x_isl100k): 0.00x
4. **mi355x_isl100k DECODE**: Activation has 71% rank imbalance (straggler: R4)
5. **mi355x_isl100k DECODE**: Triton Fused has 47% rank imbalance (straggler: R2)
6. **b300_isl100k DECODE**: Sampling has 19% rank imbalance (straggler: R2)
7. **b300_isl100k DECODE**: Triton Fused has 14% rank imbalance (straggler: R3)
8. **DECODE**: Comm time ratio (b300_isl100k/mi355x_isl100k): 0.01x

---

## Component Totals and Per-Kernel Detail (KDA-clock normalised)

> **Why the tables above this section are wrong.** They carry TWO bugs that this
> section does not.
>
> 1. **Normalisation.** They divide by the profiler's self-reported iteration count
>    (66 for MI355X, **492** for B300). That is right for MI355X but wrong for B300
>    by 2.97x -- B300 emits 3 `execute_context` annotations per real decode step.
>    This section instead uses the KDA clock: `fused_recurrent_kda` fires exactly
>    once per KDA layer per step, so step count = its call count / 69. That gives
>    **MI355X 66 steps, B300 164 steps**. Only the decode
>    `fused_recurrent_kda` may be counted -- the `chunk_*` KDA kernels are the prefill
>    path and inflate the divisor.
> 2. **Prefill leak (fixed at source 2026-09-12).** `_assign_stage_from_index` charged
>    a kernel landing in a GAP between step annotations to the NEXT step. GPU work is
>    async, so such a kernel is the PREVIOUS step's tail; at the PREFILL->DECODE
>    boundary the old rule donated prefill's tail to decode, inflating the MI355X
>    decode total by 14% and dragging 207 prefill-only `chunk_*` calls in. B300 has no
>    PREFILL steps, so the bias ran one way only.
>
> Net effect: the old tables report the total ratio as 0.21x ("we are 4.7x slower").
> **The correct ratio is 0.75x -- we are 1.34x slower.** Regenerate them with the fixed
> `trace_compare_k3.py` to make them agree.

DECODE, rank0, ISL-matched (MI355X 99,845 / B300 ~99,757). GPU kernels only.

- **MI355X total 26,642 us/step** across 2,909 calls, 113 distinct kernels
- **B300 total 19,881 us/step** across 2,750 calls, 105 distinct kernels
- **Gap 6,761 us/step (1.34x)**

| Component | MI355X us/step | calls | B300 us/step | calls | Delta | Ratio | % MI355X step | % B300 step | % of gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **MLA attention** | 3,754 | 330 | 1,942 | 327 | +1,812 | 1.93x | 14.1% | 9.8% | 27% |
| **Glue/elementwise/misc** | 1,910 | 445 | 579 | 246 | +1,331 | 3.30x | 7.2% | 2.9% | 20% |
| **Communication** | 3,818 | 410 | 2,727 | 478 | +1,091 | 1.40x | 14.3% | 13.7% | 16% |
| **KDA** | 1,753 | 138 | 1,085 | 138 | +668 | 1.62x | 6.6% | 5.5% | 10% |
| **Norm/quant** | 863 | 199 | 263 | 102 | +600 | 3.28x | 3.2% | 1.3% | 9% |
| **MoE routing/sort** | 2,679 | 368 | 2,108 | 368 | +571 | 1.27x | 10.1% | 10.6% | 8% |
| **Dense GEMM** | 8,040 | 817 | 7,640 | 889 | +400 | 1.05x | 30.2% | 38.4% | 6% |
| **MoE expert GEMM** | 3,743 | 184 | 3,458 | 184 | +284 | 1.08x | 14.0% | 17.4% | 4% |
| **Sampling** | 82 | 18 | 79 | 18 | +4 | 1.05x | 0.3% | 0.4% | 0% |
| **TOTAL** | **26,642** | 2,909 | **19,881** | 2,750 | **+6,761** | **1.34x** | 100% | 100% | 100% |

### MLA attention — MI355X 3,754 us/step vs B300 1,942 us/step (1.93x, +1,812)

| Platform | Kernel function | us/step | calls/step | us/call | % of component |
|---|---|---:|---:|---:|---:|
| MI355X | `_attn_res_kernel.kd` | 1,517.9 | 187.0 | 8.12 | 40.4% |
| MI355X | `_ZN5aiter45mla_a8w8_qh32_qseqlen4_gqaratio32_lse_cprr_psE.kd` | 1,119.8 | 29.0 | 38.61 | 29.8% |
| MI355X | `void kn_mla_reduce_v1<MlaReduceKernelV1Traits<512, 128, 1>, float, std::bfloat16_t>(MlaReduceKernelV1Params...` | 439.7 | 24.0 | 18.32 | 11.7% |
| MI355X | `void kn_get_mla_metadata_v1_2<MlaMetadataV12Traits<128, true, 0, true, false> >(MlaMetadataV1KernelParamete...` | 216.8 | 3.0 | 72.27 | 5.8% |
| MI355X | `_batched_gemm_a16wfp4_kernel_BLOCK_SIZE_M_16_BLOCK_SIZE_N_64_BLOCK_SIZE_K_256_GROUP_SIZE_M_1_NUM_KSPLIT_1_S...` | 128.4 | 24.0 | 5.35 | 3.4% |
| MI355X | `void vllm::concat_and_cache_mla_kernel<__hip_bfloat16, unsigned char, (vllm::Fp8KVCacheDataType)1>(__hip_bf...` | 123.4 | 29.0 | 4.25 | 3.3% |
| MI355X | `_batched_gemm_a16wfp4_kernel_BLOCK_SIZE_M_16_BLOCK_SIZE_N_64_BLOCK_SIZE_K_128_GROUP_SIZE_M_1_NUM_KSPLIT_1_S...` | 111.0 | 24.0 | 4.63 | 3.0% |
| MI355X | `void kn_mla_reduce_v1<MlaReduceKernelV1Traits<512, 64, 4>, float, std::bfloat16_t>(MlaReduceKernelV1Params,...` | 77.2 | 5.0 | 15.44 | 2.1% |
| MI355X | `void vllm::kimi_k3_fused_ops::fusedKimiK3MLADecodeQConcatKVCacheKernel<c10::BFloat16, true, true, true, 3>(...` | 19.6 | 5.0 | 3.92 | 0.5% |
| B300 | `kernel_cutlass_split_kv_kernel_tokenspeed_mlamla_decode_fp8BlackwellMultiHeadLatentAttentionForwardFP8_obje...` | 493.9 | 29.0 | 17.03 | 25.4% |
| B300 | `void sm100::fwd_prod_v2::attn_res_fwd_online_v2_kernel<7168, 8, 3, 1, false, true, true, true>(__nv_bfloat1...` | 135.5 | 22.0 | 6.16 | 7.0% |
| B300 | `void sm100::fwd_prod_v2::attn_res_fwd_online_v2_kernel<7168, 7, 3, 1, false, true, true, true>(__nv_bfloat1...` | 130.1 | 22.0 | 5.92 | 6.7% |
| B300 | `void sm100::fwd_prod_v2::attn_res_fwd_online_v2_kernel<7168, 6, 3, 1, false, true, true, true>(__nv_bfloat1...` | 118.2 | 22.0 | 5.37 | 6.1% |
| B300 | `void sm100::fwd_prod_v2::attn_res_fwd_online_v2_kernel<7168, 5, 3, 1, false, true, true, true>(__nv_bfloat1...` | 115.6 | 22.0 | 5.26 | 6.0% |
| B300 | `_attn_res_kernel` | 114.7 | 17.0 | 6.74 | 5.9% |
| B300 | `nvjet_sm103_tst_64x8_64x16_4x1_v_bz_NNT` | 112.6 | 24.0 | 4.69 | 5.8% |
| B300 | `void sm100::fwd_prod_v2::attn_res_fwd_online_v2_kernel<7168, 4, 3, 1, false, true, true, true>(__nv_bfloat1...` | 111.2 | 22.0 | 5.05 | 5.7% |
| B300 | `kernel_cutlass_reduction_kernel_tokenspeed_mlamla_decode_fp8BlackwellMultiHeadLatentAttentionForwardFP8_obj...` | 107.9 | 29.0 | 3.72 | 5.6% |
| B300 | `void sm100::fwd_prod_v2::attn_res_fwd_online_v2_kernel<7168, 3, 3, 1, false, true, true, true>(__nv_bfloat1...` | 102.8 | 22.0 | 4.67 | 5.3% |
| B300 | `void sm100::fwd_prod_v2::attn_res_fwd_online_v2_kernel<7168, 9, 3, 1, false, true, true, true>(__nv_bfloat1...` | 102.2 | 16.0 | 6.39 | 5.3% |
| B300 | `void sm100::fwd_prod_v2::attn_res_fwd_online_v2_kernel<7168, 2, 3, 1, false, true, true, true>(__nv_bfloat1...` | 98.8 | 22.0 | 4.49 | 5.1% |
| B300 | `nvjet_sm103_tst_16x64_64x16_4x1_v_bz_TNN` | 84.4 | 24.0 | 3.51 | 4.3% |
| B300 | `void vllm::kimi_k3_fused_ops::fusedKimiK3MLADecodeQConcatKVCacheKernel<c10::BFloat16, true, true, false>(c1...` | 84.2 | 24.0 | 3.51 | 4.3% |
| B300 | `void vllm::kimi_k3_fused_ops::fusedKimiK3MLADecodeQConcatKVCacheKernel<c10::BFloat16, true, true, true>(c10...` | 19.3 | 5.0 | 3.87 | 1.0% |
| B300 | `void vllm::concat_and_cache_mla_kernel<__nv_bfloat16, unsigned char, (vllm::Fp8KVCacheDataType)1>(__nv_bflo...` | 10.1 | 5.0 | 2.02 | 0.5% |

### Glue/elementwise/misc — MI355X 1,910 us/step vs B300 579 us/step (3.30x, +1,331)

| Platform | Kernel function | us/step | calls/step | us/call | % of component |
|---|---|---:|---:|---:|---:|
| MI355X | `void vllm::situ_and_mul_kernel<c10::BFloat16>(c10::BFloat16*, c10::BFloat16 const*, int, float, float) [clo...` | 379.3 | 93.0 | 4.08 | 19.9% |
| MI355X | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::fl...` | 365.4 | 77.0 | 4.74 | 19.1% |
| MI355X | `__amd_rocclr_copyBuffer.kd` | 274.0 | 69.0 | 3.97 | 14.3% |
| MI355X | `void at::native::vectorized_elementwise_kernel<8, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c...` | 100.1 | 24.0 | 4.17 | 5.2% |
| MI355X | `void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::TensorIteratorBase&):...` | 99.8 | 24.0 | 4.16 | 5.2% |
| MI355X | `triton_poi_fused__to_copy_cat_clamp_mul_reciprocal_view_0.kd` | 86.2 | 24.0 | 3.59 | 4.5% |
| MI355X | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<ch...` | 51.6 | 12.0 | 4.30 | 2.7% |
| MI355X | `_prepare_dflash_inputs_kernel.kd` | 38.1 | 1.0 | 38.07 | 2.0% |
| MI355X | `void at::native::(anonymous namespace)::indexSelectSmallIndex<c10::BFloat16, long, unsigned int, 2, 2, -2>(...` | 36.9 | 8.0 | 4.61 | 1.9% |
| MI355X | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul> >(in...` | 32.1 | 7.0 | 4.59 | 1.7% |
| MI355X | `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kerne...` | 29.5 | 7.0 | 4.21 | 1.5% |
| MI355X | `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_imp...` | 28.2 | 9.0 | 3.14 | 1.5% |
| MI355X | `void rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel<rocprim::ROCPRIM_400200_NS::detail::wrapped_scan...` | 24.8 | 6.0 | 4.14 | 1.3% |
| MI355X | `void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_out(c10::Scalar cons...` | 24.6 | 6.0 | 4.10 | 1.3% |
| MI355X | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::di...` | 21.5 | 4.0 | 5.36 | 1.1% |
| MI355X | `kernel.kd` | 21.3 | 5.0 | 4.27 | 1.1% |
| MI355X | `triton_poi_fused_mul_silu_slice_0.kd` | 20.3 | 5.0 | 4.05 | 1.1% |
| MI355X | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int>, std::array<char*,...` | 15.8 | 4.0 | 3.95 | 0.8% |
| MI355X | `void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul...` | 13.0 | 3.0 | 4.32 | 0.7% |
| MI355X | `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kerne...` | 12.6 | 3.0 | 4.21 | 0.7% |
| MI355X | `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_co...` | 12.5 | 3.0 | 4.18 | 0.7% |
| MI355X | `_expand_page_indices_kernel.kd` | 12.5 | 3.0 | 4.16 | 0.7% |
| MI355X | `void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, int, at::native::bina...` | 12.0 | 3.0 | 4.00 | 0.6% |
| MI355X | `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_scalar(at...` | 12.0 | 3.0 | 3.99 | 0.6% |
| MI355X | `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CU...` | 11.9 | 3.0 | 3.95 | 0.6% |
| MI355X | `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::Opaqu...` | 11.7 | 3.0 | 3.89 | 0.6% |
| MI355X | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::(a...` | 10.0 | 2.0 | 4.98 | 0.5% |
| MI355X | `_rejection_kernel.kd` | 9.3 | 1.0 | 9.26 | 0.5% |
| MI355X | `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool)...` | 9.1 | 2.0 | 4.54 | 0.5% |
| MI355X | `__amd_rocclr_fillBufferAligned.kd` | 8.5 | 2.0 | 4.27 | 0.4% |
| MI355X | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*...` | 8.5 | 2.0 | 4.26 | 0.4% |
| MI355X | `void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*...` | 8.3 | 2.0 | 4.13 | 0.4% |
| MI355X | `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_imp...` | 8.2 | 2.0 | 4.09 | 0.4% |
| MI355X | `triton_poi_fused_add_bitwise_and_bitwise_not_bitwise_or_ge_lt_mul_sub_0.kd` | 8.1 | 2.0 | 4.07 | 0.4% |
| MI355X | `_bias_kernel.kd` | 6.9 | 1.0 | 6.94 | 0.4% |
| MI355X | `_compute_local_logits_stats_kernel.kd` | 6.5 | 1.0 | 6.52 | 0.3% |
| MI355X | `void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<int>, std::array<char*, 3ul>, 4, T...` | 6.4 | 3.0 | 2.13 | 0.3% |
| MI355X | `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::Opaqu...` | 6.0 | 1.0 | 5.96 | 0.3% |
| MI355X | `_compute_slot_mappings_kernel.kd` | 5.9 | 1.0 | 5.86 | 0.3% |
| MI355X | `_scatter_num_accepted_kernel.kd` | 5.5 | 1.0 | 5.53 | 0.3% |
| MI355X | `_post_update_kernel.kd` | 5.5 | 1.0 | 5.49 | 0.3% |
| MI355X | `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::Tensor...` | 4.9 | 1.0 | 4.90 | 0.3% |
| MI355X | `_gather_block_tables_kernel.kd` | 4.5 | 1.0 | 4.55 | 0.2% |
| MI355X | `_prepare_pos_seq_lens_kernel.kd` | 4.5 | 1.0 | 4.46 | 0.2% |
| MI355X | `preprocess_mamba_align_fused_kernel.kd` | 4.4 | 1.0 | 4.40 | 0.2% |
| MI355X | `void at::native::vectorized_elementwise_kernel<16, at::native::FillFunctor<bool>, std::array<char*, 1ul> >(...` | 4.2 | 1.0 | 4.21 | 0.2% |
| MI355X | `void vllm::rotary_embedding_kernel<c10::BFloat16, float, false>(long const*, c10::BFloat16*, c10::BFloat16*...` | 4.2 | 1.0 | 4.20 | 0.2% |
| MI355X | `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_put_kernel...` | 4.2 | 1.0 | 4.19 | 0.2% |
| MI355X | `postprocess_mamba_fused_kernel.kd` | 4.1 | 1.0 | 4.11 | 0.2% |
| MI355X | `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_nocast<at::native::di...` | 4.1 | 1.0 | 4.08 | 0.2% |
| MI355X | `void at::native::(anonymous namespace)::indexSelectSmallIndex<c10::BFloat16, int, unsigned int, 2, 2, -2>(a...` | 4.0 | 1.0 | 4.05 | 0.2% |
| MI355X | `precopy_mamba_align_fused_kernel.kd` | 4.0 | 1.0 | 4.04 | 0.2% |
| MI355X | `_expand_idx_mapping_kernel.kd` | 2.6 | 1.0 | 2.61 | 0.1% |
| B300 | `void vllm::situ_and_mul_kernel<c10::BFloat16>(c10::BFloat16*, c10::BFloat16 const*, int, float, float)` | 221.8 | 93.0 | 2.38 | 38.3% |
| B300 | `triton_poi_fused_mul_sigmoid_0` | 39.6 | 24.0 | 1.65 | 6.8% |
| B300 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(...` | 32.9 | 29.0 | 1.14 | 5.7% |
| B300 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<...` | 29.4 | 7.0 | 4.19 | 5.1% |
| B300 | `_compute_local_logits_stats_kernel` | 27.7 | 1.0 | 27.74 | 4.8% |
| B300 | `void at::native::(anonymous namespace)::indexSelectSmallIndex<c10::BFloat16, long, unsigned int, 2, 2, -2>(...` | 27.1 | 8.0 | 3.39 | 4.7% |
| B300 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kern...` | 17.6 | 4.0 | 4.39 | 3.0% |
| B300 | `_prepare_dflash_inputs_kernel` | 15.5 | 1.0 | 15.52 | 2.7% |
| B300 | `void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda_scatter_gather_internal_kerne...` | 14.4 | 7.0 | 2.06 | 2.5% |
| B300 | `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kern...` | 14.1 | 8.0 | 1.76 | 2.4% |
| B300 | `memcpy32_post` | 11.9 | 6.0 | 1.98 | 2.0% |
| B300 | `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &(c10::BFloat16 vllm::silu_kernel<c10::BFloat1...` | 10.3 | 5.0 | 2.07 | 1.8% |
| B300 | `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kern...` | 9.5 | 7.0 | 1.35 | 1.6% |
| B300 | `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_imp...` | 9.3 | 3.0 | 3.10 | 1.6% |
| B300 | `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctorOnSel...` | 9.2 | 7.0 | 1.31 | 1.6% |
| B300 | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<ch...` | 9.0 | 5.0 | 1.80 | 1.6% |
| B300 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous names...` | 6.8 | 2.0 | 3.38 | 1.2% |
| B300 | `_rejection_kernel` | 6.7 | 1.0 | 6.70 | 1.2% |
| B300 | `_stage_spec_decode_metadata_kernel` | 5.8 | 3.0 | 1.93 | 1.0% |
| B300 | `_get_aligned_state_indices_kernel` | 5.7 | 3.0 | 1.91 | 1.0% |
| B300 | `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_imp...` | 5.5 | 2.0 | 2.74 | 0.9% |
| B300 | `_gather_block_tables_kernel` | 5.2 | 1.0 | 5.22 | 0.9% |
| B300 | `_post_update_kernel` | 5.2 | 1.0 | 5.21 | 0.9% |
| B300 | `_compute_slot_mappings_kernel` | 4.6 | 1.0 | 4.61 | 0.8% |
| B300 | `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool)` | 4.2 | 2.0 | 2.09 | 0.7% |
| B300 | `triton_poi_fused_add_bitwise_and_bitwise_not_bitwise_or_ge_lt_mul_sub_0` | 3.2 | 2.0 | 1.62 | 0.6% |
| B300 | `void at::native::(anonymous namespace)::indexSelectSmallIndex<c10::BFloat16, int, unsigned int, 2, 2, -2>(a...` | 3.1 | 1.0 | 3.07 | 0.5% |
| B300 | `postprocess_mamba_fused_kernel` | 3.0 | 1.0 | 2.95 | 0.5% |
| B300 | `precopy_mamba_align_fused_kernel` | 2.9 | 1.0 | 2.95 | 0.5% |
| B300 | `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_put_kernel...` | 2.7 | 1.0 | 2.73 | 0.5% |
| B300 | `preprocess_mamba_align_fused_kernel` | 2.3 | 1.0 | 2.25 | 0.4% |
| B300 | `void at::native::(anonymous namespace)::CatArrayBatchedCopy_vectorized<at::native::(anonymous namespace)::O...` | 2.2 | 1.0 | 2.23 | 0.4% |
| B300 | `void vllm::rotary_embedding_kernel<c10::BFloat16, float, false>(long const*, c10::BFloat16*, c10::BFloat16*...` | 2.0 | 1.0 | 2.03 | 0.4% |
| B300 | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int>, std::array<char*,...` | 1.7 | 1.0 | 1.74 | 0.3% |
| B300 | `_prepare_pos_seq_lens_kernel` | 1.7 | 1.0 | 1.68 | 0.3% |
| B300 | `_expand_idx_mapping_kernel` | 1.4 | 1.0 | 1.43 | 0.2% |
| B300 | `_scatter_num_accepted_kernel` | 1.4 | 1.0 | 1.42 | 0.2% |
| B300 | `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<bool>, std::array<char*, 1ul> >(i...` | 1.3 | 1.0 | 1.29 | 0.2% |
| B300 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul> >(in...` | 1.2 | 1.0 | 1.24 | 0.2% |

### Communication — MI355X 3,818 us/step vs B300 2,727 us/step (1.40x, +1,091)

| Platform | Kernel function | us/step | calls/step | us/call | % of component |
|---|---|---:|---:|---:|---:|
| MI355X | `void aiter::cross_device_reduce_2stage<std::bfloat16_t, 8, false>(aiter::RankData*, aiter::RankData*, aiter...` | 1,668.2 | 198.0 | 8.43 | 43.7% |
| MI355X | `void aiter::cross_device_reduce_1stage<std::bfloat16_t, 8, false>(aiter::RankData*, aiter::RankData*, aiter...` | 880.3 | 92.0 | 9.57 | 23.1% |
| MI355X | `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>) [clone .kd]` | 556.0 | 31.0 | 17.93 | 14.6% |
| MI355X | `mscclKernel_Sum_hip_bfloat16_Simple_false(ncclDevComm*, mscclAlgo*, mscclWork*) [clone .kd]` | 332.4 | 24.0 | 13.85 | 8.7% |
| MI355X | `_dcp_a2a_pack_send_kernel.kd` | 173.9 | 29.0 | 6.00 | 4.6% |
| MI355X | `_dcp_a2a_unpack_combine_kernel.kd` | 129.0 | 29.0 | 4.45 | 3.4% |
| MI355X | `mscclKernel_Sum_hip_bfloat16_LL_false(ncclDevComm*, mscclAlgo*, mscclWork*) [clone .kd]` | 69.6 | 5.0 | 13.91 | 1.8% |
| MI355X | `_dcp_local_seq_lens_kernel.kd` | 9.0 | 2.0 | 4.52 | 0.2% |
| B300 | `void flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel<(unsigned char)8, __nv_bfloat16, fals...` | 723.0 | 95.0 | 7.61 | 26.5% |
| B300 | `kernel_cutlass_kernel_vllmmodelskimi_k3nvidiaopscute_dsllatent_moe_tailallreduce_rmsnorm_reduce_scatter_ear...` | 591.3 | 92.0 | 6.43 | 21.7% |
| B300 | `(anonymous namespace)::direct_dcp_q_gather_multimem_kernel(uint4 const*, uint4*, unsigned int*, unsigned in...` | 356.0 | 29.0 | 12.28 | 13.1% |
| B300 | `void (anonymous namespace)::wait_lse_combine_kernel<__nv_bfloat16>(__nv_bfloat16 const*, float const*, unsi...` | 311.9 | 29.0 | 10.75 | 11.4% |
| B300 | `kernel_cutlass_kernel_vllmmodelskimi_k3nvidiaopscute_dsllatent_moe_taillamport_copyLamportCopy_object_at__t...` | 205.9 | 92.0 | 2.24 | 7.5% |
| B300 | `void (anonymous namespace)::dispatch_output_lse_kernel<float>(uint4 const*, float const*, int const*, int c...` | 124.9 | 29.0 | 4.31 | 4.6% |
| B300 | `vllm::direct_dcp::increment_epoch_kernel(long*)` | 83.3 | 58.0 | 1.44 | 3.1% |
| B300 | `(anonymous namespace)::signal_kernel(long const*, long const*, long, long)` | 82.6 | 29.0 | 2.85 | 3.0% |
| B300 | `void flashinfer::trtllm_mnnvl_allreduce::twoshotAllreduceKernel<(unsigned char)8, __nv_bfloat16, false, flo...` | 69.7 | 10.0 | 6.97 | 2.6% |
| B300 | `void flashinfer::trtllm_mnnvl_allreduce::rmsNormLamport<__nv_bfloat16, (flashinfer::trtllm_mnnvl_allreduce:...` | 61.5 | 10.0 | 6.15 | 2.3% |
| B300 | `void flashinfer::trtllm_mnnvl_allreduce::twoshotAllreduceKernel<(unsigned char)8, __nv_bfloat16, true, floa...` | 57.3 | 1.0 | 57.26 | 2.1% |
| B300 | `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 56.4 | 2.0 | 28.22 | 2.1% |
| B300 | `_dcp_local_seq_lens_kernel` | 3.4 | 2.0 | 1.70 | 0.1% |

### KDA — MI355X 1,753 us/step vs B300 1,085 us/step (1.62x, +668)

| Platform | Kernel function | us/step | calls/step | us/call | % of component |
|---|---|---:|---:|---:|---:|
| MI355X | `fused_recurrent_kda_fwd_kernel.kd` | 1,359.1 | 69.0 | 19.70 | 77.5% |
| MI355X | `_causal_conv1d_update_kernel.kd` | 394.1 | 69.0 | 5.71 | 22.5% |
| B300 | `fused_recurrent_kda_fwd_kernel` | 680.7 | 69.0 | 9.86 | 62.7% |
| B300 | `_causal_conv1d_update_kernel` | 404.5 | 69.0 | 5.86 | 37.3% |

### Norm/quant — MI355X 863 us/step vs B300 263 us/step (3.28x, +600)

| Platform | Kernel function | us/step | calls/step | us/call | % of component |
|---|---|---:|---:|---:|---:|
| MI355X | `void aiter::add_rmsnorm_quant_kernel<std::bfloat16_t, std::bfloat16_t, 256, 16, false, false, true, 1>(std:...` | 406.3 | 92.0 | 4.42 | 47.1% |
| MI355X | `layer_norm_gated_fwd_kernel.kd` | 295.5 | 69.0 | 4.28 | 34.2% |
| MI355X | `void aiter::fused_qk_rmsnorm_kernel<std::bfloat16_t, 256, 8, true, 1>(std::bfloat16_t*, std::bfloat16_t*, s...` | 100.8 | 24.0 | 4.20 | 11.7% |
| MI355X | `void aiter::add_rmsnorm_quant_kernel<std::bfloat16_t, std::bfloat16_t, 256, 32, true, false, true, 1>(std::...` | 42.9 | 10.0 | 4.29 | 5.0% |
| MI355X | `void aiter::add_rmsnorm_quant_kernel<std::bfloat16_t, std::bfloat16_t, 256, 32, false, false, true, 1>(std:...` | 13.4 | 3.0 | 4.45 | 1.5% |
| MI355X | `void vllm::rms_norm_kernel<c10::BFloat16, 8, 3, true>(c10::BFloat16*, c10::BFloat16 const*, long, long, lon...` | 4.3 | 1.0 | 4.33 | 0.5% |
| B300 | `layer_norm_gated_fwd_kernel` | 147.5 | 69.0 | 2.14 | 56.1% |
| B300 | `_fused_q_kv_rmsnorm_kernel` | 103.5 | 29.0 | 3.57 | 39.4% |
| B300 | `void vllm::rms_norm_kernel<c10::BFloat16, 8, 2, true>(c10::BFloat16*, c10::BFloat16 const*, long, long, lon...` | 9.2 | 3.0 | 3.06 | 3.5% |
| B300 | `void vllm::rms_norm_kernel<c10::BFloat16, 8, 3, true>(c10::BFloat16*, c10::BFloat16 const*, long, long, lon...` | 2.7 | 1.0 | 2.75 | 1.0% |

### MoE routing/sort — MI355X 2,679 us/step vs B300 2,108 us/step (1.27x, +571)

| Platform | Kernel function | us/step | calls/step | us/call | % of component |
|---|---|---:|---:|---:|---:|
| MI355X | `void aiter::grouped_topk_kernel<float, float __vector(4), 1, true, true, false>(float*, float const*, float...` | 1,078.2 | 92.0 | 11.72 | 40.3% |
| MI355X | `_ZN5aiter30fused_mx_quant_moe_sort_kernelIDF16bDB8_Li256ELi16EEEvPT0_PhPKT_PKiS9_PKfiiiiiiiii.kd` | 691.5 | 92.0 | 7.52 | 25.8% |
| MI355X | `void aiter::opus_moe_sorting_entry<aiter::MoeSortingMultiPhaseKernel_P23<aiter::MoeSortingProblemMp<int, fl...` | 515.6 | 92.0 | 5.60 | 19.2% |
| MI355X | `void aiter::opus_moe_sorting_entry<aiter::MoeSortingMultiPhaseKernel_P0_v2<aiter::MoeSortingProblemMp<int, ...` | 393.4 | 92.0 | 4.28 | 14.7% |
| B300 | `void moe::dev::routing::routingCustom::routingIndicesClusterKernel<moe::dev::routing::routingCustom::Kernel...` | 921.3 | 92.0 | 10.01 | 43.7% |
| B300 | `void moe::dev::routing::routingCustom::routingIndicesBlockScoresKernel<moe::dev::routing::routingCustom::Ke...` | 583.6 | 92.0 | 6.34 | 27.7% |
| B300 | `void moe::dev::finalize::finalizeKernel<moe::dev::finalize::KernelParams<cutlass::bfloat16_t, cutlass::bflo...` | 420.1 | 92.0 | 4.57 | 19.9% |
| B300 | `kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeLinearKernel_object_at__tenso...` | 183.1 | 92.0 | 1.99 | 8.7% |

### Dense GEMM — MI355X 8,040 us/step vs B300 7,640 us/step (1.05x, +400)

| Platform | Kernel function | us/step | calls/step | us/call | % of component |
|---|---|---:|---:|---:|---:|
| MI355X | `Cijk_Alik_Bljk_BSS_BH_Bias_S_HA_S_SAV_UserArgs_MT32x16x128_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_C...` | 1,387.8 | 92.0 | 15.08 | 17.3% |
| MI355X | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x512_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CAD...` | 1,254.7 | 69.0 | 18.18 | 15.6% |
| MI355X | `hgemm_bf16_16x64x64x7_SPK4_W1x2x1_BLDS1_TN_AS1_0.kd` | 1,156.9 | 92.0 | 12.57 | 14.4% |
| MI355X | `hgemm_bf16_16x64x64x7_SPK8_W1x2x1_BLDS1_TN_AS1_0.kd` | 917.2 | 116.0 | 7.91 | 11.4% |
| MI355X | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x512_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CAD...` | 759.0 | 92.0 | 8.25 | 9.4% |
| MI355X | `hgemm_bf16_16x64x64x8_SPK2_W1x2x1_BLDS1_TN_AS1_0.kd` | 698.4 | 93.0 | 7.51 | 8.7% |
| MI355X | `hgemm_bf16_16x64x64x8_SPK1_W1x2x1_BLDS1_TN_AS1_0.kd` | 459.0 | 92.0 | 4.99 | 5.7% |
| MI355X | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x128_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CAD...` | 320.0 | 69.0 | 4.64 | 4.0% |
| MI355X | `hgemm_bf16_16x64x64x7_SPK7_W1x2x1_BLDS1_TN_AS1_0.kd` | 221.6 | 24.0 | 9.23 | 2.8% |
| MI355X | `void wvSplitK_hf_sml_<__hip_bfloat16, 64, 2, 16, 8, 2, 1>(int, int, int, int, int, int, __hip_bfloat16 cons...` | 149.7 | 7.0 | 21.39 | 1.9% |
| MI355X | `hgemm_bf16_16x64x64x5_SPK4_W1x2x1_BLDS1_TN_AS1_0.kd` | 128.2 | 24.0 | 5.34 | 1.6% |
| MI355X | `_ZN5aiter37bf16gemm_fp32bf16_tn_32x64_pf3_splitkE.kd` | 106.4 | 10.0 | 10.64 | 1.3% |
| MI355X | `void gemm_a16w16_flatmm_splitk_kernel<opus_flatmm_splitk_traits_gfx950<256, opus::seq<32, 128, 64>, opus::t...` | 100.7 | 1.0 | 100.65 | 1.3% |
| MI355X | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT96x16x256_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CAD...` | 98.9 | 2.0 | 49.45 | 1.2% |
| MI355X | `hgemm_bf16_16x64x128x6_SPK4_W1x1x2_BLDS1_TN_AS1_0.kd` | 60.5 | 5.0 | 12.09 | 0.8% |
| MI355X | `hgemm_bf16_16x64x64x6_SPK7_W1x2x1_BLDS1_TN_AS1_0.kd` | 44.7 | 5.0 | 8.94 | 0.6% |
| MI355X | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT128x32x128_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CA...` | 40.3 | 5.0 | 8.05 | 0.5% |
| MI355X | `Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS...` | 28.5 | 5.0 | 5.71 | 0.4% |
| MI355X | `void splitk_reduce_kernel<16, 64, std::bfloat16_t, false, std::bfloat16_t, true>(opus_splitk_ws_handle cons...` | 27.5 | 6.0 | 4.59 | 0.3% |
| MI355X | `void gemm_a16w16_flatmm_splitk_kernel<opus_flatmm_splitk_traits_gfx950<256, opus::seq<64, 32, 64>, opus::tu...` | 26.7 | 5.0 | 5.34 | 0.3% |
| MI355X | `hgemm_bf16_16x256x64x4_SPK7_W1x2x1_BLDS1_TN_AS1_0.kd` | 24.9 | 1.0 | 24.89 | 0.3% |
| MI355X | `hgemm_bf16_16x64x64x6_SPK2_W1x2x1_BLDS1_TN_AS1_0.kd` | 15.5 | 1.0 | 15.50 | 0.2% |
| MI355X | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CA...` | 13.2 | 1.0 | 13.23 | 0.2% |
| B300 | `void fused_a_gemm_kernel<1, 3584, 7168, 16, 8, 256, 16>(__nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16...` | 1,522.7 | 92.0 | 16.55 | 19.9% |
| B300 | `nvjet_sm103_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 1,356.3 | 71.0 | 19.10 | 17.8% |
| B300 | `void fused_a_gemm_kernel<1, 1536, 7168, 16, 8, 256, 16>(__nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16...` | 937.1 | 116.0 | 8.08 | 12.3% |
| B300 | `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_ll_bf16_splitkLLBf16SplitK_object_at__tensorp...` | 703.1 | 92.0 | 7.64 | 9.2% |
| B300 | `kernel_cutlass_kernel_vllmmodelskimi_k3nvidiaopscute_dsllatent_moe_tailfused_add_multicast_gemmFusedAddMult...` | 646.4 | 92.0 | 7.03 | 8.5% |
| B300 | `nvjet_sm103_tst_64x8_64x16_4x1_v_bz_TNT` | 633.3 | 95.0 | 6.67 | 8.3% |
| B300 | `void fused_a_gemm_kernel<1, 7168, 768, 16, 8, 256, 3>(__nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 c...` | 473.9 | 92.0 | 5.15 | 6.2% |
| B300 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, _...` | 344.7 | 78.0 | 4.42 | 4.5% |
| B300 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x1_tn_align2>(cutlass_80_wmma...` | 211.1 | 69.0 | 3.06 | 2.8% |
| B300 | `void fused_a_gemm_kernel<1, 2112, 7168, 16, 8, 256, 16>(__nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16...` | 207.1 | 24.0 | 8.63 | 2.7% |
| B300 | `void fused_a_gemm_kernel<1, 2304, 1536, 16, 8, 256, 6>(__nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 ...` | 131.6 | 24.0 | 5.48 | 1.7% |
| B300 | `nvjet_sm103_tst_64x8_64x16_4x2_h_bz_TNT` | 114.5 | 10.0 | 11.45 | 1.5% |
| B300 | `nvjet_sm103_tst_128x8_64x12_2x1_v_bz_splitK_TNT` | 82.2 | 1.0 | 82.18 | 1.1% |
| B300 | `nvjet_sm103_tst_64x16_64x16_4x1_v_bz_TNT` | 66.9 | 10.0 | 6.69 | 0.9% |
| B300 | `nvjet_sm103_tst_384x8_64x4_2x1_v_bz_TNT` | 66.6 | 7.0 | 9.52 | 0.9% |
| B300 | `nvjet_sm103_tst_64x8_64x16_1x2_h_bz_splitK_TNT` | 56.9 | 5.0 | 11.37 | 0.7% |
| B300 | `nvjet_sm103_tst_192x16_64x8_2x1_2cta_v_bz_splitK_TNT` | 48.6 | 1.0 | 48.63 | 0.6% |
| B300 | `nvjet_sm103_tst_8x64_64x16_4x1_v_bz_TNN` | 19.8 | 5.0 | 3.95 | 0.3% |
| B300 | `nvjet_sm103_tst_64x8_64x16_2x2_h_bz_NNT` | 17.4 | 5.0 | 3.48 | 0.2% |

### MoE expert GEMM — MI355X 3,743 us/step vs B300 3,458 us/step (1.08x, +284)

| Platform | Kernel function | us/step | calls/step | us/call | % of component |
|---|---|---:|---:|---:|---:|
| MI355X | `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_xcd4_situv2_sb4p0_slb25p0_v2out_v32.kd` | 2,358.8 | 92.0 | 25.64 | 63.0% |
| MI355X | `gemm2_a4w4_port_hmax8192_imax8192_bm32_bn128_bk128_atomic_a8_g2ks2_bhoist_apf_spart4x2_v2.kd` | 1,383.7 | 92.0 | 15.04 | 37.0% |
| B300 | `bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x8x512_s3_et128x8_m128x8x32_c1x1x1_rM_TN_transOut_schPd2x1x...` | 2,051.8 | 92.0 | 22.30 | 59.3% |
| B300 | `bmm_Bfloat16_MxE2m1MxE4m3_Fp32_Ab32_Bb32_t128x8x512_s3_et128x8_m128x8x32_c1x1x1_rM_TN_transOut_schPd2x1x2x3...` | 1,406.5 | 92.0 | 15.29 | 40.7% |

### Sampling — MI355X 82 us/step vs B300 79 us/step (1.05x, +4)

| Platform | Kernel function | us/step | calls/step | us/call | % of component |
|---|---|---:|---:|---:|---:|
| MI355X | `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, unsigned i...` | 36.8 | 7.0 | 5.26 | 44.6% |
| MI355X | `_gumbel_sample_kernel.kd` | 28.9 | 7.0 | 4.12 | 35.0% |
| MI355X | `_resample_kernel.kd` | 4.7 | 1.0 | 4.68 | 5.7% |
| MI355X | `_combine_sampled_and_draft_tokens_kernel.kd` | 4.5 | 1.0 | 4.46 | 5.4% |
| MI355X | `_insert_resampled_kernel.kd` | 4.3 | 1.0 | 4.29 | 5.2% |
| MI355X | `_get_num_sampled_and_rejected_kernel.kd` | 3.4 | 1.0 | 3.37 | 4.1% |
| B300 | `_resample_kernel` | 28.7 | 1.0 | 28.65 | 36.4% |
| B300 | `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, unsigned i...` | 24.5 | 7.0 | 3.50 | 31.2% |
| B300 | `_gumbel_sample_kernel` | 15.9 | 7.0 | 2.27 | 20.2% |
| B300 | `_combine_sampled_and_draft_tokens_kernel` | 3.5 | 1.0 | 3.47 | 4.4% |
| B300 | `_get_num_sampled_and_rejected_kernel` | 3.0 | 1.0 | 3.05 | 3.9% |
| B300 | `_insert_resampled_kernel` | 3.0 | 1.0 | 3.04 | 3.9% |

### Provenance for this section

| item | value |
|---|---|
| MI355X trace | `/dev/shm/prof_isl100k/dp0_pp0_tp0_dcp0_ep0_rank0.*.pt.trace.json.gz` |
| B300 trace | `~/work/b300/profile_20260911_102437/traces/dp0_pp0_tp0_dcp0_ep0_rank0.*` |
| extraction | `/dev/shm/_mla_raw.py` — reuses `trace_compare_k3.py`'s `_parse_gpu_steps` / `_build_step_index` / `_assign_stage_from_index`, so the DECODE window matches the tables above |
| this section | `/dev/shm/_mk_component_section.py` |
| config | DCP8 + DSpark K=7 + DRAM offload, TP8/EP1, ROCM_AITER_MLA fp8 asm, conc-1 |
| caveat | Profiler inflation differs by platform (~+13% ours, ~+27% B300 at this ISL), so absolute us are inflated on BOTH sides. Ratios and call counts are the trustworthy part. |
| date | 2026-09-12 |
| known inference | B300's MLA BMM1/BMM2 absorb is attributed to `nvjet_*_NNT` + `nvjet_*_TNN` (24 calls each = 24 MLA layers, bmm transpose layouts), matching our `batched_gemm_a16wfp4` (23 calls x2). Inferred from call count and layout, NOT confirmed. |
