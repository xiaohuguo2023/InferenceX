# Trace Comparison: K3-TP8-8k1k

## Configuration

| | K3-TP8-8k1k |
|---|---|
| World size | 8 |
| Decode iterations | 58 |

## PREFILL

### K3-TP8-8k1k Per-Rank Breakdown (us)

| Category | Sub-kernel | R0 | R1 | R2 | R3 | R4 | R5 | R6 | R7 | Imbal% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **GEMM** | *(total)* | 8610869.6 | 8488322.9 | 8423680.7 | 8467688.9 | 8560462.6 | 8523168.0 | 8554191.3 | 8528764.6 | 2.2% |
| | hipBLASLt (Cijk) | 5080405.0 | 4990633.3 | 4947523.9 | 4968043.8 | 5046056.2 | 5025427.3 | 5046803.1 | 5024256.7 | |
| | AITER flydsl MoE | 3513605.2 | 3480624.9 | 3459214.1 | 3482950.7 | 3496913.5 | 3480481.7 | 3490280.7 | 3487378.8 | |
| | AITER wfp4 batched | 14576.2 | 14806.4 | 14688.2 | 14446.8 | 15230.5 | 14997.1 | 14841.4 | 14846.0 | |
| | AITER hgemm bf16 | 1455.6 | 1451.7 | 1440.4 | 1430.5 | 1454.7 | 1454.5 | 1452.7 | 1455.2 | |
| | rocBLAS splitK | 701.4 | 684.0 | 688.1 | 693.1 | 683.3 | 684.9 | 688.2 | 701.9 | |
| | Triton GEMM (aiter) | 126.2 | 122.5 | 126.0 | 124.0 | 124.5 | 122.4 | 125.3 | 126.0 | |
| **Communication** | *(total)* | 7964104.2 | 8199760.4 | 8041554.2 | 4409723.2 | 8003125.3 | 7242827.6 | 8098589.1 | 7484453.3 | 51.0% :warning: |
| | CustomAR 2-stage | 7961414.8 | 8196492.6 | 8038423.5 | 4406916.0 | 8000164.0 | 7239773.5 | 8095826.0 | 7481721.5 | |
| | NCCL/RCCL | 2689.4 | 3267.8 | 3130.7 | 2807.3 | 2961.3 | 3054.1 | 2763.1 | 2731.8 | |
| **Attention** | *(total)* | 1500527.7 | 1469300.1 | 1468419.7 | 1464729.1 | 1476929.4 | 1471013.8 | 1492935.0 | 1491765.7 | 2.4% |
| | MLA attn residual | 910563.5 | 885861.9 | 884860.6 | 882632.4 | 888370.8 | 884843.0 | 906246.8 | 906490.0 | |
| | MLA gluon (decode) | 300544.2 | 299835.8 | 300927.0 | 300625.5 | 301267.5 | 300289.1 | 300596.8 | 300196.4 | |
| | aiter attn | 256959.9 | 251363.6 | 250516.4 | 249606.2 | 254791.2 | 253847.0 | 253846.9 | 253038.7 | |
| | MLA merge states | 32460.1 | 32238.8 | 32115.6 | 31865.0 | 32499.9 | 32034.8 | 32244.5 | 32040.6 | |
| **KDA Linear Attn** | *(total)* | 1269249.8 | 1259119.1 | 1257583.1 | 1258695.3 | 1267873.4 | 1262395.2 | 1263208.1 | 1262637.7 | 0.9% |
| | KDA gated-delta | 759064.0 | 754407.3 | 752441.8 | 753560.1 | 757749.6 | 754447.1 | 754901.1 | 754866.9 | |
| | KDA gate/state | 246458.0 | 243148.8 | 244691.5 | 243381.8 | 246339.0 | 246255.5 | 246314.4 | 245848.6 | |
| | KDA conv1d | 169964.3 | 167976.2 | 166979.2 | 168764.8 | 170073.2 | 168234.7 | 168436.4 | 168300.4 | |
| | KDA GLA | 93763.5 | 93586.8 | 93470.6 | 92988.6 | 93711.6 | 93457.9 | 93556.2 | 93621.9 | |
| **Other** | *(total)* | 1084426.0 | 1079913.9 | 1082075.6 | 1075670.1 | 1089862.9 | 1084276.8 | 1088840.7 | 1088146.2 | 1.3% |
| | void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add... | 418484.4 | 419076.5 | 420103.4 | 420070.2 | 419743.0 | 417152.0 | 422282.4 | 419774.1 | |
| | void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_ker... | 160653.4 | 158865.4 | 159721.6 | 157726.5 | 159739.4 | 159589.6 | 158596.1 | 160509.5 | |
| | void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat... | 100985.0 | 99612.1 | 101847.9 | 101929.2 | 103917.1 | 104465.9 | 103222.4 | 103014.2 | |
| | void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_ker... | 65782.4 | 64960.9 | 65049.7 | 64903.2 | 65984.8 | 65421.5 | 66213.8 | 65858.2 | |
| | void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kerne... | 52067.2 | 51955.7 | 51284.9 | 50828.8 | 51903.4 | 51883.0 | 51748.5 | 51354.2 | |
| | void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<f... | 47906.9 | 47859.5 | 47692.6 | 47531.3 | 47776.7 | 47683.0 | 47777.5 | 47659.1 | |
| | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native... | 45629.2 | 45559.2 | 45161.5 | 44209.0 | 45623.8 | 45182.7 | 45493.8 | 45221.2 | |
| | void at::native::vectorized_elementwise_kernel<4, at::native::sigmoid_kernel_... | 37018.3 | 37081.9 | 36782.7 | 36612.2 | 36871.0 | 36934.0 | 36959.8 | 37033.4 | |
| | void at::native::vectorized_elementwise_kernel<4, at::native:: | 35119.9 | 35091.1 | 35066.4 | 34721.5 | 35104.0 | 35182.9 | 35047.2 | 35008.6 | |
| | void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_k... | 28570.0 | 27955.3 | 27835.3 | 28560.6 | 27989.8 | 27909.3 | 28635.5 | 28541.9 | |
| | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSe... | 16810.3 | 17047.3 | 16955.9 | 16411.5 | 18405.4 | 17067.1 | 17392.7 | 17800.5 | |
| | void at::native::vectorized_elementwise_kernel<4, at::native::rsqrt_kernel_cuda | 16264.1 | 16045.6 | 15942.3 | 15732.6 | 17179.4 | 16589.2 | 16383.4 | 17290.0 | |
| | void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_... | 12884.9 | 12770.3 | 12557.3 | 12446.7 | 12814.7 | 12677.5 | 12756.0 | 12725.2 | |
| | void vllm::concat_and_cache_mla_kernel<__hip_bfloat16, __hip_bfloat16,  | 12427.1 | 12381.0 | 12385.7 | 12401.2 | 12377.3 | 12446.7 | 12418.2 | 12389.4 | |
| | void at::native::vectorized_elementwise_kernel<8, at::native::BinaryFunctor<c... | 10970.5 | 10983.6 | 10913.5 | 11038.2 | 11061.0 | 11201.2 | 10958.5 | 10961.1 | |
| | void vllm::gather_and_maybe_dequant_cache<__hip_bfloat16, __hip_bfloat16,  | 7487.4 | 7467.9 | 7432.8 | 7330.4 | 7582.6 | 7509.2 | 7482.1 | 7473.0 | |
| | void gqa_d192_v128_kernel<opus_gqa_d192_traits<32, 64, 8, true, true> > | 4791.5 | 4735.5 | 4690.3 | 4692.8 | 4758.2 | 4746.2 | 4723.6 | 4726.3 | |
| | mask_empty_context_kernel | 2957.8 | 2990.9 | 3026.2 | 2964.0 | 3224.4 | 3088.5 | 3059.9 | 3168.9 | |
| | void at::native::vectorized_gather_kernel<16, long> | 1289.7 | 1165.5 | 1175.9 | 1118.3 | 1226.9 | 1187.3 | 1146.4 | 1219.0 | |
| | _zero_kv_blocks_kernel | 1203.4 | 1165.8 | 1159.4 | 1161.8 | 1157.7 | 1144.6 | 1177.0 | 1145.6 | |
| | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add... | 1121.5 | 1127.5 | 1166.1 | 592.6 | 1199.5 | 1150.2 | 1185.8 | 1176.9 | |
| | void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<int>... | 866.9 | 872.6 | 907.5 | 434.6 | 935.4 | 890.3 | 919.5 | 907.1 | |
| | void at::native::vectorized_elementwise_kernel<4, at::native::compare_scalar_... | 658.6 | 656.1 | 676.3 | 439.1 | 699.5 | 671.2 | 685.3 | 676.7 | |
| | void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctorOnSelf_ad... | 566.7 | 573.5 | 586.3 | 280.6 | 612.5 | 567.4 | 601.3 | 589.4 | |
| | void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<i... | 406.5 | 397.7 | 420.9 | 239.0 | 425.9 | 400.8 | 418.2 | 408.6 | |
| | _compute_slot_mapping_kernel | 384.8 | 381.9 | 379.9 | 389.5 | 382.9 | 385.9 | 381.3 | 384.0 | |
| | void rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel<rocprim::ROCPRIM_4... | 367.3 | 373.0 | 371.0 | 307.5 | 376.5 | 375.8 | 386.1 | 364.5 | |
| | void at::native:: | 349.6 | 361.0 | 366.8 | 296.6 | 364.1 | 361.8 | 371.0 | 350.0 | |
| | _expand_page_indices_kernel | 231.9 | 231.7 | 238.5 | 214.9 | 246.9 | 242.4 | 240.2 | 243.7 | |
| | void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctorOnSe... | 168.7 | 167.6 | 177.0 | 85.6 | 179.2 | 169.6 | 177.2 | 172.0 | |
| **MoE Routing** | *(total)* | 793953.2 | 789149.8 | 792336.8 | 790348.7 | 796029.9 | 792572.7 | 793621.2 | 793426.5 | 0.9% |
| | MoE reduction | 493285.7 | 490628.8 | 491494.7 | 491932.8 | 493284.1 | 491488.4 | 493299.8 | 492506.9 | |
| | MoE sort mxfp4 | 177966.4 | 178280.4 | 180977.0 | 179864.3 | 181560.4 | 180237.6 | 179742.4 | 180747.2 | |
| | AITER grouped_topk | 122701.1 | 120240.5 | 119865.1 | 118551.6 | 121185.4 | 120846.6 | 120578.9 | 120172.3 | |
| **Memory** | *(total)* | 450554.6 | 453609.8 | 452599.4 | 454359.2 | 454863.0 | 452037.1 | 452337.2 | 454309.0 | 1.0% |
| | Memcpy | 425205.9 | 428655.2 | 426480.7 | 430180.9 | 428275.8 | 425567.4 | 425869.2 | 426522.2 | |
| | Fill | 24893.7 | 24495.5 | 25577.2 | 23778.3 | 26036.3 | 25938.1 | 25959.1 | 27243.2 | |
| | Memset | 455.0 | 459.1 | 541.5 | 399.9 | 550.9 | 531.5 | 508.8 | 543.6 | |
| **Normalization** | *(total)* | 286616.3 | 284440.8 | 282322.8 | 281075.3 | 282983.6 | 281824.5 | 287513.7 | 286549.6 | 2.3% |
| | Add+RMSNorm+quant | 286616.3 | 284440.8 | 282322.8 | 281075.3 | 282983.6 | 281824.5 | 287513.7 | 286549.6 | |
| **Triton Fused** | *(total)* | 63341.7 | 62953.3 | 62781.4 | 62464.9 | 63880.2 | 65277.6 | 62543.7 | 63190.7 | 4.4% |
| | Triton fused op | 63341.7 | 62953.3 | 62781.4 | 62464.9 | 63880.2 | 65277.6 | 62543.7 | 63190.7 | |
| **Quantization** | *(total)* | 55118.1 | 55028.5 | 54631.7 | 55537.6 | 53625.3 | 56236.3 | 54656.4 | 55364.6 | 4.7% |
| | dynamic per-group quant | 55118.1 | 55028.5 | 54631.7 | 55537.6 | 53625.3 | 56236.3 | 54656.4 | 55364.6 | |
| **Sampling** | *(total)* | 1258.2 | 1257.0 | 1255.3 | 1234.1 | 1274.5 | 1260.4 | 1267.1 | 1266.3 | 3.2% |
| | ArgMax/Reduce | 1258.2 | 1257.0 | 1255.3 | 1234.1 | 1274.5 | 1260.4 | 1267.1 | 1266.3 | |
| **TOTAL** | | 22080019.6 | 22142855.4 | 21919240.7 | 18321526.4 | 22050910.0 | 21232889.9 | 22149703.6 | 21509874.2 | 17.9% |

> **Straggler**: R6 (22149704 us) — 3.4% above mean. Main cause: Communication (51% imbalanced)
>
> **Comm/Total ratio** (bottleneck rank): 37.5%

## DECODE

### K3-TP8-8k1k Per-Rank Breakdown (per-iter avg us)

| Category | Sub-kernel | R0 | R1 | R2 | R3 | R4 | R5 | R6 | R7 | Imbal% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **GEMM** | *(total)* | 14746.6 | 14691.9 | 14702.0 | 14796.8 | 14823.9 | 14826.6 | 14789.8 | 14890.3 | 1.3% |
| | AITER flydsl MoE | 6126.1 | 6100.9 | 6090.3 | 6126.8 | 6123.3 | 6098.6 | 6106.2 | 6142.1 | |
| | hipBLASLt (Cijk) | 3786.0 | 3767.5 | 3774.6 | 3804.8 | 3813.8 | 3817.0 | 3798.3 | 3819.5 | |
| | AITER hgemm bf16 | 2266.8 | 2260.5 | 2259.4 | 2277.3 | 2270.2 | 2266.8 | 2273.5 | 2298.8 | |
| | rocBLAS splitK | 1357.8 | 1353.4 | 1367.5 | 1364.6 | 1368.4 | 1390.6 | 1367.9 | 1380.7 | |
| | Triton GEMM (aiter) | 912.6 | 907.3 | 902.5 | 914.4 | 935.4 | 934.5 | 933.1 | 931.8 | |
| | AITER wfp4 batched | 297.3 | 302.4 | 307.6 | 308.8 | 312.9 | 319.1 | 311.0 | 317.4 | |
| **Other** | *(total)* | 8154.2 | 8185.0 | 8855.4 | 9009.4 | 9089.3 | 9203.8 | 9054.3 | 9338.0 | 13.4% :warning: |
| | void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat... | 1679.1 | 1660.6 | 1807.6 | 1820.2 | 1850.8 | 1872.7 | 1840.7 | 1887.1 | |
| | void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add... | 1386.5 | 1415.5 | 1500.7 | 1508.5 | 1522.4 | 1545.3 | 1526.0 | 1565.2 | |
| | __amd_rocclr_copyBuffer | 1147.3 | 1160.0 | 1267.4 | 1285.9 | 1289.6 | 1309.8 | 1285.2 | 1336.1 | |
| | void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_ker... | 1013.6 | 1016.1 | 1101.8 | 1130.5 | 1150.6 | 1157.4 | 1155.0 | 1184.2 | |
| | void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_k... | 349.8 | 352.2 | 381.0 | 384.6 | 384.3 | 399.9 | 381.1 | 398.4 | |
| | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native... | 359.9 | 359.3 | 375.6 | 384.9 | 384.0 | 392.2 | 383.3 | 397.5 | |
| | void at::native::vectorized_elementwise_kernel<4, at::native::sigmoid_kernel_... | 339.0 | 334.3 | 374.0 | 386.3 | 388.5 | 384.4 | 381.4 | 397.3 | |
| | void at::native::vectorized_elementwise_kernel<4, at::native::rsqrt_kernel_cuda | 337.8 | 337.7 | 364.4 | 385.2 | 385.6 | 385.3 | 386.0 | 397.2 | |
| | void at::native::vectorized_elementwise_kernel<4, at::native:: | 335.4 | 339.5 | 372.8 | 381.5 | 390.2 | 391.9 | 381.5 | 396.8 | |
| | void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<f... | 333.1 | 337.5 | 366.9 | 385.7 | 384.1 | 388.7 | 384.2 | 393.1 | |
| | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSe... | 338.4 | 338.3 | 369.6 | 383.0 | 382.4 | 387.8 | 378.5 | 391.9 | |
| | void at::native::vectorized_elementwise_kernel<8, at::native::BinaryFunctor<c... | 124.8 | 124.3 | 134.4 | 134.1 | 135.3 | 139.6 | 132.9 | 139.5 | |
| | void vllm::concat_and_cache_mla_kernel<__hip_bfloat16, __hip_bfloat16,  | 129.8 | 127.5 | 136.8 | 136.5 | 134.7 | 137.8 | 133.9 | 139.4 | |
| | void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_... | 118.0 | 121.5 | 131.6 | 131.3 | 134.1 | 135.2 | 133.9 | 137.1 | |
| | void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_ker... | 54.7 | 54.3 | 56.8 | 56.1 | 56.1 | 56.9 | 56.1 | 57.8 | |
| | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add... | 25.5 | 25.8 | 27.8 | 28.4 | 28.3 | 29.4 | 27.7 | 29.1 | |
| | _Z32gemm_a16w16_flatmm_splitk_kernelI32opus_flatmm_splitk_traits_gfx950ILi256... | 18.7 | 18.8 | 18.6 | 18.7 | 18.6 | 18.7 | 18.6 | 18.8 | |
| | void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<int>... | 14.6 | 14.5 | 16.2 | 16.4 | 17.2 | 17.3 | 16.8 | 17.4 | |
| | void at::native:: | 9.5 | 8.5 | 9.1 | 9.1 | 9.6 | 9.4 | 9.2 | 9.7 | |
| | void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kerne... | 7.0 | 6.9 | 7.7 | 7.7 | 8.0 | 8.1 | 7.9 | 8.1 | |
| | void at::native::vectorized_gather_kernel<16, long> | 7.4 | 7.5 | 7.8 | 7.8 | 7.8 | 7.9 | 7.8 | 8.1 | |
| | _expand_page_indices_kernel | 5.2 | 5.1 | 5.6 | 5.7 | 5.7 | 5.9 | 5.6 | 5.8 | |
| | void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctorOnSelf_ad... | 4.9 | 4.9 | 5.4 | 5.6 | 5.8 | 5.7 | 5.7 | 5.9 | |
| | _compute_slot_mapping_kernel | 5.1 | 5.1 | 5.6 | 5.7 | 5.6 | 5.9 | 5.5 | 5.8 | |
| | void rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel<rocprim::ROCPRIM_4... | 5.0 | 5.0 | 5.6 | 5.4 | 5.6 | 5.7 | 5.5 | 5.8 | |
| | _Z20splitk_reduce_kernelILi16ELi64EDF16bLb0EDF16bLb0EEvPK21opus_splitk_ws_han... | 2.5 | 2.5 | 2.7 | 2.7 | 2.6 | 2.8 | 2.6 | 2.7 | |
| | void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda... | 1.7 | 1.7 | 1.8 | 1.8 | 1.8 | 1.9 | 1.8 | 1.9 | |
| **Attention** | *(total)* | 6782.0 | 6749.0 | 6797.0 | 6798.8 | 6822.5 | 6808.6 | 6818.3 | 6813.2 | 1.1% |
| | MLA gluon (decode) | 5441.8 | 5413.0 | 5442.5 | 5439.0 | 5452.9 | 5430.7 | 5439.1 | 5426.8 | |
| | MLA attn residual | 1340.2 | 1336.0 | 1354.5 | 1359.7 | 1369.6 | 1377.8 | 1379.2 | 1386.3 | |
| **Communication** | *(total)* | 4902.4 | 4951.8 | 3909.9 | 3562.5 | 3455.7 | 3280.0 | 3547.0 | 3030.1 | 50.2% :warning: |
| | CustomAR 2-stage | 3879.6 | 3939.9 | 3182.7 | 2766.6 | 2729.6 | 2704.1 | 2855.1 | 2384.0 | |
| | CustomAR 1-stage | 986.3 | 975.2 | 692.7 | 761.5 | 692.3 | 542.8 | 658.0 | 613.8 | |
| | NCCL/RCCL | 36.6 | 36.7 | 34.5 | 34.3 | 33.7 | 33.1 | 33.9 | 32.3 | |
| **MoE Routing** | *(total)* | 3178.0 | 3184.5 | 3265.2 | 3254.8 | 3330.2 | 3320.3 | 3330.3 | 3340.9 | 5.0% |
| | MoE sort mxfp4 | 1834.9 | 1837.0 | 1924.1 | 1920.4 | 1951.3 | 1940.1 | 1958.4 | 1961.9 | |
| | AITER grouped_topk | 1176.8 | 1179.8 | 1168.1 | 1166.9 | 1188.0 | 1189.0 | 1191.8 | 1187.2 | |
| | MoE reduction | 166.4 | 167.6 | 173.0 | 167.6 | 190.9 | 191.2 | 180.1 | 191.9 | |
| **Normalization** | *(total)* | 1692.0 | 1689.8 | 1826.5 | 1824.8 | 1828.6 | 1880.4 | 1817.3 | 1894.7 | 11.3% :warning: |
| | Add+RMSNorm+quant | 1692.0 | 1689.8 | 1826.5 | 1824.8 | 1828.6 | 1880.4 | 1817.3 | 1894.7 | |
| **KDA Linear Attn** | *(total)* | 1002.3 | 1014.4 | 1030.4 | 1021.8 | 1028.0 | 1044.1 | 1023.4 | 1044.4 | 4.1% |
| | KDA gated-delta | 631.4 | 630.3 | 637.0 | 631.3 | 635.1 | 643.4 | 636.8 | 638.3 | |
| | KDA conv1d | 370.9 | 384.0 | 393.4 | 390.6 | 392.8 | 400.7 | 386.6 | 406.0 | |
| **Triton Fused** | *(total)* | 466.9 | 457.3 | 520.0 | 512.5 | 518.4 | 530.2 | 522.2 | 537.5 | 15.8% :warning: |
| | Triton fused op | 466.9 | 457.3 | 520.0 | 512.5 | 518.4 | 530.2 | 522.2 | 537.5 | |
| **Memory** | *(total)* | 168.1 | 168.8 | 185.1 | 187.7 | 192.8 | 195.8 | 187.4 | 196.8 | 15.5% :warning: |
| | Memcpy | 107.5 | 108.0 | 119.1 | 120.4 | 124.3 | 126.2 | 119.9 | 126.3 | |
| | Fill | 52.5 | 52.7 | 57.2 | 58.2 | 59.5 | 60.5 | 58.4 | 61.0 | |
| | Memset | 8.0 | 8.1 | 8.9 | 9.0 | 9.1 | 9.0 | 9.1 | 9.5 | |
| **Sampling** | *(total)* | 22.0 | 21.8 | 21.8 | 21.8 | 21.6 | 21.9 | 22.2 | 21.9 | 2.8% |
| | ArgMax/Reduce | 22.0 | 21.8 | 21.8 | 21.8 | 21.6 | 21.9 | 22.2 | 21.9 | |
| **TOTAL** | | 41114.5 | 41114.3 | 41113.2 | 40990.7 | 41111.0 | 41111.8 | 41112.2 | 41107.8 | 0.3% |

> All ranks balanced (max 0.0% above mean)
>
> **Comm/Total ratio** (bottleneck rank): 11.6%

## Summary

No significant findings.

---

## Interpretation — the AR "imbalance" is spin-wait, not a comm bottleneck

Analysis via `trace_compare_k3.py` (K3-patched fork of `~/work/vllm_traces_tp_varied/trace_compare.py`)
over all 8 rank traces (`kimik3_traces/`, 8k/1k conc32, 58 prefill + 58 decode steps).

### DECODE (per-iter avg — the TPOT-relevant stage)
- **Per-rank TOTAL is balanced to 0.3%** — every rank spends ~41,110 µs/iter. No end-to-end straggler.
- **Communication alone shows 50.2% imbalance** (`CustomAR 2-stage`: R1 3940 µs … R7 2384 µs), but this is
  **spin-wait, not comm cost**: the 2-stage custom all-reduce is a **barrier**. Ranks that finish compute
  early sit in the AR spinning until the last rank arrives.
  - **R0/R1**: highest AR (~4900 µs) **but lowest** elementwise "Other" (~8150 µs) → they finish early, spin longest.
  - **R7**: lowest AR (3030 µs) **but highest** "Other" (9338 µs), Norm (1895 µs), GEMM (14890 µs) → it is the
    compute straggler; everyone waits for it.
  - The AR barrier exactly **absorbs** the compute skew → TOTAL flat. **Comm/total at the bottleneck rank is only 11.6%**.
- **The real (modest) compute imbalance is elementwise + norm**, not the heavy math:
  | Category | Imbal% | note |
  |---|---:|---|
  | GEMM (MoE flydsl + hipBLASLt + hgemm) | 1.3% | balanced |
  | Attention (MLA gluon + residual) | 1.1% | balanced |
  | KDA linear attention | 4.1% | balanced |
  | MoE routing/sort | 5.0% | balanced |
  | **Normalization** (`add_rmsnorm_quant`) | **11.3%** | R0/R1 low, R2–R7 high |
  | **"Other" (elementwise/copy)** | **13.4%** | R0/R1 low, R2–R7 high |
  | **Communication (AR)** | **50.2%** | **spin-wait absorbing the above** |
  - The skew is a **2-vs-6 rank-group pattern** (R0,R1 consistently below R2–R7 on elementwise/norm/memcpy) —
    suggestive of a placement/NUMA/XCD asymmetry in the elementwise+norm path, not the sharded GEMM/attention.

### PREFILL (8k)
- Total imbalance **17.9%**, straggler **R6** (+3.4% above mean); comm-heavy (**comm/total 37.5%**).
- **R3 is the AR outlier** (`CustomAR 2-stage` 4.4 M µs vs ~8 M on other ranks) → R3 arrives *last* each AR
  (little spin), the other 7 spin-wait on it. Prefill is where the big AR time lives (matches the earlier
  8k-dominated 34% AR and the `merge_attn_states`/eager-MLA CPU bubbles).

### Bottom line
1. **K3 TP8 decode is compute-bound and well-balanced end-to-end (0.3%).** The 34–50% AR figure is **spin-wait**,
   not a communication bottleneck — confirming the earlier CPU-bubble analysis (AR time inflated by cross-rank desync).
2. **Lowering the AR share in decode requires reducing the compute skew that ranks wait on** — the ~13%
   elementwise + ~11% norm imbalance (R0/R1 vs R2–R7), not faster all-reduce. AR volume itself is already on
   the AITER custom path with `fuse_allreduce_rms=True`.
3. **In prefill, comm is genuinely large (37.5%)** and gated by the slowest rank per all-reduce (R3 straggler);
   that is the 8k-prefill barrier cost, orthogonal to decode.

Artifacts: `docs/kimik3_tp8_rank_imbalance.md` (this file), `kimik3_tp8_rank_imbalance.csv`,
`trace_compare_k3.py`.
