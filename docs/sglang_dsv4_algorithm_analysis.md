# SGLang DeepSeek-V4 — Decode Algorithm Cost Analysis (MI355X)

Each GPU kernel in SGLang's DeepSeek-V4 **decode** path is attached to a step of the model's forward algorithm, with its share of total decode GPU compute.

| Field | Value |
|---|---|
| Model | `deepseek-ai/DeepSeek-V4-Pro` (FP4 MoE + FP8 attn, FP8 KV) |
| Engine | SGLang 0.5.14 (`--attention-backend dsv4`), TP=8 + DP-attention, MI355X |
| Source | decode CUDA-graph **capture** trace (`--enable-profile-cuda-graph`), TP-0 rank |
| Kernels | 85 distinct, 15,925 launches, 106.9 ms total GPU time |

> **Method/caveat:** times are from the decode graph-*capture* trace (eager, `record_shapes`), summed over captured batch sizes 1–64; they are a structural proxy for the decode compute mix (which kernel costs what fraction), not a steady-state wall-clock at one batch size. Capture-time CPU/API artifacts (`hipPointerGetAttribute`, `aten::*`) are excluded (GPU-kernel events only). TP-0 rank; ranks are symmetric.

## Compute backends in play

SGLang's DeepSeek-V4 path is a mosaic of kernel backends. Each kernel below is
tagged with its origin, verified from the image (AITER `.co`/hsaco files, AITER
`flydsl_cache`, AITER `jit/module_*.so`, `ck::` symbols, Tensile `Cijk_`, SGLang
Triton/`gluon`, and SGLang `jit_kernel/csrc/deepseek_v4/*`):

| Tag | Backend | What it is |
|---|---|---|
| **AITER-asm** | AITER assembly (`hsa/gfx950/*.co`) | hand-written MI350 assembly GEMMs (`fp8gemm_bf16_blockscale_BpreShuffle`, `bf16gemm_fp32bf16`) |
| **AITER-flydsl** | AITER `flydsl` DSL codegen (`ops/flydsl`, `jit/flydsl_cache`) | DSL-generated MFMA kernels — MoE experts (`mfma_moe1/2` = `flydsl_moe1/2`) and bf16 split-K `hgemm` |
| **AITER-jit** | AITER JIT C++/HIP (`aiter/jit/module_*.so`) | `mhc_*` (MLA), `opus_moe_sorting`, `topk_softplus`, `*_quant`, `add_rmsnorm_quant` |
| **CK** | Composable Kernel (`ck::…`) | `ck::kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_preshuffle` |
| **hipBLASLt** | hipBLASLt / Tensile (`Cijk_…`) | some dense projection shapes |
| **SGL-triton** | SGLang Triton (`layers/attention/dsv4/…`, fused ops) | `_paged_decode_split/reduce` (dsv4 unified-KV attn), fused rms/rope/silu/quant |
| **SGL-gluon** | SGLang Triton **gluon** dialect | `_gluon_deepgemm_fp8_paged_mqa_logits` (DSA indexer logits) |
| **SGL-jit** | SGLang jit_kernel C++/HIP (`jit_kernel/csrc/deepseek_v4/*.cuh`) | compressed flash decode (`flash_c128/c4`), indexer Q/rope/topk, norm+rope |
| torch / rocPRIM / RCCL | PyTorch native, rocPRIM sort, RCCL | elementwise/copy, MoE-sort primitives, TP/DP collectives |

### Cost by compute backend

| Backend | % of decode compute | time (ms) |
|---|---|---|
| **AITER — JIT C++/HIP** (`aiter::`) | 20.8% | 22.19 |
| **AITER — flydsl** (MoE experts + bf16 hgemm) | 20.0% | 21.34 |
| **AITER — asm/hsaco** (fp8/bf16 blockscale GEMM) | 16.3% | 17.41 |
| SGLang Triton | 12.5% | 13.38 |
| hipBLASLt (Tensile) | 10.9% | 11.67 |
| Composable Kernel (CK) | 8.4% | 9.02 |
| PyTorch native (`at::native`) | 5.7% | 6.14 |
| SGLang jit_kernel (C++/HIP) | 4.3% | 4.54 |
| RCCL | 0.6% | 0.64 |
| SGLang gluon (Triton) | 0.5% | 0.50 |
| rocPRIM | 0.1% | 0.07 |

> **AITER in its three forms (asm + flydsl + JIT) is ~57% of decode compute** — the
> dominant backend. Dense/MLA GEMM is split across AITER-asm, CK, and hipBLASLt;
> MoE experts and bf16 projections are AITER-flydsl; MLA compression, MoE
> routing/sort, and quant are AITER-JIT. SGLang's own code is the attention
> machinery: Triton for the dsv4 paged-decode attention and fused norm/rope/quant,
> C++/HIP jit_kernel for the DeepSeek Sparse Attention path, and one gluon kernel
> for the indexer logits.

## DeepSeek-V4 decode forward pass (algorithm → kernels → backend)

Per transformer layer, SGLang executes the following steps; each kernel is tagged
with its backend (see above).

1. **Input RMSNorm + FP8 quant** — normalize hidden state and quantize activations
   for the FP8 projection GEMMs
   [`_fused_rms_fp8_group_quant` **SGL-triton**, `dynamic_per_group_scaled_quant` **AITER-jit**].
2. **MLA (Multi-head Latent Attention)**
   - Down-project hidden → compressed latent, then up-project to per-head Q/K/V
     [`aiter::fp8gemm…BpreShuffle` **AITER-asm**, `ck…b_preshuffle` **CK**, `Cijk…` **hipBLASLt**,
     `hgemm_bf16…` **AITER-flydsl**].
   - Latent-compression math and output recombination
     [`mhc_pre_gemm_sqrsum`, `mhc_pre_big_fuse`, `mhc_post` — all **AITER-jit** (`module_mhc`)].
   - QK-norm + RoPE, write compressed KV to the paged cache
     [`_fused_qk_norm_rope_store`, `apply_rotary_emb_flat` **SGL-triton**;
     `fused_norm_rope_flashmla` **SGL-jit**].
   - Core attention over paged KV
     [`_paged_decode_split` → `_paged_decode_reduce` — **SGL-triton** (dsv4 unified-KV backend)].
3. **DeepSeek Sparse Attention (lightning indexer)** — a lightweight path that picks
   which past tokens each query attends to, keeping attention sub-quadratic:
   indexer-Q rope+hadamard+quant [`fused_q_indexer_rope_hadamard_quant` **SGL-jit**]
   → MQA logits [`_gluon_deepgemm_fp8_paged_mqa_logits` **SGL-gluon**]
   → top-k token select [`deepseek_v4_topk_transform` **SGL-jit**]
   → compressed flash decode [`flash_c128_decode`/`flash_c4_decode` **SGL-jit**].
4. **Post-attention RMSNorm + quant** [`add_rmsnorm_quant` **AITER-jit** (`module_rmsnorm_quant`)].
5. **MoE FFN**
   - Router/gating selects top-k experts per token [`topk_softplus` **AITER-jit** (`module_moe_topk`)].
   - Token sort/permute + mx-quant groups tokens by expert
     [`fused_mx_quant_moe_sort`, `opus_moe_sorting` **AITER-jit** (`module_moe_sorting_opus`)].
   - Expert GEMM1 gate/up (FP8×FP4) + activation + Expert GEMM2 down
     [`mfma_moe1_silu_mul_afp8_wfp4`, `mfma_moe2_afp8_wfp4` **AITER-flydsl** (2-stage `fmoe`);
     `_fused_clamp_silu_mul` **SGL-triton**] → combine [`moe_reduction` **SGL-triton**].
   - The always-on **shared expert** runs as dense GEMMs (AITER-asm/CK/flydsl; folded
     into the Dense-GEMM bucket).
6. **Residual adds** [`at::native` **torch**] and, across ranks, **TP/DP collectives**
   [`ncclDevKernel` **RCCL**].

## Cost by algorithm step

| Algorithm step | % of decode compute | time (ms) |
|---|---|---|
| Dense GEMM (proj + shared-expert) | 42.7% | 45.65 |
| MLA attention | 20.0% | 21.35 |
| MoE experts | 14.3% | 15.33 |
| MoE routing | 7.4% | 7.88 |
| Norm + quant | 6.7% | 7.20 |
| Elementwise / misc | 5.7% | 6.09 |
| DSA sparse-attn indexer | 2.4% | 2.55 |
| Communication (TP/DP) | 0.6% | 0.64 |
| Other | 0.2% | 0.21 |
| **Total** | **100.0%** | **106.90** |

## Per-function cost (attached to algorithm step)

### Dense GEMM (proj + shared-expert) — 42.7% of decode compute

| Function (kernel) | % total | time (ms) | launches |
|---|---|---|---|
| AITER fp8 blockscale (BpreShuffle) <br/>`aiter::fp8gemm_bf16_blockscale_BpreShuffle_32x128` | 15.87% | 16.966 | 1,402 |
| hipBLASLt (Tensile Cijk) <br/>`Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x16x128_MI16x1…` | 10.92% | 11.673 | 372 |
| CK blockscale (b_preshuffle) <br/>`void ck::kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_preshu…` | 8.44% | 9.017 | 608 |
| bf16 hgemm (splitK) <br/>`hgemm_bf16_16x64x128x5_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 7.48% | 7.992 | 1,092 |

### MLA attention — 20.0% of decode compute

| Function (kernel) | % total | time (ms) | launches |
|---|---|---|---|
| core paged decode (split-KV) <br/>`_paged_decode_split_kernel` | 3.89% | 4.163 | 366 |
| MLA compress: mhc_pre_big_fuse <br/>`void aiter::mhc_pre_big_fuse_kernel<bool _Accum, int, ELi4E, 2, …` | 3.46% | 3.702 | 732 |
| MLA compress: mhc_pre_gemm_sqrsum <br/>`_ZN5aiter26mhc_pre_gemm_sqrsum_kernelIDF16bLi4ELi64ELi16ELi128EE…` | 3.39% | 3.627 | 732 |
| QK-norm + RoPE + KV store <br/>`_fused_qk_norm_rope_store_kernel` | 2.37% | 2.528 | 732 |
| flash MLA decode (compressed) <br/>`void (anonymous namespace)::fused_norm_rope_flashmla<float, ((an…` | 2.33% | 2.492 | 918 |
| core paged decode (reduce) <br/>`_paged_decode_reduce_kernel` | 2.29% | 2.447 | 366 |
| MLA output: mhc_post <br/>`_ZN5aiter15mhc_post_kernelIDF16bLi4ELi4ELi1024ELb0EEEvPT_S2_S2_P…` | 2.24% | 2.393 | 732 |

### MoE experts — 14.3% of decode compute

| Function (kernel) | % total | time (ms) | launches |
|---|---|---|---|
| expert GEMM1 gate/up (FP8xFP4) <br/>`mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async…` | 7.04% | 7.528 | 366 |
| expert GEMM2 down (FP8xFP4) <br/>`mfma_moe2_afp8_wfp4_bf16_cshuffle_t32x256x256_vscale_fix3_fp4opt…` | 5.86% | 6.266 | 366 |
| expert activation (clamp/silu/mul) <br/>`_fused_clamp_silu_mul_kernel_BLOCK_SIZE_N_4096_QUANT_BLOCK_SIZE_…` | 1.06% | 1.129 | 366 |
| expert output reduction <br/>`moe_reduction_kernel_plain_bf16_topk6_md7168` | 0.38% | 0.406 | 122 |

### MoE routing — 7.4% of decode compute

| Function (kernel) | % total | time (ms) | launches |
|---|---|---|---|
| token sort/permute <br/>`void aiter::opus_moe_sorting_entry<aiter::MoeSortingKernel<aiter…` | 3.07% | 3.279 | 610 |
| mx-quant + MoE sort <br/>`_ZN5aiter30fused_mx_quant_moe_sort_kernelIDF16bDB8_Li256ELi32EEE…` | 2.23% | 2.386 | 366 |
| gating (softplus top-k) <br/>`void aiter::topk_softplus_kernel_opt<hip_bfloat16, hip_bfloat16,…` | 2.07% | 2.215 | 348 |

### Norm + quant — 6.7% of decode compute

| Function (kernel) | % total | time (ms) | launches |
|---|---|---|---|
| per-group activation quant <br/>`_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32EL…` | 3.15% | 3.369 | 912 |
| fused RMSNorm + fp8 group-quant <br/>`_fused_rms_fp8_group_quant_kernel` | 2.45% | 2.615 | 732 |
| add-RMSNorm + quant <br/>`_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi32ELb0ELb0…` | 1.14% | 1.214 | 372 |

### Elementwise / misc — 5.7% of decode compute

| Function (kernel) | % total | time (ms) | launches |
|---|---|---|---|
| copy / cast <br/>`void at::native::elementwise_kernel_manual_unroll<128, 8, at::na…` | 2.88% | 3.080 | 996 |
| fill <br/>`void at::native::vectorized_elementwise_kernel<8, at::native::Fi…` | 1.57% | 1.676 | 823 |
| elementwise add <br/>`void at::native::vectorized_elementwise_kernel<8, at::native::CU…` | 1.05% | 1.120 | 414 |
| elementwise (other) <br/>`void at::native::index_elementwise_kernel<128, 4, at::native::gp…` | 0.20% | 0.209 | 78 |

### DSA sparse-attn indexer — 2.4% of decode compute

| Function (kernel) | % total | time (ms) | launches |
|---|---|---|---|
| indexer Q rope+hadamard+quant <br/>`(anonymous namespace)::fused_q_indexer_rope_hadamard_quant_kerne…` | 0.81% | 0.864 | 180 |
| MQA logits (deepgemm) <br/>`_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle` | 0.47% | 0.499 | 180 |
| top-k token select <br/>`(anonymous namespace)::deepseek_v4_topk_transform_kernel((anonym…` | 0.45% | 0.485 | 198 |
| KV compress tail fill <br/>`_fill_compress_tail_kernel` | 0.39% | 0.420 | 186 |
| indexer norm+rope <br/>`void (anonymous namespace)::fused_norm_rope_indexer<float, ((ano…` | 0.27% | 0.284 | 180 |

### Communication (TP/DP) — 0.6% of decode compute

| Function (kernel) | % total | time (ms) | launches |
|---|---|---|---|
| NCCL/RCCL collective <br/>`ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 0.60% | 0.638 | 6 |

### Other — 0.2% of decode compute

| Function (kernel) | % total | time (ms) | launches |
|---|---|---|---|
| void rocprim::ROCPRIM_400200_NS::detail::trampolin <br/>`void rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel<rocpr…` | 0.06% | 0.067 | 24 |
| void at::native:: <br/>`void at::native::(anonymous namespace)::indexSelectSmallIndex<c1…` | 0.03% | 0.034 | 6 |
| memcpy_triton_kernel <br/>`memcpy_triton_kernel` | 0.03% | 0.033 | 12 |
| void at::native::reduce_kernel<512, 1, at::native: <br/>`void at::native::reduce_kernel<512, 1, at::native::ReduceOp<long…` | 0.02% | 0.023 | 6 |
| host::compress::plan_compress_decode_kernel <br/>`host::compress::plan_compress_decode_kernel(host::compress::Deco…` | 0.02% | 0.021 | 12 |
| _init_compressed_attn_metadata_kernel <br/>`_init_compressed_attn_metadata_kernel` | 0.02% | 0.017 | 6 |
| _v4_paged_decode_indices_kernel <br/>`_v4_paged_decode_indices_kernel` | 0.02% | 0.017 | 6 |

## Key observations

- **AITER is the dominant backend — ~57% of decode compute** across its three
  forms: JIT C++/HIP (20.8%, MLA `mhc_*` + routing/sort + quant), flydsl DSL codegen
  (20.0%, MoE experts + bf16 `hgemm`), and hand-written assembly (16.3%, fp8/bf16
  blockscale GEMM). SGLang's own kernels handle attention: Triton (12.5%) for the
  dsv4 paged-decode + fused norm/rope/quant, C++/HIP jit_kernel (4.3%) for the
  DeepSeek Sparse Attention path, and one gluon kernel for the indexer logits.
  hipBLASLt (10.9%) and CK (8.4%) cover the remaining dense-GEMM shapes.
- **Dense linear projections dominate decode (42.7%)**, far above the MoE experts
  (14.3%). At decode batch sizes each routed expert sees very few tokens, so the
  expert GEMMs are small; the always-run MLA q/kv/o projections + the shared
  expert are the real matmul cost. This is the opposite of the prefill/large-batch
  regime where MoE experts dominate.
- **All dense GEMM is weight-preshuffled** — `aiter::fp8gemm_bf16_blockscale_BpreShuffle`
  (15.9%) + `ck…b_preshuffle` (8.4%), with hipBLASLt `Cijk` (10.9%) and bf16 `hgemm`
  splitK (7.5%) for the remaining shapes. (vLLM uses non-preshuffled `GridwiseGemmMultiD_ABScale`
  here — see `vllm_vs_sglang_decode_kernels`.)
- **MLA attention is 20%, fragmented into many small kernels**: the `mhc_*`
  compression helpers alone are ~9% (`mhc_pre_big_fuse` 3.5% + `mhc_pre_gemm_sqrsum`
  3.4% + `mhc_post` 2.2%), core paged decode ~6.2%, and QK-norm/RoPE ~4.7%.
- **DeepSeek Sparse Attention is cheap at decode (2.4%)** — the lightning indexer
  (MQA-logits + top-k select + compressed flash) adds little per-token cost, which
  is its design intent; its payoff is bounding attention cost on long context.
- **MoE routing overhead (7.4%) is ~half the expert compute (14.3%)** — token
  sort/permute (3.1%), mx-quant+sort (2.2%), and gating (2.1%) are a large tax
  relative to the actual expert matmuls at decode batch.
- **Quantization is pervasive (6.7%)**: a per-group activation quant precedes
  essentially every GEMM (`dynamic_per_group_scaled_quant` 3.2%, plus fused
  RMSNorm+quant variants), reflecting the FP8/FP4 mixed-precision recipe.
- **Communication looks tiny (0.6%)** only because this is a per-rank graph-capture
  trace; steady-state TP=8/DP-attention all-reduce/gather cost is not represented
  here (it lands outside the captured decode graph).

## Reproduce

```
# in xguo-sgl, server launched with --enable-profile-cuda-graph + SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE=1
python sglang_dsv4_algo_analysis.py --md docs/sglang_dsv4_algorithm_analysis.md \
    sgl_capture/graph_capture_profile/cuda_graph_capture-DecodeCudaGraphRunner-TP-0.json.gz
```
See `docs/sglang_kernel_profiling.md` for how the capture trace is produced.
