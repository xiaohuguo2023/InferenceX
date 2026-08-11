# Kimi-K3 FP4 — conc-48 DSpark decode-step GPU kernel profile (MI355X)

| Field | Value |
|---|---|
| Model | Kimi-K3 FP4 (mxfp4 / a8w4 MoE, fp8 attn, **fp8 KV**) |
| Hardware | 8× AMD Instinct MI355X (gfx950), **TP=8** |
| Engine | vLLM ROCm nightly `v0.26.1rc1.dev306+gcb8104839`, torch 2.11, ROCm 7.2.3 |
| Spec decode | **DSpark**, `num_speculative_tokens=2` (draft forced causal), `ROCM_AITER_MLA` target+draft |
| Serve config | mandated recipe: `max_num_seqs=64`, `MNBT=16384`, `gpu_mem=0.95`, `FULL_AND_PIECEWISE`, KV pinned 32 GiB |
| Cudagraph | capture-size fix applied → **28 FULL** decode graphs (adds {12,36}; conc-4/12 bubble fixed) |
| Workload | ISL 68,089 (63,911-tok cached prefix / 4,089 suffix), single wave of 48 reqs, OSL 900 |
| Profile window | torch profiler, `start_profile`→3 s→`stop_profile` taken ~45 s in (steady-state **pure decode**) |
| Trace | rank0 (`dp0_pp0_tp0`); TP=8 is symmetric so rank0 is representative |

> **Isolation method:** the stage analyzer splits prefill/decode on the `execute_context(ctx)`
> annotation (`ctx>0 ⇒ prefill`), which cannot separate DSpark's *large-KV-context* decode from
> prefill. So decode is isolated at the **workload** level: a single wave (request-count == concurrency)
> prefills once, then all 48 requests decode together with no further prefills — the window is then
> **100% decode** by construction. Confirmed: analyzer reports 0.0 ms prefill / 3134 ms decode.

## Overall — steady-state decode

- **100% decode / 0% prefill** window (3.0 s wall).
- Aggregate rank0 GPU-kernel time: **3134.1 ms**, **174,892** kernel launches, over **46** decode steps.
- ≈ **68 ms of GPU-kernel time per decode step** (consistent with conc-48 ITL p50 ≈ 77 ms; the
  ~9 ms gap is host/launch + all-reduce sync not on the GPU timeline).
- Each decode step is a DSpark spec **verify** of M = 48×(1+2) = **144** tokens.

## Decode — kernel time by category

| Category | time (ms) | % | launches |
|---|---|---|---|
| Other (elementwise / copy / misc) | 1161.8 | 37.1% | 50,968 |
| **GEMM — MoE experts** (AITER ASM FP8×FP4) | 782.3 | 25.0% | 4,232 |
| **GEMM — dense/linear** (hipBLASLt + skinny) | 626.2 | 20.0% | 39,238 |
| **Communication** (TP=8 custom all-reduce) | 254.2 | 8.1% | 13,432 |
| Memory / elementwise | 156.8 | 5.0% | 33,994 |
| Quantization (per-group FP8 act quant) | 87.3 | 2.8% | 19,826 |
| MoE routing / sorting | 63.4 | 2.0% | 12,696 |
| Normalization | 1.0 | 0.0% | 230 |
| Activation | 1.0 | 0.0% | 230 |
| RoPE | 0.2 | 0.0% | 46 |

## Decode — by kernel backend (cross-check, backend_breakdown.py)

| Backend | time (ms) | % |
|---|---|---|
| AITER flydsl (DSL-gen MFMA) — MoE experts | 1014.9 | 32.5% |
| AITER JIT C++/HIP (`aiter::`) — MLA, all-reduce, rmsnorm, quant, sort | 818.4 | 26.2% |
| other / unclassified (torch elementwise/copy) | 756.0 | 24.2% |
| hipBLASLt (Tensile) — dense/linear GEMM | 384.5 | 12.3% |
| PyTorch native (`at::native`) | 132.8 | 4.2% |
| RCCL | 10.9 | 0.3% |
| AITER asm (hsaco) | 9.2 | 0.3% |
| rocPRIM | 0.5 | 0.0% |

## Decode — top kernels (rank0)

| time (ms) | % | launches | kernel |
|---|---|---|---|
| 782.3 | 25.0% | 4,232 | `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256 …` — **MoE stage-1 GEMM+SiLU (FP8×FP4)** |
| 297.0 | 9.5% | 1,104 | `aiter::mla_a8w8_qh16_qseqlen4_gqaratio16_v3_ps` — **MLA decode/verify attn (qlen 4)** |
| 291.5 | 9.3% | 4,232 | `opus_moe_stage2_a8w4_decode_kernel_gfx950 …` — **MoE stage-2 GEMM (a8w4)** |
| 243.3 | 7.8% | 13,340 | `aiter::cross_device_reduce_2stage<…8…>` — **TP=8 custom all-reduce** |
| 77.0 | 2.5% | 230 | `aiter::mla_a8w8_qh16_qseqlen2_gqaratio16_ps` — MLA attn (qlen 2) |
| 50.5 | 1.6% | 4,232 | `aiter::grouped_topk_kernel` — MoE routing top-k |
| 39.3 | 1.3% | 8,694 | `aiter::add_rmsnorm_quant_kernel<…32…>` — fused add+RMSNorm+quant |
| 25.3 | 0.8% | 4,232 | `aiter::opus_moe_sorting_entry …` — MoE token sort |
| 18.8 | 0.6% | 4,232 | `aiter::dynamic_per_group_scaled_quant_kernel` — pre-GEMM FP8 act quant |

## Key observations (Kimi-K3 specific)

- **MoE is the dominant compute:** stage-1 `mfma_moe1_silu_mul` (25.0%) + stage-2 `opus_moe_stage2`
  (9.3%) = **~34%** of decode GPU time, all on the AITER assembly **FP8×FP4** path. This is the
  irreducible expert-GEMM cost at 144 tokens/step.
- **MLA attention (spec verify) ≈ 12%:** `mla_a8w8_qh16_qseqlen4_v3_ps` (9.5%) is the main decode
  attention; the `qseqlen4` reflects DSpark's padded query length. `qseqlen2` (2.5%) is the residual
  non-padded path. This is the kernel fed by `get_mla_metadata_v1` — now on the captured FULL-graph
  fast path at every mandated concurrency after the {12,36} capture-size fix.
- **TP=8 all-reduce ≈ 8%:** `cross_device_reduce_2stage` (243 ms, 13k launches) — the custom AITER
  all-reduce, well below prefill's share (decode activations are small).
- **Dense/linear GEMM ≈ 20% (hipBLASLt 12.3%):** this bucket still contains the **untuned
  `6288×7168` shapes that fall back to `torch solution:0`** (N%64=16 → no ASM tile). Tuning these
  (task #50) is the main remaining decode-GEMM win.
- **"Other"/elementwise is the single largest bucket (~37% time, ~51k launches) and dominates launch
  count** — the decode step is partly **launch-bound** on many tiny elementwise/copy/quant kernels.
  Fusion (or fewer, larger kernels) is the biggest structural opportunity if per-step decode latency
  matters beyond GEMM tuning.

---
*Tables from `analyze_dsv4_trace.py`; backend split from `backend_breakdown.py`. The analyzer's built-in
header/observations are DSV4 boilerplate and have been replaced here with the correct Kimi-K3 run metadata
and findings. Trace: `kimik3_traces_c48/dp0_pp0_tp0_…pt.trace.json.gz` (rank0, 215 MB).*
