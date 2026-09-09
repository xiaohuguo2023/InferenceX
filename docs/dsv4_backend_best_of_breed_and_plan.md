# DeepSeek-V4 on MI355X — Best-of-breed per component + improvement plan (vLLM / SGLang / Atom)

Basis: the conc=64 1k/1k decode function-by-function comparison
(`vllm_sglang_atom_dsv4_conc64_compare.md`). Per-function **absolute time = function%
× real TPOT**, valid for the **runtime** traces (vLLM TPOT 33.8 ms, Atom TPOT 23.4 ms).
SGLang is a graph-capture trace, so its per-function *ms* are directional; its kernel
*identities* are solid.

## Fastest implementation per component

| Component | vLLM (ms/tok) | Atom (ms/tok) | Fastest | Winning kernel |
|---|---|---|---|---|
| **Dense/MLA-proj GEMM** | 7.00 | **3.21** | **Atom** | CK `kernel_gemm_xdl…b_preshuffle` + hipBLASLt (weight-preshuffled) |
| **MoE expert GEMM** | 6.87 | **3.75** | **Atom** | CK `kernel_moe_mxgemm_2lds` (GridwiseMoeGemmMX, B-preshuffled) |
| **Attention (MLA/DSA)** | 6.26 | **5.95** | ~tie (Atom) | shared `aiter::mhc_*`; Atom `_sparse_attn_ragged_varlen_triton` |
| **MoE routing/sort** | 3.01 | **0.94** | **Atom** | `ck_tile::MoeSortingMultiPhase` + `aiter::mxfp4_moe_sort` |
| **Quantization** | 1.15 | **0.73** | **Atom** | same `aiter::dynamic_per_group_scaled_quant` (fewer launches) |
| **Communication** | **2.50** | 3.68 | **vLLM** | `cross_device_reduce_2stage` custom AR only (no RCCL) |
| **Elementwise** | 5.62 | 4.88 | **SGLang** | most-fused (SGLang 6.1% share vs 16–21%); Atom hurt by `CatArrayBatchedCopy` |
| **MLA compression** | — | — | tie | universal `aiter::mhc_pre_gemm_sqrsum/pre_big_fuse/post` |

**Summary:** Atom has the fastest **GEMM** (dense + MoE), **routing**, and **quant**;
vLLM has the fastest **communication** (custom-AR-only vs Atom's RCCL-heavy path); SGLang
has the best **operator fusion** (lowest elementwise). MLA compression is identical
(AITER `mhc_*`) everywhere.

## Why Atom's GEMM wins
- **Dense:** weight-**preshuffled** CK GEMM avoids the runtime B-layout conversion vLLM
  pays (vLLM uses non-preshuffled `GridwiseGemmMultiD_ABScale`). SGLang also preshuffles.
- **MoE:** Atom runs experts on **Composable Kernel `kernel_moe_mxgemm_2lds`**, which at
  conc64 beats the **AITER `mfma_moe` flydsl** path vLLM and SGLang use (3.75 vs 6.87 ms/tok).
  This is the single largest per-component gap.

---

## Improvement plan

### vLLM — largest headroom (~10 ms/tok = ~30% TPOT)
1. **Weight-preshuffled dense GEMM** — adopt CK `b_preshuffle` / AITER `fp8gemm…BpreShuffle`
   (as SGLang/Atom). Dense 7.0 → ~3.2 ms/tok. *(optimization Target #7)*
2. **CK MoE MXGEMM for experts** — replace AITER `mfma_moe` flydsl with CK
   `kernel_moe_mxgemm_2lds` (Atom's path). MoE 6.9 → ~3.8 ms/tok.
3. **Faster MoE routing/sort** — adopt `ck_tile::MoeSortingMultiPhase` (Atom). Routing
   3.0 → ~0.9 ms/tok.
4. **Fuse activation quant into the GEMM prologue** — 1.15 → ~0.7 ms/tok.
5. Keep vLLM's already-best **custom-AR-only communication**.
- **Projected TPOT:** 33.8 → ~**24 ms** (≈ +40% throughput), i.e. reach Atom-class GEMM.

### Atom — already fastest per token; fix comm + elementwise (~3–4 ms/tok)
1. **Communication** — make the AITER `cross_device_reduce_2stage` custom AR the primary
   path and drop the RCCL `ncclDevKernel` (12% of decode). 3.7 → ~2.5 ms/tok (vLLM-class).
2. **Reduce concat/elementwise** — `CatArrayBatchedCopy` is ~10% of decode; fuse the
   concatenations / avoid materializing. Elementwise 4.9 → ~3 ms/tok.
3. TTFT already best (217 ms) — no action.
- **Projected TPOT:** 23.4 → ~**20 ms**.

### SGLang — best fusion & peak throughput; close the MoE and TTFT gaps
1. **Evaluate CK MoE MXGEMM** (Atom's `kernel_moe_mxgemm_2lds`) vs its current AITER
   `mfma_moe` flydsl experts — same opportunity as vLLM #2.
2. **Keep** its wins: preshuffled dense GEMM + aggressive norm/rope/quant fusion (lowest
   elementwise) + custom-AR communication.
3. **TTFT** — the peak-throughput conc64 recipe has very high TTFT (~1400 ms); the
   `--enable-prefill-delayer` / lower-latency variant already hits TPOT ~21 ms at 652 tok/s.
   Tune toward that when TTFT matters.

## Cross-backend "dream stack" (best kernel per component)
| Component | Adopt from | Kernel |
|---|---|---|
| Dense GEMM | Atom/SGLang | CK / AITER **preshuffled** blockscale |
| MoE experts | Atom | CK `kernel_moe_mxgemm_2lds` |
| MoE routing | Atom | `ck_tile::MoeSortingMultiPhase` |
| Quant | Atom | `aiter::dynamic_per_group_scaled_quant` (fused) |
| Communication | vLLM/SGLang | `cross_device_reduce_2stage` custom AR (no RCCL) |
| Elementwise/fusion | SGLang | fused norm+rope+quant, no concat materialization |
| MLA compression | any | `aiter::mhc_*` (already universal) |

A backend combining **Atom's GEMM + routing + quant**, **vLLM/SGLang's custom-AR comm**,
and **SGLang's fusion** would project to **~18–20 ms/tok TPOT** — below every current
backend.

## Caveats
- vLLM & Atom per-function ms are anchored to their **real conc64 TPOT** (comparable);
  SGLang per-function ms are **directional** (graph-capture proportions).
- Traces/serving points are the best-throughput conc64 recipe per backend; other recipe
  variants trade TTFT vs TPOT (esp. SGLang).
- Projections assume a clean kernel swap with no new overhead; each should be A/B-measured.
  The GEMM/MoE swaps are the highest-confidence (direct kernel-time deltas); fusion and
  comm changes need validation.
