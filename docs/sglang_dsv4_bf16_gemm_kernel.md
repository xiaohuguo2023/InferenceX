# SGLang DeepSeek-V4 (MI355X) — which AITER kernel serves the bf16/fp16 GEMM

## Answer
SGLang routes every **unquantized bf16 (a16w16) GEMM** in DSV4 through **AITER's tuned-GEMM
dispatcher `aiter.tuned_gemm.tgemm.mm`** (compiled op **`aiter::gemm_a16w16`**). That dispatcher
then auto-selects, **per shape/tune**, among three actual kernels — all confirmed present in the
SGLang DSV4 trace:

| Dispatched kernel | Library | Trace evidence (calls) |
|---|---|---|
| `Cijk_Alik_Bljk_BBS_BH_…` | **hipBLASLt (Tensile)** — default `hipb_mm`, solidx 0 | many (≈2.6k across tiles) |
| `hgemm_bf16_*` (e.g. `32x64x128x5_SPK4_…`) | **AITER flydsl** (`flydsl_hgemm`) | ≈1.3k across tile variants |
| `aiter::_gemm_a16w16_asm` | **AITER ASM** (`gemm_a16w16_asm`) | 668 |
| (skinny/small-M) `wvSplitK` | AITER skinny GEMM (`skinny_gemm`, solidx 2) | shape-gated (not prominent here) |

Top-level dispatcher op `aiter::gemm_a16w16` appears **2,912×**; the GPU work underneath splits
across the hipBLASLt / flydsl / asm kernels above.

## Call path (source)
1. **Linear layers** — `UnquantizedLinearMethod.apply` →
   `python/sglang/srt/layers/quantization/unquant.py:202`:
   ```python
   elif _use_aiter and type(layer.weight.data) is torch.Tensor:
       return tgemm.mm(x, layer.weight, bias, otype=x.dtype)
   ```
2. **DSV4 MoE router/gate** (the main bf16 GEMM in DSV4) —
   `python/sglang/srt/models/deepseek_v2.py:528`:
   ```python
   logits = aiter_dsv3_router_gemm(hidden_states, self.weight)
   ```
   → `python/sglang/srt/layers/rocm_linear_utils.py:14`:
   ```python
   from aiter.tuned_gemm import tgemm
   return tgemm.mm(hidden_states, weight.detach(), otype=hidden_states.dtype)
   ```
3. Both funnel into `aiter/tuned_gemm.py::mm` → compiled `gemm_a16w16` (op annotation
   `aiter/tuned_gemm.py(260): gemm_a16w16`).

## How `tgemm.mm` picks the kernel (`aiter/tuned_gemm.py`)
- Reads a per-shape tuned config (`bf16_tuned_gemm.csv`); the row's `libtype` selects the backend:
  - `hipblaslt` → `hipb_mm(inp, weights.t(), solidx, …)` → **`Cijk_*`** Tensile kernel (**default**, solidx 0).
  - `asm` → **`gemm_a16w16_asm`** (`aiter/ops/gemm_op_a16w16.py:60`) → `aiter::_gemm_a16w16_asm`.
  - `flydsl` → `flydsl_hgemm` (`aiter.ops.flydsl.gemm_kernels`) → **`hgemm_bf16_*`**.
  - `skinny` → `skinny_gemm` → **`wvSplitK`** (small-M shapes, solidx 2).
  - untuned/gfx-fallback → `F.linear` (torch).
- The `Bf16GemmBackend` enum in `unquant.py` (`AUTO` / `CUTEDSL`) only matters on NVIDIA — `CUTEDSL`
  requires SM100 (Blackwell); on MI355X the AITER path (`_use_aiter`) is taken and `tgemm.mm` is used.

## Where bf16 GEMM appears in DSV4
DSV4 is FP4-MoE + FP8 dense/MLA-proj, so bf16 (a16w16) GEMM is a small slice — chiefly the
**MoE router/gate** (`aiter_dsv3_router_gemm`) and any remaining unquantized linears. These are the
GEMMs that land on `aiter::gemm_a16w16` and thus on hipBLASLt / flydsl-hgemm / asm-a16w16.

## Source of evidence
- Trace: `profile_dsv4_fp4_sglang_tp8-dpatrue_conc256_mi355x_PROFILE.trace.json.gz` (SGLang
  Mechanism-B graph-capture trace, TP8 DP-attn conc256).
- Source: `~/work/sglang` (unquant.py, rocm_linear_utils.py, deepseek_v2.py) and
  `~/work/aiter/aiter/tuned_gemm.py`.

## Bottom line
**The AITER kernel for SGLang DSV4 bf16 GEMM is `aiter::gemm_a16w16` (the `aiter.tuned_gemm.tgemm.mm`
dispatcher)**, which resolves per shape to **hipBLASLt `Cijk_`** (default), **AITER flydsl `hgemm_bf16`**,
or **AITER ASM `gemm_a16w16_asm`** — all three are active in the DSV4 trace, with hipBLASLt the most common.
