# DeepSeek-V4-Pro on MI355X (vLLM) — Optimization Targets

Evidence base: per-stage torch-profiler breakdowns at concurrency 4 / 32 / 256
(`docs/dsv4_fp4_mi355x_conc{4,32,256}_1k1k_profile.md`), the serving `server.log`
tuning warnings, and a cross-backend kernel comparison against the published
`mi355x/atom` DeepSeek-V4 trace (`InferenceX-trace-storage`).

Config profiled: `deepseek-ai/DeepSeek-V4-Pro`, TP=8, FP4 MoE + FP8 attn, FP8 KV,
AITER MoE backend, cudagraph FULL_AND_PIECEWISE, ISL=OSL=1024. Decode dominates
the steady state (83–95% of GPU time at these shapes).

Kernels at ~4–5 µs average duration are **launch-bound** (a HIP kernel launch is
~5 µs), so their cost is dominated by count, not work — fusion / launch reduction
is the lever, not faster math.

---

## Performance headroom (why this matters)

Pulled from the public dashboard API (`/api/v1/benchmarks?model=DeepSeek-V4-Pro`,
DB dump 2026-07-13). dsv4 1k/1k, **throughput per GPU (tok/s/GPU)**.

**Same hardware (MI355X), by backend, matched concurrency — the profiled vLLM path is the slowest AMD backend:**

| conc | vLLM (profiled) | SGLang | Atom |
|---|---|---|---|
| 32 | 288 | **509** | 417 |
| 256 | 855 | **1722** | 1096 |
| 512 | 1128 | **2389** | 1396 |
| 2048 | — | **3675** | 3728 |

→ **vLLM trails SGLang by ~1.8–2.1× and Atom by ~1.2–1.4× on the *same* GPU.** The
single biggest, cheapest win is closing the vLLM↔SGLang *software* gap — not hardware.
The kernel-level targets below are exactly where that ~2× is hiding.

**Cross-vendor peak tput/GPU (best backend each, fp4 unless noted):**

| MI355X (sglang) | B200 (vllm) | B300 (sglang) | GB300 (dynamo-trt) |
|---|---|---|---|
| 3,675 | 7,211 (1.96×) | 10,515 (2.86×) | 11,846 (3.22×) |

→ MI355X's *best* backend is ~0.5× B200 / ~0.31× GB300 at peak. Caveat: the NV peaks
are extreme-batch disagg configs (conc 8192, multi-second TTFT); at an interactive
point (median TPOT ≤ 25 ms) the gap is MI355X-sglang 1,117 vs B300-trt 4,590 vs
GB300 7,965 — same direction. **No NV V4 kernel traces exist**, so the cross-vendor
gap can't be traced kernel-vs-kernel; the within-MI355X vLLM↔SGLang gap can, and the
targets below explain it.

---

## Target #0 — close the vLLM↔SGLang gap on MI355X (HIGHEST leverage)
SGLang already achieves ~2× the profiled vLLM throughput on identical silicon, so
~2× is provably on the table in software. Either serve dsv4 on SGLang, or port
SGLang's wins into the vLLM path — which decompose into targets #1–#6 below
(GEMM tuning, quant/elementwise fusion, MLA/MoE launch reduction). Treat #1–#6 as
"how vLLM catches up to SGLang," with SGLang as the reference implementation.

### What SGLang/Atom do that vLLM's recipe doesn't (traced/confirmed this session)
Profiling SGLang (`lmsysorg/sglang-rocm:v0.5.14-…-20260706`, TP=8, its recipe) and
diffing against the vLLM traces + the published `mi355x/atom` V4 trace surfaced
three concrete levers, each mapping to a target below:

1. **Weight-preshuffled GEMM (CONFIRMED across BOTH faster backends).** SGLang runs
   `aiter::fp8gemm_bf16_blockscale_BpreShuffle_128x128` and
   `ck::…_multi_d_blockscale_b_preshuffle`; Atom runs the same `b_preshuffle` CK
   GEMM. vLLM runs the **non-preshuffled** `GridwiseGemmMultiD_ABScale` and eats a
   runtime B-layout conversion. → **Target #7, now high-confidence.**
2. **Tuned dsv4 GEMM configs shipped in-image.** SGLang's AITER loads
   `dsv4_bf16_tuned_gemm.csv` (+ model_configs). vLLM's image logged **1,488
   `not found tuned config … use default config!`** warnings. → **Target #1.**
3. **DP-attention + two-batch-overlap.** SGLang's TP=8 recipe uses
   `--enable-dp-attention --enable-two-batch-overlap --enable-prefill-delayer`;
   the vLLM config is **pure TP** (`dp-attn: false`). Two-batch-overlap hides comm
   behind compute — directly attacking the all-reduce cost. → **new Target #8.**

> Profiling caveat: SGLang decode is CUDA-graph-captured (kernels hidden inside the
> replayed graph → the DECODE trace is ~83% memcpy/launch), and the EXTEND window was
> dominated by DP-attention NCCL waits. So the SGLang *time-%* breakdown isn't usable;
> the **kernel *names*** (above) are, and they corroborate #1/#7/#8.

### Target #8 — DP-attention + two-batch-overlap in the vLLM recipe
vLLM exposes `--data-parallel-size` (DP-attention) via the existing `DP_ATTENTION`
knob, but the published `dsv4-fp4-mi355x-vllm` config leaves it off. SGLang's TP=8
win comes substantially from DP-attention + two-batch-overlap. **Action:** A/B the
vLLM recipe with `DP_ATTENTION=true` + comm/compute overlap at conc ≥ 64.

---

## Prioritized targets

### 1. Tune the AITER a8w8 block-scale GEMM (HIGH impact, LOW effort)
`server.log` emits **1,488** `not found tuned config in a8w8_blockscale_tuned_gemm.csv,
will use default config!` warnings across ~62 distinct shapes — every
N/K ∈ {768/7168, 7168/384, 2048/7168} at M = 1…8192. The a8w8 block-scale GEMM is
1.1–2.4% of decode and runs on **default (untuned) tiles**.
- **Action:** run AITER's offline GEMM tuner for these exact M/N/K on gfx950 and
  ship the populated `a8w8_blockscale_tuned_gemm.csv` in the image.
- **Why easy:** offline, no model/code change; also silences 8× (bf16) config
  misses. Pure win on the dense/MLA projection path.

### 2. Fuse activation quant into GEMM prologues (HIGH impact, MED effort)
`aiter::dynamic_per_group_scaled_quant` = **2.25M launches**, ~4.1 µs each,
3.2% of decode — a *separate* kernel before essentially every GEMM.
- **Action:** fuse the per-group FP8 activation quant into the consuming GEMM's
  prologue (or the producing op's epilogue) so the quant is not its own launch.
- **Why:** classic launch-bound epilogue/prologue fusion; removes millions of
  tiny dispatches.

### 3. Cut elementwise/copy launch storm (HIGH impact, MED effort)
Memory/elementwise is the **largest non-attention decode bucket by time (~22%)**
and **by far the largest by launch count (~12M, ~5 µs each)** — `direct_copy`,
`vectorized_elementwise` (dtype casts, adds).
- **Action:** lean harder on torch.compile fusion for the decode graph, remove
  redundant bf16↔fp8/fp32 conversions, fuse residual-adds into adjacent kernels.
- **Why:** biggest single time bucket outside attention/GEMM; almost pure
  overhead at decode batch sizes.

### 4. Reduce MLA kernel fragmentation (MED–HIGH impact, MED effort)
MLA decode is split into three helpers each launched ~1M times, ~4.5–5.3 µs:
`aiter::mhc_pre_gemm_sqrsum`, `mhc_pre_big_fuse`, `mhc_post` — plus DSA
`_sparse_attn_decode_partial` + `_decode_reduce` + `_save_partial_states` and
`_fused_kv_compress_norm_rope_insert`. Attention (MLA/DSA) is 17–21% of decode.
- **Action:** fuse the `mhc_*` pre/post helpers into the MLA attention kernel and
  merge the sparse-attn partial+reduce passes where possible.
- **Why:** many small launches on the critical decode path.

### 5. Trim MoE routing/sort overhead at low concurrency (MED impact, MED effort)
MoE routing is **12.5% of decode at conc4**, falling to 6.5% at conc256 — it is
overhead that dominates when token counts per expert are small. Multiple passes:
`opus_moe_sorting` (multi-phase P0/P23), `fused_mx_quant_moe_sort`,
`topKPerRowDecode`, `topkGatingSoftplusSqrt`.
- **Action:** collapse the multi-phase sort, fuse gating+topk+sort, or skip
  sorting below a token threshold. Targets **low-latency / low-conc** serving.

### 6. Overlap TP=8 all-reduce (MED impact, HIGH effort) — concurrency- & stage-dependent
Custom all-reduce (`cross_device_reduce_2stage`) grows **4.4% → 8.8%** of decode
from conc4 → conc256, and is **~23% of prefill** (large activation tensors).
- **Action:** overlap AR with compute (comm/compute pipelining), and evaluate
  QuickReduce / one-shot AR for the dominant message sizes. Matters most for
  high-throughput decode and for prefill TTFT.

### 7. Adopt weight-preshuffled CK GEMM (INVESTIGATE — cross-backend lead)
The published **Atom** V4 build uses `ck::…_blockscale_b_preshuffle` (weight
pre-swizzled), whereas vLLM uses non-preshuffled `GridwiseGemmMultiD_ABScale` for
the dense/MLA projections (11–14% of decode, the single biggest dense kernel).
- **Action:** evaluate CK B-preshuffle (offline weight layout) in the vLLM path to
  drop runtime layout conversion.
- **Caveat:** the Atom trace is a short, comm-dominated capture — treat this as a
  *lead to A/B*, not a quantified win.

---

## Not worth chasing
- **`wvSplitK`** (thin-M skinny GEMM) is decode-only and vanishes by conc256
  (1.9% → 0.1%). It is already doing its job (low-batch decode); no action.
- **RoPE / Normalization** are each ~1–3% and already fused (CK-tile RMSNorm,
  fused QK-rope+cache). Low return.

## Priority ordering (impact ÷ effort)
1. Tune a8w8 block-scale GEMM (#1) — quickest win.
2. Fuse activation quant (#2) and elementwise storm (#3) — largest launch-bound time.
3. MLA fragmentation (#4) and MoE-sort at low conc (#5).
4. AR overlap (#6) and CK preshuffle A/B (#7) — larger efforts / need validation.

## Gaps in the evidence
- No V4 B200/GB300 traces are published (only DeepSeek-R1 there), so no
  cross-vendor V4 kernel comparison is possible without running those recipes.
- The Atom V4 comparison window is not steady-state; a matched-window Atom re-profile
  would make targets #5/#7 quantitative.
