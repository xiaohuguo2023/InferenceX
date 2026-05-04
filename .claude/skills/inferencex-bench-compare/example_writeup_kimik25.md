# Kimi-K2.5 fp4 (MXFP4) — MI355x ours vs InferenceX dashboard vs B200/B300

## Setup

- **Model**: `amd/Kimi-K2.5-MXFP4` (~590 GB / 64 shards)
  - Originally downloaded to `/data/amd/Kimi-K2.5-MXFP4`, then `mv`'d to `/home/xiaohugu/Kimi-K2.5-MXFP4` (same filesystem, instant) so it's visible inside `xguo-comms4`.
- **Hardware**: 8× MI355X (single node)
- **Sweep script**: `sweep_kimik25_widegraph_default_mi355x.sh` (widegraph-default style — AITER env on, no explicit `--compilation-config`). Now supports `TP_FILTER` and `OUT_BASE` env-var overrides.
- **Two containers used** (image-version explained below):
  - `xguo-kimi-bench` — `rocm/vllm-dev:nightly_main_20260224`, image-baked vLLM **0.16.0rc2.dev445** → ran TP=4 ✓
  - `xguo-comms4` — same image but with editable `/home/work/vllm` install of vLLM **0.19.1.dev210** → ran TP=8 ✓
- **Combined sweep dir** (after merge): `/home/xiaohugu/work/sweep_kimik25_output/sweep_kimik25_widegraph_default_20260501-075311/` (TP=4 from kimi-bench + TP=8 from comms4)
- **Plot script**: `comparison_plots/plot_kimik25_compare.py`

## Why two containers / two vLLM versions

The image-baked vLLM 0.16 has an AITER-MLA assertion that rejects 8 heads/shard:

```
File "vllm/v1/attention/backends/mla/rocm_aiter_mla.py", line 214
    assert num_heads == 16 or num_heads == 128
```

Kimi-K2.5 has 64 attention heads. TP=4 → 16/shard ✓; TP=8 → 8/shard rejected. Two paths around it:

1. **Disable AITER MLA** on 0.16 (`VLLM_ROCM_USE_AITER_MLA=0`) → falls back to TRITON_MLA. Works, but TPOT 2–3× worse at long ISL than what's achievable.
2. **Use vLLM 0.19** (xguo-comms4's editable `/home/work/vllm` checkout) → assertion is gone, AITER MLA accepts 8 heads/shard.

Used path 2 for the production TP=8 numbers reported below. Path 1 numbers are kept (in the dropped section near the bottom) as a quantification of the AITER-MLA-vs-TRITON-MLA gap.

## Coverage

| TP | (ISL, OSL) | CONC | vLLM | Status |
|---|---|---|---|---|
| 4 | (1024,1024), (1024,8192), (8192,1024) | 4, 8, 16, 32, 64 | 0.16 | ✅ all 15 |
| 8 | same | same | 0.19 | ✅ all 15 |

Per-combo wall: TP=4 310–3143 s; TP=8 (v0.19) 277–2518 s (faster than the v0.16 fallback's 310–3901 s — AITER MLA is faster on the long-output combos too). Dominant cost is the 1k/8k cluster.

## Image-version table (still matters for the dashboard comparison)

| Source | vLLM | Note |
|---|---|---|
| Ours TP=4 | 0.16.0rc2.dev445.rocm700 | image-baked, AITER MLA on |
| Ours TP=8 | 0.19.1rc1.dev210.rocm700 | editable from `/home/work/vllm`, AITER MLA on |
| IX MI355x cfg 672 (TP=4) | 0.18.0 | dashboard pin |
| IX MI355x cfg 603 (TP=8) | 0.18.0 | dashboard pin |
| IX B200 cfg 635/636/811 | 0.17.0 (NV) | |
| IX B300 cfg 813/814 | 0.19.0-cu130 (NV) | |

We're slightly newer than the dashboard on the AMD side. Numbers reproduce the dashboard within ~10%, sometimes a touch faster (most visible at TP=8 where we're on .dev210 vs dashboard 0.18).

## Headline plots

**Per-metric grids** (throughput / TTFT / TPOT × ISL/OSL):

- `comparison_plots/kimik25_mi355x_vs_b200_tp4.png` — full grid, all 4 series
- `comparison_plots/kimik25_mi355x_vs_b200_tp8.png` — full grid; B200/B300 only have CONC=4 (dashboard sparse)

**Pareto curves** (throughput-per-GPU vs interactivity or end-to-end latency, CONC sweep along each curve):

- Interactivity (Y: tok/s/GPU, X: tok/s/user = 1000/TPOT). Up-and-right is better.
  - `kimik25_pareto_1k1k_tp{4,8}.png`
  - `kimik25_pareto_8k1k_tp{4,8}.png`
- E2EL (Y: tok/s/GPU, X: mean E2EL in s). Up-and-left is better.
  - `kimik25_pareto_e2el_1k1k_tp{4,8}.png`
  - `kimik25_pareto_e2el_8k1k_tp{4,8}.png`

**Data**

- `comparison_plots/kimik25_data_table.csv` — per-(isl,osl,tp,conc) numbers across all sources

## Findings — TP=4

### Ours vs IX MI355x dashboard — sanity check

Within ~5–10% throughput at every CONC; near-identical TTFT/TPOT. Recipe + image reproduce the dashboard.

| (ISL, OSL) | CONC | Ours tput | IX MI355x tput | Δ |
|---:|---:|---:|---:|---:|
| 1024/1024 | 4   | 611  | 671  | -8.9% |
| 1024/1024 | 64  | 3585 | 3872 | -7.4% |
| 8192/1024 | 4   | 2440 | 2589 | -5.7% |
| 8192/1024 | 64  | 10299 | 10790 | -4.5% |

### MI355x vs B200/B300 — the gap

B200/B300 throughput is **~50% higher** than MI355x at every CONC. TPOT ~30% lower; TTFT ~40% lower.

| (ISL, OSL) | CONC | MI355x ours | B200 vllm | B300 vllm | B200/MI355x |
|---:|---:|---:|---:|---:|---:|
| 1024/1024 | 4  | 611  | 946  | 986   | **1.55×** |
| 1024/1024 | 64 | 3585 | 5414 | 5609  | **1.51×** |
| 8192/1024 | 4  | 2440 | 3869 | 4024  | **1.59×** |
| 8192/1024 | 64 | 10299 | 15324 | 18135 | **1.49×** |

Flat across CONC and ISL/OSL — steady-state per-token compute, not a startup artifact. Different from gptoss-fp4 (where MI355x and B200 vLLM tracked within ~10% at low CONC; only B200 TRT ran away). For Kimi-K2.5 vLLM, **B200 wins everywhere on TP=4**.

### Dashboard anomaly to flag

B200 cfg 635 at TP=4 ISL=8192 CONC=64 reports **TTFT=4600 ms** (vs B300 866 ms, B200 at lower CONC ~570 ms). 5× outlier with no supporting trend; throughput at that row (15324) is sensible. Almost certainly a stuck-prompt warmup. Worth filing back.

## Findings — TP=8 (vLLM 0.19, AITER MLA on)

### Ours vs IX MI355x — dashboard reproduced

With AITER MLA on, TP=8 results track the dashboard within **~5% at every CONC** for both 1k/1k and 8k/1k:

| (ISL, OSL) | CONC | Ours tput | IX tput | Δ tput | Ours TPOT | IX TPOT | Δ TPOT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024/1024 | 4   | 714   | 692   | +3.2%  | 10.72 | 11.14 | -3.8% |
| 1024/1024 | 16  | 2021  | 1981  | +2.0%  | 15.41 | 15.63 | -1.4% |
| 1024/1024 | 64  | 4608  | 4580  | +0.6%  | 26.85 | 26.89 | -0.1% |
| 8192/1024 | 4   | 2941  | 2691  | **+9.3%** | 11.61 | 12.64 | -8.1% |
| 8192/1024 | 16  | 7350  | 6960  | +5.6%  | 18.56 | 19.56 | -5.1% |
| 8192/1024 | 64  | 13159 | 13450 | -2.2%  | 41.85 | 41.08 | +1.9% |

Often slightly faster than the dashboard (we're on .dev210, dashboard pinned 0.18 — small drift). **Apples-to-apples reproduction.**

### TP=4 vs TP=8 on MI355x — does going wider help?

Using IX MI355x dashboard (cfg 672 vs 603, both vLLM 0.18, apples-to-apples):

| (ISL, OSL) | CONC | TP=4 (cfg 672) | TP=8 (cfg 603) | TP=8 / TP=4 |
|---:|---:|---:|---:|---:|
| 1024/1024 | 64 | 3872  | 4580  | **1.18×** |
| 8192/1024 | 64 | 10790 | 13450 | **1.25×** |

Going from 4 → 8 GPUs gives only **+18–25% throughput**, not 2×. Throughput per GPU **drops** at TP=8 — Kimi-K2.5 has a real comm tax. **For perf-per-GPU, TP=4 is the better operating point on MI355x.**

### MI355x vs B200/B300 at TP=8 — dashboard sparse, one-point comparison

Dashboard B200 cfg 636 and B300 cfg 814 only have CONC=4 for TP=8:

| (ISL, OSL) | CONC | MI355x ours | B200 IX | B300 IX | B200/MI355x |
|---:|---:|---:|---:|---:|---:|
| 1024/1024 | 4 | 714  | 1091 | 1117 | 1.53× |
| 8192/1024 | 4 | 2941 | 3943 | 4543 | 1.34× |

Same shape as TP=4 (B200 ~1.5× MI355x), but at 8k/1k the gap is narrower (1.34× vs 1.59× at TP=4) — TP=8 helps MI355x close some of the gap on long prefill. Dashboard hasn't run high-CONC B200 TP=8, so we can't say if that gap stays narrow.

### How much does AITER MLA matter at TP=8? — and where does the perf actually come from?

We have THREE TP=8 runs at the same combo, isolating one variable at a time. Headline numbers:

| (ISL, OSL) | CONC | v0.16 + Triton tput | v0.19 + AITER tput | Δ | v0.16 TPOT | v0.19 TPOT | Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024/1024 | 4   | 477   | 714   | **+50%**  | 16.2 | 10.7 | -34% |
| 1024/1024 | 64  | 4070  | 4608  | +13%      | 30.3 | 26.8 | -11% |
| 1024/8192 | 4   | 157   | 395   | **+152%** | 28.2 | 11.2 | -60% |
| 1024/8192 | 64  | 1618  | 2672  | **+65%**  | 42.8 | 26.2 | -39% |
| 8192/1024 | 4   | 982   | 2941  | **+200%** | 35.2 | 11.6 | -67% |
| 8192/1024 | 64  | 9022  | 13159 | +46%      | 61.0 | 41.9 | -31% |

**This is NOT pure AITER-MLA-vs-Triton.** It's "vLLM 0.16 + Triton MLA" → "vLLM 0.19 + AITER MLA" — two changes layered. To isolate, we ran a **smoke test** at the worst combo (8k/1k CONC=4) with v0.19 + Triton MLA forced (`VLLM_ROCM_USE_AITER_MLA=0`):

| Run | vLLM | MLA backend | tput | TPOT | TTFT |
|---|---|---|---:|---:|---:|
| **A** v0.16 + Triton | 0.16.dev445 | TRITON_MLA (forced) | 982  | 35.2 | 415 |
| **B** v0.19 + Triton | 0.19.dev210 | TRITON_MLA (forced, smoke test) | **1990** | **17.3** | 316 |
| **C** v0.19 + AITER  | 0.19.dev210 | AITER MLA (default)             | 2941 | 11.6 | 281 |

Decomposition of the headline 3× tput gap:

| Step | Δ tput | Δ TPOT | What changed |
|---|---:|---:|---|
| A → B | **+103%** (2.03×) | -51% | vLLM version, **MLA backend held constant** |
| B → C | +48% (1.48×) | -33% | MLA backend, **vLLM version held constant** |
| **A → C** | **+200%** (3.0×) | -67% | both layered |

**vLLM 0.16 → 0.19 alone delivers ~2× throughput — independent of which MLA backend you pick.** AITER MLA on top adds another ~1.5×. So the original framing "AITER MLA is what makes TP=8 fly" was only ~half the story.

**Where the v0.16 → v0.19 Triton-on-Triton 2× comes from** (best guesses; would need per-kernel profiling to nail exactly):

1. **AITER MoE / RMSNorm / decode-attn kernel updates** — on in BOTH runs. Between 0.16 and 0.19, AITER had ~3 months of weekly drops; MoE FP4 GEMM and RMSNorm both got tile-shape and instruction-scheduling improvements. For Kimi-K2.5 (256 experts, 8 active), MoE is a large slice of the per-token cost.
2. **Triton MLA kernel itself** — `triton_mla.py` evolved across versions: better tile-size autotuning, MLA matrix-absorption fusion, fewer host-device sync points.
3. **Cudagraph capture coverage** — v0.19 captures 51 sizes in `FULL_AND_PIECEWISE` mode; v0.16 likely captures fewer, forcing eager fallback on uncovered shapes.
4. **V1 engine refactor + async-scheduling improvements** — between 0.16 and 0.19 the scheduler/batching path got reworked.

**Where the AITER MLA → +48% on top comes from** (clean isolation, MLA backend is the only delta):

- Hand-tuned MFMA tile shapes for `kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128` (Triton's autotuner doesn't pick these well).
- MLA matrix-absorption fusion likely complete in AITER, partial in Triton.
- Latent K/V decompression kept in registers vs spilled to HBM.

**Rough attribution at 8k/1k CONC=4:**

| Source | Contribution to 3× tput | Why |
|---|---:|---|
| AITER MoE / RMSNorm / decode-attn improvements | ~1.4× | MoE is a large per-token slice; same backend on both 0.16/0.19 but newer kernels |
| Triton MLA kernel + cudagraph + scheduler in 0.19 | ~1.4× | Half the prefill cost; backend held constant, so this is the MLA Triton kernel itself + plumbing |
| AITER MLA over Triton MLA | ~1.5× | The ISA-tuned MLA kernel |

(1.4 × 1.4 × 1.5 ≈ 2.94. Loose multiplicative buckets — illustrative, not measured per-kernel.)

**Updated takeaway**: don't run Kimi-K2.5 on the v0.16 image at all — vLLM-version + AITER kernel improvements together deliver the 3× lift; AITER MLA is the cherry on top.

## Pareto observations

### TP=4, 8k/1k (`kimik25_pareto_8k1k_tp4.png`)

B200 and B300 sit visibly **above and to the right** of MI355x at every CONC. There's no MI355x operating point where it matches B200/B300 on either throughput-per-GPU or interactivity. Ours (red) tracks IX MI355x (orange) closely; ~5–10% lower per-GPU throughput at the same TPOT.

### TP=4, 1k/1k (`kimik25_pareto_1k1k_tp4.png`)

Same shape — B200/B300 dominate at every CONC. The gap is roughly constant on the log-log axes (~50% throughput, ~30% TPOT advantage).

### TP=8, 8k/1k (`kimik25_pareto_8k1k_tp8.png`)

Ours (red) sits **slightly above** IX MI355x (orange dashed) across the whole curve — confirming the v0.19 .dev210 perf bump over the dashboard's pinned 0.18 holds across operating points. B200/B300 are CONC=4-only points; their per-GPU throughput at CONC=4 is similar to MI355x at CONC=8–16, while their interactivity is ~2× higher. **Without high-CONC dashboard B200 data we can't say where the curves cross.**

### TP=8, 1k/1k (`kimik25_pareto_1k1k_tp8.png`)

Ours and IX MI355x are essentially overlapping. B200/B300 single point sits in the same cluster as MI355x at low CONC.

### E2EL variants (`*_pareto_e2el_*.png`)

Same conclusions, just plotted with E2EL on X. Up-and-left is better on these. B200/B300 are upper-left of MI355x at TP=4; at TP=8 the dashboard sparsity makes the cross-vendor read inconclusive.

## What's missing / next steps

1. **B200/B300 TP=8 at higher CONC** is a dashboard gap — only CONC=4 currently. Until that fills in, TP=8 cross-vendor pareto is one point per NV vendor.
2. **Fix the v0.16 image** to either bump the head-count assertion or expose a CLI knob for the MLA backend, so users on the pinned image don't hit this. Issue worth filing against the AITER repo / vLLM ROCm image.

## Source files

- Sweep: `sweep_kimik25_widegraph_default_mi355x.sh` (supports `TP_FILTER`, `OUT_BASE`, `MODEL` env vars)
- Plot script: `comparison_plots/plot_kimik25_compare.py`
- IX dump: `/tmp/inferencex_dump/inferencex-dump-2026-04-27/` (public, no auth — released weekly at https://github.com/SemiAnalysisAI/InferenceX-app/releases)
- Sweep outputs:
  - TP=4 + TP=8 (production): `/home/xiaohugu/work/sweep_kimik25_output/sweep_kimik25_widegraph_default_20260501-075311/`
  - TP=8 v0.16 fallback (archived): same dir, `v016_tp8_archive/` subdir
  - TP=8 v0.19 source: `/home/xiaohugu/work/sweep_kimik25_output/v019_tp8/sweep_20260501-211604/`
- Configs of interest:
  - MI355x vllm: cfg 672 (TP=4), 603 (TP=8), 681 (TP=4 1k/8k rerun)
  - B200 vllm: cfg 811 (TP=4 newer), 635 (TP=4 older fills gaps), 636 (TP=8 sparse — CONC=4 only)
  - B300 vllm: cfg 813 (TP=4), 814 (TP=8 sparse — CONC=4 only)
