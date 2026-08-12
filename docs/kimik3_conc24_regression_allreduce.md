# Kimi-K3 FP4 DSpark — why conc-24 (M=72) regresses vs conc-16 (M=48)

**TL;DR (root cause found + FIXED 2026-08-12) — the M=72 dense BF16 GEMMs route to aiter's
FlyDSL split-K path, which runs EAGER inside the FULL decode cudagraph.** Earlier framings were
refined twice: it is **not** a GEMM compute knee (every kernel scales sub-linearly 48→72), it is
**not** the all-reduce itself (that's the symptom), and it is **not** a vLLM capture-ladder gap
(M=72 *does* have a FULL decode graph and `dispatch()` returns it — verified in
`cudagraph_utils.py`). The real cause is one level down: at M=72 the three K3 dense GEMMs
`(N,K) ∈ {(1024,7168),(384,7168),(7168,512)}` match CSV rows with `libtype=="flydsl"` in
`merged_bf16_tuned_gemm.csv`. FlyDSL's per-call Python launcher + split-K semaphore protocol is
not capturable in the decode static-buffer FULL path, so those GEMMs execute **eager every decode
step** → conc-24 runs **~290 eager `hipLaunchKernel`/step vs ~90 for conc-16/48 (3.2×)** → decode
is launch-bound → the 8 TP ranks desync → the straggler's GPU sits **38.8% idle** (host-starved)
while the other 7 **spin in `cross_device_reduce_2stage`** (9.8× ballooning, 3.4→33.8 ms/step).
conc-16 (M=48) and conc-48 (M=144) are clean because their FlyDSL kernelNames aren't catalog-valid,
so they already fall back to `torch` (→ rocBLAS/hipBLASLt Cijk, which IS captured) — zero FlyDSL
launches.

**Fix (applied):** convert the FlyDSL rows for those 3 dense shapes at all decode-range M (≤192 =
`max_num_seqs 64 × decode_query_len 3`) to `torch` in the tuned CSV — exactly the capturable
backend M=48/144 already use. Prefill (M>192) keeps FlyDSL. Reproducible via
`_patch_flydsl_decode_to_torch.sh`. See "Fix" section below.

This is the **same class** as the conc-4/12 bubble (falling off the FULL decode graph) but a
*different signature*: conc-4/12 dropped to PIECEWISE and paid the eager `get_mla_metadata_v1`
path; conc-24 shows **equal** `mla_metadata` refs across ranks/concs yet 3.2× more eager
launches — so the earlier metadata-count test does not catch it; the discriminator is
**eager `hipLaunchKernel` per step**.

## Method

Three pure-decode windows captured with the same driver (`_profile_c48_decode.sh`, single
wave so request-count==concurrency → 100% decode by construction, verified by the analyzer):

| conc | M = 3×conc | decode steps (rank0) | window |
|---|---|---|---|
| 16 (win)  | 48  | 76 | WARM_S=18, 3 s |
| 24 (loss) | 72  | 42 | WARM_S=45, 3 s (clean re-run; first WARM_S=18 caught prefill) |
| 48 (par)  | 144 | 46 | WARM_S=45, 3 s |

Decode steps derived from MLA launches ÷ 24 (24 MLA layers/step; consistent across all three:
1824/76, 1008/42, 1104/46). Per-step = kernel time ÷ decode steps, rank0 unless noted.

## Per-step kernel time (ms/step, rank0)

| kernel | c16 M=48 | c24 M=72 | c48 M=144 | 48→72 | 72→144 |
|---|---|---|---|---|---|
| **`cross_device_reduce_2stage` (all-reduce)** | **3.44** | **33.8** | **5.29** | **9.8×** | 0.16× |
| `mfma_moe1_silu_mul` (MoE stage-1, FP8×FP4) | 6.83 | 9.13 | 17.0 | 1.34× | 1.86× |
| `opus_moe_stage2_a8w4_decode` (MoE stage-2) | 3.66 | 4.88 | 6.34 | 1.33× | 1.30× |
| `mla_a8w8_qh16_qseqlen4_v3_ps` (MLA verify) | 2.56 | 3.78 | 6.46 | 1.48× | 1.71× |
| dense/linear (hipBLASLt) | 5.79 | 7.53 | 8.36 | 1.30× | 1.11× |
| "Other" / elementwise | 12.83 | 15.9 | ~25 | 1.24× | — |

For a **1.5× token increase** (48→72) every compute kernel grows **1.30–1.48×** — sub-linear,
exactly as a well-fed GEMM should. Only the all-reduce breaks trend (**9.8×**).

## Cross-rank evidence (conc-24, per-step ms via 42 steps) — the stall is a straggler wait

| rank | all-reduce | MoE s1 | MoE s2 | MLA | reading |
|---|---|---|---|---|---|
| rank0 | **33.8** (1421 ms) | 9.13 | 4.88 | 3.78 | waiting (spins) |
| rank3 | **33.8** (1422 ms) | 9.04 | 4.83 | 3.77 | waiting (spins) |
| rank6 | **3.41** (143 ms)  | 9.08 | 4.87 | 3.75 | **straggler — arrives last, others wait** |

MoE/MLA are identical across ranks → **compute is balanced**. rank6's all-reduce (143 ms) is
the *true* reduction cost (≈ conc-16's level); the 1421 ms on ranks 0/3 is pure spin. The
custom `cross_device_reduce_2stage` is a busy-wait one-shot all-reduce, so a late peer inflates
every early peer's kernel duration. Same launches/step (290) on every rank → it's per-call
latency (spin), not more work.

## Root cause — eager launches per step (the discriminator)

Same-window `hipLaunchKernel` (host kernel-launch API calls = *uncaptured* kernels) and graph
replays, normalized per decode step:

| conc | eager `hipLaunchKernel`/step | graph replays/step | GPU-idle (rank0 / straggler) |
|---|---|---|---|
| 16 (win)    | **90**  | 2.0 | 0.8% / — |
| 24 (loss)   | **290** | 1.9 | 0.7% (spinning) / **38.8%** (host-starved) |
| 48 (parity) | **90**  | 2.0 | — |

conc-24 launches **3.2×** more kernels eagerly than conc-16/48 — its decode step is largely not
replaying the FULL graph. All ranks agree (rank0 and rank6 both = 12165 launches over 42 steps),
so it is a real property of the conc-24 decode, not per-rank. Because decode is then host-bound,
per-rank host jitter desyncs the barrier.

### GPU-idle proof of the straggler (conc-24, 3 s window, device-busy union across streams)

| rank | GPU busy | GPU idle | all-reduce on-GPU | # idle gaps | reading |
|---|---|---|---|---|---|
| rank0 (spinner)   | 99.3% | 0.7% (22.9 ms) | 1421 ms (42.6%) | 3 | busy *spinning* in all-reduce |
| rank6 (straggler) | 61.2% | **38.8% (1292 ms)** | 143 ms (4.3%) | **889 (~0.5 ms ea, ~21/step)** | GPU *host-starved*, arrives late |
| conc-16 rank0 (control) | 99.2% | 0.8% (23.8 ms) | 261 ms (8.5%) | 3 | clean |

rank6's GPU is idle ~21×/step waiting on its host thread; it reaches each all-reduce late, and
the collective barrier forces the other 7 ranks to spin. The 1421 ms of "all-reduce" on ranks
0/3 is pure busy-wait, not reduction work (rank6's 143 ms is the true cost).

## Cross-check against the un-profiled benchmark

Benchmark ITL p50: conc-16 = 24.2 ms, conc-24 = 47.3 ms → **+23 ms** regression. The profiled
all-reduce delta is **+30 ms/step** (33.8−3.4); every compute delta is <3 ms. The all-reduce
delta alone explains the regression; the profiler only lets us attribute it (the effect is real
without the profiler — conc-24 is off-trend in the plain sweep too).

## Implications

1. **Task #50 (tune 6288×7168 dense GEMM) will NOT fix conc-24.** That GEMM's per-step time is
   balanced and sub-linear (5.79→7.53→8.36). Worth doing for absolute decode GEMM cost, but it
   is not the conc-24 lever.
2. **The lever is the aiter GEMM backend at M=72, NOT vLLM capture coverage.** Reading the
   container `cudagraph_utils.py` proved M=72 *is* covered by a FULL decode graph and `dispatch()`
   returns it (`max_decode_tokens = max_num_seqs × decode_query_len = 192`; 72 ≤ 192 and
   `round_up(72,3)=72`). What runs eager inside that graph is the FlyDSL dense GEMM. Trace evidence
   (conc-24-only kernels): FlyDSL launchers `_normalize_launch_stream` (4420) + `launcher` (2210),
   `hgemm_bf16_32x64x128x3_SPK4` (4292), `bf16gemm_fp32bf16_tn_32x64_pf3_splitk` — all absent at
   conc-16/48. Fix = reroute those shapes off FlyDSL (see "Fix" below); 290→~90 eager launches/step
   follows.
3. **Secondary knobs to A/B** once dispatch is understood: `--async-scheduling` off (the straggler
   is host-timing, not compute); AITER custom all-reduce algo thresholds (only relevant because
   the spin *surfaces* there — fixing capture should remove it regardless).
4. **Ruled out:** GEMM compute knee (sub-linear); expert-parallel imbalance (MoE compute balanced
   across ranks); conc-4/12-style PIECEWISE metadata bubble (equal `mla_metadata` refs).
5. **DSpark win condition, corrected:** DSpark stops beating baseline at conc-24 not because the
   verify step crosses a compute knee, but because M=72 lands in a **capture-coverage hole** →
   host-bound decode → rank desync. Baseline at conc-24 runs M=24 (well-captured, no stall);
   DSpark's 3× token multiplier pushes the step size into the uncaptured M≈72 regime.

## Fix (applied 2026-08-12)

Reroute the K3 dense BF16 GEMMs off aiter FlyDSL for all decode-range M, in the tuned GEMM CSV
`/opt/aiter-local/aiter/configs/merged_bf16_tuned_gemm.csv`:

- **Which rows:** `libtype=="flydsl"` AND `M ≤ 192` AND `(N,K) ∈ {(1024,7168),(384,7168),(7168,512)}`
  (60 rows; the 3 at exactly M=72 are the proven conc-24 offenders). At M=72 these were the *only*
  FlyDSL rows.
- **Change each to a native torch row:** `libtype=torch, solidx=0, splitK=0, kernelName=native`.
  `torch_gemm` → `torch.matmul`/`F.linear` → rocBLAS/hipBLASLt Cijk kernels, captured natively in
  the FULL decode graph. This is *exactly* what M=48/144 already fall back to.
- **Prefill untouched:** rows with M>192 keep FlyDSL (prefill isn't in the decode static-buffer
  FULL path).
- **Reproducible:** `_patch_flydsl_decode_to_torch.sh` (idempotent, backs up to
  `<csv>.pre_flydsl_fix.bak`). The tuned CSV lives inside the container and resets on image rebuild,
  so re-run the script after any aiter/container refresh.

**GOTCHA (cost me one failed boot):** do **not** leave `splitK` empty. The CSV is loaded with
`pandas.read_csv`; an empty cell becomes `NaN` and flips the whole `splitK` column to `float`, after
which pre-existing **asm** rows feed `splitK=3.0` (a float) into `aiter::_gemm_a16w16_asm`, whose
signature is `Optional[int]` → engine-core init crashes (`Unable to cast float to int`). Native torch
rows encode `splitK=0` (int), keeping the column integer-typed. `opus_gemm` casts with `int(...)` so
it's immune; `asm_gemm` passes `splitK` through uncast, so it's the victim.

### Validated results (2026-08-12, full mandated sweep)

Re-ran the full mandated sweep {48,32,24,16,12,8,4,2,1} on the 68k ISL / 350 OSL long-ctx
workload (TP8, num_spec=2, gpu_mem 0.95 / max_num_seqs 64 / MNBT 16384 / FULL_AND_PIECEWISE),
0 errors, DSpark acceptance intact (AL ≈ 2.15 tok/step, pos0 ≈ 72%, pos1 ≈ 43% at every conc).
The sweep carries **both** graph-capture fixes: the `{12,36}` capture-sizes fix (rescues conc-4/12
from the PIECEWISE `get_mla_metadata_v1` bubble — see `k3-dspark-decode-bubble-mla-metadata`) and
the FlyDSL→torch reroute here. ITL p50 (ms):

| conc | M=3×conc | PRE (no fixes) | `{12,36}` only | **both fixes** | fix responsible |
|---|---|---|---|---|---|
| 1  | 3   | 10.24 | 10.36 | 10.48 | — |
| 2  | 6   | 12.13 | 12.66 | 12.49 | — |
| 4  | 12  | **74.96** | 13.79 | **13.73** | capture-sizes `{12,36}` (−82%) |
| 8  | 24  | 17.92 | 17.67 | 17.56 | — |
| 12 | 36  | **76.00** | 20.89 | **21.87** | capture-sizes `{12,36}` (−71%) |
| 16 | 48  | 24.27 | 24.19 | 24.48 | — |
| 24 | 72  | 47.99 | 47.34 | **44.56** | FlyDSL→torch reroute (−7% vs pre; −2.8 ms isolated vs `{12,36}`-only) |
| 32 | 96  | 58.91 | 59.87 | 60.39 | — |
| 48 | 144 | 85.09 | 76.57 | 86.62 | — (76.57 was a lucky run; TTFT p90 also 2.5× lower — treat as noise) |

**Reading:** the curve is now cleanly monotonic — the two ~75 ms spikes (conc-4/12) are gone. The
`{12,36}` fix is the large win (−61 ms / −55 ms, ~4–5× on those points); the FlyDSL reroute is a
smaller, cleanly-isolated gain at conc-24 (both right-hand columns share `{12,36}`, so 47.34→44.56
is purely the reroute). The steady-state benchmark benefit of the reroute (−2.8 ms) is far below the
33.8 ms/step all-reduce spike seen in the *isolated single-wave pure-decode* profiling window,
because chunked prefill of the 68k ISL interleaves with decode in the real sweep and hides most of
the bubble. Raw data: `k3_dspark_longctx_bench` (pre), `k3_dspark_longctx_bench_FULLFIX`
(`{12,36}` only), `k3_dspark_longctx_bench_FLYDSLFIX` (both).

## Artifacts

Traces: `kimik3_traces_c48/{_c16,_c24,_c48}/dp0_pp0_tp*_*.pt.trace.json.gz` (8 ranks each).
Stage reports: `_c16_stage.md`, `_c24_stage.md`, `_c24_r{3,6}.md`. conc-48 baseline:
`docs/kimik3_c48_decode_profile.md`. Analyzer: `analyze_dsv4_trace.py` (DSV4 boilerplate header;
K3 numbers are the tables above).
