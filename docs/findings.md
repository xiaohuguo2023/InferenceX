# Findings log

## 2026-08-16 — Phase 0: decompose the 35 GiB KV reserve (measure-first)

**Boot:** K3 DSpark nspec=7, TP8, gfx950, no-pin (`KV_CACHE_MEMORY=none`) +
`VLLM_KV_RUNTIME_RESERVE_GIB=35`, mandated config (GPU_MEM=0.95, seqs=64, MNBT=16384,
FULL_AND_PIECEWISE). Serve on :8890. Method: boot-log memory summary + 1 Hz `rocm-smi`
VRAM sampling around a real 68K prefill (token-id driver, no aiperf) and a conc-24 load.

### Per-rank boot decomposition (from `gpu_worker.py:862`, TP0 representative; ±0.4 GiB across 8 ranks)
| component | GiB |
|---|---|
| physical card | 287.98 |
| requested (0.95×) | 273.59 |
| consumed (weights + non-torch) | 200.78 |
| profile_run peak_activation | 5.5 |
| **CUDA graph pool** (`capture_model()` → "Graph capturing … took") | **11.18** |
| KV allocated (reserve=35 applied) | 32.3 |

Sizing math verified exactly: `available_kv = 273.59 − (200.78+5.5) − 0[V2 graph stub] − 35[reserve] = 32.3` ✓.
V2 `profile_cudagraph_memory()` returns 0, so the reserve is literally standing in for the
missing graph accounting — **plus more**.

### Runtime VRAM (1 Hz sampler)
- idle post-capture, pre-prefill: **253.4–254.8 GiB** (~33 GiB free)
- single real 68K prefill (0 cached, 4.76 s): peak **+5.64 GiB** — **matches** profile_run's 5.5 GiB estimate.
  The +5.64 GiB is **persistent** (does not free after the request) → off-torch backend scratch
  allocated on first real prefill, invisible to `profile_run` (which frees its dummy before KV sizing).
- conc-24 (24 distinct resident 68K seqs, 64 decode steps w/ spec verify, 109 s): peak **267.2 GiB**,
  **min free 20.8 GiB**. Peak − pre-prefill-idle = **12.38 GiB** (= 5.64 persistent scratch + 6.74 conc-scaled activation).

### The 35 GiB reserve decomposition (per rank)
| bucket | GiB | % | who accounts it |
|---|---|---|---|
| CUDA graph pool | 11.18 | 32% | **PR1** (V2 `profile_cudagraph_memory` stub → real value) |
| persistent first-prefill scratch | 5.64 | 16% | **PR2** (backend workspace declaration / init-alloc) |
| high-conc activation under-count + margin | 18.18 | 52% | **PR3** (generic high-conc/mixed `MemoryProfileShape`) |

### Verdicts (this is what Phase 0 was for)
1. **The plan's hypothesis is REFUTED.** The graph pool is only **32%** of the reserve, not "most of it."
   The majority (68%) is genuine activation/scratch that `profile_run`'s single-prefill dummy under-counts.
2. **PR1 alone does NOT free any KV.** Accounting the 11.18 GiB graph pool while preserving the same real
   headroom just means `reserve' = 35 − 11.18 = 23.82` → KV stays **32.3 GiB**. PR1 is a
   correctness/transparency change (move a precisely-measurable quantity out of a magic number), not an
   immediate perf win. Verified numerically.
3. **Growing KV requires PR2 + PR3**, each validated at conc-48, because the under-count is real
   (explains why KV=56 OOMed at conc-48 — memory `k3-kv-pin-vram-ceiling`: vLLM's own "fit requested"=56.13
   suggestion omits the ~5.6 scratch + ~14 high-conc activation ≈ 20 GiB, so safe KV ≈ 36, ≈ the current 32.3).
4. **The reserve=35 is well-calibrated, not wildly over-provisioned** (~4 GiB true margin at conc-48).

### Implication for the user's fork (immediate perf vs long-term project)
This is a **long-term recorded project**: PR1 (transparency) → PR2 (scratch) → PR3 (high-conc profile shape),
each generic + upstreamable + validated at conc-48. There is **no quick reserve→0 flip** and no free KV from
PR1 alone. The only immediate lever would be re-tuning the constant down by the measured conc-48 slack
(~10 GiB), but that keeps a magic number, which is what we're trying to remove.

Artifacts: `_phase0/{vram_sampler.sh,prefill68k.py,vram.tsv,markers.txt}`, `serve_k3_bench_spec7.log`.

### DECISION (2026-08-16, user): STOP — record as project, no code change now
Two facts settle it: (1) the 35 GiB is genuinely needed — measured under-count = graph 11.18 +
scratch 5.64 + hi-conc activation 18.18 = **exactly 35**, so KV is **invariant** under any faithful
re-accounting (move a term out of the reserve, drop the reserve by the same amount → KV stays 32.3).
No KV to free on this box. (2) V2 has **none** of V1's throwaway-graph-profiling seam
(`cudagraph_dispatcher`/`_warmup_and_capture`/`CUDAGraphWrapper` pool-swap/`graph_capture`); it uses an
opaque `cudagraph_manager.capture()`. A faithful pre-sizing graph profile = reimplement throwaway
capture+teardown against V2, and V1's own `shutdown()` warns delayed ROCm graph destruction surfaces
HSA faults on next startup. So the "faithful profiling pass" is a **correctness/portability** change
(compute ~35 instead of hardcoding, to be right on other models/concurrencies) with **zero perf payoff
here** and real boot-fragility risk. **Retain `VLLM_KV_RUNTIME_RESERVE_GIB=35`.** PR1/PR2/PR3 (tasks
#88/#89/#90) stay recorded as the upstream project, gated on a future model/config change or an upstream
appetite for the V2 profiling reimplementation — not being implemented now.

## 2026-08-17 — Session: Codegen
- Extended the common eager fused all-reduce + RMSNorm helper to dispatch to AITER on ROCm, allowing K3 DSpark's existing unreduced row-parallel path to use the fused primitive without `torch.compile`.
