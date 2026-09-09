# Kimi-K3 KV-offload host-pool sizing (MI355X FP4 DSpark)

**Date:** 2026-08-14 · TP8 MI355X gfx950, vLLM 0.26.1rc1, DSpark num_spec=2, native KV
offload. Measured with the lightweight no-aiperf probe (`_offload_smoke.sh` +
`_offload_probe.py`), which times three TTFT points on one serve — `gpu_hit` (resident
floor) / `reload` (evicted→host) / `cold` (recompute ceiling) — plus `/metrics` byte deltas.

## TL;DR

**The host pool must be ≥ the reused-prefix working set. If it is smaller, offload
silently degrades to no-offload (reload → cold) *and still pays the full write cost* — the
worst of both worlds.** 64 GB is not "smaller/cheaper offload"; below the working set it is
zero offload with extra write traffic.

## 64 GB vs 256 GB — apples-to-apples (P=100 distinct ~8k-tok prefixes, ~128 GB working set, GPU_MEM=0.90)

| | pool = 64 GB | pool = 256 GB |
|---|---|---|
| gpu_hit (resident floor) | 211.6 ms | 217.7 ms |
| **reload** (evicted → host) | **509.9 ms** | **233.5 ms** |
| cold (recompute ceiling) | 510.3 ms | 512.7 ms |
| read benefit (cold − reload) | **0.3 ms (0%)** | **279.1 ms (54%)** |
| external_prefix_cache_hits Δ | 0 | +27,648 |
| load_bytes CPU→GPU Δ | 0 | +6.45 GB |
| store_bytes GPU→CPU (prime) | 128 GB | 128 GB |
| **VERDICT** | **DEAD** (reload == cold) | **ALIVE** (reload ≈ resident) |

- **64 GB is DEAD here:** it wrote 128 GB into a 64 GB pool → overflowed and evicted the
  earliest prefixes off the host tier too. Re-requests find nothing to reload → full
  recompute. Offload buys 0 ms while still paying 128 GB of write traffic.
- **256 GB is ALIVE:** reload sits ~16 ms above the resident floor and **279 ms / 54%
  below recompute**; read path provably live (+27,648 external hits, +6.45 GB CPU→GPU).

**Not a 64-GB defect — a working-set relation.** At a smaller working set (P=30, ~30 GB)
the *same* 64 GB pool was ALIVE (reload 234.2 ≈ gpu_hit 227.9, 56% saved). 64 GB failed at
P=100 only because 64 GB < the 128 GB working set.

## Sizing rule

> **pool_bytes ≥ steady-state distinct-reused-prefix footprint** (headroom ~1.3×).

Measure the footprint live — it is the **resident write volume of distinct evicted
prefixes**, i.e. `kv_offload_total_bytes_total{transfer_type="GPU_to_CPU"}` once the
eviction set has stabilised (steady state, ≥900 s in the agentic scenario). **Do NOT size
from `CPU_to_GPU` (reload) volume** — that is cumulative read throughput (a hot prefix
reloads many times) and vastly overstates the footprint.

## Sizing for the real agentic recipe (not the synthetic scenario)

The synthetic aiperf agentic scenario has **≈0 prefix reuse** (writes >1 TB, reads back
~0.04%), so it is the wrong instrument for both proving *and* sizing offload — its footprint
is meaningless. Size against the **real multi-turn shape** instead:

- **Colleague-mirror 24-session A/B (conc16, 24×8192 real-shaped prefixes):** distinct
  resident footprint ≈ **65.5 GB store (GPU→CPU)**; drove 752k external hits and 250 GB of
  *cumulative* CPU→GPU reload; delivered the validated **TTFT −12% p50 / −29% p90 / −19%
  avg**. Its 65.5 GB footprint *just* fit a 64 GiB pool → ALIVE. Bump the session count,
  the prefix length, or the concurrency and it would cross 64 GB and die exactly like the
  P=100 probe.

**Recommendation for the FP4 MI355X recipe: default the host pool to 256 GB.** It covers
the 128 GB probe working set outright and leaves comfortable headroom over the 65.5 GB
real-24-session footprint for higher concurrency / longer contexts. Only drop to 64 GB when
the reused footprint is *proven* < ~50 GB (few sessions, short shared prefixes) — and know
it fails silently (DEAD + wasted writes) the moment the workload exceeds it. When in doubt,
oversize: an over-large pool costs host RAM only; an under-sized one costs the entire
offload benefit plus the write traffic.

## Measured across the agentic sweep conc 1→24 (2026-08-14, GPU KV cap ≈ 2.2 M tok)

From `server_metrics_export.json` of the synthetic-2.51 sweep. Two things are real, one is a trap.

| run | GPU-KV pk | GPU-KV avg | gpu prefix hit | offload store | offload load | ext hit | **peak pool use** |
|---|---|---|---|---|---|---|---|
| c1  base | 23% | 8%  | 94.5% | — | — | — | — |
| c2  base | 38% | 10% | 93.2% | — | — | — | — |
| c4  base | 64% | 23% | 94.6% | — | — | — | — |
| c8  base | 100% | 57% | 71.8% | — | — | — | — |
| c16 base | 100% | 83% | **8.8%** | — | — | — | — |
| c24 base | 100% | 85% | **11.2%** | — | — | — | — |
| c8  OFF | 100% | 47% | 78.8% | 1,323 GB | 285 GB | 9.6% | **0.3%** |
| c16 OFF | 100% | 81% | 8.9% | 3,346 GB | 188 GB | 1.8% | **0.1%** |

1. **Real GPU-KV eviction cliff at conc-8.** GPU KV saturates (100%) from conc-8 up; GPU
   prefix-cache hit rate collapses **94% → 9%** between conc-4 and conc-16. This is the
   pressure offload exists to relieve.
2. **The synthetic sweep CANNOT size the pool.** While writing **1.3–3.3 TB** of evicted
   KV, peak host-pool utilization is **0.1–0.3%** and external-hit is **1.8–9.6%** — near-zero
   reuse, so the pool never accumulates a resident footprint. `store_bytes` here is pure
   write-churn. **Do not size (or judge) offload from this scenario.** Confirms the rule
   above with data: footprint must come from a reuse-bearing workload.

**So the only valid footprint we have is still the real colleague-mirror 24-session shape
(65.5 GB).** A conc-48 pool size must be measured on that shape scaled up (or the
`_offload_smoke.sh` probe with P scaled to match conc-48's reused working set) — NOT by
re-running the synthetic aiperf sweep, which will just report <0.3% pool use again.

## Reproduce

```bash
# 64 GB pool, 128 GB working set (DEAD):
P=100 KV_OFFLOADING_SIZE=64  GPU_MEM=0.90 bash _offload_smoke.sh
# 256 GB pool, same working set (ALIVE, 54% recompute saved):
P=100 KV_OFFLOADING_SIZE=256 GPU_MEM=0.90 bash _offload_smoke.sh
```

Verdict liveness gate (both must hold): `load_bytes(CPU_to_GPU) Δ > 0` **and**
`external_prefix_cache_hits Δ > 0`. If either is 0, the pool is undersized and offload is
dead regardless of the TTFT numbers.
