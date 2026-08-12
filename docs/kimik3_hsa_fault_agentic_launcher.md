# Kimi-K3 DSpark agentic — cluster HSA fault review & resolution

**Scope:** the multi-node agentic fan-out driven by
`benchmarks/single_node/agentic/kimik3_fp4_mi355x_vllm_dspark.sh` (KV-offload, full
~131K context, B300-MTP/B200 comparison). NOT the single-box sweep driven by
`_run_agentic_dspark.sh` → `_serve_k3_bench_spec.sh`, which is healthy (see below).

## Symptom (as reported from the cluster)

- Points c1 (n034) and c16 (n061) died with **Memory access fault by GPU node-N …
  Reason: Unknown → HSA_STATUS_ERROR_EXCEPTION 0x1016**, queue aborts. Earlier n235
  crashes share the signature. c2/c4/c8/c24 kept running.
- Two fresh, different nodes, identical signature ⇒ **systemic software fault, not bad
  hardware** (n235 was not special). ✅ agrees with our evidence.
- Reproduces at **concurrency 1** ⇒ rules out the split-K / multi-stream *decode*
  contention theory (no batching pressure at c1). ✅ agrees — the conc-24 all-reduce
  stall (`docs/kimik3_conc24_regression_allreduce.md`) is a different, decode-only bug.
- Last work logged before each fault: a burst of large **BF16 prefill GEMMs
  M≈2620–2750, N=7168, K=35840** on the **untuned `torch solution:0`** fallback across
  all 8 TP ranks.

## Two findings from the report — one confirmed, one corrected

### Finding A (CONFIRMED) — the FlyDSL→torch reroute was never active on the agentic path

`kimik3_fp4_mi355x_vllm_dspark.sh` launches `vllm serve` itself and set every AITER env
var **except `AITER_CONFIG_GEMM_BF16`**. Only `_serve_k3_bench_spec.sh` exported it
(lines 116–119). With it unset, aiter re-merges `configs/model_configs/*bf16_tuned_gemm*.csv`
into `/tmp/aiter_configs/bf16_tuned_gemm.csv` and reads **that**, not our patched
`/opt/aiter-local/aiter/configs/merged_bf16_tuned_gemm.csv`.

Verified on-box (container `k3-dspark-benchmark`):

| CSV | decode-range FlyDSL rows | who reads it |
|---|---|---|
| `/opt/aiter-local/.../merged_bf16_tuned_gemm.csv` (we patch this) | **925** | serve script (export set) |
| `/tmp/aiter_configs/bf16_tuned_gemm.csv` (auto-merged) | **985** | cluster launcher (export unset) |

The 60-row delta is exactly the decode FlyDSL→torch reroute
(`docs/kimik3_conc24_regression_allreduce.md`). So on the cluster the reroute was silently
inactive — a real **decode-path perf/routing bug**. **This is not the c1 crash cause**
(the crash is in prefill; the reroute only touches decode-range M ≤ 192).

### Finding B (CORRECTED) — the untuned N=7168, K=35840 prefill GEMM is NOT the culprit

The report attributed the crash to that GEMM on `torch solution:0`. Disproven directly:

- **Neither CSV has any K=35840 row** (`grep -c 35840` = 0 in both). So that GEMM falls to
  `torch solution:0` **regardless** of the `AITER_CONFIG_GEMM_BF16` export — Finding A's fix
  cannot change its behavior.
- **Our healthy single-box serve runs the identical shape with zero faults.** The live
  `serve_k3_bench_spec2.log` shows `M:16384, N:7168, K:35840` BF16 → "not found tuned
  config … using torch solution:0" on **all 8 ranks** — an even larger M than the cluster's
  ~2620–2750 — and the box completed agentic points c1–c8 clean, **no HSA fault**.

⇒ `N=7168,K=35840` on `torch solution:0` is a **logging red herring**: it is merely the last
kernel printed before an **asynchronous** GPU fault surfaced from elsewhere. Async HSA faults
are routinely misattributed to whatever kernel was in flight when the queue aborted.

## What actually differs: clean box vs crashing cluster

Same container, same weights, same DSpark config, same untuned GEMM — the divergence is the
**serving/memory configuration** in the two launchers:

| knob | `_serve_k3_bench_spec.sh` (clean) | cluster `..._vllm_dspark.sh` (crashes) |
|---|---|---|
| KV offload | **none** (KV pinned 34 GiB) | **SimpleCPUOffloadConnector, DRAM offload (`KV_OFFLOADING`)** |
| context | 1,048,576 cap, no offload | **full ~131K, offload_mode=on** |
| gpu-mem util | 0.95 | **0.88** |
| max-num-seqs / MNBT | 64 / 16384 | 16 / 4096 |
| GEMM catalog | patched (export set) | auto-merged (export unset → Finding A) |

The prime suspect for the c1 fault is therefore the **KV-offload path under full-context
memory pressure** (`SimpleCPUOffloadConnector` host↔device copy, or an allocation that OOBs
at gpu-mem 0.88 with 131K context) — exercised heavily at c1 with a long prompt, independent
of concurrency — **not** the GEMM. This still needs confirmation from the cluster logs (see
next-steps).

## Resolution

1. **[applied] Plumb the patched GEMM catalog into the agentic launcher.** Added the
   `AITER_CONFIG_GEMM_BF16` export to `kimik3_fp4_mi355x_vllm_dspark.sh` (mirrors
   `_serve_k3_bench_spec.sh:116-119`, overridable via `AITER_MERGED_GEMM_CSV`, warns if
   missing). Fixes Finding A so the decode FlyDSL→torch reroute is actually in effect. **This
   is a decode-perf/correctness fix; it does not by itself stop the c1 prefill fault.**

2. **[investigate] The real fault — KV-offload / full-context memory path.** Confirm on the
   cluster before more fan-out spend:
   - `grep -iE "SimpleCPUOffload|kv_connector|offload|OOM|out of memory|hipMalloc|HSA_STATUS" server.log`
     around the fault; check whether the fault PC is in the offload connector, not a GEMM.
   - A/B a single c1 point with **`KV_OFFLOAD_BACKEND` unset** (GPU-resident, shorter context)
     — if it stops faulting, the offload path is confirmed as the cause.
   - Check headroom: gpu-mem **0.88 + full 131K context + DSpark verify arena** may be too
     tight; try 0.85 or a smaller `MAX_MODEL_LEN` as a bisection probe.
   - Rule the GEMM out explicitly: it already runs clean at larger M on the single box, so do
     **not** spend on tuning `N=7168,K=35840` to "fix the crash" (tuning it is still worth doing
     for prefill throughput — task #50 — but it is not the fault lever).

## One-liners to reproduce the checks

```bash
# Finding A — which catalog is live in a worker:
docker exec <ctr> bash -lc 'pid=$(pgrep -f "vllm serve"|head -1); tr "\0" "\n" </proc/$pid/environ | grep AITER_CONFIG_GEMM_BF16'
# FlyDSL row delta between patched vs auto-merged:
docker exec <ctr> bash -lc 'for f in /opt/aiter-local/aiter/configs/merged_bf16_tuned_gemm.csv /tmp/aiter_configs/bf16_tuned_gemm.csv; do printf "%s  flydsl=%s\n" "$f" "$(grep -c flydsl "$f")"; done'
# Finding B — the "culprit" GEMM runs clean on the single box:
docker exec <ctr> bash -lc 'grep "N:7168, K:35840" /workspace/serve_k3_bench_spec2.log | head; grep -ciE "HSA_STATUS_ERROR|Memory access fault" /workspace/serve_k3_bench_spec2.log'
```
