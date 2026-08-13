---
name: profile-decode-bubble
description: Torch-profile a Kimi-K3 DSpark decode step on MI355X to find the per-step bottleneck behind an ITL/TTFT knee, and tell a real GPU cost apart from a host launch bubble (a spinning collective is usually a symptom, not the disease). Use for the K3 fp8-asm ROCM_AITER_MLA long-ctx test case whenever decode ITL jumps non-linearly with concurrency, a collective (custom all-reduce) dominates a trace, GPU sits at low power reading "100% busy", or a colleague reports a per-step "bubble". Reuses the in-repo profiling + trace-analysis scripts; do not hand-roll new trace parsers.
---

# Profile the K3 DSpark decode bubble

The K3 FP4 DSpark decode path on MI355X (TP8, `ROCM_AITER_MLA`, num_spec=2) shows
**non-linear ITL/TTFT knees** as concurrency rises. The usual first read — "the custom
all-reduce is 59% of decode, so communication is the bottleneck" — is almost always
**wrong**. This skill is the measured workflow that separates the *disease* (a real
per-step cost) from the *symptom* (7 ranks spinning in a barrier waiting for 1 laggard rank
whose host thread fell behind).

Everything runs inside the `k3-dspark-benchmark` container. `/workspace` is a bind-mount of
the repo root, so scripts edited on the host are live in the container.

## Core mental model

- DSpark decode processes **M = 3×conc** tokens/step (`uniform_decode_query_len = 1+num_spec = 3`).
  conc16→M48, conc24→M72, conc48→M144. Always reason in M, not conc.
- TP8 uses aiter's **custom** all-reduce `cross_device_reduce_2stage` (a **busy-wait** 2-stage
  reduce), **not** RCCL/nccl. A rank that finishes its real work early **spins inside this
  kernel** until the slowest rank arrives. That spin is charged as GPU "busy" time on the
  early ranks → the collective looks huge in aggregate, but it is *wait*, not work.
- So a high "Communication %" in an aggregate report is the **signature of launch
  imbalance**, not a comms problem. Confirm by checking per-rank idle (below).
- The torch profiler inflates decode ITL ~4× (decode is launch-bound) and *widens* the
  arrival gap at the barrier. Trust **relative** comparisons across M and **per-rank** idle,
  not the absolute collective %.

## Reuse these scripts — do not write new trace parsers

| script | what it does |
|---|---|
| `_profile_knee_c162448.sh` | drives the whole capture: serves FULLFIX nspec-2 once with the torch profiler, runs single-wave pure-decode aiperf loads at conc {16,24,48}, POSTs `/start_profile`+`/stop_profile`, moves per-rank traces into `traces_knee/c<conc>/`, then runs both analyzers |
| `_gpu_gap.py <trace.gz>` | **the disease/symptom discriminator.** Per-rank GPU-stream union busy vs idle, count of gaps, and on-GPU `cross_device_reduce` time. Run it on **every rank** — the laggard is the one with low busy% + many gaps |
| `_nightly_fusion_bubble.py <trace.gz>` | part (B) attributes each host gap to the kernel that follows it ("GPU waited to launch this") → names the bubble. part (A) sizes elementwise/glue fusion headroom |
| `analyze_dsv4_trace.py --md <out.md> <traces...>` | stage-split (PREFILL/DECODE) category breakdown; **correctly** buckets the custom all-reduce as "Communication" |
| `backend_breakdown.py <traces...>` | backend-level %. **Caveat:** it lumps the custom all-reduce into the generic "AITER JIT C++/HIP (aiter::)" bucket and only counts nccl-named kernels as RCCL — do not read "RCCL 0.4%" as "no comms stall" |

## The workflow

### 1. Capture pure-decode traces at several M

Use single-wave loads (`--concurrency C --request-count C`): all requests prefill once
against the shared 63.9k prefix, then decode together → the profiled window is **pure
steady-state decode at M=3×conc**.

```bash
docker exec k3-dspark-benchmark bash -lc 'cd /workspace && bash _profile_knee_c162448.sh'
```

Two capture gotchas baked into that script — preserve them if you adapt it:
- **OSL must outlast the warm+profile window.** A single wave finishes in ~`prefill + OSL*ITL`.
  At conc-16 (~24 ms ITL) OSL=600 ends in ~19 s, before the profile fires → 0 traces.
  Use `OSL≈1500`, `WARM_S≈12` (prefill drains in ~5 s), `PROFILE_S≈3`.
- The relaunch does `rm -rf $PROFILE_ROOT`. If you already have good traces for one conc,
  copy them out first (e.g. `_c48_traces_saved/`) before re-running other concs.

### 2. Reproduce the knee, then find the laggard rank

Run `_gpu_gap.py` on **all 8 ranks** at the fast M and the knee M:

```bash
docker exec k3-dspark-benchmark bash -lc '
  cd /workspace
  for t in traces_knee/c24/dp0*rank*.pt.trace.json.gz; do python3 _gpu_gap.py "$t"; done
  for t in _c48_traces_saved/dp0*rank*.pt.trace.json.gz;  do python3 _gpu_gap.py "$t"; done'
```

Read it like this:
- **All ranks ~99% busy, low all-reduce %** → no bubble at this M (healthy, lockstep).
- **One rank low busy% + hundreds/thousands of gaps, while the other 7 are ~99% busy with a
  huge on-GPU `cross_device_reduce`** → **found it.** The low-busy rank is the disease; the 7
  "busy" ranks are spinning in the barrier waiting for it. (Real example: at M=144, rank3 was
  40% busy / 1386 gaps while ranks 0-2,4-7 were 99% busy with all-reduce = 61% of wall. At
  M=72 all ranks were 99% busy — the bubble only appears past the knee.)

### 3. Name the bubble — attribute the laggard's gaps

Run `_nightly_fusion_bubble.py` on the **laggard** trace and read section (B):

```bash
docker exec k3-dspark-benchmark bash -lc '
  cd /workspace
  R=$(ls _c48_traces_saved/dp0*tp3*rank3*.pt.trace.json.gz)
  python3 _nightly_fusion_bubble.py "$R" 2>/dev/null | sed -n "/(B) CPU-BUBBLE/,\$p"'
```

The "idle attributed to the kernel AFTER the gap" table names the host stalls. Kernel-name
decoder:
- `Cijk_…` = hipBLASLt/**Tensile** dense GEMM (eager fallback), `hgemm_bf16` = bf16 dense
  GEMM. Repeated `add_rmsnorm_quant → Cijk_` gaps of ~0.5 ms = **per-launch host dispatch
  stall on an eager/untuned dense-GEMM path** (feeds GEMM-tuning work).
- `opus_moe_sorting`, `grouped_topk`, `fused_mx_quant_moe_sort` = MoE routing/dispatch host cost.
- Many thousands of individually-launched kernels each with a host gap before them = the
  decode step is **not** replaying as one FULL cudagraph for those shapes (dispatch fell to
  piecewise/eager). Collapsing the step into a replayed graph removes the per-launch cost —
  usually the bigger win than tuning the individual GEMMs.

### 4. Cross-check the stage breakdown (optional, for the write-up)

```bash
docker exec k3-dspark-benchmark bash -lc 'cd /workspace && python3 analyze_dsv4_trace.py --md _knee_c48_report.md _c48_traces_saved/dp0*rank*.pt.trace.json.gz'
```

Compare **per-launch** kernel time across M for honesty: real compute (MoE ASM GEMM, MLA)
scales ~1.4–2.0× when M doubles; the all-reduce ballooning ~20× per launch is spin, not work.

## Verdict template

> The all-reduce is a **symptom**: at the knee, rank N goes launch-bound (X% idle, K host
> gaps), and the other 7 ranks spin in `cross_device_reduce` waiting for it. The disease is
> the per-step **host launch chain** on that rank — dominated by `<top gap kernels>`. The
> lever is `<eager dense-GEMM tuning / collapse step into FULL cudagraph>`, not the collective.

## Don't
- Don't conclude from `backend_breakdown.py`'s low "RCCL %" that there's no comms stall — it
  mislabels the custom all-reduce.
- Don't treat a high aggregate "Communication %" as a comms problem before checking per-rank
  `_gpu_gap.py`.
- Don't guess GEMM shapes from the untuned-gemm log path before locating the bubble; find the
  laggard and attribute its gaps first, then extract the exact M/N/K of the named kernels.
- Don't change the mandated DSpark config for the agentic sweep (gpu_mem 0.95 / max_num_seqs 64
  / MNBT 16384 / FULL_AND_PIECEWISE / KV pin). Profiling uses the same serve; only OSL/warm/
  profile-window knobs differ.
- Don't use `--enforce-eager` (loses perf; only for debugging). `pkill "vllm serve"` does not
  free VRAM — kill `vllm|EngineCore|VllmWorker|multiprocessing.spawn` and wait for VRAM drain.
