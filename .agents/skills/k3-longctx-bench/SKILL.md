---
name: k3-longctx-bench
description: Run the Kimi-K3 DSpark long-context spec-decode benchmark on MI355X (the "conc1-conc48 sweep") and produce the perf + acceptance table. This is the ISL-68k (63,911-tok cached prefix) / OSL-350, concurrency 48→1 workload defined by K3_Attention_Benchmark_Instructions.md, with the ATOM report mi355x_atom0807docker_specdecode7.md as the reference numbers. Use whenever the user asks to "benchmark", "run the conc1-48 / conc-sweep", "get perf numbers", or "run the k3 long-ctx bench". Reuses the in-repo serve + aiperf + table scripts; do not hand-roll a new harness.
---

# K3 long-context spec-decode sweep (`k3-longctx-bench`)

The canonical Kimi-K3 DSpark benchmark on MI355X (TP8, gfx950, `ROCM_AITER_MLA` fp8-asm).
Workload is byte-identical to the ATOM report and the instructions doc: **ISL 68,089**
(63,911-tok cached prefix, pool of 1, + 4,089 suffix) / **OSL 350**, concurrency sweep
**48,32,24,16,12,8,4,2,1** with zip request counts **240,160,120,80,60,40,20,10,5**,
`warmup=16`, `ignore_eos`, `min=max=350`, `random_seed=42`.

Everything runs inside the `k3-dspark-benchmark` container; `/workspace` is a bind-mount of
the repo root, so host edits are live. Reference docs: `~/mi355x_atom0807docker_specdecode7.md`
(nspec=7 reference numbers), `~/work/InferenceX-dspv4/K3_Attention_Benchmark_Instructions.md`
(workload spec + DSpark configs).

## Two knobs to confirm with the user before launching

1. **NUM_SPEC** — `2` (our validated recipe, AL~2.5) or `7` (matches the ATOM reference report
   directly). The doc lists both. `2` fails on the *stock* ATOM image (missing gqa16/qSeqLen2/
   causal0 asm kernel) but **runs on ours** because the draft is forced causal (causal=1 path).
2. **Acceptance method** — `block` (real target-vs-draft verify, honest internal number) or
   `synthetic` (golden AL, AgentX-comparable — what PR#2508 / the B300 baseline publish; set
   `SYNTHETIC_ACCEPT_LEN`: 2.51 for nspec2, 3.00 for nspec3, 3.84 for nspec7). Leave unset for
   block. This materially changes ITL/throughput — always confirm.

Backend is **not** a question: use `ROCM_AITER_MLA` fp8-asm (the MI355X perf path all our GEMM
tuning targets). The instructions doc says TRITON_MLA, but our whole recipe + retune is on the
asm path; TRITON_MLA would throw the tuning away. Note the divergence, keep the asm path.

## The three scripts — do not rewrite them

| script | role |
|---|---|
| `_serve_k3_bench_spec.sh` | serve the DSpark target+draft. Self-backgrounds vllm and polls `/health` (blocks ≤30 min). Params: `NUM_SPEC PORT GPU_MEM MAX_NUM_SEQS MNBT` + optional `SYNTHETIC_ACCEPT_LEN`, `CAPTURE_SIZES`, `KV_OFFLOADING_SIZE`. Wires the flydsl-folded GEMM CSV via `AITER_CONFIG_GEMM_BF16` automatically. |
| `_dspark_longctx_bench.sh` | drive the aiperf sweep. Snapshots `/metrics` before/after each point for exact acceptance deltas. Params: `PORT ROOT PAIRS` (default = full 48→1). |
| `_dspark_perf_diag.py` | **PRIMARY latency/interactivity scanner** — the point of this bench for agentic IX. Reads each point's *profiling*-phase json (explicitly excludes warmup) + /metrics, prints the interactivity table (TTFT/ITL/TPOT p50/p90/avg, per-user tok/s, cache%, AL, OSL), SLA gates (claw <25ms / chat <66.7ms), and AUTO-FLAGS perf issues: ITL non-monotonic (decode BUBBLE), ITL knee/cliff, ITL tail (p90/p50), TTFT cliff, prefix-cache eviction, invalid OSL, low AL. `python3 _dspark_perf_diag.py <ROOT> --tp 8`. Run this FIRST. |
| `_dspark_perf_table.py` | full perf + acceptance + per-draft-position detail table (throughput-oriented, for the record + ATOM compare). `python3 _dspark_perf_table.py <ROOT> --tp 8`. |

## Mandated server config (do NOT change these)

`GPU_MEM=0.95 MAX_NUM_SEQS=64 MNBT=16384 CUDAGRAPH_MODE=FULL_AND_PIECEWISE` + the default
`KV_CACHE_MEMORY` pin (32 GiB). **`MAX_NUM_SEQS` must be ≥ the top concurrency (64 ≥ 48)** or the
conc-48/32/24 points silently cap. The script's own defaults are the *conservative* fp8-asm ones
(seqs16/MNBT4096/gpu_mem0.88) — override them. See memory `k3-dspark-mandated-config-runs`.

Decode M = `(1+NUM_SPEC)×conc`. The `CAPTURE_SIZES` default ladder already covers every needed M
for both nspec=2 (M=3×conc → {3,6,12,24,36,48,72,96,144}) and nspec=7 (M=8×conc →
{8,16,32,64,96,128,192,256,384}). The {12,36} entries are the item-1 decode-bubble fix (task #51);
keep them. If a new concurrency C is added, ensure a size `s` with `round_up(s, 1+NUM_SPEC)==(1+NUM_SPEC)*C` exists.

## Workflow

### 1. Verify the box is free + prereqs staged
`pgrep -f "vllm serve"` **self-matches your own shell** (the pattern is in your cmdline) → always
cross-check with VRAM and `/health`. A real serve pins ~200 GiB/GPU; 0.3 GiB = idle context = free.
```bash
docker exec k3-dspark-benchmark bash -lc '
  rocm-smi --showmeminfo vram | awk "/Used Memory/{printf \"%.1f \", \$NF/1073741824}"; echo
  for p in 8888 8889 8890; do curl -sf -m3 -o /dev/null -w "$p:%{http_code} " http://127.0.0.1:$p/health; done; echo
  ls -d /dev/shm/hf-cache/models--moonshotai--Kimi-K3/snapshots/*/ | head -1
  ls -d /dev/shm/hf-cache/models--Inferact--Kimi-K3-DSpark/snapshots/*/ | head -1'
```
Also confirm the draft is causal (`dflash_config.causal==true`; the serve script refuses otherwise)
and the vLLM #50183 NaN-argmax guard is present (grep count==2 in
`.../vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`) — it resets on rebuild-from-base
(memory `k3-rejection-sampler-nan-argmax-0x1016`).

### 2. Launch the serve (background; cold ≈19 min, warm ≈5 min)
```bash
docker exec k3-dspark-benchmark bash -lc 'cd /workspace &&
  NUM_SPEC=7 PORT=8890 GPU_MEM=0.95 MAX_NUM_SEQS=64 MNBT=16384 \
  setsid nohup bash _serve_k3_bench_spec.sh > /workspace/_serve_launch.out 2>&1 &'
```
Poll `serve_k3_bench_spec<N>.log` for `Application startup complete` / `/health`=200. Expect a clean
45-PIECEWISE + ~40-FULL capture. **Benign** startup noise: two `Failed to import Triton kernels ...
triton_kernels.matmul_ogs` ERRORs (the mxfp4 path we don't use). **Real** faults to catch fast:
`cannot get heuristic kernel` (asm MLA lookup miss — a qSeqLen/causal coverage gap), `died
unexpectedly`, `HSA_STATUS`/`0x1016`, `out of memory`.

### 3. Run the sweep (background; ~20–30 min full)
```bash
docker exec k3-dspark-benchmark bash -lc 'cd /workspace &&
  ROOT=/workspace/k3_dspark_longctx_bench PORT=8890 \
  setsid nohup bash _dspark_longctx_bench.sh > /workspace/_bench_run.out 2>&1 &'
```
Watch to completion with a backgrounded poll loop (grep `BENCH DONE`); check every point prints
`aiperf rc=0`. conc-48 runs first (heaviest). For a single point: `PAIRS="48 240"`.

### 4. Scan for perf issues FIRST, then build the full table
The focus of this bench is **ITL / TTFT / TPOT / interactivity → quickly identify perf issues that
hurt the agentic IX benchmark**. So run the diagnostic first:
```bash
docker exec k3-dspark-benchmark bash -lc 'cd /workspace &&
  python3 _dspark_perf_diag.py /workspace/k3_dspark_longctx_bench --tp 8'
```
It prints the interactivity table + SLA gates and **auto-flags** anomalies. Read the flags — they tell
you *where* and *what kind* of problem, so you know what to chase (and, just as important, what NOT to):
- **ITL non-monotonic (BUBBLE)** → a lower conc slower than a higher one = PIECEWISE/eager-attention
  fallback (the `get_mla_metadata_v1` bubble). Check FULL-decode capture at `M=(1+NUM_SPEC)×conc` and
  the `{12,36}`/capture-size ladder. A CLEAN monotonic ITL curve means the capture fix is working.
- **ITL knee/cliff** → super-linear decode cost at a transition → `profile-decode-bubble` skill.
- **TTFT cliff + cache eviction at the same transition** → the knee is a **memory** problem, not a
  kernel one: KV pressure evicts the shared prefix → prefill recompute. Lever = KV offload / prefix
  retention / pool sizing (memory `k3-offload-read-path-VALIDATED`, `k3-ttft-knee-prefix-cache-eviction`).
- **invalid OSL / low AL** → the point didn't hold; don't trust its latencies.

Then build the full acceptance/throughput table for the record + ATOM compare:
```bash
docker exec k3-dspark-benchmark bash -lc 'cd /workspace &&
  python3 _dspark_perf_table.py /workspace/k3_dspark_longctx_bench --tp 8'
```
Sanity-check `osl≈350` and ISL≈68,089 at every point (else the shape didn't hold). Compare against
the ATOM reference table (§4 of the specdecode7 doc) for the nspec=7 arm.

## Teardown
Stop the serve cleanly (graceful drain) when done — do **not** SIGKILL a native-offload serve
(memory `vram-leak-zombie-init-root-cause` / `kill-vllm-clean-gpu`). `pkill "vllm serve"` does not
free VRAM. If VRAM is left pinned, `docker restart k3-dspark-benchmark` reaps zombies and `/dev/shm`
weights survive.

## Don't
- **Don't enable KV offload (`KV_OFFLOADING_SIZE`) for this sweep.** MEASURED negative (nspec=7,
  2026-08-15): on the pool-of-1 single-shared-63.9k-prefix shape, offload makes high-conc TTFT
  **2.8–7× worse** (conc-48 5266→29023 ms), `external_prefix_cache_hits_total`=0 despite ~4.9 TB
  CPU→GPU thrash, and collapses the on-GPU prefix cache% from ~90% to 8–24% (the connector evicts the
  resident prefix). Offload only helps the agentic multi-distinct-prefix corpus, not this microbench
  (memory `k3-offload-harmful-on-longctx-bench-shape`, `k3-offload-read-path-VALIDATED`).
- Don't run with `MAX_NUM_SEQS<48` for the full sweep (caps top concurrencies).
- Don't use `--enforce-eager` (loses perf; debug only; memory `no-eager-mode`).
- Don't use aiperf for a quick yes/no — the 900s+warmup makes it ~1 hr; for smoke use a direct
  driver (memory `no-aiperf-for-smoke`). This skill IS the full aiperf benchmark, so it's fine here.
- Don't switch to TRITON_MLA "because the doc says so" — the asm path is the tuned perf path.
- Don't assume block vs synthetic; the numbers differ — confirm which the user wants.
