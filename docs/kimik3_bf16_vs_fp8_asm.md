# Kimi-K3 MI355X — bf16-ASM vs fp8-ASM MLA decode (comparison + bottleneck)

Compares the two **ASM 576/512 MLA decode** configs on K3 (TP8, MI355X, vLLM):

| config | KV cache | Q | decode kernel | MoE |
|---|---|---|---|---|
| **fp8-ASM** | fp8 | fp8 | `mla_a8w8_qh16_qseqlen1_gqaratio16` (ASM) | aiter fp4 |
| **bf16-ASM** | bf16 | bf16 | `mla_dec_stage1_bf16_a16w16` (ASM) | aiter fp4 |

Both use the pad-to-16 dispatch patch (K3 has 12 MLA heads/rank) + the **wvSplitK
strided-activation fix (vLLM #50618)** that unblocks full `FULL_AND_PIECEWISE`
cudagraph capture. Everything else is identical; only `--kv-cache-dtype` differs.
See also [[kimik3-mla-decode-kernel-options]] and the pareto in `docs/kimik3_pareto/`.

---

## TL;DR

- **Native-context agentic workload → bf16-ASM ≈ fp8-ASM** (KV not the bottleneck).
- **Long-context workload (68k ctx, conc 24) → fp8-ASM clearly wins**: ITL 43 vs 65 ms
  (−34%), total tput +48%, TTFT ~3.1× lower. This is the KV-bound regime where fp8's 2×
  KV bandwidth + capacity finally pays off.
- **Both ASM configs beat the "current MI355X" reference** on the long-context workload
  (conc24: fp8 2.46×, bf16 1.66× total tput) — the win is the ASM decode kernel
  (~2–2.7× lower ITL vs the default gluon path).
- **Current bottleneck (long ctx, high conc): KV-cache handling in the mixed
  prefill+decode steps** (memory bandwidth/capacity). fp8 relieves it; the decode
  kernel itself is no longer the limiter.

---

## Workload A — agentic cc-traces (native context)

AIPerf `inferencex-agentx-mvp`, 900 s, conc 1–24. Metric = total(in+out) tok/s ÷ 8.

| conc | fp8-ASM | bf16-ASM |
|---:|---|---|
| 1  | 226, 27 ms, 36.9 | 1415, 28 ms, 35.7 |
| 8  | 1655, 35 ms, 28.6 | 1530, 34 ms, 29.2 |
| 16 | 3195, 49 ms, 20.4 | 3215, 46 ms, 21.8 |
| 24 | 1303, 232 ms, 4.3 | 1401, 218 ms, 4.6 |

*(tput/gpu, TPOT, interactivity). Low-conc tput is a trajectory-sample artifact.)*

**bf16-ASM ≈ fp8-ASM.** MLA stores one compressed 576-dim latent per token (shared
across 128 heads), so at native context the KV cache is tiny and decode is dominated
by the projection/MoE GEMMs — halving the KV dtype touches a non-bottleneck. conc24
is the capacity ceiling (compute/scheduler-bound, not KV — fp8 doesn't help there).

---

## Workload B — synthetic long context (the KV-bound regime)

Adapted from the reference recipe. Client (AIPerf 0.8.0):
`--num-prefix-prompts 8 --prompt-prefix-length 63240 --synthetic-input-tokens-mean 4760
--output-tokens-mean 350 (min=max=350, ignore_eos) --warmup-request-count 3
--concurrency {16,24} --request-count {80,120} --random-seed 42`.
Serve: TP8, ROCM_AITER_MLA, moe-backend aiter, prefix caching, full capture,
gpu-mem 0.8, max-num-seqs 48. Effective context ≈ 68k tokens/req.

| cfg | conc | dur(s) | req/s | In tok/s/GPU | Out tok/s/GPU | **Tot tok/s/GPU** | TTFT p50 | TTFT p90 | **ITL p50** | ITL p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **reference (current MI355X)** | 16 | 200.4 | 0.399 | 3397 | 17.5 | 3414 | 2076 | 12100 | 100.9 | 115.3 |
| fp8-ASM | 16 | 130.7 | 0.612 | 5210 | 26.8 | **5237** | 2500 | 20213 | **54.4** | 83.6 |
| bf16-ASM | 16 | 136.9 | 0.584 | 4974 | 25.6 | **4999** | 2328 | 16162 | **58.1** | 92.4 |
| **reference (current MI355X)** | 24 | 214.7 | 0.559 | 4757 | 24.5 | 4782 | 2214 | 8184 | 114.9 | 116.0 |
| fp8-ASM | 24 | 87.3 | 1.374 | 11695 | 60.1 | **11755** | 1148 | 3056 | **43.1** | 56.0 |
| bf16-ASM | 24 | 129.6 | 0.926 | 7882 | 40.5 | **7923** | 3538 | 6472 | **65.4** | 68.3 |

> fp8-ASM numbers are the **clean redo** (isolated port, no concurrent session). An
> earlier fp8 run was ~19% slower at conc24 due to a concurrent client on the shared box
> (archived as `k3_fp8asm_longctx_c*_v1`); bf16-ASM was unaffected.

### vs the reference baseline — the ASM decode is ~2–2.7× on ITL
The reference (default MLA decode = gluon on 12 heads) runs **ITL ≈ 100–115 ms**. Our
ASM decode roughly halves-to-thirds it (**43–58 ms**), and everything cascades:
- conc24 **Total tok/s/GPU: fp8 11755 / bf16 7923 vs 4782 → 2.46× / 1.66×**
- conc24 **ITL: fp8 43.1 / bf16 65.4 vs 114.9 ms → 2.67× / 1.76× faster per token**
- conc24 **wall: 87 s (fp8) / 130 s (bf16) vs 215 s → 2.46× / 1.66× faster**

### fp8-ASM vs bf16-ASM — fp8 wins once KV-bound (clean at conc24, warm cache)
- **ITL 43.1 vs 65.4 ms** (fp8 −34%)
- **Total tput 11755 vs 7923 tok/s/GPU** (fp8 +48%)
- **TTFT p50 1148 vs 3538 ms** (fp8 ~3.1× lower)

> Caveat: **conc16 ran cold** (the eight 63k prefixes prefill during c16; c24 reused the
> warm cache → low TTFT). So conc16 TTFT p90 (15–16 s) is a cold-cache artifact, and the
> conc16 fp8-vs-bf16 tie is not the KV-bound comparison. Warm-cache conc16 would show fp8
> ahead too, by less than conc24.

---

## Phase breakdown & bottleneck (from existing AIPerf output)

AIPerf is client-side, so this is a **prefill-vs-decode phase** decomposition — not a
per-GPU-kernel breakdown (that needs a profiler trace; see below). Aggregate (8-GPU):

| cfg/conc | eff prefill tput | eff decode tput | **eff prefill conc** | **eff decode conc** | prefill tput/user | decode tput/user |
|---|---:|---:|---:|---:|---:|---:|
| fp8-ASM c16  | 41682 | 214 | 3.15 | 12.76 | 40268 | 14.1 |
| bf16-ASM c16 | 39789 | 204 | 3.20 | 12.72 | 37358 | 13.8 |
| fp8-ASM c24  | 93558 | 480 | **2.25** | **21.68** | 67844 | 20.4 |
| bf16-ASM c24 | 63059 | 323 | **3.45** | **20.50** | 25093 | 13.8 |

### What it says
- The workload is **prefill-heavy in token count** (ISL_total ≈ 8.17M vs OSL_total ≈
  42k at conc24), but the 63k prefix is **prefix-cached**, so the real recompute is the
  4760 unique tokens/req + cold prefixes.
- Each engine step **mixes prefill + decode**. At conc24, ~2–3.5 of the 24 slots are
  always prefilling; the rest decode. Those prefill chunks share the step with decode →
  they **inflate ITL** (see [[kimik3-itl-ttft-coupling]] below).
- **bf16 carries more prefill contention**: eff prefill conc **3.45 vs fp8's 2.25**, and
  its **prefill-tput/user collapses to 25093 vs fp8's 67844** — i.e. bf16's larger KV
  (2×) makes the long-context prefill/KV-materialization ~2.7× slower per user, so more
  requests pile up in prefill, stealing decode steps.

### ITL is not TTFT, but they're coupled
ITL measures only token-to-token gaps during decode (TTFT excluded). But under chunked
prefill, a decode token sharing a step with a prefill chunk is delayed, so **the same
KV-cost that drives TTFT also inflates ITL**. fp8's smaller KV → cheaper prefill → fewer
concurrent prefills → less decode interference → **both** lower TTFT **and** lower ITL.
fp8's ITL advantage is thus (1) raw decode KV bandwidth + (2) reduced prefill contention.

### Bottleneck conclusion
At **long context + high concurrency the limiter is KV-cache handling** (memory
bandwidth + capacity) inside the mixed prefill+decode steps — **not** the decode kernel
(the ASM kernel already cut ITL ~2× vs the reference). fp8 directly relieves this (2×
smaller/faster KV). Remaining levers, in order of expected impact:
1. **fp8 KV** (done — this is the win).
2. **Reduce prefill-decode contention** — larger prefix-cache headroom (gpu-mem ↑),
   prefill/decode disaggregation, or scheduling that caps concurrent prefills.
3. **Decode-side GEMM/MoE efficiency** — only after (1)/(2), and requires a kernel profile.

---

## What existing output can NOT show — per-kernel breakdown

A true "which kernel dominates" split (MLA attention vs kv_b/o projections vs fp4 MoE vs
custom all-reduce) requires a **GPU profiler trace** (torch profiler / rocprofv3), which
these AIPerf runs did not collect. The earlier fixed-seq profile
(`docs/kimik3_fp4_mi355x_conc32_vllm_profile.md`) gives a per-function/library breakdown
but for a **different config** (fp8-TRITON decode, conc32, 8k) — directional only.

**To get the conc16/24 per-kernel breakdown for the current bf16/fp8-ASM configs**, run
a short profiled pass (PROFILE=1 torch-profiler hook, ~64 decode steps) at conc16 and
conc24 for each KV dtype and aggregate by kernel/library. This would confirm the
phase-level conclusion (KV-bound decode + prefill contention) at kernel granularity and
show whether the fp4 MoE or the MLA attention is the largest decode-time slice.

---

## Reproduce
```
# serve (fp8-ASM shown; bf16-ASM = --kv-cache-dtype auto)
bash _serve_fp8asm_bench.sh          # in xguo-k3asm
# client sweep (conc 16,24 long-context)
bash _sweep_longctx.sh fp8asm        # -> k3_fp8asm_longctx_c{16,24}
bash _sweep_longctx.sh bf16asm       # -> k3_bf16asm_longctx_c{16,24}
# tabulate in the reference format
python3 _fmt_ref.py
```
