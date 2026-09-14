# Kimi-K3 on MI355X — two-week summary (2026-09-01 → 2026-09-14)

Goal throughout: close the conc-1 gap to B300 on the IX agentic benchmark.
The two metrics are **tok/s/chip** and **p90 interactivity**; prefill/TTFT is not
in either.

44 commits of ours in the window (the remainder are other teams' merges).

---

## 1. DCP8 + DSpark + DRAM offload: from not-booting to a valid long run

Start of the window, DCP8 with the DSpark draft did not survive an agentic run
on the asm fp8 MLA path. It now does, and produces valid CI-gated numbers.

What had to be fixed, each one a separate root cause:

| fix | what was wrong |
|---|---|
| draft parallel config | `create_draft_parallel_config` silently drops `decode_context_parallel_size`, so the draft lost DCP — a boot blocker |
| acceptance cp_size | `speculator.py` passed the *global* `cp_size` instead of `cp_sizes[gid]`, so acceptance was computed against the wrong context-parallel size |
| sharded draft | ported ATOM's sharded draft; took DCP8 acceptance length to 2.5–2.8 |
| KV-split cap | see §2 — the single biggest perf item of the window |
| symm_mem teardown | SRCU exit deadlock; wedged ranks that cannot be killed |

`DRAFT-REPL` (replicating the draft group) was built, debugged over four
commits, and then **retired** — upstream #53598 fixed the cache cliff it
existed to work around. The harnesses were retargeted at the asm cprr route.

## 2. Measured perf results

**KV-split cap — the big one.** Our own `_mla_max_split_per_batch = 32` was
applied to the DCP path only, starving decode at low batch. MLA decode is
lopsided (a few query rows against tens of thousands of KV rows), so at batch 1
the only parallelism is splitting the KV axis:

| cap | 32 | 64 | 128 | **256 (CU count)** | 320/384/512 |
|---|---|---|---|---|---|
| decode | 456 µs | 242 µs | 141 µs | **103 µs** | 103–106 µs |

It is a cap, not a mandate — at batch 8/14 the split count changes nothing.
End-to-end at conc-1: **ITL p90 9.43 → 8.18 ms, intvty p90 106.0 → 122.2**,
closing 71% of the DCP8-vs-DCP1 deficit.

**MLA reduce scratch: 9.35 → 2.38 GiB**, by giving aiter the tight split bound
(`max` → `min`) now that the cap is passed consistently to both the sizing and
the runtime call.

**wvSplitK skinny dead zone.** `n` in 6..9 fell out of the fast path. The N≤5
limit turned out to be register spill, not a correctness bound — clamping
`YTILE` lets it serve 6..9. Made stock-safe and env-tunable.

**Closed as dead ends** (measured, not assumed): attn_res launch tuning (1.00×),
ATOM's `_ATTN_RES_CONFIGS` (1.00×), aiter #4572 attn_res (parity), one-shot
all-reduce (1.11×, ~160 µs), exp2/tanh substitution (Triton already emits
`v_exp_f32`; no `v_tanh` on gfx950), MLA split count (256 already optimal),
multi-stream overlap at M=8 (a net loss — AMD explained the CU-contention
mechanism), one-shot MoE sort (slower, and hangs), a8w4 vs a16w4 MoE (identical).

## 3. The B300 comparison, and the analyser that produces it

`docs/k3_vs_b300_conc1_isl100k.md` (81 KB) — conc-1 ISL-100k, per-rank decode
breakdown, component totals, and a per-kernel section per component.

| component | MI355X µs | B300 µs | ratio | % of gap |
|---|---:|---:|---:|---:|
| MLA attention | 3,754 | 1,942 | 1.93× | **27%** |
| Communication | 3,818 | 2,727 | 1.40× | 16% |
| Glue/elementwise | 997 | 187 | **5.34×** | 12% |
| KDA | 1,753 | 1,085 | 1.62× | 10% |
| Norm/quant | 863 | 263 | **3.28×** | 9% |
| MoE routing/sort | 2,679 | 2,108 | 1.27× | 8% |
| Dense GEMM | 8,040 | 7,640 | 1.05× | 6% |

**Read this correctly:** it is a kernel-sum decomposition. Kernel-sum gap is
**26,642 vs 19,881 = 1.34×**; wall-clock step p50 is **27.23 vs 15.53 ms =
1.75×**. The difference is a ~4.4 ms *overlap* term B300 gets and we do not, so
the table explains the 1.34×, not the whole deficit.

The components with real headroom are the **high-ratio, launch-heavy** ones
(Glue 5.34×, Norm/quant 3.28× on 199 calls vs B300's 102), not the big-µs ones
that are already at parity.

**The analyser is now tested, because it was wrong.** It charged gap kernels to
the *next* step instead of the previous step's async tail, leaking prefill into
decode and inflating MI355X by ~14%. Fixed at source, plus 95 mutation-verified
tests (`test_trace_compare_k3.py`) covering the component classifier and the
stage split.

## 4. Upstreaming the DCP work

Branch `xguo/rocm-mla-dcp-cprr-verify`, 4 commits, +1077/−9 across 5 files,
rebased on current `origin/main`, DCO signed, ruff clean.

The feature: stock upstream routes **all** DCP verify through Triton
(`#51705`'s segmented route), because a spec-decode step always has `qlen > 1`.
So DCP + spec decode cannot reach the AITER ASM MLA path at all. This PR adds
the round-robin (cprr) asm route that can.

**Tests: 540 passed** (505 pre-existing + 35 new) against **505 on the
unmodified base** — no regressions. The new GPU test drives the real metadata
builder and the real asm kernel, simulating all 8 DCP ranks in one process, and
carries a positive control (wrong `cp_rank` must blow the error up).

Running upstream's own suite against the branch found **a real bug in our
code**: `_build_decode` built `g_kv_indptr` whenever `dcp_world_size > 1`, but
only the asm route allocates the buffer — so any DCP user on the *segmented*
route hit an assert. Our serve testing could never have found it, because we
always take the asm route.

Also verified the PR stands alone: reverting both python hunks of our aiter
patch leaves the suite at **540 passed**, identical. So upstream CI on stock
aiter will pass.

## 5. Corrections issued

Recorded because each one would have shipped a wrong number:

- **Retracted** a reported aiter cprr kernel bug — it was a harness artifact.
- **Corrected** the claimed aiter→vLLM PR ordering dependency. The aiter change
  is gated by `max_split_per_batch > 0` with default `-1`, so it is inert for
  any caller that does not pass a cap. Either order is safe.
- **Fixed** the trace analyser's prefill→decode leak (≈14% inflation).
- **Corrected** interactivity: the IX table's `frITL` is
  `full_decode_duration / OSL`, **not** aiperf's `inter_token_latency`. They
  differ ~9%. Mixing them reported a conc-1 arm as 140.1 when it was 133.0 — and
  on the wrong metric that arm sat *below* baseline, so the mistake **inverted
  the sign** of the verdict.

## 6. Benchmark infrastructure

Committed repro launcher for conc-1/2/4, `SERVE_ONLY` + `EXTRA_VLLM_ARGS`, fast
preflight on the prerequisites the launcher cannot fix, an A/B reporter that
computes frITL and prints a workload-identity check, and a fix for
`apply_vllm_patch` being unable to detect an already-applied *rebase variant*
(which force-applied the base patch on top and aborted the run).

Recurring tax, now documented: LMCache's `nightly-rocm` index is not immutable —
dev89, dev105 and dev134 have all been withdrawn under us.

## 7. Open / queued

| item | state |
|---|---|
| push the DCP PR | **ready**, needs approval; suite should be re-run on the current rebase |
| arm B (segmented route) A/B | staged; baseline has a shelf life (see below) |
| DSpark torch.compile A/B (2 arms) | staged, from upstream PR #56664 |
| LMCache IPC-event lifetime fix | local patch, should be a PR to LMCache |
| vLLM offloading eagle-prefix veto | local patch, should be a PR to vLLM |
| conc-1 glue Stage 2 | ~1.0–1.1 ms/iter of metadata-build launches, untouched |

**Shelf life:** upstream PR #56664 puts `@support_torch_compile` on
`K3DSparkModel`, which compiles the draft by default at mode 3. Once it merges,
every conc-1 number we hold — including arm A's `intvty p90 133.04` — becomes a
pre-change measurement.
