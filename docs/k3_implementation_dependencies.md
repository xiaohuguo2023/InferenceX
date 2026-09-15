# Kimi-K3 MI355X — implementation dependencies

Everything the DCP8 + DSpark + DRAM-offload recipe depends on, in one place:
upstream PRs we **consume**, PRs we **own**, and patches still carried locally.

Verified 2026-09-14 against vLLM `origin/main` and image
`vllm/vllm-openai-rocm:nightly-2671fedfc7ae604761990603fc736c0c4f21de57`
(vLLM `0.29.1rc1.dev9+g2671fedfc`, ROCm 7.2.3).

**All PR numbers, CI results and merge states re-verified 2026-09-15** by
querying GitHub — see §2.

Supersedes the wave planning in `patches/k3-dcp8/UPSTREAM.md`, which was
surveyed 2026-09-03 and has since gone stale in two places (V8 is dead, V6 is
filed).

---

## 1. Upstream vLLM PRs we consume — all MERGED

These are hard dependencies: the recipe does not work without them. All are
present in the pinned image, so nothing here needs chasing.

| PR | title | why we need it |
|---|---|---|
| **#51705** | [ROCm][MLA][DCP] Support causal multi-token verification | the DCP verify framework our asm route plugs into |
| **#53598** | [ROCm][DSpark][DCP] Serve prefix cache hits under DCP for Kimi-K3 | fixes the DCP prefix-cache cliff; closed the ~20pp agentic cache deficit |
| **#55472** | [Bugfix][Spec Decode] Preserve target parallel config (DCP) for DSpark | boot blocker — without it the draft loses DCP and asserts on `dcp_manager` |
| **#52047** | [Bugfix][AMD] Annotate draft KV cache groups on the hybrid grouping path | required once past #47891 for KV offload + DSpark |
| **#50183** | [Bugfix][Spec Decode] Fix NaN handling in rejection sampler tl.argmax | DSpark `0x1016` crash; resets on any rebuild-from-base |
| **#53388** | [Feature][Spec] Support disabling trailing prefix-cache block dropping | landed after our first survey; supersedes part of our offload work |
| **#53407** | [Bugfix][MRV2] Dispatch uniform decode to a padded FULL cudagraph | **ours, merged** — keeps uniform decode on FULL graphs instead of eager |

## 2. PRs we own

Status verified 2026-09-15 by querying GitHub directly.

### vLLM — 8 open, 3 merged

| PR | state | size | content | blocker |
|---|---|---|---|---|
| **#56991** | OPEN | +232/−3 | [ROCm][Bugfix][DCP] Validate the retention interval against the right block size | `pre-run-check` (see below) |
| **#56906** | OPEN | +137/−0 | [ROCm][Model][DCP] Default Kimi-K3's DCP combine to a2a | `pre-run-check` |
| **#56869** | OPEN / **draft** | +131/−0 | [Bugfix][Spec Decode][DCP] Align ranks before DFlash speculator graph capture | `pre-run-check`; device test is done, so it can be promoted to ready |
| **#56861** | OPEN | +1074/−9 | [ROCm][MLA] Add an AITER ASM round-robin decode route for DCP multi-token verify | `pre-run-check` |
| **#55118** | OPEN / **draft** | +31/−31 | [Bugfix][KV Offload] Don't pop a verified full-attn eagle prefix chunk | **CONFLICTING — needs rebase** |
| **#53487** | OPEN / **draft** | +100/−16 | [Kimi-K3] Split the KDA mixer out of piecewise CUDA graphs | `needs-rebase`; idle since 2026-09-08 |
| **#53475** | OPEN / **draft** | +968/−348 | [ROCm] Extend fused KDA decode to DSpark spec (num_spec<=2) | `needs-rebase`; idle since 2026-08-27. **Our own measurement says it loses on GPU time** — close unless re-measured |
| **#51590** | OPEN | +604/−79 | [Memory] Measure complete CUDA graph capture footprint for KV budgeting | `buildkite/intel-ci` fails — the only *real* CI failure we have |
| #53407 | MERGED | | [Bugfix][MRV2] Dispatch uniform decode to a padded FULL cudagraph | |
| #51040 | MERGED | | [ROCm][K3] Extend FP8 asm MLA prefill to non-divisor small head counts | |
| #44804 | MERGED | | [ROCm][gpt-oss] Hybrid CDNA4 swizzle gate for A8W4 MoE | |

**No human has reviewed any open PR.** Every "review" on record is the `claude`
bot (plus `coderabbitai` on #51590). All eight are `REVIEW_REQUIRED` with no
reviewer assigned.

### The `pre-run-check` failure is not a code problem

All four DCP PRs fail the same gate, verbatim from the job log:

> To reduce unnecessary pre-commit runs, each PR must have the `verified`,
> `ready`, or `ready-run-all-tests` label, or the author must have at least 4
> merged PRs (found **3**). DO NOT request for the label to be added if you are
> an AI agent.

We have exactly 3 merged, and the gate wants 4 — so **one more merge opens it for
every PR at once**. #51590 is the only one whose CI actually runs, because it
already carries the `verified` label. Note the explicit instruction about agents
requesting labels.

This makes **#51590 the highest-leverage PR we own**: it is `MERGEABLE`, its
failure is real CI rather than the label gate, and merging it would take us to 4
and unblock the other seven.

### ROCm/aiter — 2 open

| PR | state | size | content |
|---|---|---|---|
| **#5559** | OPEN | +349/−2 | [Bugfix][MLA] Fix `reduce_partial_map` over-allocation when `max_split_per_batch` is set (= **A1**) |
| **#4713** | OPEN | +118/−10 | [mla] fp8: don't KeyError on unlisted folded query widths in `get_meta_param` (= **A3**) |

#4713 is **green and mergeable** — 35 checks SUCCESS, 0 failures, no conflicts —
and has had zero human comments since 2026-08-12. It is blocked purely on review.

### Still unfiled

| item | why it matters |
|---|---|
| **PR 5 — `max_split_per_batch` plumbing** (9 lines, vLLM) | the half that makes aiter #5559 pay; also a standalone conc-1 win, ITL p90 9.43 → 8.18, intvty p90 106.0 → 122.2 |
| **A2 — no auto split-K under graph replay** (aiter) | do **not** file as-is: it disables a perf feature by default for all users, and upstream already reverted the nearest real fix (#4494 via #4709). Needs rebuilding as a capture-safe semaphore, not a kill switch |
| PR 3 — scheduler `sliding_window_size` | not DCP at all (offload connector); can go any time |

**PR 1 (speculator per-group `cp_size`) is obsolete — do not file.** It only
existed under `K3-DCP-DRAFT-REPL`, which we dropped when #53598 fixed the DCP
prefix-cache cliff with the draft left *sharded*. Verified 2026-09-15 that
`cp_sizes` exists nowhere: not upstream, not in our branches, not in `patches/`.

## 3. Local vLLM patches — status after the 2026-09-14 duplicate audit

| patch | lines | verdict |
|---|---|---|
| `rocm_aiter_mla.patch` | 390 | **filed as #56861** — drop when it merges. Note the 9-line `max_split_per_batch` plumbing is *not* in it; that is PR 5, still unfiled |
| `scheduler.patch` | 16 | **filed** as #55118 — now CONFLICTING, needs a rebase |
| `speculator.patch` | 10 | **filed as #56869** (draft). Quiesce ranks before DCP speculator capture; boot blocker (GPU fault). `torch.cuda` → `torch.accelerator` (RFC #30679) done |
| `config.patch` | 27 | **filed as #56906.** Perf, measured — default K3's DCP combine to `a2a`. Per combine call on MI355X: T=5 1.13x, T=48 1.15x, T=144 1.35x vs `ag_rs`, cos-similarity >= 0.999994. One `all_to_all_single` instead of `allgather(lse)` + `reduce_scatter(out)`, and combine runs per MLA layer per decode step |
| `retention_alignment.patch` | 57 | **filed as #56991.** Bug fix — retention interval is validated against `scheduler_block_size` unconditionally, but the granularity a prefix-cache hit actually lands at is `hash_block_size` when fine-grained partial-hash hits are on, which under DCP is `scheduler_block_size // dcp_world_size`. Affects any DCP user, not just us |
| `cp_common.patch` | 125 | **NOT part of the K3 story — does not block anything.** Two unrelated changes that share a file: (a) an NVLS multicast probe skip on ROCm, which our own notes say *not* to file (justified only by our unmerged a2a port, and #33274 shows someone already tried and failed); (b) ordered symm_mem teardown, a genuine lifecycle bug that wedges a whole box with unkillable D-state ranks. Neither is on our path — DCP dispatches PYNCCL, so no symm_mem mesh is ever built. (b) deserves a standalone PR on its own merits, whenever; it is not K3 work and nothing waits on it |
| `speculative_draft_dcp.patch` | 13 | **DEAD** — #55472 merged and is in our image, with a more robust fix. Delete |
| `dcp_a2a_pack_mask.patch` | 69 | **probably dead** — upstream ships `tests/v1/attention/test_dcp_a2a_pack_mask.py`. Confirm, then delete |

**Correction (2026-09-14):** the first version of this table called `config.patch`
"convenience only" and `retention_alignment` "not load-bearing". Both were wrong.
"Not load-bearing **for us**" is a statement about our configuration, not about
whether the change is a real upstream bug — and `config.patch` was carrying a
measured perf result the summary had simply dropped. Judge each patch on its own
merits before deciding it is not worth filing.

## 4. Non-vLLM dependencies

### ROCm/aiter — `patches/k3-dcp8/aiter/`
A1-A4 are **live in the image** — verified 2026-09-15 by grepping
`/opt/aiter-local` in `k3-2671`, not inferred from the patch files. A5 is
conditional and is currently **not** applied (see its row).

| id | patch | change | PR | load-bearing |
|---|---|---|---|---|
| A1 | `0001` | `attention.py`: fix `reduce_partial_map` over-allocation when `max_split_per_batch` is set (`max`→`min`) | **[#5559](https://github.com/ROCm/aiter/pull/5559)** OPEN | **yes** — MLA reduce scratch 9.35 → 2.38 GiB; what let FULL cudagraphs fit under DCP |
| A2 | `0001` | `asm_gemm_a16w16.cu`: don't auto-select split-K under graph replay (`AITER_ALLOW_SPLITK`) | **UNFILED — do not file as-is** | **yes** — boot blocker, all waves spin forever at seqs=64 warmup |
| A3 | `0001` | `mla.py`: fp8 MLA `get_block_n_fp8` fallback + 80/96/112 entries | **[#4713](https://github.com/ROCm/aiter/pull/4713)** OPEN | **yes** — `KeyError` on any unlisted `nhead * max_seqlen_q`. NOT on the cprr path (verified), so not a dependency of our vLLM PR |
| A4 | `0002` | K3 bf16 tuned-GEMM rows (CSV) | **UNFILED** (data, probably not upstreamable as-is) | **yes** — 371 conc-1 tuned-config misses; absence HSA-faults the launcher |
| A5 | `0003` | `flydsl/kernels/buffer_ops.py`: 0.3.2 `aux` attr | **UNFILED**, and not upstreamable — it is a local compat shim | **conditional, NOT applied today.** Only needed on an image whose flydsl is >= 0.3.2 while the transplanted aiter was built against 0.3.0; both `k3-2671` and `k3-r72` ship flydsl **0.3.0**, so the shim is correctly absent. Applied by hand (`git apply`), never automatically — **re-check on every image bump**, because aiter is transplanted rather than rebuilt and a flydsl minor bump silently breaks it with `TypeError: RawPtrBufferStoreOp.__init__() takes 5 positional arguments but 6 were given` at KV-cache profiling, taking every worker with it |

**So two of the five carried aiter changes have a PR: #5559 and #4713.** A2, A4
and A5 have none, and A5 is a version-compat shim that should never be filed.

**Correction (2026-09-15):** an earlier revision of this row claimed A5 was live
in the image. It is not — that was a grep for `aux` matching unrelated lines. The
real marker is `aux=aux_attr`, absent from both containers.

**A2 must not be filed in its current form.** It disables a perf feature by
default for every aiter user via an env opt-in, and upstream already merged and
then **reverted** the nearest real fix (#4494, reverted by #4709). Rebuilding it
as a capture-safe semaphore — the way #4715 does for FlyDSL — is the route, not a
kill switch.

### Our other aiter PRs — NOT K3 dependencies

Listed so nobody re-derives whether they matter to the recipe. None is in
`patches/`, and #4715's marker is absent from the image (checked).

| PR | state | why it is not a dependency |
|---|---|---|
| #5487 | DRAFT | K3 latent FHMoE **prototype**; not carried, not on the serving path |
| #4715 | DRAFT | FlyDSL split-K capture-safe semaphore — **not applied in our image**; it is the model for rebuilding A2, not a dependency |
| #4647 | OPEN | FlyDSL MoE stage-1 scratch reuse; not carried |
| #4108 | OPEN, **CHANGES_REQUESTED since 2026-08-02** | A8W4 CDNA4 scale addressing; not carried. Blocked on us for 6 weeks |

Merged and therefore already in the image: #3580, #3428, #3372 (gfx950 MoE A8W4
tuned configs / dispatch), #1653, #1607, #1464, #725, #659.

A1 and vLLM's `max_split_per_batch` are **order-independent** — the aiter change
is gated on `max_split_per_batch > 0` with default `-1`, so it is inert for any
caller that does not pass a cap to the sizing call. Landing either first is safe;
the memory reclaim needs both.

**aiter #4494 was REVERTED upstream** — keep our local split-K guard.

### LMCache — `patches/k3-lmcache/`
`0001-retain-exported-ipc-events.patch`. Callers export an event's IPC handle
then drop their reference, so CPython destroys the event on return; the platform
only defers the real destroy while recorded work is outstanding, so once the copy
completes the handle stops being openable and the importer gets
`hipErrorInvalidValue`. Our fix is a bounded retain ring
(`LMCACHE_EVENT_IPC_RETAIN`).

**Do NOT file this — upstream LMCache PR #5116 supersedes it**
(https://github.com/LMCache/LMCache/pull/5116, `[fix][mp] keep the server's
exported completion event alive (ROCm 10.0)`, +110/-17, **OPEN** as of
2026-09-15, author sammshen). Same root cause, better fix:

| | ours | #5116 |
|---|---|---|
| interprocess event creations | **one per transfer** | one per registered context |
| live ROCr signals | up to **4096** (the ring) | one per context |
| tuning knob | `LMCACHE_EVENT_IPC_RETAIN` | none needed |

It gives each registered context one long-lived `completion_event`, re-recorded
and re-exported per transfer. Because a context's transfers share a stream, a
late-imported handle is conservative rather than early. That removes the
per-transfer allocation our patch keeps, so it should also be *faster* on the
offload hot path — unmeasured, but mechanical. The tradeoff is slight
over-synchronisation: a consumer may wait on later transfers than its own.

**Status: not in anything we run.** #5116 is unmerged; the ROCm-10 image ships
lmcache **0.5.3** with neither fix, and our pinned `0.5.6.dev3+rocm7.2` has only
ours (applied by the recipe at `kimik3_fp4_mi355x_mtp.sh:342`).

**Keep our patch until #5116 merges and reaches the AMD nightly**, then delete
it — which also ends the re-pin tax below.

**We should comment on #5116 with our data.** It states that "CUDA and ROCm 7.2
tolerated violations". We measured otherwise on ROCm 7.2.3: 17 min of clean
serving, then `EngineDeadError` and a 10.4% client error rate (see the recipe
comment at :338). That is independent corroboration and strengthens the case for
merging.

Also: LMCache's `nightly-rocm` index is **not immutable**. dev89, dev105 and
dev134 have all been withdrawn under us. Current pin `0.5.6.dev3+rocm7.2`.

### Model revisions — pin these, `main` has moved
| model | revision | note |
|---|---|---|
| `moonshotai/Kimi-K3` | `a590ce090cb049c93a33dfe8c208ec652aa20503` | `refs/main` is now `f831ab66…` — an unpinned fetch gets **different weights** |
| `Inferact/Kimi-K3-DSpark` | `cf6b8244620e7ea4b0651d214f28e89eac75bed6` | |

Re-stage after any reboot with `~/work/devshm_scripts/_fetch_k3.sh` (`/dev/shm`
is tmpfs; a reboot wipes 1.5 TB and resets the mount to 1.5 T — remount at
`size=2560G`, note `2.5T` is rejected).

## 5. Watch list

| PR | why it matters to us |
|---|---|
| **#56664** (open) | `@support_torch_compile` on `K3DSparkModel`. No `enable_if`, so at our `mode:3` it compiles the draft **by default**. On merge, every conc-1 number we hold becomes a pre-change measurement |
| **#56723** (open, draft, maintainer-authored) | "Replicated DFlash/DSpark drafts were still treated as DCP-sharded under DCP > 1" — directly adjacent to our ATOM-sharded draft; may overlap what we carry |
| **#54627** (open, fululi12) | "Apply `prefill_schedule_interval` outside data parallelism". `EngineCore._should_throttle_prefills()` returns `False` unconditionally, so the option is **silently a no-op for us today** — verified in our pinned image at `v1/engine/core.py:584`. Its stated motivating deployment is "a single engine using decode context parallelism (`--decode-context-parallel-size 8`)", i.e. exactly our shape. Directly targets prefill leaking into decode, which is our diagnosed conc-1 interactivity cause |
| **#54625** (open, fululi12) | "Add cache-aware admission ordering". Admits requests whose prefix is already resident ahead of cold ones within a bounded look-ahead, so a cold admission cannot evict blocks a queued request still needs. Matches our TTFT-knee / prefix-cache-eviction finding. Flags `--cache-aware-admission-window` (0) and `--cache-aware-admission-threshold` (0.0), both off by default; requires `fcfs` + prefix caching |

**All three of #56664, #54625 and #54627 are by the same author (fululi12)** — a
coordinated K3 agentic effort. Engage with these rather than duplicating them.

### Scheduler flags: where they are expected to pay

At **conc-1** the client keeps one request in flight, so the waiting queue is
near-empty (#54625 has little to reorder) and a request's decode cannot begin
until its own prefill completes (#54627 has little to defer). Both should matter
much more at **conc 8+**, where prefill and decode genuinely compete — which is
where our sweep is weakest. Both are off by default, so trying them is a
serve-flag experiment, not a code risk.

## 6. Standing lesson

`UPSTREAM.md` was written 2026-09-03 and by 09-14 had two entries that were no
longer true — one fixed upstream, one already filed by us. **Check every patch
against current `origin/main` and the PR search before proposing work on it.**
The check that caught #55472 took two minutes and saved a duplicate PR.

---

## 7. Decomposition review

Each PR must clear three bars: **independent** (lands alone, no ordering
constraint), **testable** (a test that fails if the change is reverted), and
**one clear category** with evidence — enablement, bug fix, or measured perf.

| PR | category | evidence it is real | independent | testable |
|---|---|---|---|---|
| **#56861** cprr route + split cap | **enablement** | ASM MLA path is unreachable under DCP + spec decode; every step has qlen>1 so all decode goes to Triton | yes | yes — 35 tests, mutation-checked |
| **speculator barrier** (built) | **bug fix** | boot blocker; GPU fault at capture | yes | unit (ordering + skip-without-DCP), mutation-checked |
| **#55118** offload eagle prefix | **bug fix** | drops a verified prompt chunk, vetoes <=1-chunk prefixes | yes | unit |
| **`config.patch`** a2a default | **perf** | 1.13x–1.35x per combine call, cos-sim >= 0.999994 | yes | unit on default resolution |
| **`retention_alignment`** | **bug fix** | validates against the wrong granularity under DCP | yes | unit — pure block-size arithmetic |
| **`cp_common`** | **bug fix** | symm_mem teardown wedges the box | yes | **weak** — teardown race; needs an honest caveat, as the speculator PR does |
| **LMCache IPC events** | **bug fix** | exported handle stops being openable -> `hipErrorInvalidValue` | yes (other repo) | unit on the retain ring |

### Is #56861 correctly bundled?

It carries two things — the cprr route (enablement) and `max_split_per_batch`
(perf). They are **not** separable: `forward_mqa` has
`assert decode.mla_num_kv_splits > 0` *inside* the `if asm_dcp_heads:` branch,
and the cap is only set when `dcp_world_size > 1` and only consumed on the asm
path. The cap does nothing without the route. One feature, one PR.

### The one real decomposition defect

`patches/k3-dcp8/aiter/0001-k3-dcp8-code.patch` bundles **three unrelated fixes**
in one file, touching three subsystems:

| | file | category |
|---|---|---|
| A1 | `aiter/ops/attention.py` — `reduce_partial_map` over-allocation | perf/memory (9.35 -> 2.38 GiB) |
| A2 | `csrc/py_itfs_cu/asm_gemm_a16w16.cu` — no auto split-K under graph replay | bug fix (boot blocker, waves spin forever) |
| A3 | `aiter/mla.py` — `get_block_n_fp8` fallback + 80/96/112 | bug fix (`KeyError`) |

These share nothing but our repo. They should be **three separate aiter PRs**.
A4 (tuned-GEMM CSV) is data and is probably not upstreamable as-is.

**A1 is now filed as [aiter #5559](https://github.com/ROCm/aiter/pull/5559)** (open,
+349/−2, 3 files). A2 and A3 are still to be split out, duplicate-checked first.

### Not PRs

`speculative_draft_dcp` (dead) and `dcp_a2a_pack_mask` (probably dead) are
deletions, not submissions — about 82 lines we can stop carrying.
