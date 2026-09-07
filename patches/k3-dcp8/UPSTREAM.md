# Upstream PR plan

We cannot pin the image. Every change in this bundle has to land upstream or be
carried forever, so this is the plan for getting each one in — and, first, the
decision about which ones are worth filing at all.

Surveyed 2026-09-03. **Criticality audited 2026-09-05** against the config the
agentic benchmark actually runs, which changed the picture for four items.
The four upstream PRs this bundle *consumes* — vLLM #50183, #52047, #53598 and
#51705 — are all merged, so there is nothing to chase there.

## The criticality filter

The bundle grew during debugging, and debugging finds real bugs in paths you
later stop using. "Is this a genuine upstream bug?" and "does our benchmark
depend on it?" are different questions, and only the second one earns a place
in the first wave. So each item below is checked against a live DCP8 serve
(`results_ixci/c12_prof2_dcp8/vllm_command.txt` and `server.log`):

| id | change | file | size | load-bearing for us? | evidence |
|---|---|---|---|---|---|
| V6 | asm round-robin CP path for DCP multi-token verify | `mla/rocm_aiter_mla.py` | ~+352/−9 | **yes** | 16 `cprr` hits in the serve log; the alternative is TRITON_MLA, measured +44% ITL / −43% tok/s |
| V8 | propagate DCP settings into the draft's `ParallelConfig` | `config/speculative.py` | +3/−0 | **yes** | boot blocker under DCP8 + DSpark: `assert isinstance(self.dcp_manager, MLADCPManager)` |
| V7 | ~~let a same-process offload connector keep `interleave=1`~~ | `config/vllm.py` | +8/−1 | **retired** | upstream narrowed the realignment to `NixlConnector`; patch **deleted** 2026-09-07 |
| V3 | quiesce ranks before DCP speculator graph capture | `dflash/speculator.py` | +8/−0 | **yes** | boot blocker — GPU fault at capture without it |
| A1 | take the tighter split-tile bound when a cap is supplied | `ops/attention.py` | +9/−1 | **yes** | 9.35 GiB → 2.38 GiB fp32 MLA reduce scratch; this is what let FULL cudagraphs fit under DCP |
| A2 | ASM a16w16: don't auto-select split-K under graph replay | `asm_gemm_a16w16.cu` | +39/−2 | **yes** | boot blocker — all waves spin forever at seqs=64 warmup |
| A3 | fp8 MLA `get_block_n_fp8` fallback + 80/96/112 entries | `aiter/mla.py` | +2/−1 | **yes** (fallback) | `KeyError` crash on any unlisted `nhead * max_seqlen_q` |
| A4 | K3 bf16 tuned-GEMM rows | `configs/*.csv` | +765/−694 | **yes** | 371 conc-1 tuned-config misses; absence also HSA-faults the agentic launcher |
| V2 | default Kimi-K3 to the `a2a` DCP combine | `models/config.py` | +24/−0 | **convenience** | we pass `--dcp-comm-backend a2a` explicitly; the flag already does this |
| V1 | don't veto ≤1-chunk prefixes on the full-attention EAGLE offload path | `offloading/scheduler.py` | +13/−1 | **no** | we run `LMCacheMPConnector`, not vLLM's native offloading connector; conc 1–4 run no offload at all |
| V4 | ordered symmetric-memory teardown | `ops/cp_common.py` | ~+80 | **no** | DCP group dispatches `PYNCCL`; no symm_mem mesh is ever built |
| V5 | skip the NVLS multicast probe on ROCm | `ops/cp_common.py` | ~+16 | **no** | same — the probe only runs when symm_mem is available |

Two consequences worth stating plainly:

- **V4 is the best piece of engineering in the bundle and is not on our critical
  path.** It is a genuine kernel-level lifecycle bug that wedges the whole box,
  affecting any ROCm DCP user who enables symm_mem. We stopped being one when we
  standardised on `a2a`. File it — but on its own merits, not as part of the K3
  performance story, and not ahead of the seven items our numbers depend on.
- **V1 and V5 should not be filed as-is.** V1 fixes a path we no longer exercise
  and carries real supersession risk (#53388 landed after our pinned image). V5
  is justified in-tree by our own unmerged a2a port, has no upstream-only
  rationale written, and #33274 shows someone already tried guarding this probe
  and failed. Neither is performance or functionality critical for us.

---

## Wave 1 — the seven the numbers depend on

### V6 · asm round-robin CP path for DCP multi-token verify
`v1/attention/backends/mla/rocm_aiter_mla.py`, ~+352/−9, behind
`VLLM_ROCM_AITER_MLA_DCP_VERIFY`.

Upstream's DCP multi-token verify is Triton, so stock vLLM runs the entire DCP
decode on Triton under DSpark. Our head-to-head has the causal-asm draft beating
TRITON_MLA by +44% ITL and −43% tok/s for +1.5% acceptance length. This is the
single largest performance item on either list, and the reason the whole bundle
exists.

*The most contested file we touch.* **#54899** OPEN (okorzh-amd, native-tile head
pad) **blocks this** — route through its `get_actual_mla_num_heads` rather than
duplicating the pad in `_NATIVE_CPRR_HEADS`. **#51705** MERGED 08-31 (causal
multi-token verification) is what V6 builds on. **#51171** MERGED 08-30 (FULL
cudagraphs for AITER MLA spec decode) and **#51647** MERGED 08-18 (pad
non-aligned AITER MLA heads, which #54899 extends) are the near neighbours.
**#51040** MERGED 08-26 is ours and worth citing as precedent. **#53815 /
#53816 / #53814 / #53587** are seungrokj's four same-day closures in this exact
area — ask why before writing. **#52377** MERGED (sparse MLA metadata after the
DCP Manager refactor).

Ours already open, chase rather than refile: **#51590**, **#53475** (draft),
**#53487** (draft).

Split it. ~352 lines behind one env var is not a reviewable unit; the head-count
handling, the global page-indptr construction and the kernel-selection change
are three separable stories.

### V8 · propagate DCP settings into the draft's `ParallelConfig`
`config/speculative.py`, +3/−0. **New 2026-09-07.**

`create_draft_parallel_config` builds a fresh `ParallelConfig` for the draft and
copies only tp/pp/executor/loading-workers/all-reduce/nsight/placement. It drops
`decode_context_parallel_size`, `cp_kv_cache_interleave_size` and
`dcp_comm_backend`. The K3 draft's MLA layer builds its `MLADCPManager` only when
`parallel_config.decode_context_parallel_size > 1` (`models/kimi_k3/nvidia/mla.py`),
so it ends up with `dcp_manager=None` — but the metadata builder reads the
**global** DCP group (`mla_attention.py`, `get_dcp_group().world_size` = 8) and
trips `assert isinstance(self.dcp_manager, MLADCPManager)`. Any DCP8 + DSpark
serve fails to boot without this.

Propagating the three fields is also what gives the ATOM-style **sharded** draft,
which is the configuration DCP8 was validated on (see
`k3-dcp-atom-sharded-draft-port`). Upstream framing: the draft inherits the
target's parallelism everywhere else; DCP being omitted looks like an oversight
from when DCP predated speculative support, not a deliberate choice. Small,
self-contained, and the assert makes the failure mode concrete — this is the
easiest of the vLLM items to argue.

### V7 · ~~let a same-process offload connector keep `interleave=1`~~ (RETIRED)
`config/vllm.py`, +8/−1. **Deleted 2026-09-07 — do not file.**

The premise no longer holds. Upstream's `adjust_dcp_kv_cache_interleave_size` now
returns early unless the transfer config `has_connector("NixlConnector")`, so an
LMCache offload connector never triggers the realignment and
`cp_kv_cache_interleave_size` stays 1 on its own. Verified on
`nightly-rocm100-e962733e`. The patch and `VLLM_DCP_KEEP_INTERLEAVE` are both
gone from the recipe.

### V3 · quiesce ranks before DCP speculator graph capture
`v1/worker/gpu/spec_decode/dflash/speculator.py`, +8/−0.

The sharded draft runs a CP collective inside speculator capture, so ranks must
be aligned before any rank begins. Without it, capture GPU-faults. Boot-time
only, so zero steady-state cost.

*Related:* **#54277** MERGED 08-29 (FlashInfer MLA for DSpark drafting) — same
file, same DCP × DSpark-draft intersection, and it establishes that a DCP-aware
draft is a supported configuration. **#48329** OPEN is the same bug class (spec
decode × cudagraph capture). Also **#54282**, **#53694** MERGED.

Smallest, cleanest story in the bundle. File it first as the icebreaker.

### A1 · take the tighter split-tile bound when a per-batch cap is supplied
`aiter/ops/attention.py`, +9/−1. **Highest value on the aiter list.**

`get_mla_metadata_info_v1` computes a loose `max_split_tiles` assuming an
unbounded per-batch split budget, then does `max(loose, tight)` when the caller
supplies a cap. The loose estimate always wins, so **the cap has no effect** —
aiter `max()`es its own tight bound away. Flipping to `min()` reclaims the fp32
MLA reduce scratch from 9.35 GiB to 2.38 GiB, which is what made FULL cudagraphs
fit under DCP.

Validated over 1030 (batch, qlen, ragged-kv) shapes on gfx950, both cprr and
non-cprr: 0 violations, worst actual/bound 0.998. Sound only when the same
`max_split_per_batch` reaches `get_mla_metadata_v1` at build time — so it must
land with V6, and the PR must say so.

*Related:* **#4729** and **#4796** MERGED restructure this exact code, so A1 must
be **rewritten against post-refactor main, not cherry-picked**. **#4227** MERGED
(forward compatibility for `get_mla_metadata_v1`) is the precedent for changing
this contract. **#4964** MERGED may shift the cprr fold path underneath it.

### A2 · ASM a16w16: don't auto-select split-K under graph replay
`csrc/py_itfs_cu/asm_gemm_a16w16.cu`, +39/−2.

The split-K a16w16 ASM kernels reduce partial-K through a per-(device, stream)
atomic semaphore whose "last workgroup reduces" protocol assumes a zero counter
at launch. Under cudagraph replay plus multi-stream drafting that invariant
breaks, the reduction never fires, and all waves spin forever (GPU 100%, ~310W)
at seqs=64 warmup. Eager single launches are fine, which is why it only bites
in-serve.

*The full history matters here.* **#3288** MERGED 05-20 (original per-stream
semaphore workspace) → **#4494** MERGED 08-12 (capture-safe, +91, with
`test_gemm_a16w16_graph.py`) → **#4709** MERGED 08-12 (straight revert, −91, same
day, empty body). Ours: **#4715** OPEN does this properly for the FlyDSL variant.

**Lead with the layer distinction.** Every one of those PRs lives in Python
(`gemm_op_a16w16.py`, `gemm_kernels.py`); ours is the only change in
`csrc/py_itfs_cu/asm_gemm_a16w16.cu`, untouched since 2026-04-15 (#2221). A2 is
a kernel-*selection* guard, not another workspace fix — that framing is its best
chance of surviving review given the revert. Ask what broke in #4494 first.

### A3 · fp8 MLA `get_block_n_fp8` fallback
`aiter/mla.py`, +2/−1.

`get_block_n_fp8[int(nhead * max_seqlen_q)]` raises `KeyError` on any unlisted
product. Our **#4713** OPEN already carries the `.get(..., 64)` fallback, so
**do not file separately** — the remaining delta is the explicit 80/96/112
entries, which are tuning rather than a crash fix. Fold them into #4713.

Clean up first: the in-tree edit has broken indentation, inserted above `8: 64`
at the wrong level. Justify the values against **#4521** MERGED 08-06 and
**#4430** MERGED 08-05, which are where these folded widths come from.

### A4 · K3 bf16 tuned-GEMM rows
`aiter/configs/model_configs/*.csv`, +765/−694.

Closes 371 conc-1 tuned-config misses; their absence also HSA-faults the agentic
launcher. Performance critical, but **not fileable as-is**: the diff is mostly
*rewritten* rows, and `kimi`, `kimik2` and `qwen3_5_397b` show deletions only
(−1, −3, −17) — we would be removing other models' tuned rows. Audit and
restrict to K3 additions before filing.

*Related:* nothing competes for these CSVs. Our merged **#3580**, **#3428**,
**#3372** are the format precedent — follow #3580's shape: additions only, plus
a note on how the rows were generated.

---

## Wave 2 — real, but not ours to justify on performance grounds

### V4 · ordered symmetric-memory teardown
`v1/attention/ops/cp_common.py`, ~+80. Not on our critical path; file anyway.

Exported symm_mem buffers are released in arbitrary order at teardown, so an
exporter routinely releases while peers still hold imports.
`amdgpu_bo_release_notify()` then loses its `dma_resv_trylock` on the aliased
private resv — its premise that nobody else holds a pointer is false for an
exported buffer, because `ttm_bo_individualize_resv` returns early when
`resvp == &_resv` — skips `amdgpu_amdkfd_remove_all_eviction_fences()`, and the
buffer strands at refcount 7 with live eviction fences. That blocks KFD delayed
restore and wedges the exiting rank inside the *global* `mmu_notifier` SRCU
section, queueing every other exiting process on the box behind it. The fix is
ordering, not freeing: drop imports everywhere, barrier, then drop exports.

Measured: 24 orphaned dma-bufs and ~22.5 s KFD eviction per run.

*Related:* the file is new and quiet — its only commit is **#52839** MERGED
08-20, which created it. Low conflict risk but no established reviewer; tag that
author. Adjacent: **#42993** MERGED (symm_mem gate fix — precedent that lifecycle
fixes land here), **#48880** OPEN (NVSHMEM backend, would inherit this bug),
**#50505**.

### V2 · default Kimi-K3 to the `a2a` DCP combine
`model_executor/models/config.py`, +24/−0. A default, not a capability.

Measured on MI355X, `a2a` beats the default `ag_rs` at every token count
(0.107→0.095 ms at T=5, 0.136→0.101 at T=144, so 1.13×→1.35× as T grows), cosine
similarity ≥ 0.999994. One `all_to_all_single` instead of
`allgather(lse) + reduce_scatter(out)` per MLA layer per decode step.

We pass `--dcp-comm-backend a2a` on the command line, so this changes nothing for
our numbers — it makes the good default automatic for everyone else. Cheap, well
evidenced, near-zero risk. File it with wave 1 if there is capacity; drop it
without regret if there is not.

*Related:* **#50382** MERGED 08-21 (default query replication for GLM sparse
attention) is a near-exact structural precedent — a per-model DCP default set
from the same file. Backend context: **#48248** OPEN (FlashInfer fused A2A),
**#54889** OPEN (fuse empty-shard LSE mask into the A2A pack kernel — same path),
**#48247** OPEN (AITER custom AG/RS).

---

## Not filing

**V1 · full-attention EAGLE prefix veto** (`offloading/scheduler.py`, +13/−1).
Upstream decrements `num_hit_chunks` unconditionally for `is_eagle_unverified`.
That trim is only correct for the sliding-window path, which over-queries one
chunk; the full-attention EAGLE group runs a prefix scan, never over-queries,
and never stores a volatile chunk, so the decrement drops a verified prompt
chunk. It is a genuine bug and it is unit-testable — but we run
`LMCacheMPConnector`, so we no longer exercise the file, and **#53388** MERGED
09-01 (disabling trailing prefix-cache block dropping) may already have changed
the surrounding logic. Revisit only if we move back to the native connector, and
re-read the file on current main first. (**#51161** MERGED, chunked local
attention in the same scheduler, is the closest precedent for "this group needs
a different trim rule". Also **#50507**, **#51243**, **#51100**.)

**V5 · skip the NVLS multicast probe on ROCm** (`ops/cp_common.py`, ~+16). The
probe's rendezvous is a collective that some ranks can return before reaching,
so rank asymmetry strands peers; it also leaks its tensor and handle per call.
But our in-tree justification cites our own unmerged a2a port, which is not an
argument upstream can act on, and **#33274** CLOSED shows a previous attempt to
guard this probe that did not land. Not critical for us, and not currently
arguable. Read #33274 before spending any further effort.

---

## The recipe is a separate PR, to InferenceX

`benchmarks/single_node/agentic/kimik3_fp4_mi355x_mtp.sh` (+117) and
`configs/amd-master.yaml` (+30) are not upstream vLLM/aiter work — they go to
InferenceX as the K3 FP4 MI355X agentic recipe, and they carry the three-tier
concurrency matrix.

This one has an unresolved conflict: upstream PR **#2810** (`c355f549a`) landed a
matrix that runs conc 4–14 on DCP1 with `dram-utilization 0.60`, while ours runs
those on DCP8. Our measurement says their choice would cost us — at conc 12,
DCP1 gives per-user p50 48.0/46.7 against DCP8's 55.2/52.8 and 263/206 tok/s
against 308/307, over two independent repeats. File with that data attached.
Note also that their 0.60 DRAM utilisation cannot be backed on a box that stages
weights in `/dev/shm`; ours is 0.268/0.343 for that reason.

---

## Landscape findings that shape the plan

1. **We are not alone on the AITER MLA DCP file.** seungrokj (AMD) filed and
   self-closed four PRs in this exact area on 2026-08-24/26 — **#53587** and
   **#53814** (`[ROCm] Allow full CUDA graphs with DCP for the a2a backend`,
   +8/−2 on `platforms/rocm.py`) and **#53815** / **#53816** (`[ROCm][MLA] ...
   AITER MLA under DCP`, +67/−3). All closed same-day, unmerged, reason not
   recorded. Add okorzh-amd's open **#54899** and that is three parties on one
   file. Coordinate before writing V6.
2. **Our aiter base (`55dbc4f47`) is ~3 weeks behind on the files we patch.**
   **#4729** (08-14) and **#4796** (08-18) refactored and de-torched the MLA
   metadata and reduce modules that A1 edits, and **#4964** (08-28) rebuilds
   `qo_indptr` and handles `g_kv_indptr` for cp round-robin nhead folding —
   which overlaps our `K3-DCP-GKV-PERSIST` work and may subsume part of V6.
   Rebase onto current main before writing any aiter PR.
3. **The ASM split-K fix was already merged and reverted.** **#4494** landed
   2026-08-12 and **#4709** reverted it the *same day*, with an empty body and no
   discussion. Ask what broke before rebuilding on that approach.

---

## Order

Before writing anything: fetch and rebase both repos, and ask the three
coordination questions above — they can invalidate V6 and A1 outright.

1. **V3** — smallest, cleanest, no dependencies. Icebreaker.
2. **A1** — needs the post-#4729/#4796 rewrite; start early because it gates V6.
3. **V8** — one three-line omission with a hard assert behind it. File next to V3.
4. **A3** — fold into the open #4713. Nearly free.
5. **A2** — only after asking what broke in #4494.
6. **V6** — split into three; blocked on #54899 and on A1 landing.
7. **A4** — after the additions-only audit.
8. **V2**, then **V4** — independent of the above, file when there is capacity.

Confidence, highest first: V2 (exact precedent), V8 (three lines, hard assert),
V4 (quiet file, real root cause), V3 (#54277 legitimises it), A1 (needs the
rebase), V6 (three-way overlap), A2 (one revert already), A4 (needs audit).

Note the tension in that ordering: the items we are most confident will land are
not the items our numbers depend on. V6 and A1 carry the performance story and
are the two hardest to get in. Plan for them taking the longest.
