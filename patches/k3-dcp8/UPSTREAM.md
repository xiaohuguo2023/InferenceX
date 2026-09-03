# Upstream status of the patches in this bundle

None of the changes here are upstream yet. This is the inventory of what we owe
vLLM and aiter, with the related open/merged PRs for each, so the work can be
picked up without re-doing the survey.

Surveyed 2026-09-03. The four upstream PRs this bundle *consumes* — vLLM #50183,
#52047, #53598 and #51705 — are all **merged**, so there is nothing to chase there.

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

## vLLM

| # | change | file | size |
|---|---|---|---|
| V1 | don't veto ≤1-chunk prefixes on the full-attention EAGLE offload path | `offloading/scheduler.py` | +13/−1 |
| V2 | default Kimi-K3 to the `a2a` DCP combine | `models/config.py` | +24/−0 |
| V3 | quiesce ranks before DCP speculator graph capture | `dflash/speculator.py` | +8/−0 |
| V4 | ordered symmetric-memory teardown | `ops/cp_common.py` | ~+80 |
| V5 | skip the NVLS multicast probe on ROCm | `ops/cp_common.py` | ~+16 |
| V6 | asm round-robin CP path for DCP multi-token verify | `mla/rocm_aiter_mla.py` | ~+352/−9 |

**V1** — upstream decrements `num_hit_chunks` unconditionally for
`is_eagle_unverified`. That trim is only correct for the sliding-window path,
which over-queries one chunk; the full-attention EAGLE group runs a prefix scan,
never over-queries, and never stores a volatile chunk, so the decrement drops a
verified prompt chunk. Genuine bug, unit-testable.
*Related:* **#53388** MERGED 09-01 (disabling trailing prefix-cache block
dropping) — nearest prior art and the real supersession risk; it lands *after*
our pinned image, so re-read the file on current main first. **#51161** MERGED
(chunked local attention in the same scheduler) is the closest precedent for
"this group needs a different trim rule". Also **#50507**, **#51243**, **#51100**.

**V2** — measured on MI355X, `a2a` beats the default `ag_rs` at every token count
(1.13×→1.35× as T grows), cosine similarity ≥ 0.999994. One `all_to_all_single`
instead of `allgather(lse) + reduce_scatter(out)` per MLA layer per decode step.
*Related:* **#50382** MERGED 08-21 (default query replication for GLM sparse
attention) is a near-exact structural precedent — a per-model DCP default set
from the same file. Backend context: **#48248** OPEN (FlashInfer fused A2A),
**#54889** OPEN (fuse empty-shard LSE mask into the A2A pack kernel — same path),
**#48247** OPEN (AITER custom AG/RS).

**V3** — the sharded draft runs a CP collective inside speculator capture, so
ranks must be aligned before any rank starts. Boot-time only.
*Related:* **#54277** MERGED 08-29 (FlashInfer MLA for DSpark drafting) — same
file, same DCP × DSpark-draft intersection, and it establishes that a DCP-aware
draft is a supported configuration. **#48329** OPEN is the same bug class (spec
decode × cudagraph capture). Also **#54282**, **#53694** MERGED.

**V4** — the strongest single contribution. Exported symm_mem buffers are
released in arbitrary order at teardown, so an exporter routinely releases while
peers still hold imports. `amdgpu_bo_release_notify()` then loses its
`dma_resv_trylock` on the aliased private resv — its premise that nobody else
holds a pointer is false for an exported buffer, because
`ttm_bo_individualize_resv` returns early when `resvp == &_resv` — skips
`amdgpu_amdkfd_remove_all_eviction_fences()`, and the buffer strands at refcount
7 with live eviction fences. That blocks KFD delayed restore and wedges the
exiting rank inside the *global* `mmu_notifier` SRCU section, queueing every
other exiting process on the box behind it. The fix is ordering, not freeing:
drop imports everywhere, barrier, then drop exports. Affects any ROCm DCP user.
*Related:* the file is new and quiet — its only commit is **#52839** MERGED 08-20,
which created it. Low conflict risk but no established reviewer; tag that author.
Adjacent: **#42993** MERGED (symm_mem gate fix — precedent that lifecycle fixes
land here), **#48880** OPEN (NVSHMEM backend, would inherit this bug), **#50505**.

**V5** — the probe's rendezvous is a collective that some ranks can return before
reaching, so rank asymmetry strands peers; it also leaks its tensor and handle per
call. Lowest-confidence item: our in-tree justification cites our own unmerged a2a
port, so the rationale needs rewriting around upstream-only facts.
*Related:* **#33274** CLOSED — someone already tried guarding this probe and it
did not land. **Read it before spending effort.**

**V6** — upstream's DCP multi-token verify is Triton, so stock vLLM runs the whole
DCP decode on Triton under DSpark. This adds the asm route behind
`VLLM_ROCM_AITER_MLA_DCP_VERIFY`.
*Related, the most contested file on either list:*
**#54899** OPEN (okorzh-amd, native-tile head pad) — **blocks V6**; route through
its `get_actual_mla_num_heads` rather than duplicating the pad.
**#51705** MERGED 08-31 (causal multi-token verification) — V6 builds on it.
**#51171** MERGED 08-30 (FULL cudagraphs for AITER MLA spec decode).
**#51647** MERGED 08-18 (pad non-aligned AITER MLA heads) — the prior art #54899
extends and our `_NATIVE_CPRR_HEADS` duplicates.
**#51040** MERGED 08-26 — **ours**, good precedent to cite.
**#53815 / #53816 / #53814 / #53587** CLOSED unmerged (seungrokj).
**#52377** MERGED (sparse MLA metadata after the DCP Manager refactor).

Ours already open, chase rather than refile: **#51590**, **#53475** (draft),
**#53487** (draft).

---

## aiter

| # | change | file | size |
|---|---|---|---|
| A1 | take the tighter split-tile bound when a per-batch cap is supplied | `ops/attention.py` | +9/−1 |
| A2 | ASM a16w16: don't auto-select split-K under graph replay | `asm_gemm_a16w16.cu` | +39/−2 |
| A3 | fp8 MLA `get_block_n_fp8` entries for 80/96/112 | `aiter/mla.py` | +2/−1 |
| A4 | K3 bf16 tuned-GEMM rows | `configs/model_configs/*.csv` | +765/−694 |

**A1 — highest value on either list.** `get_mla_metadata_info_v1` computes a loose
`max_split_tiles` assuming an unbounded per-batch split budget, then does
`max(loose, tight)` when the caller supplies a cap. The loose estimate always
wins, so **the cap has no effect** — aiter `max()`es its own tight bound away.
Flipping to `min()` reclaims the fp32 MLA reduce scratch from 9.35 GiB to 2.38
GiB, which is what unblocked FULL cudagraphs under DCP. Validated over 1030
(batch, qlen, ragged-kv) shapes on gfx950, both cprr and non-cprr: 0 violations,
worst actual/bound 0.998. Sound only when the same `max_split_per_batch` reaches
`get_mla_metadata_v1` at build time — so it must land with **V6**.
*Related:* **#4729** and **#4796** MERGED restructure this exact code, so A1 must
be **rewritten against post-refactor main, not cherry-picked**. **#4227** MERGED
(forward compatibility for `get_mla_metadata_v1`) is the precedent for changing
this contract. **#4964** MERGED may shift the cprr fold path underneath it.

**A2** — the split-K a16w16 ASM kernels reduce partial-K through a per-(device,
stream) atomic semaphore whose "last workgroup reduces" protocol assumes a zero
counter at launch. Under cudagraph replay plus multi-stream drafting that
invariant breaks, the reduction never fires, and all waves spin forever. Eager is
fine, so it only bites in-serve.
*Related, the full history:* **#3288** MERGED 05-20 (original per-stream semaphore
workspace) → **#4494** MERGED 08-12 (capture-safe, +91, with
`test_gemm_a16w16_graph.py`) → **#4709** MERGED 08-12 (straight revert, −91, same
day, empty body). Ours: **#4715** OPEN does this properly for the FlyDSL variant.
**Layer distinction worth making in the PR:** every one of those lives in Python
(`gemm_op_a16w16.py`, `gemm_kernels.py`); ours is the only change in
`csrc/py_itfs_cu/asm_gemm_a16w16.cu`, untouched since 2026-04-15 (#2221). So A2 is
a kernel-*selection* guard, not another workspace fix — that framing is its best
chance of surviving review given the revert.

**A3** — `get_block_n_fp8[int(nhead * max_seqlen_q)]` raises `KeyError` on any
unlisted product. Our **#4713** OPEN already carries the `.get(..., 64)` fallback;
the remaining delta is the explicit 80/96/112 entries, which are tuning rather
than a crash fix, so fold them into #4713 instead of filing separately. The
in-tree edit has broken indentation (inserted above `8: 64` at the wrong level) —
clean that up first.
*Related:* **#4521** MERGED 08-06 and **#4430** MERGED 08-05 are where these folded
widths come from; justify the values against those kernels' coverage.

**A4** — closes 371 conc-1 tuned-config misses. **Not fileable as-is:** the diff is
mostly *rewritten* rows, and `kimi`, `kimik2` and `qwen3_5_397b` show deletions
only (−1, −3, −17), i.e. we would be removing other models' tuned rows. Audit and
restrict to K3 additions first.
*Related:* nothing competes for these CSVs. Our merged **#3580**, **#3428**,
**#3372** are the format precedent — follow #3580's shape: additions only, plus a
note on how the rows were generated.

Ours already open: **#4713**, **#4715** (draft), **#4647**, **#4108**.

---

## Suggested order

Before writing anything: fetch and rebase both repos, re-read
`offloading/scheduler.py` on current main (#53388), and ask the three
coordination questions above.

First wave — **V1, V2, V3, A1**: small, self-contained, clean bug stories, no
cross-dependencies except A1's pairing note with V6.

Confidence, highest first: V2 (exact precedent), V4 (quiet file, real root cause),
V3 (#54277 legitimises it), A1 (needs the rebase), V1 (supersession risk), V6
(three-way overlap), A2 (one revert already), V5 (#33274 failed), A4 (needs audit).
