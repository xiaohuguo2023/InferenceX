# DSpark non-causal path — bring-up findings (bonus experiment)

**Date:** 2026-08-09
**Machine:** MI355X ×8 (gfx950), TP8
**Container:** `xguo-k3nc` (sibling of the shipped `xguo-k3nightly`) — vLLM
0.26.1rc1.dev306+gcb8104839, ROCm 7.2.3, **triton 3.7.0**
**Goal:** revert the shipped *forced-causal* DSpark draft to its real **non-causal**
form and measure whether forcing causal cost us acceptance length (shipped N=2 =
2.32 mean accept len). Reference: seungrokj's PR #2508.

## TL;DR (RESOLVED 2026-08-09)

The non-causal DSpark path now **runs and is measured**. Net answer to "did
forcing causal leave acceptance on the table?": **yes — 2.32 → 2.50 mean accept
len at N=2** (~0.18 tok/verify, ~9 pts/token: 65.8% → 75.2% per-token accept).

Two blockers were cleared:

1. **Triton gluon-compile blocker — FIXED.** The gluon MLA verify kernel failed
   to JIT under the *upstream* `triton 3.7.0` (`expected offsets type layout to be
   BlockedLayout or SliceLayout`). `xguo-k3nc` had drifted onto upstream triton;
   restoring the **ROCm-fork** `triton 3.7.0+amd.rocm7.2.0.git89002410` (copied
   from the pristine `xguo-k3nightly`) fixed it. That fork relaxes the check to
   `isinstance(offsets.type.layout, DistributedLayout)`, and AITER's
   `gl.DistributedLinearLayout` q_pe offset *is* a `DistributedLayout` subclass.
   No kernel hack needed — the ROCm fork was always the intended triton.
2. **gluon 256-workgroup cap.** `mla_gluon[bh16bn128]` requires
   `B * NUM_KV_SPLITS <= 256`; a cudagraph capture bucket of 320 tripped it.
   Capping capture at 256 (`max_cudagraph_capture_size=256`, `MAX_CG=256`) fixed
   it — real decode batch maxes at 192, so no runtime path exceeds the cap.

**Measured (N=2, 24-gen temp=0, mean accept len = accepted/drafts + 1):**

| verify path | causal | non-causal |
|---|---|---|
| asm **q-row-fold** (shipped) | **2.32** (65.8%/tok) | impossible — rejects non-causal |
| gluon `bh16bn128` (ROCm-fork triton) | ~2.3 | **2.50** (75.2%/tok) ✓ |
| asm **flatten** (this session's attempt) | 1.85 ✗ | 1.82 ✗ |

The gluon non-causal **2.50** is the correct answer (spec decode is loss-less, so
the numbers-differ-by-window structure is the correctness signal, cross-checked
below).

## UPDATE — the ATOM path shows asm CAN do non-causal (my flatten was just wrong)

Cross-checking `~/mi355x_atom0807docker_specdecode7.md` (vLLM-ATOM, AITER PR
#4565 `mmd/dev/mla_bf16_non_mask`) reframes the section below. ATOM runs the
DSpark verify **on the fp8 asm MLA kernel with `causal:0`** — its startup error
for N=2 is literally `q_type:fp8 kv_type:fp8 gqa:16 ps:1 prefill:0 causal:0
qseqlen:2 ... cannot get heuristic kernel`. So the asm kernel *is* the intended
non-causal verify path; ATOM's N=2 only failed because its image lacked the
prebuilt `gqa16/qSeqLen2/causal0` kernel (its table had qSeqLen2 as causal=1
only). N=4 and N=7 dispatch qSeqLen=4/causal=0, which it *did* have, so they ran.

**Two things this tells us for our stack (`xguo-k3nc`):**

1. **The asm decode path is non-causal by construction.** `asm_mla.cu`'s
   decode-stage-1 entry hardcodes `prefill=0, causal=0` and selects the kernel
   from `config_max_seqlen_q = max_seqlen_q`. There is no causal knob to fight —
   it always picks a `causal=0` kernel. (`aiter/mla.py:mla_decode_fwd` →
   `mla_decode_stage1_asm_fwd`.)
2. **We already ship the kernels ATOM's N=2 lacked.** `hsa/gfx950/mla/mla_asm.csv`
   lists, for `fp8,fp8,gqa16,prefill0,causal0`: qSeqLen ∈ {1, **2**, 4} (ps=0 and
   ps=1). The gqa16 remap in `asm_mla.cu:974` only fires for gqa 32/64/128, so
   gqa16+qSeqLen2 stays and resolves to `mla_a8w8_qh16_qseqlen2_gqaratio16_ps.co`.

**So my asm-flatten failure was a design bug, not a kernel limitation.** Flatten
submitted each verify row as its own `max_seqlen_q=1` decode → the kernel selects
the **qSeqLen=1** kernel → N independent single-token decodes → no in-block
bidirectional attention → garbage (1.82/1.85, identical causal/non-causal because
neither honors a block). The **correct** call is a single **native multi-token**
decode: `max_seqlen_q=N`, `qo_indptr` strided by N per request (per-request KV,
NOT per-row-flattened KV) → selects the `qSeqLen=N, causal=0` kernel → true
non-causal block verify, exactly like ATOM. This would let the asm path
**replace gluon** (no 256-workgroup cap, no ROCm-triton dependency) for N∈{2,4}
(N=7 folds to qSeqLen4). **Not yet implemented/measured** — supersedes the
"refuted" verdict below, which only refuted the *flatten* form.

## [SUPERSEDED by the ATOM update above] Can the fp8 asm MLA replace gluon? — the flatten form: NO

To dodge gluon's 256-cap + ROCm-triton dependency, I tried routing the small-head
(<16) multi-token verify onto the fp8 **asm** `mla_decode_fwd` in a **flatten
form**: each verify row submitted as its own qlen=1 asm decode
(`qo_indptr=arange`, `max_seqlen_qo=1`), so the asm kernel's built-in
global-position causal mask is a no-op and the per-row flat KV window alone would
set causality. Gated by `VLLM_ROCM_AITER_MLA_ASM_PADDING=asm`; 5 edits in
`rocm_aiter_mla.py` (backup `.pre_asmflatten.bak`), patcher `_patch_asmflatten.py`.

**It is broken.** Non-causal gave **1.82**, and a clean diagnostic — the *same
asm-flatten path on the causal draft config* — gave **1.85**, NOT the shipped
asm-native causal 2.32. That causal≈non-causal equality is decisive: causal and
non-causal build *different* flat KV windows (truncated vs. full-range), so if the
kernel honored the per-row window the two would diverge. They don't → **the
asm-flatten path collapses/ignores the per-row KV window** (the asm decode kernel
indexes batch↔KV differently than the flatten form assumes). So asm cannot serve
the multi-token verify as a gluon drop-in. The 5 asm-flatten edits are left in the
container behind the `=asm` env gate but are **not** the shipped path; the gluon
hybrid is. Reverting `VLLM_ROCM_AITER_MLA_ASM_PADDING` away from `asm` (or
restoring `.pre_asmflatten.bak`) selects gluon.

## What was already in place (base cb810), confirmed this session

The full non-causal *wiring* already exists in the base image and only needed the
AITER backend to opt in:

- `kimi_k3/nvidia/dspark_mla.py:56` already passes `non_causal_multi_token_decode=True`.
- `kimi_k3/nvidia/mla.py:346` forwards it into the `MLAAttentionSpec`.
- `mla_attention.py` base `build()` reads it (`self.non_causal_multi_token_decode`)
  and, when `common_attn_metadata.causal is False`, takes a dedicated non-causal
  branch (requires a *uniform* query block — DSpark verify is uniform) and sets
  `causal=False` on the metadata.

The AITER backend simply declined to participate: `AiterMLABackend` inherited
`supports_non_causal()=False`, and `AiterMLAMetadataBuilder` inherited
`supports_non_causal_multi_token_decode=False`, so `build()` raised before ever
reaching `_build_decode`. That's exactly the original blocker.

## The 4 edits that unblocked the backend (in `xguo-k3nc` only)

File: `vllm/v1/attention/backends/mla/rocm_aiter_mla.py`
(backup: `rocm_aiter_mla.py.pre_noncausal.bak`)

1. `AiterMLABackend.supports_non_causal() -> True`.
2. `AiterMLAMetadataBuilder.supports_non_causal_multi_token_decode: ClassVar = True`.
3. `_build_decode`: when `self.non_causal_multi_token_decode`, build the flat
   verify view with **every row spanning its request's full range**
   (`per_req_len.unsqueeze(1).expand(-1, qlen)`) instead of the causal-truncated
   `per_req_len - (qlen-1) + causal_offset[t]`. The existing
   `_expand_page_indices_kernel` needs **no change** — it already fills each row
   with the first `flat_kv_indptr[r+1]-flat_kv_indptr[r]` tokens of request
   `r=row//QLEN`'s blocks, so a full-range `row_len` yields bidirectional
   in-block attention.
4. `forward_mqa`: dropped the `assert attn_metadata.causal` in the small-head
   multi-token verify path (the flat view is now correct for both causal target
   verify and non-causal draft verify).

After these, `AiterMLABackend.supports_non_causal()` and the builder ClassVar both
read `True`, the module imports, and serving proceeds **past** backend selection
all the way into `capture_model()`.

## [RESOLVED — historical] The gluon kernel vs triton 3.7.0 blocker

**This was the upstream-triton drift, fixed by restoring the ROCm-fork triton
(see TL;DR). Kept below for the record.**

At cudagraph capture, JIT-compiling the gluon verify kernel fails:

```
aiter/ops/triton/gluon/mla_gluon.py:379  (buffer_load_to_shared for q_pe)
triton .../amd/cdna4/async_copy.py:105
ValueError: expected offsets type layout to be BlockedLayout or SliceLayout
```

Cause: in `mla_gluon.py`, `blocked_q_nope` is a `gl.BlockedLayout` (so
`offs_q_nope`'s SliceLayout offset is accepted), but `blocked_q_pe` is a
`gl.DistributedLinearLayout` (lines 194/227). Triton 3.7.0's
`gl.amd.cdna4.async_copy.buffer_load_to_shared` requires the *offsets* tensor to
carry `BlockedLayout` or `SliceLayout`; the q_pe offset derived from a
`DistributedLinearLayout` is rejected. The q_nope load compiles; the q_pe load
(line 379) is the first to fail.

Why it was latent until now: the **shipped forced-causal** benchmark routed the
12-head fp8 verify to the **asm q-row-fold** (per PR #51011), never to this gluon
kernel. seungrokj's hybrid (transcribed as step 7 of `_pr2508_apply_patches.sh`)
forces verify onto gluon — which is what surfaced the incompatibility. Both
`xguo-k3nc` and the shipped `xguo-k3nightly` run triton **3.7.0**, so this is not a
version drift I can revert; the kernel itself needs a fix (or a different triton
that accepts `DistributedLinearLayout` offsets in `buffer_load_to_shared`).

## [historical] Why not just hack the kernel to get a number

**Moot — the fix was the correct ROCm-fork triton, not a kernel hack, so the
gluon 2.50 number IS trustworthy. Kept for the record.**


Speculative decoding is **loss-less by construction** — the target verifies every
drafted token — so **even a buggy non-causal draft kernel would still pass GSM8K**
while producing a **meaningless acceptance length**. Changing `blocked_q_pe` from
`DistributedLinearLayout` to a `BlockedLayout` (to satisfy the compiler) could
silently change the kernel's MFMA/matmul semantics, and I have **no correctness
oracle** for it (GSM8K won't catch it) and **no reference** (PR #2508 unreachable,
GitHub egress down). A hand-patched kernel would yield an *untrustworthy* number,
which is worse than reporting none. So the bring-up was stopped here deliberately.

## [DONE] Reproduce the non-causal 2.50

1. In `xguo-k3nc`, ensure the **ROCm-fork** triton is installed
   (`triton 3.7.0+amd.rocm7.2.0.git89002410`) — not upstream 3.7.0. (Restored
   from `xguo-k3nightly` via tar stream; upstream backed up as `triton.upstream.bak`.)
2. Keep the 4 vLLM non-causal edits (backup `.pre_noncausal.bak`); leave
   `VLLM_ROCM_AITER_MLA_ASM_PADDING` **unset/`auto`** (NOT `asm`) so verify routes
   to gluon, not the broken asm-flatten.
3. Draft config pristine non-causal (`config.json.orig.bak`, 1251 B, no `causal`
   key). Serve `MAX_CG=256 NUM_SPEC=2 PORT=8890 bash _serve_k3_nc.sh`; confirm
   FULL_AND_PIECEWISE capture (no eager, no fault).
4. Run the 24-gen temp=0 workload, scrape `vllm:spec_decode_*`:
   drafts 2424 / draft_tokens 4848 / accepted 3645 → **mean accept len 2.50
   (75.2%/tok)** vs shipped forced-causal 2.32 (65.8%/tok).

## Artifacts

- `_serve_k3_nc.sh` — non-causal serve (draft on ROCM_AITER_MLA hybrid,
  `VLLM_ROCM_AITER_MLA_ASM_PADDING=asm`, fp8 KV, FULL_AND_PIECEWISE, no eager).
- `_pr2508_apply_patches.sh` — the #2508 transcription (patches #4474/#4494/
  #50578/#51171/#50619 + mla_gluon batch256/dequant + KDA fix + hybrid quant).
  **Gap:** it never ported a triton-3.7.0-compatible gluon verify kernel.
- Container `xguo-k3nc`: 4 non-causal edits live in
  `rocm_aiter_mla.py` (backup `.pre_noncausal.bak`).
- Shipped deliverables untouched and complete: `K3_DSpark_Benchmark_Report.md`,
  `K3_Baseline_Benchmark_Report.md`, `dspark.md`, `docs/DSpark_Tutorial.md`.
