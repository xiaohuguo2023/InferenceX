#!/usr/bin/env python3
"""Port the Kimi-K3 DSpark DCP recipe onto nightly-ba07e4a4 (vLLM dev1046).

Background
----------
Moving from nightly-5a4c8d99 (dev942) to nightly-ba07e4a4 (dev1046), a
pristine-vs-pristine diff shows:

  rocm_aiter_mla.py          IDENTICAL  -> now carried here as hunk D
  speculator.py              IDENTICAL  -> patched file copied verbatim
  offloading/scheduler.py    IDENTICAL  -> patched file copied verbatim
  mla_attention.py           drifted 10 lines  -> re-apply hunks A + B here
  models/kimi_k3/nvidia/mla.py  drifted 27 lines -> re-apply hunk C here
  v1/attention/ops/dcp_utils.py -> RENAMED to ops/dcp.py, rewritten (797 lines)

The old container's dcp_utils.py edits were shape-trace logging plus the
env-gated ``K3_DCP_A2A_SO`` direct-a2a loader -- diagnostics only, and the
speculator fault reproduces on stock PYNCCL a2a.  So ops/dcp.py is left
pristine and nothing is ported there.

The old mla_attention.py also carried ``K3_DCP_DIAG`` / ``K3_DCP_SYNC``
scaffolding.  That is deliberately NOT ported: K3_DCP_SYNC was shown to be
non-causal for the boot wedge, and the diagnostics were slated for removal.

"Copied verbatim" was fine while a patched container was alive, and stopped
being fine when one of them was queued for deletion.  As of 2026-08-25 the
biggest of those files is carried here as hunk D so this script alone can
reconstruct the port from a pristine image.  Re-verified against f94666:
rocm_aiter_mla.py, ops/dcp.py and ops/cp_common.py are byte-identical
dev1046 -> f94666, and all three anchors below still match exactly once.

This script is 4 of the 5 steps.  The full chain:

  0. _k3_dspark_fp8asm_apply_patches.sh -- MUST RUN FIRST as of the 1dc464d
     rebase.  Two of its sub-steps edit rocm_aiter_mla.py (0/7 skip-k3-fp8-ps
     and 4b envs->os.environ), so hunk D's base md5 below is the POST-fp8asm
     file, not the pristine image.  Both edits are disjoint from the DCP code;
     the ordering is enforced, not merely assumed -- run this script first and
     the md5 guard refuses rather than half-patching.
  0b. /opt/aiter-local     -- NOT scripted; still a hand transplant.  Stock
     aiter ships cprr asm kernels for bf16/bf16 only, so the fp8 DCP path
     SIGABRTs in get_heuristic_kernel_mla without it.  See memory
     k3-dcp-f94666-port-chain.  MOVED AHEAD of this script on 2026-08-31:
     hunk P1 patches a file inside that tree, so the tree has to exist first.
  1. this script            -- hunks L/E/J/N/P + hunk D (rocm_aiter_mla.py)
  2. _patch_pad128.py       -- pad 96 heads -> 128 instead of folding
  3. k3_dcp_direct_hip/_patch_dcp_skip_multicast_probe.py

What is ported
--------------
L. rocm_aiter_mla.py -- ATOM parity: undo the one replicated-draft collapse
   that lives inside hunk D, so the DSpark draft is DCP-SHARDED exactly like
   the target.  This REPLACES the former hunks A/B/C/F (draft attention pinned
   to dcp=1, vLLM #51705) and G1-G6/H/I (per-group CP degree so the draft's KV
   was written replicated), all deleted 2026-08-30.  See the long note on the
   hunk: everything ATOM patches in for this is already upstream here.

E. models/config.py -- direct-a2a combine as the DCP default.

J. platforms/rocm.py -- env opt-out (K3_DCP_ALLOW_FULL_CUDAGRAPH=1) of the
   blanket "DCP => PIECEWISE" cudagraph downgrade. Default-off; unset is
   upstream behaviour. Added 2026-08-30.

N. dflash/speculator.py -- quiesce + barrier before speculator cudagraph
   capture.  Under the hunk-L sharded draft the capture contains a CP
   collective, so staggered rank arrival is a hard GPU fault at boot.
   Boot-time only, measured boot-to-ready unchanged.  Added 2026-08-31.

P. the MLA reduce-scratch tight bound (3 sites, 2 trees).  Sizes the fp32
   reduce scratch from the real global split cap instead of an unbounded
   estimate: 9.35 -> 2.38 GiB, 6.97 GiB/rank, which is what makes FULL
   cudagraphs affordable under DCP.  P3 is the correctness-critical half --
   sizing and build must agree unconditionally, because an undersized reduce
   buffer faults the GPU rather than raising.  Added 2026-08-31.

Q/R/S. Three guards, added 2026-08-31.  They change no behaviour on a correct
   config; each converts a SILENT wrongness mode into a loud failure:
     Q  config/vllm.py     -- refuse a block-level KV interleave under spec
                              decode.  Any KV connector (offload as well as
                              P/D) silently pins cp_kv_cache_interleave_size to
                              the block size, and validate_block_size() skips
                              its compatibility assert in the same case.
     R  rocm_aiter_mla.py  -- assert the reduce-scratch split cap is passed on
                              every DCP build (the invariant hunk P depends on;
                              violating it faults the GPU, it does not raise).
                              Plus an opt-in K3_DCP_CHECK_REDUCE=1 deep check.
     S  rocm_aiter_mla.py  -- state lse_base_on_e explicitly instead of
                              inheriting it; a flipped default is a silent
                              ln(2) error in the cross-rank merge.

D. rocm_aiter_mla.py -- the DCP implementation itself (25 hunks, +568/-58):
   the round-robin CP decode, the fold/pad head handling and the direct-a2a
   combine.  Applied as an exact unified diff, guarded at both ends by md5.

Idempotent; --revert restores the pristine text.
"""

import argparse
import hashlib
import os
import sys

SITE = "/usr/local/lib/python3.12/dist-packages"
MODELS_CFG = SITE + "/vllm/model_executor/models/config.py"
ROCM_PLAT = SITE + "/vllm/platforms/rocm.py"
SPEC_PATH = SITE + "/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py"
VLLM_CFG = SITE + "/vllm/config/vllm.py"
# NOT under SITE: the hand-transplanted aiter tree (chain step 0b), which
# shadows the pip aiter.  Patching the pip copy would be a silent no-op.
AITER_ATTN = "/opt/aiter-local/aiter/ops/attention.py"

# --- hunk L: ATOM parity -- the DSpark draft is DCP-SHARDED --------------------
#
# SUPERSEDES hunks A/B/C/F and G1-G6/H/I (the "replicated draft" design, vLLM
# #51705), deleted 2026-08-30.
#
# #51705 pins the non-causal DSpark draft's attention to dcp=1 and writes its KV
# replicated on every rank, which makes the draft the only KV group whose
# geometry differs from what the rest of the stack assumes.  Every DCP bug we
# chased had that same shape -- the cp_sizes[gid] slot-mapping mix-up, the stale
# g_kv_indptr, the block-table width -- and a FULL cudagraph bakes the odd
# geometry in at capture time.  ~/work/ATOM runs DCP8 + DSpark on MI355X with
# GSM8K parity and has ZERO draft special-cases: the draft is sharded exactly
# like the target and correctness comes from the cross-rank LSE merge.
#
# The three things ATOM patches in are ALL already upstream in this vLLM:
#   * the CP-aware draft-input Triton kernel -- prepare_dflash_inputs() takes
#     cp_rank/cp_size/cp_interleave and computes DCP-local slots itself
#     (gpu/spec_decode/dflash/speculator.py:530-599).  This is ATOM's
#     dspark_dcp_patch.py, which is therefore redundant for us.
#   * the draft-step DCP-local seq lens -- gpu/spec_decode/speculator.py:296
#     derives them into a persistent buffer when the target's are stale.  This
#     is ATOM's attention/metadata.py:1567-1585.
#   * supports_non_causal_multi_token_dcp = True on AiterMLAMetadataBuilder
#     (rocm_aiter_mla.py:322), which opens the mla_attention.py:2154 gate.
#
# So the port is subtraction, not addition: drop every draft special-case and
# let the uniform DCP path run.  The one collapse that does not live in a SITES
# hunk is inside hunk D (the aiter builder __init__), so it is undone here as
# its own anchor-guarded site rather than by editing the md5-guarded diff --
# revert() unwinds SITES before D, so D's md5 still matches on the way out.
L_ANCHOR = """        # Replicated draft group (Kimi-K3 DSpark): no non-causal cprr asm
        # kernel exists, so run the draft at dcp=1 (vLLM #51705). Causal
        # target verify is unaffected (dcp stays = decode_context_parallel).
        if self.non_causal_multi_token_decode and self.dcp_world_size > 1:
            self.dcp_world_size = 1
            self.dcp_virtual_block_size = self.dcp_local_block_size
"""

L_PATCH = """        # K3-DCP-ATOM: the DSpark draft is DCP-SHARDED like the target, so no
        # group is collapsed to dcp=1 here.  The #51705 replicated-draft
        # collapse used to sit at exactly this spot and is deliberately gone;
        # see the hunk L note in _port_dcp_nightly_ba07e4a4.py.
"""

# --- hunk J: opt out of the blanket DCP full-cudagraph downgrade --------------
#
# platforms/rocm.py unconditionally rewrites FULL_AND_PIECEWISE -> PIECEWISE
# whenever decode_context_parallel_size > 1, with no backend check. On the
# asm-MLA + a2a path that guard is measurably wrong: with it bypassed, BOTH the
# FULL and PIECEWISE ladders capture 100%, the a2a does not hang, and conc-1
# step time collapses 182.8 -> 33.0 ms (5.5x). An ITL linear fit attributes the
# whole DCP regression to it -- +67.4 ms FIXED per step (~1104 us/MLA layer,
# far too large for an RCCL a2a; it is per-launch host dispatch) while the
# MARGINAL per-verify-token cost is 0.51x baseline, i.e. sharded KV is working
# exactly as designed.
#
# Kept as an ENV OPT-IN rather than a deletion: unset reproduces upstream
# behaviour byte for byte, so this cannot silently change a non-DCP or
# non-K3 run. Narrowing the guard properly (condition it on the attention
# backend's declared cudagraph support) is the upstream fix; this is the probe.
#
# CAVEAT, unresolved as of 2026-08-30: forcing FULL also drops draft acceptance
# 2.40 -> 1.66, and the damage compounds with draft depth (pos1 67.9 -> 39.9%,
# pos5 2.8 -> 0.5%) rather than being flat, so it is NOT the zero-fill
# signature of the cp_size bug above. Leading hypothesis is that our base image
# predates vLLM #51171 (no get_cudagraph_support in rocm_aiter_mla.py), so
# forcing FULL bypasses the guard without the capture-safety machinery #51171
# adds. Do not enable this for accuracy runs until that is settled.

J_ANCHOR = """        if compilation_config.cudagraph_mode.has_full_cudagraphs():
            # decode context parallel does not support full cudagraphs
            if parallel_config.decode_context_parallel_size > 1:
"""

J_PATCH = """        # K3-DCP-FULLGRAPH: the DCP downgrade below is a blanket guard with no
        # backend check. On the asm-MLA + a2a path it is measurably unnecessary
        # (both ladders capture; 5.5x step time), so allow an explicit opt-out.
        # Unset => upstream behaviour, unchanged.
        import os as _os

        _k3_allow_full_dcp = _os.environ.get("K3_DCP_ALLOW_FULL_CUDAGRAPH", "0") == "1"
        if compilation_config.cudagraph_mode.has_full_cudagraphs():
            # decode context parallel does not support full cudagraphs
            if (
                parallel_config.decode_context_parallel_size > 1
                and not _k3_allow_full_dcp
            ):
"""

# --- hunk D: the whole rocm_aiter_mla.py DCP implementation -------------------
#
# This is the bulk of the port and it used to live NOWHERE in the repo -- only
# inside containers k3-nightly4-test / k3-nightly5-test, one of which was queued
# for deletion. Carried here as a unified diff so the port is self-contained.
#
# REBASED 2026-08-29 onto nightly-6d4562c (0.28.1rc1.dev87), was f94666/dev1046.
# The md5 guard below did its job: it refused to apply and forced this re-derive.
#
# The diff shrank from 25 hunks to 18. Nothing was lost. Upstream drifted by
# 12 hunks (1406 -> 1527 lines) between f94666 and 6d4562c, and every one of
# those 12 is the fp8-prefill 16-head replicate-pad for K3's 12 heads/rank --
# our own work, upstreamed, down to keeping the `_real_nhead` variable name and
# the `out.view(...).copy_(out_3d[:, :_real_nhead, :])` unpad. The 7 hunks that
# stopped applying were that same pad work, so they were DROPPED as redundant
# rather than re-derived; `get_fp8_prefill_num_heads()` / `get_mla_padded_q()` /
# `is_valid_num_heads()` now provide it natively. The remaining 18 hunks are the
# DCP implementation proper and are untouched by the drift -- verified disjoint,
# not merely "applied clean". Same reason `_patch_fp8asm.py`,
# `_patch_fp8_prefill.py` and `_patch_ps_metadata16.py` are now obsolete.
#
# The two md5s turn "upstream has not moved again" into an assertion rather than
# a hope: if it moves, this refuses to apply rather than half-patching a file
# nobody will think to check.
MLA_PATH = os.path.join(SITE, "vllm/v1/attention/backends/mla/rocm_aiter_mla.py")
MLA_BASE_MD5 = "d3d5307ccb8f25f43a7d65f7bbab831e"    # 1dc464d AFTER _k3_dspark_fp8asm_apply_patches.sh
MLA_RESULT_MD5 = "8df22f8a588b07f3b7224f63e785a25d"  # rebased onto 1dc464d, 2079 lines
MLA_MARK = "_dcp_fold_factor"  # present iff the DCP implementation is in

MLA_DIFF = r'''--- rocm_aiter_mla.py.orig
+++ rocm_aiter_mla.py
@@ -2,6 +2,8 @@
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 
 import functools
+
+from vllm.distributed.parallel_state import get_dcp_group
 from dataclasses import dataclass
 from pathlib import Path
 from typing import ClassVar, Final
@@ -259,6 +261,25 @@
     use_gluon_decode: bool = False
     # Whether persistent MLA metadata was computed
     has_persistent_metadata: bool = False
+    # DCP round-robin: GLOBAL per-request page indptr (cumsum of the global
+    # seq lens) + CP group geometry the asm kernel needs for global-position
+    # causal masking. cp_world_size==1 leaves the non-DCP path untouched.
+    g_kv_indptr: torch.Tensor | None = None
+    cp_world_size: int = 1
+    cp_rank: int = 0
+    # DCP head-fold pseudo-batch (qlen>=3 verify only). When fold_qo_indptr is
+    # not None, forward_mqa folds q's dcp_heads into fold_factor groups of
+    # fold_heads, runs the native cprr kernel over these pseudo-requests, then
+    # un-folds o/lse. All fold_* indptrs describe the fold_num_reqs*fold_factor
+    # pseudo-batch; the fields above stay built for the real (unfolded) batch.
+    fold_factor: int = 1
+    fold_heads: int = 0
+    fold_num_reqs: int = 0
+    fold_qo_indptr: torch.Tensor | None = None
+    fold_kv_indptr: torch.Tensor | None = None
+    fold_kv_indices: torch.Tensor | None = None
+    fold_kv_last: torch.Tensor | None = None
+    fold_g_kv_indptr: torch.Tensor | None = None
 
 
 @dataclass
@@ -295,6 +316,10 @@
     #  https://github.com/vllm-project/vllm/issues/22945
     _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
     query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM
+    # DSpark draft KV spec is non_causal_multi_token_decode=True; the aiter
+    # round-robin decode applies causal masking on GLOBAL positions inside
+    # the kernel (g_kv_indptr), so the non-causal-draft DCP gate is safe.
+    supports_non_causal_multi_token_dcp: ClassVar[bool] = True
 
     @staticmethod
     def _uniform_padded_mtp_qo_len(
@@ -343,10 +368,24 @@
         vllm_config: VllmConfig,
         device: torch.device,
     ):
+        # AITER varlen round-robin CP requires interleave==1; flipping this
+        # frees the DSpark causal multi-token DCP gate in the builder base.
+        _cp_interleave = vllm_config.parallel_config.cp_kv_cache_interleave_size
         super().__init__(
-            kv_cache_spec, layer_names, vllm_config, device, AiterMLAMetadata
-        )
-
+            kv_cache_spec,
+            layer_names,
+            vllm_config,
+            device,
+            AiterMLAMetadata,
+            supports_dcp_with_varlen=(_cp_interleave == 1),
+        )
+
+        # Replicated draft group (Kimi-K3 DSpark): no non-causal cprr asm
+        # kernel exists, so run the draft at dcp=1 (vLLM #51705). Causal
+        # target verify is unaffected (dcp stays = decode_context_parallel).
+        if self.non_causal_multi_token_decode and self.dcp_world_size > 1:
+            self.dcp_world_size = 1
+            self.dcp_virtual_block_size = self.dcp_local_block_size
         self.compilation_config = vllm_config.compilation_config
         self.decode_attn_out_dtype = vllm_config.model_config.dtype
 
@@ -389,11 +428,50 @@
 
         from aiter import dtypes, get_mla_metadata_info_v1
 
+        # Under DCP the base class all-gathers query heads across the CP group
+        # before forward_mqa, so the asm decode processes num_heads*dcp heads
+        # over this rank's KV shard. dcp_world_size==1 collapses to num_heads,
+        # keeping the non-DCP metadata sizing byte-identical.
+        self._decode_num_heads = self.num_heads * self.dcp_world_size
+        try:
+            self._dcp_rank = (
+                get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
+            )
+        except AssertionError:
+            self._dcp_rank = 0
+        import logging as _lg
+        _lg.getLogger('vllm').warning(
+            'DCP-BLD layer=%s non_causal=%s num_heads=%s dcp=%s decode_num_heads=%s',
+            self.kv_cache_spec.__class__.__name__, self.non_causal_multi_token_decode,
+            self.num_heads, self.dcp_world_size, self._decode_num_heads)
+        # kv-split count for the round-robin CP metadata; must match the
+        # num_kv_splits the persistent buffers below are sized with.
+        self._mla_max_split_per_batch = 32
         # Keep metadata sizing consistent with the padded tensor shape passed
         # to mla_decode_fwd.
         self._num_attention_heads = AiterMLAHelper.get_actual_mla_num_heads(
-            self.num_heads
-        )
+            self._decode_num_heads
+        )
+        # DCP head-fold pseudo-batch. dcp_heads = num_heads*dcp_world_size (e.g.
+        # 96 = 12*8) is not a native cprr asm count {16,32,64,128}; aiter would
+        # fold it to gqa16 internally but with metadata built for the UNfolded
+        # count, which is numerically broken under round-robin CP. Instead we
+        # expand each request into F pseudo-requests of the largest native count
+        # that divides dcp_heads (96 -> 32, F=3): fewest KV re-reads with no
+        # padding-head waste (MEASURED fastest across conc 24..70). Only the
+        # qlen>=3 verify path folds; qlen==1 decode already works natively.
+        self._dcp_fold_factor = 1
+        self._dcp_fold_heads = 0
+        _NATIVE_CPRR = (16, 32, 64, 128)
+        if self.dcp_world_size > 1 and self._decode_num_heads not in _NATIVE_CPRR:
+            for _nat in (64, 32, 16):
+                if (
+                    self._decode_num_heads > _nat
+                    and self._decode_num_heads % _nat == 0
+                ):
+                    self._dcp_fold_heads = _nat
+                    self._dcp_fold_factor = self._decode_num_heads // _nat
+                    break
         kv_cache_dtype_str = getattr(vllm_config.cache_config, "cache_dtype", "auto")
         if kv_cache_dtype_str in ("fp8", "fp8_e4m3", "fp8_e5m2"):
             kv_cache_dtype_str = "fp8"
@@ -415,6 +493,31 @@
         # wrong split/reduce metadata for the gfx950 fp8 nhead=32 fold path.
         self._mla_q_dtype = q_dtype
         self._mla_kv_dtype = kv_dtype
+        def _mla_meta_info(nreq, nheads):
+            return get_mla_metadata_info_v1(
+                nreq,
+                self._mtp_decode_qlen,
+                nheads,
+                q_dtype,
+                kv_dtype,
+                is_sparse=False,
+                fast_mode=True,
+                num_kv_splits=self._mla_max_split_per_batch,
+                intra_batch_mode=False,
+            )
+
+        _mla_meta = _mla_meta_info(max_num_reqs, self._num_attention_heads)
+        if self._dcp_fold_factor > 1:
+            # The qlen>=3 verify path builds metadata for max_num_reqs*F
+            # pseudo-requests at nhead=fold_heads; size buffers to the max of
+            # both the native (qlen==1) and the folded shapes.
+            _mla_meta_fold = _mla_meta_info(
+                max_num_reqs * self._dcp_fold_factor, self._dcp_fold_heads
+            )
+            _mla_meta = tuple(
+                (max(a[0], b[0]), a[1])
+                for a, b in zip(_mla_meta, _mla_meta_fold)
+            )
         (
             (work_meta_data_size, work_meta_data_type),
             (work_indptr_size, work_indptr_type),
@@ -422,15 +525,7 @@
             (reduce_indptr_size, reduce_indptr_type),
             (reduce_final_map_size, reduce_final_map_type),
             (reduce_partial_map_size, reduce_partial_map_type),
-        ) = get_mla_metadata_info_v1(
-            max_num_reqs,
-            self._mtp_decode_qlen,
-            self._num_attention_heads,
-            q_dtype,
-            kv_dtype,
-            is_sparse=False,
-            fast_mode=True,
-        )
+        ) = _mla_meta
         self._mla_work_meta_data = torch.empty(
             work_meta_data_size, dtype=work_meta_data_type, device=device
         )
@@ -484,6 +579,39 @@
             self.qo_indptr = torch.zeros(
                 max_num_reqs + 1, dtype=torch.int32, device=device
             )
+
+        # Static pseudo-batch buffers for the DCP head-fold verify path. Each of
+        # up to max_num_reqs requests expands to F pseudo-requests, so indptrs
+        # are max_num_reqs*F+1. kv_indices holds this rank's local shard repeated
+        # F times; the local shard is (global/dcp_world_size), so its F copies
+        # never exceed the global max_num_pages -> reuse that size. Filled
+        # in-place by _build_decode (outside cudagraph capture) each step.
+        if self._dcp_fold_factor > 1:
+            _pb = max_num_reqs * self._dcp_fold_factor
+            self._fold_qo_indptr = torch.zeros(
+                _pb + 1, dtype=torch.int32, device=device
+            )
+            self._fold_kv_indptr = torch.zeros(
+                _pb + 1, dtype=torch.int32, device=device
+            )
+            self._fold_g_kv_indptr = torch.zeros(
+                _pb + 1, dtype=torch.int32, device=device
+            )
+            self._fold_kv_indices = torch.zeros(
+                max_num_pages, dtype=torch.int32, device=device
+            )
+            self._fold_kv_last = torch.ones(
+                _pb, dtype=torch.int32, device=device
+            )
+
+        # --- K3 VAMAP: record metadata-buffer VA ranges for fault attribution ---
+        try:
+            import k3_vamap as _k3_vamap
+
+            _k3_vamap.register_builder(self)
+        except Exception:
+            pass
+        # --- end K3 VAMAP ---
 
     def _init_fp8_prefill_ps_buffers(
         self,
@@ -706,6 +834,125 @@
         metadata.fp8_prefill_max_q_len = prefill.max_query_len
         metadata.fp8_prefill_num_partial_tiles = num_partial_tiles
 
+    def _build_fold_pseudo_metadata(
+        self,
+        num_reqs: int,
+        max_qo_len: int,
+        paged_kv_indptr: torch.Tensor,
+        paged_kv_indices: torch.Tensor,
+        g_kv_indptr: torch.Tensor,
+    ):
+        """DCP head-fold: expand the real batch into fold_factor pseudo-requests
+        per request (one per head-group) and build nhead=fold_heads round-robin
+        cprr metadata for it.  Runs during metadata build (eager, OUTSIDE
+        cudagraph capture) and fills the static fold buffers in place; the
+        captured forward only reads them.  Returns the sliced pseudo tensors."""
+        from aiter import get_mla_metadata_v1
+
+        device = self.device
+        F = self._dcp_fold_factor
+        pb = num_reqs * F
+
+        # qo: uniform max_qo_len tokens per pseudo-request.
+        fold_qo = self._fold_qo_indptr[: pb + 1]
+        fold_qo.copy_(
+            torch.arange(
+                0,
+                (pb + 1) * max_qo_len,
+                step=max_qo_len,
+                dtype=torch.int32,
+                device=device,
+            )
+        )
+
+        # local shard kv indptr: per-request local len repeated F times, cumsum.
+        local_len = paged_kv_indptr[1 : num_reqs + 1] - paged_kv_indptr[:num_reqs]
+        pseudo_local = local_len.repeat_interleave(F)
+        fold_kv = self._fold_kv_indptr[: pb + 1]
+        fold_kv[0] = 0
+        fold_kv[1:].copy_(torch.cumsum(pseudo_local, 0, dtype=torch.int32))
+
+        # global kv indptr (round-robin causal masking): per-request global len
+        # repeated F times, cumsum.
+        global_len = g_kv_indptr[1 : num_reqs + 1] - g_kv_indptr[:num_reqs]
+        pseudo_global = global_len.repeat_interleave(F)
+        fold_g = self._fold_g_kv_indptr[: pb + 1]
+        fold_g[0] = 0
+        fold_g[1:].copy_(torch.cumsum(pseudo_global, 0, dtype=torch.int32))
+
+        # kv indices: each request's local shard repeated F times, laid out
+        # [request][group] (matching the q fold in forward_mqa). Fully
+        # vectorized gather -- no host loop.
+        total = int(fold_kv[pb].item())
+        _cap = self._fold_kv_indices.numel()
+        assert total <= _cap, (
+            f"[FOLD-OOB] total={total} > _fold_kv_indices cap={_cap} "
+            f"pb={pb} num_reqs={num_reqs} max_qo_len={max_qo_len}"
+        )
+        import os as _os
+        if _os.environ.get("K3_FOLD_DIAG"):
+            W = self.dcp_world_size
+            _exp_local_max = int(((global_len + W - 1) // W).max())
+            _local_max = int(local_len.max())
+            logger.warning(
+                "[FOLD-BLD] pb=%d total=%d cap=%d num_reqs=%d qlen=%d W=%d "
+                "local[max=%d,sum=%d] global[max=%d,sum=%d] "
+                "exp_local_max=%d %s | wmd=%d wis=%d wip=%d rip=%d rfm=%d rpm=%d",
+                pb, total, _cap, num_reqs, max_qo_len, W,
+                _local_max, int(local_len.sum()),
+                int(global_len.max()), int(global_len.sum()),
+                _exp_local_max,
+                ("MISMATCH!" if _local_max > _exp_local_max else "ok"),
+                self._mla_work_meta_data.numel(), self._mla_work_info_set.numel(),
+                self._mla_work_indptr.numel(), self._mla_reduce_indptr.numel(),
+                self._mla_reduce_final_map.numel(),
+                self._mla_reduce_partial_map.numel(),
+            )
+        fold_idx = self._fold_kv_indices
+        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
+            fold_idx.fill_(-1)
+        if total > 0:
+            base = (
+                paged_kv_indptr[:num_reqs].repeat_interleave(F).to(torch.int64)
+            )
+            within = torch.arange(
+                total, device=device, dtype=torch.int64
+            ) - fold_kv[:pb].to(torch.int64).repeat_interleave(pseudo_local)
+            src = base.repeat_interleave(pseudo_local) + within
+            fold_idx[:total].copy_(paged_kv_indices[src])
+
+        fold_last = self._fold_kv_last[:pb]  # always ones (page_size==1)
+
+        get_mla_metadata_v1(
+            fold_qo,
+            fold_kv,
+            fold_last,
+            self._dcp_fold_heads,
+            1,
+            False,
+            self._mla_work_meta_data,
+            self._mla_work_info_set,
+            self._mla_work_indptr,
+            self._mla_reduce_indptr,
+            self._mla_reduce_final_map,
+            self._mla_reduce_partial_map,
+            page_size=1,
+            kv_granularity=16,
+            max_seqlen_qo=max_qo_len,
+            uni_seqlen_qo=max_qo_len,
+            fast_mode=True,
+            dtype_q=self._mla_q_dtype,
+            dtype_kv=self._mla_kv_dtype,
+            max_split_per_batch=self._mla_max_split_per_batch,
+            intra_batch_mode=False,
+            is_cp_round_robin=True,
+        )
+        import os as _os2
+        if _os2.environ.get("K3_FOLD_SYNC") and not torch.cuda.is_current_stream_capturing():
+            torch.cuda.synchronize()
+            logger.warning("[FOLD-BLDSYNC] post OK pb=%d total=%d qlen=%d", pb, total, max_qo_len)
+        return fold_qo, fold_kv, fold_idx, fold_last, fold_g
+
     def _build_decode(
         self,
         block_table_tensor: torch.Tensor,
@@ -820,6 +1067,18 @@
                 else:
                     qo_indptr = query_start_loc_device[: 1 + num_kernel_reqs]
 
+        # DCP round-robin: build the GLOBAL per-request page indptr the asm
+        # kernel needs for global-position causal masking over this rank's
+        # local KV shard (page_size==1, so pages==tokens).
+        g_kv_indptr = None
+        if self.dcp_world_size > 1 and dcp_tot_seq_lens_device is not None:
+            g_kv_indptr = torch.cat(
+                [
+                    torch.zeros(1, dtype=torch.int32, device=device),
+                    dcp_tot_seq_lens_device.cumsum(dim=0, dtype=torch.int32),
+                ]
+            )
+
         # Only the asm decode consumes the schedule, so gate on the routing
         # rather than on num_heads >= 16, which denies it to a padded rank
         # running the same asm kernels. The two predicates are disjoint --
@@ -843,33 +1102,74 @@
             and max_qo_len >= 1
             and max_qo_len <= self._mtp_decode_qlen
         )
+        do_fold = False
+        fold_qo = fold_kv = fold_idx = fold_last = fold_g = None
         if use_persistent_metadata:
             from aiter import get_mla_metadata_v1
 
-            uni_qo_len = (
-                max_qo_len if pad_uniform_mtp or torch.all(qo_len == max_qo_len) else -1
-            )
-            get_mla_metadata_v1(
-                qo_indptr,
-                paged_kv_indptr,
-                paged_kv_last_page_len,
-                self._num_attention_heads,
-                1,
-                True,
-                self._mla_work_meta_data,
-                self._mla_work_info_set,
-                self._mla_work_indptr,
-                self._mla_reduce_indptr,
-                self._mla_reduce_final_map,
-                self._mla_reduce_partial_map,
-                page_size=1,
-                kv_granularity=16,
-                max_seqlen_qo=max_qo_len,
-                uni_seqlen_qo=uni_qo_len,
-                fast_mode=True,
-                dtype_q=self._mla_q_dtype,
-                dtype_kv=self._mla_kv_dtype,
-            )
+            uniform = pad_uniform_mtp or bool(torch.all(qo_len == max_qo_len))
+            uni_qo_len = max_qo_len if uniform else -1
+            # DCP head-fold verify path: dcp_heads (e.g. 96) is not a native
+            # cprr count, so expand each request into fold_factor pseudo-requests
+            # at nhead=fold_heads (native) instead of letting aiter fold with
+            # mismatched metadata. Only for the uniform qlen>=3 spec-verify
+            # shape; qlen==1 decode already runs natively at dcp_heads.
+            import os as _osf
+            do_fold = (
+                self._dcp_fold_factor > 1
+                and g_kv_indptr is not None
+                and uniform
+                and int(max_qo_len) >= 3
+                and not _osf.environ.get("K3_NO_FOLD")
+            )
+            # Round-robin CP applies causality via global positions + the qlen
+            # window inside the kernel, so is_causal must be False under DCP.
+            cprr_kwargs = {}
+            is_causal = True
+            if g_kv_indptr is not None:
+                is_causal = False
+                cprr_kwargs = dict(
+                    max_split_per_batch=self._mla_max_split_per_batch,
+                    intra_batch_mode=False,
+                    is_cp_round_robin=True,
+                )
+            if do_fold:
+                (
+                    fold_qo,
+                    fold_kv,
+                    fold_idx,
+                    fold_last,
+                    fold_g,
+                ) = self._build_fold_pseudo_metadata(
+                    num_kernel_reqs,
+                    int(max_qo_len),
+                    paged_kv_indptr,
+                    paged_kv_indices,
+                    g_kv_indptr,
+                )
+            else:
+                get_mla_metadata_v1(
+                    qo_indptr,
+                    paged_kv_indptr,
+                    paged_kv_last_page_len,
+                    self._num_attention_heads,
+                    1,
+                    is_causal,
+                    self._mla_work_meta_data,
+                    self._mla_work_info_set,
+                    self._mla_work_indptr,
+                    self._mla_reduce_indptr,
+                    self._mla_reduce_final_map,
+                    self._mla_reduce_partial_map,
+                    page_size=1,
+                    kv_granularity=16,
+                    max_seqlen_qo=max_qo_len,
+                    uni_seqlen_qo=uni_qo_len,
+                    fast_mode=True,
+                    dtype_q=self._mla_q_dtype,
+                    dtype_kv=self._mla_kv_dtype,
+                    **cprr_kwargs,
+                )
             has_persistent_metadata = True
 
         # Small-head multi-token verify uses mla_gluon's 4-D MTP entry over the
@@ -892,6 +1192,17 @@
             else:
                 min_kv_seq_len = int(per_req_len.min().item())
 
+        import os as _os
+        if _os.environ.get("K3_FOLD_DIAG"):
+            logger.warning(
+                "[FOLD-DEC] full_cg=%s num_reqs=%d max_qo_len=%d "
+                "g_kv=%s dcp_tot=%s persistent=%s do_fold=%s",
+                self.compilation_config.cudagraph_mode.has_full_cudagraphs(),
+                num_kernel_reqs, int(max_qo_len),
+                g_kv_indptr is not None, dcp_tot_seq_lens_device is not None,
+                has_persistent_metadata, do_fold,
+            )
+
         attn_metadata = AiterMLADecodeMetadata(
             block_table=block_table_tensor,
             seq_lens=seq_lens_for_kernel,
@@ -905,6 +1216,17 @@
             use_gluon_decode=use_gluon_decode,
             attn_out_dtype=self.decode_attn_out_dtype,
             has_persistent_metadata=has_persistent_metadata,
+            g_kv_indptr=g_kv_indptr,
+            cp_world_size=self.dcp_world_size,
+            cp_rank=self._dcp_rank,
+            fold_factor=(self._dcp_fold_factor if do_fold else 1),
+            fold_heads=(self._dcp_fold_heads if do_fold else 0),
+            fold_num_reqs=(num_kernel_reqs if do_fold else 0),
+            fold_qo_indptr=fold_qo,
+            fold_kv_indptr=fold_kv,
+            fold_kv_indices=fold_idx,
+            fold_kv_last=fold_last,
+            fold_g_kv_indptr=fold_g,
         )
 
         return attn_metadata
@@ -1089,6 +1411,17 @@
         # Undo the tile-padding from get_mla_padded_q: the real heads are the
         # first num_heads.
         return o[:, :num_heads, :]
+
+    @staticmethod
+    def get_mla_unpadded_lse(num_heads: int, lse: torch.Tensor) -> torch.Tensor:
+        # lse is [tokens, padded_heads]; undo the same head padding as the o
+        # tensor so the DCP cross-shard merge sees exactly num_heads columns.
+        m = AiterMLAHelper.get_actual_mla_num_heads(num_heads)
+        if num_heads == m:
+            return lse
+        if m % num_heads == 0:
+            return lse[:, :: m // num_heads]
+        return lse[:, :num_heads]
 
     @staticmethod
     def use_gluon_decode(num_heads: int, max_qo_len: int, kv_cache_dtype: str) -> bool:
@@ -1132,6 +1465,10 @@
 
 
 class AiterMLAImpl(MLACommonImpl[AiterMLAMetadata]):
+    # Under DCP the base decode path all-gathers query heads and needs each
+    # rank to return its partial LSE for the cross-shard online-softmax merge.
+    can_return_lse_for_decode: bool = True
+
     def __init__(
         self,
         num_heads: int,
@@ -1161,6 +1498,10 @@
             **mla_args,
         )
         AiterMLAHelper.check_num_heads_validity(num_heads)
+
+        # Heads seen by the asm decode after the base class' DCP query
+        # all-gather (num_heads*dcp). Collapses to num_heads when DCP is off.
+        self._decode_num_heads = self.num_heads * self.dcp_world_size
 
         unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
         if any(unsupported_features):
@@ -1519,8 +1860,106 @@
         assert isinstance(q, torch.Tensor)
         B = q.shape[0]
 
-        mla_padded_q = AiterMLAHelper.get_mla_padded_q(self.num_heads, q)
-        mla_num_heads = AiterMLAHelper.get_actual_mla_num_heads(self.num_heads)
+        # DCP head-fold verify path. dcp_heads (e.g. 96) is not a native cprr
+        # count, so the metadata builder expanded the batch into
+        # fold_num_reqs*fold_factor pseudo-requests of fold_heads (native) each.
+        # Fold q the same [request][group][token] way, run the native round-robin
+        # kernel, then un-fold o/lse back to dcp_heads. Pure permutes/copies =
+        # cudagraph-capture safe (no host work). Verify is rectangular
+        # (B == fold_num_reqs*qlen), the same invariant the gluon-verify path
+        # above relies on.
+        if decode.fold_qo_indptr is not None:
+            fold_f = decode.fold_factor
+            fold_h = decode.fold_heads
+            nr = decode.fold_num_reqs
+            qlen = int(decode.max_qo_len)
+            dcp_heads = self.num_heads * self.dcp_world_size
+            head_dim_q = q.shape[-1]
+            assert B == nr * qlen, (B, nr, qlen)
+            q_f = (
+                q.reshape(nr, qlen, fold_f, fold_h, head_dim_q)
+                .permute(0, 2, 1, 3, 4)
+                .reshape(nr * fold_f * qlen, fold_h, head_dim_q)
+                .contiguous()
+            )
+            o_f = torch.empty(
+                nr * fold_f * qlen,
+                fold_h,
+                self.kv_lora_rank,
+                dtype=decode.attn_out_dtype,
+                device=q.device,
+            )
+            kv_buffer = kv_c_and_k_pe_cache.view(-1, 1, 1, head_dim_q)
+            import os as _os
+            _fsync = _os.environ.get("K3_FOLD_SYNC") and not (
+                torch.cuda.is_current_stream_capturing()
+            )
+            if _fsync:
+                logger.warning(
+                    "[FOLD-SYNC] pre B=%d nr=%d qlen=%d fold_f=%d fold_h=%d "
+                    "kvidx_total=%d q_f=%s",
+                    B, nr, qlen, fold_f, fold_h,
+                    int(decode.fold_kv_indptr[-1]), tuple(q_f.shape),
+                )
+            # RAW aiter cprr call (byte-for-byte the validated micro-repro):
+            # num_kv_splits/intra_batch_mode/page_size/nhead_kv MUST match the
+            # fold metadata (max_split_per_batch=32, intra_batch_mode=False) or
+            # the reduce-step split layout is inconsistent -> OOB. The vLLM
+            # custom op forwards none of these, which is why the wrapped path
+            # faulted. Returns (logits, lse); o_f is written in place.
+            from aiter.mla import mla_decode_fwd as _raw_mla_decode_fwd
+            _, lse_f = _raw_mla_decode_fwd(
+                q_f,
+                kv_buffer,
+                o_f,
+                decode.fold_qo_indptr,
+                decode.fold_kv_indptr,
+                decode.fold_kv_indices,
+                decode.fold_kv_last,
+                max_seqlen_q=qlen,
+                page_size=1,
+                nhead_kv=1,
+                sm_scale=self.scale,
+                num_kv_splits=32,
+                q_scale=layer._q_scale,
+                kv_scale=layer._k_scale,
+                intra_batch_mode=False,
+                return_lse=True,
+                g_kv_indptr=decode.fold_g_kv_indptr,
+                cp_world_size=decode.cp_world_size,
+                cp_rank=decode.cp_rank,
+                work_meta_data=attn_metadata.work_meta_data,
+                work_indptr=attn_metadata.work_indptr,
+                work_info_set=attn_metadata.work_info_set,
+                reduce_indptr=attn_metadata.reduce_indptr,
+                reduce_final_map=attn_metadata.reduce_final_map,
+                reduce_partial_map=attn_metadata.reduce_partial_map,
+            )
+            if _fsync:
+                torch.cuda.synchronize()
+                logger.warning("[FOLD-SYNC] post OK nr=%d qlen=%d", nr, qlen)
+            o_out = (
+                o_f.reshape(nr, fold_f, qlen, fold_h, self.kv_lora_rank)
+                .permute(0, 2, 1, 3, 4)
+                .reshape(nr * qlen, dcp_heads, self.kv_lora_rank)
+                .contiguous()
+            )
+            lse_out = (
+                lse_f.reshape(nr, fold_f, qlen, fold_h)
+                .permute(0, 2, 1, 3)
+                .reshape(nr * qlen, dcp_heads)
+                .contiguous()
+            )
+            return o_out, lse_out
+
+        # Under DCP q already carries the all-gathered heads (num_heads*dcp).
+        # Read dcp_world_size LIVE (not the __init__-cached _decode_num_heads):
+        # the replicated non-causal draft group has its impl.dcp_world_size
+        # forced to 1 by the layer AFTER construction, so the cached value is
+        # stale for the draft. Collapses to num_heads when DCP is off.
+        dcp_heads = self.num_heads * self.dcp_world_size
+        mla_padded_q = AiterMLAHelper.get_mla_padded_q(dcp_heads, q)
+        mla_num_heads = AiterMLAHelper.get_actual_mla_num_heads(dcp_heads)
         o = torch.empty(
             B,
             mla_num_heads,
@@ -1552,17 +1991,89 @@
                 reduce_partial_map=attn_metadata.reduce_partial_map,
             )
 
-        rocm_aiter_ops.mla_decode_fwd(
-            mla_padded_q,
-            kv_buffer,
-            o,
-            self.scale,
-            decode.qo_indptr,
-            decode.max_qo_len,
-            decode.paged_kv_indptr,
-            decode.paged_kv_indices,
-            decode.paged_kv_last_page_len,
-            **mla_kwargs,
-        )
-
-        return AiterMLAHelper.get_mla_unpadded_o(self.num_heads, o), None
+        # DCP: surface the per-(token, head) LSE and drive the round-robin
+        # global-position causal mask over this rank's KV shard. The base
+        # MLACommonImpl.forward merges the per-rank partials via online softmax.
+        final_lse = None
+        if self.dcp_world_size > 1:
+            final_lse = torch.empty(
+                B, mla_num_heads, dtype=torch.float32, device=q.device
+            )
+            # The gqa16 fp8 round-robin (cprr) + lse asm kernel exists ONLY at
+            # config_max_seqlen_q==4 (real max_qo_len in {3,4}, or >4 in
+            # persistent mode). There is NO cprr kernel for max_qo_len in {1,2}.
+            # Those tiny shapes never occur at real spec-decode verify
+            # (qo == 1+num_spec >= 3) -- only during PIECEWISE cudagraph capture,
+            # where attention runs eager and its output is discarded. So request
+            # the cprr/lse kernel only when the shape maps to the valid kernel;
+            # otherwise fall back to the plain decode kernel and return a zero
+            # lse purely to satisfy the base DCP-merge assert (whose merged
+            # output is discarded during capture).
+            if decode.max_qo_len >= 3:
+                mla_kwargs["final_lse"] = final_lse
+                mla_kwargs["g_kv_indptr"] = decode.g_kv_indptr
+                mla_kwargs["cp_world_size"] = decode.cp_world_size
+                mla_kwargs["cp_rank"] = decode.cp_rank
+            else:
+                final_lse.zero_()
+
+        if self.dcp_world_size > 1:
+            # Keep _aiter_ops.py pristine: all DCP decode goes through raw aiter
+            # (the custom op forwards no final_lse/g_kv/cp/num_kv_splits). This
+            # branch is only reached for dcp>1 at qlen<3 or non-uniform batches
+            # (capture / no-spec); the qlen>=3 target verify folds and returns
+            # earlier. Correctness of this branch is not on the num_spec>=2
+            # production path -- it only must not crash.
+            from aiter.mla import mla_decode_fwd as _raw_mla_decode_fwd
+            _rk = dict(mla_kwargs)
+            _fl = _rk.pop("final_lse", None)
+            _gk = _rk.pop("g_kv_indptr", None)
+            _cw = _rk.pop("cp_world_size", 1)
+            _cr = _rk.pop("cp_rank", 0)
+            _qs = _rk.pop("q_scale", None)
+            _ks = _rk.pop("kv_scale", None)
+            _ret = _raw_mla_decode_fwd(
+                mla_padded_q,
+                kv_buffer.view(-1, 1, 1, mla_padded_q.shape[-1]),
+                o,
+                decode.qo_indptr,
+                decode.paged_kv_indptr,
+                decode.paged_kv_indices,
+                decode.paged_kv_last_page_len,
+                max_seqlen_q=decode.max_qo_len,
+                page_size=1,
+                nhead_kv=1,
+                sm_scale=self.scale,
+                num_kv_splits=32,
+                q_scale=_qs,
+                kv_scale=_ks,
+                intra_batch_mode=False,
+                return_lse=(_fl is not None),
+                g_kv_indptr=_gk,
+                cp_world_size=_cw,
+                cp_rank=_cr,
+                **_rk,
+            )
+            if _fl is not None and isinstance(_ret, tuple) and _ret[1] is not None:
+                _fl.copy_(_ret[1].reshape(_fl.shape))
+        else:
+            rocm_aiter_ops.mla_decode_fwd(
+                mla_padded_q,
+                kv_buffer,
+                o,
+                self.scale,
+                decode.qo_indptr,
+                decode.max_qo_len,
+                decode.paged_kv_indptr,
+                decode.paged_kv_indices,
+                decode.paged_kv_last_page_len,
+                **mla_kwargs,
+            )
+
+        o_out = AiterMLAHelper.get_mla_unpadded_o(dcp_heads, o).contiguous()
+        if final_lse is not None:
+            lse_out = AiterMLAHelper.get_mla_unpadded_lse(
+                dcp_heads, final_lse
+            ).contiguous()
+            return o_out, lse_out
+        return o_out, None
'''


def _parse_hunks(diff_text):
    """Yield (old_start_0based, old_lines, new_lines) for a unified diff."""
    hunks, it = [], iter(diff_text.splitlines(True))
    next(it), next(it)  # the ---/+++ headers
    cur = None
    for ln in it:
        if ln.startswith("@@"):
            if cur:
                hunks.append(cur)
            # "@@ -l,s +l,s @@" -- only the old-side start is needed; the new side
            # is whatever the +/space lines spell out.
            start = int(ln.split()[1][1:].split(",")[0])
            cur = (start - 1, [], [])
        elif cur is not None:
            tag, body = ln[0], ln[1:]
            if tag == " ":
                cur[1].append(body)
                cur[2].append(body)
            elif tag == "-":
                cur[1].append(body)
            elif tag == "+":
                cur[2].append(body)
    if cur:
        hunks.append(cur)
    return hunks


def _splice(src, forward):
    """Apply MLA_DIFF exactly -- no fuzz, no offset search."""
    lines, out, i = src.splitlines(True), [], 0
    for start, old, new in _parse_hunks(MLA_DIFF):
        if not forward:
            # Reverting walks the same hunks with the sides swapped, but the
            # recorded starts are old-side while `lines` is now the NEW file, so
            # each one has to be shifted by the net growth the earlier hunks
            # introduced. At this point `i` counts new-file lines consumed and
            # len(out) counts old-file lines emitted, so their difference is
            # exactly that running delta.
            old, new = new, old
            start += i - len(out)
        if lines[start:start + len(old)] != old:
            return None, "hunk at line %d does not match its context" % (start + 1)
        out.extend(lines[i:start])
        out.extend(new)
        i = start + len(old)
    out.extend(lines[i:])
    return "".join(out), None


def apply_mla_diff() -> int:
    label = "rocm_aiter_mla.py hunk D (DCP implementation, 18 hunks)"
    if not os.path.exists(MLA_PATH):
        print("  !! missing %s" % MLA_PATH)
        return 1
    src = open(MLA_PATH).read()
    if MLA_MARK in src:
        print("  already applied: %s" % label)
        return 0
    got = hashlib.md5(src.encode()).hexdigest()
    if got != MLA_BASE_MD5:
        print("  !! rocm_aiter_mla.py is not the expected pristine file")
        print("     expected md5 %s, got %s" % (MLA_BASE_MD5, got))
        print("     -> upstream drifted; re-derive the diff, do not force it")
        return 1
    result, err = _splice(src, True)
    if err:
        print("  !! %s" % err)
        return 1
    got = hashlib.md5(result.encode()).hexdigest()
    if got != MLA_RESULT_MD5:
        print("  !! patched result md5 %s != expected %s" % (got, MLA_RESULT_MD5))
        return 1
    open(MLA_PATH, "w").write(result)
    print("  applied: %s" % label)
    return 0


def revert_mla_diff() -> int:
    label = "rocm_aiter_mla.py hunk D (DCP implementation)"
    if not os.path.exists(MLA_PATH):
        print("  not applied: %s (file missing)" % label)
        return 0
    src = open(MLA_PATH).read()
    if MLA_MARK not in src:
        print("  not applied: %s" % label)
        return 0
    # Gate the reverse on the exact result md5 rather than just trying it: if the
    # file does not match, someone stacked _patch_pad128.py (or the padding hunks)
    # on top, and a reverse would silently mangle their edits.
    if hashlib.md5(src.encode()).hexdigest() != MLA_RESULT_MD5:
        print("  !! %s has been modified since it was applied" % label)
        print("     refusing to reverse -- reinstall the file from the image")
        return 1
    result, err = _splice(src, False)
    if err or hashlib.md5(result.encode()).hexdigest() != MLA_BASE_MD5:
        print("  !! reverse failed (%s)" % (err or "md5 mismatch"))
        return 1
    open(MLA_PATH, "w").write(result)
    print("  reverted: %s" % label)
    return 0


# --- hunk E -----------------------------------------------------------------
# Make `a2a` the DCP combine default for K3, mirroring how GlmMoeDsaForCausalLM
# opts in. Upstream's global default is "ag_rs" (parallel.py set_dcp_defaults),
# and set_dcp_defaults only fills fields left None -- and vllm.py runs the model
# hook (:1127) BEFORE the global default (:1131) -- so this wins while an
# explicit `--dcp-comm-backend ag_rs` on the command line still overrides it.
#
# The K3 class already defines verify_and_update_model_config (the ModelConfig
# hook); this adds the separate VllmConfig hook, which the base class stubs out
# as a no-op, so both are dispatched.
E_ANCHOR = '''                quant_config["quant_method"] = "mxfp4"


class GptOssForCausalLMConfig(VerifyAndUpdateConfig):'''

E_PATCH = '''                quant_config["quant_method"] = "mxfp4"

    # --- K3 #51705: prefer the a2a DCP combine over the default ag_rs ---
    # MEASURED on MI355X (gfx950, 8-rank xGMI), wall ms per combine call:
    #   T=5   ag_rs 0.107  a2a 0.095 (1.13x)
    #   T=48  ag_rs 0.111  a2a 0.097 (1.15x)
    #   T=144 ag_rs 0.136  a2a 0.101 (1.35x)
    # a2a wins at every token count and the margin grows with T: it is one
    # all_to_all_single instead of allgather(lse) + reduce_scatter(out). combine
    # runs per MLA layer per decode step, so this multiplies by the layer count.
    # cos-similarity vs ag_rs was >= 0.999994 across the sweep.
    #
    # Deliberately NOT setting q_replicate=True (GlmMoeDsa does): that changes
    # weight loading and is an independent, unmeasured perf question.
    #
    # This does NOT select the symmetric-memory `direct` path. On ROCm
    # direct_cp_enabled() falls back to current_platform.is_cuda() (False), so
    # get_direct_dcp_a2a_workspace() returns None and _init_combine falls
    # through to dcp_a2a_lse_reduce. direct is reachable only by explicitly
    # setting VLLM_USE_DIRECT_DCP_A2A=1, and it should not be: its kernel-issued
    # peer stores move 17.8 GB/s at T=144 where RCCL moves 140 GB/s over the
    # same fabric (a single xGMI pair does 60 GB/s), and it strands 24 orphaned
    # dma-bufs per run.
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        vllm_config.parallel_config.set_dcp_defaults(comm_backend="a2a")


class GptOssForCausalLMConfig(VerifyAndUpdateConfig):'''


# --- hunk K -----------------------------------------------------------------
# K3-DCP-GKV-PERSIST. Under DCP the asm `cprr` decode applies causality from
# *global* positions, which it reads out of `g_kv_indptr` -- and it reads it
# inside the region a FULL cudagraph captures. Hunk D built that tensor with a
# fresh `torch.cat` on every build, so the graph replayed against whatever
# allocation happened to exist at capture time while each real step wrote its
# lengths somewhere the graph never looks. The verify mask was then computed
# against stale global positions: target output silently wrong, and the DSpark
# draft rejected almost everywhere (conc-1 block AL 2.40 PIECEWISE -> 1.09 FULL,
# pos-1 acceptance 67% -> 7.6%). PIECEWISE hid it because nothing is replayed.
#
# Fix = what the fold path already does one screen below (`_fold_g_kv_indptr`)
# and what ATOM does in `atom/plugin/vllm/attention/metadata.py`: preallocate
# once, fill in place, hand the kernel a stable address.
#
# Applies to MLA_PATH *after* hunk D, so it must stay in SITES (which apply()
# runs after apply_mla_diff, and revert() unwinds before revert_mla_diff).
K_ALLOC_ANCHOR = (
    "        # Static pseudo-batch buffers for the DCP head-fold verify path. Each of\n"
)
K_ALLOC_PATCH = (
    "        # K3-DCP-GKV-PERSIST: stable address for the global page indptr the\n"
    "        # cprr kernel reads *inside* the captured region. Zero-initialized;\n"
    "        # element 0 is the indptr base and is never written again.\n"
    "        self._g_kv_indptr_buf = None\n"
    "        if self.dcp_world_size > 1:\n"
    "            self._g_kv_indptr_buf = torch.zeros(\n"
    "                max_num_reqs + 1, dtype=torch.int32, device=device\n"
    "            )\n"
    "\n"
    "        # Static pseudo-batch buffers for the DCP head-fold verify path. Each of\n"
)

K_BUILD_ANCHOR = (
    "        g_kv_indptr = None\n"
    "        if self.dcp_world_size > 1 and dcp_tot_seq_lens_device is not None:\n"
    "            g_kv_indptr = torch.cat(\n"
    "                [\n"
    "                    torch.zeros(1, dtype=torch.int32, device=device),\n"
    "                    dcp_tot_seq_lens_device.cumsum(dim=0, dtype=torch.int32),\n"
    "                ]\n"
    "            )\n"
)
K_BUILD_PATCH = (
    "        g_kv_indptr = None\n"
    "        if self.dcp_world_size > 1 and dcp_tot_seq_lens_device is not None:\n"
    "            # K3-DCP-GKV-PERSIST: fill the preallocated buffer in place. A\n"
    "            # torch.cat here returns a new tensor each build, which a FULL\n"
    "            # cudagraph cannot follow -- see the hunk K note above.\n"
    "            assert self._g_kv_indptr_buf is not None\n"
    "            _ngk = dcp_tot_seq_lens_device.shape[0]\n"
    "            g_kv_indptr = self._g_kv_indptr_buf[: _ngk + 1]\n"
    "            g_kv_indptr[1:].copy_(\n"
    "                dcp_tot_seq_lens_device.cumsum(dim=0, dtype=torch.int32)\n"
    "            )\n"
)


# --- hunk M -----------------------------------------------------------------
# K3-DCP-RPMSLICE. aiter sizes its fp32 reduce scratch off the *capacity* of the
# reduce_partial_map it is handed:
#
#     logits = torch.empty((reduce_partial_map.size(0) * max_seqlen_q,
#                           1, nhead, v_head_dim), fp32)   # aiter/mla.py:843
#
# `self._mla_reduce_partial_map` is a worst-case buffer, sized once at builder
# init for `max_num_reqs` requests at `_mtp_decode_qlen`. Under DSpark that
# qlen is NOT 1+num_spec: `parallel_drafting` is True for dspark/dflash, so
# `_init_reorder_batch_threshold` (v1/attention/backend.py:641) sets
#     reorder_batch_threshold = 1 + 2*num_spec = 15   at num_spec=7
# while the verify the kernel actually runs is qlen 8. `build()` then handed
# aiter the whole 4785-entry buffer, so every decode call allocated
#     4785 * 8 * 128 * 512 * 4 B = 9.347 GiB
# of transient fp32 scratch -- at conc-1 as much as at conc-48. MEASURED: that
# is the exact "Tried to allocate 9.35 GiB" that killed the 48->1 sweep at
# conc-48 (Worker_TP0..7, 08-31 08:20:49) with only ~6.3 GiB free.
#
# REJECTED FIX (kept as a warning): slicing that buffer down to a per-build
# "live" tile count from `get_mla_metadata_info_v1(num_kernel_reqs, max_qo_len,
# ...)` FAULTS THE GPU. Under DCP the real build passes
#     max_split_per_batch=self._mla_max_split_per_batch, is_cp_round_robin=True
# (the `cprr_kwargs` block), and the info helper has NO `is_cp_round_robin`
# parameter at all -- it structurally cannot model round-robin tiling, so its
# estimate is not an upper bound on what `get_mla_metadata_v1` writes. The
# slice came out short and the reduce kernel walked off the end:
# "Memory access fault by GPU node-7 ... Reason: Unknown" during the DSpark
# speculator capture (08-31 09:21:44). Do not retry that route.
#
# Fix instead: shrink the CAPACITY, by sizing it at the qlen decode can really
# present. Same `get_mla_metadata_info_v1` call as before, one input corrected,
# so the layout maths the builder does is unchanged -- there is no new
# assumption about packing. `1 + num_spec` (8) is the verify width the target
# actually runs; the extra `num_spec` in `reorder_batch_threshold` is the
# parallel draft's own forward, a separate KV group with its own builder.
# 4785 -> 2552 tiles: 9.35 GiB -> 4.98 GiB of transient fp32 scratch per decode
# call, against the 6.24 GiB that was free when conc-48 died.
#
# `_mtp_decode_qlen` has exactly two readers -- this sizing call and the
# `max_qo_len <= self._mtp_decode_qlen` persistent-metadata gate -- so lowering
# it keeps the two self-consistent: anything wider than 8 now takes the
# non-persistent path rather than overflowing a buffer sized for 8.
#
# `K3_NO_QLEN_TRIM=1` restores the old sizing for A/B.
#
# Applies to MLA_PATH *after* hunk D -> must live in SITES.
M_QLEN_ANCHOR = (
    "        self._mtp_decode_qlen = self.reorder_batch_threshold or 1\n"
)
M_QLEN_PATCH = (
    "        self._mtp_decode_qlen = self.reorder_batch_threshold or 1\n"
    "        # K3-DCP-QLENTRIM: for a parallel drafter (dspark/dflash)\n"
    "        # reorder_batch_threshold is 1 + 2*num_spec = 15 at num_spec=7,\n"
    "        # but the target only ever verifies 1 + num_spec = 8. The extra\n"
    "        # width inflates every persistent metadata buffer below, and aiter\n"
    "        # sizes a transient fp32 reduce scratch off reduce_partial_map's\n"
    "        # full length (aiter/mla.py:843) -- 9.35 GiB per decode call at\n"
    "        # nhead=128, which OOMs conc-48. See the hunk M note in\n"
    "        # _port_dcp_nightly_ba07e4a4.py.\n"
    "        import os as _osq\n"
    "        if not _osq.environ.get('K3_NO_QLEN_TRIM'):\n"
    "            _spec_cfg = getattr(vllm_config, 'speculative_config', None)\n"
    "            _nspec = getattr(_spec_cfg, 'num_speculative_tokens', None)\n"
    "            if _nspec:\n"
    "                self._mtp_decode_qlen = min(\n"
    "                    self._mtp_decode_qlen, 1 + int(_nspec)\n"
    "                )\n"
)


# --- hunk N: barrier before the speculator cudagraph capture -----------------
#
# Under hunk L the DSpark draft is DCP-SHARDED, so its MLA runs a CP collective
# INSIDE this capture. Ranks reach capture() staggered by ~1s, so a rank can
# begin capturing a collective its peers have not entered yet -- a rank-ordering
# race that surfaces as a hard GPU fault at boot with a fresh address and pid
# every run. The pre-hunk-L replicated draft (vLLM #51705) ran at dcp=1 with no
# collective, so this path was never exercised.
#
# Evidence it is a race, not an OOB: AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1
# boots reliably. A config that only fails when kernels overlap is ordering.
#
# Boot-time only; measured boot-to-ready unchanged at ~280s. With it the full
# 48->1 sweep runs 9/9 clean at the mandated config; without it every DCP boot
# faults. The guards keep it safe for non-distributed and single-rank runs.
N_ANCHOR = (
    "    def capture(self) -> None:\n"
    '        logger.info("Capturing model for %s speculator...", self._speculator_name)\n'
)
N_PATCH = (
    "    def capture(self) -> None:\n"
    '        logger.info("Capturing model for %s speculator...", self._speculator_name)\n'
    "        # K3-DCP-SPECCAP-BARRIER: under DCP the ATOM-sharded draft runs a CP\n"
    "        # collective inside this capture, so every rank must be quiesced and\n"
    "        # aligned before any rank starts capturing. Boot-time only.\n"
    "        import torch.distributed as _dist\n"
    "        torch.cuda.synchronize()\n"
    "        if _dist.is_available() and _dist.is_initialized():\n"
    "            _dist.barrier()\n"
    "        torch.cuda.synchronize()\n"
)


# --- hunk P: size the MLA reduce scratch from the real split cap -------------
#
# aiter's mla_decode_fwd sizes its fp32 `logits` scratch straight off
# reduce_partial_map.size(0) (aiter/mla.py:843). Under DCP the head count pads
# to 128 instead of 16, so at the K3 serving shape (batch 64, qlen 15) the loose
# fast_mode estimate of 4785 tiles becomes
#
#     4785 * 8 * 128 * 512 * 4 B = 9.35 GiB
#
# per rank -- 58x the non-DCP 0.16 GiB, and exactly what makes FULL cudagraphs
# unaffordable under DCP (captures cleanly, then dies at runtime).
#
# aiter already computes the correct tighter bound. The metadata kernel's split
# budget is GLOBAL -- min(cu_num, max_split_per_batch * batch_size), see
# csrc/kernels/mla/metadata/v1_2_device.cuh:560-562 -- so
#
#     reduce_partial_map <= tile_cnt + per_tile_cap        (= 960 + 256 = 1216)
#
# but it was combined with max(), which can only raise, so the tight bound could
# never win. Three defects had to be fixed together; any one alone is inert:
#
#   P1. aiter: max() -> min(), so a supplied cap actually binds.
#   P2. vLLM: pass the cap to the SIZING call as max_split_per_batch=. It was
#       passed as num_kv_splits=, which the info helper ignores for this.
#   P3. vLLM: pass the cap on EVERY DCP build, not just the round-robin one.
#
# P3 is the correctness-critical half. The non-round-robin fallback (taken when
# dcp_tot_seq_lens_device is None) otherwise writes up to 7.7x the sized entries
# -- measured -- and an undersized reduce buffer FAULTS THE GPU rather than
# raising, so sizing and build must agree unconditionally.
#
# Blast radius: the tight branch requires max_split_per_batch > 0, and P2 passes
# -1 when dcp_world_size == 1. The only other caller of the info helper,
# rocm_aiter_mla_sparse.py:437, passes no cap and is unaffected.
#
# Validation (gfx950, serve-free single-GPU driver, fp8/fp8, num_heads 128):
# 1030 shapes comparing measured reduce_indptr.max() against the new bound ->
# 0 violations, worst actual/bound 0.998 (reached but never exceeded, i.e. the
# bound is exact, not lucky). Control with no cap at build time: 119/380 shapes
# violate, worst 7.7x. End-to-end 960 cases with a 4096-entry canary tail, both
# round-robin and fallback: 0 overflows, 0 canary writes, worst fill 1.000.
#
# Effect: reduce_partial_map 4785 -> 1216, scratch 9.35 -> 2.38 GiB, capture
# 22.47 -> 18.29 GiB, zero runtime OOM.
#
# P1 targets AITER_ATTN (chain step 0b); P2/P3 target MLA_PATH *after* hunk D.
P1_ANCHOR = (
    "        per_tile_cap = min(max_splits, max_split_per_batch * batch_size)\n"
    "        max_split_tiles = max(max_split_tiles, tile_cnt + per_tile_cap)\n"
)
P1_PATCH = (
    "        per_tile_cap = min(max_splits, max_split_per_batch * batch_size)\n"
    "        # This is a strictly TIGHTER bound than the fast_mode estimate above,\n"
    "        # which assumes an unbounded per-batch split budget. When the caller\n"
    "        # supplies a cap the kernel cannot exceed tile_cnt + per_tile_cap, so\n"
    "        # take the min -- max() would let the loose estimate always win and the\n"
    "        # cap would have no effect. Measured over 1030 (batch, qlen, ragged-kv)\n"
    "        # shapes on gfx950, both cprr and non-cprr: 0 violations, worst\n"
    "        # actual/bound 0.998. NOTE: only valid because the same\n"
    "        # max_split_per_batch is passed to get_mla_metadata_v1 at build time.\n"
    "        max_split_tiles = min(max_split_tiles, tile_cnt + per_tile_cap)\n"
)

P2_ANCHOR = (
    "                fast_mode=True,\n"
    "                num_kv_splits=self._mla_max_split_per_batch,\n"
    "                intra_batch_mode=False,\n"
)
P2_PATCH = (
    "                fast_mode=True,\n"
    "                # Must match the cap passed to get_mla_metadata_v1 at build\n"
    "                # time, or the persistent reduce_partial_map is mis-sized.\n"
    "                # num_kv_splits= was a no-op here: the info helper keys the\n"
    "                # tight bound off max_split_per_batch.\n"
    "                max_split_per_batch=(\n"
    "                    self._mla_max_split_per_batch if self.dcp_world_size > 1 else -1\n"
    "                ),\n"
    "                intra_batch_mode=False,\n"
)

P3_ANCHOR = (
    "            if g_kv_indptr is not None:\n"
    "                is_causal = False\n"
    "                cprr_kwargs = dict(\n"
    "                    max_split_per_batch=self._mla_max_split_per_batch,\n"
    "                    intra_batch_mode=False,\n"
    "                    is_cp_round_robin=True,\n"
    "                )\n"
)
P3_PATCH = (
    "            if self.dcp_world_size > 1:\n"
    "                # reduce_partial_map is sized with this cap in effect, so EVERY\n"
    "                # DCP build must pass it -- including the non-round-robin\n"
    "                # fallback taken when dcp_tot_seq_lens_device is None. Without\n"
    "                # it that path writes up to 7.7x the sized entries (measured).\n"
    "                cprr_kwargs = dict(\n"
    "                    max_split_per_batch=self._mla_max_split_per_batch,\n"
    "                    intra_batch_mode=False,\n"
    "                )\n"
    "            if g_kv_indptr is not None:\n"
    "                is_causal = False\n"
    '                cprr_kwargs["is_cp_round_robin"] = True\n'
)


# --- hunk Q: refuse a block-level KV interleave under spec decode ------------
#
# The qlen>1 verify cprr MLA kernel assumes TOKEN-level round-robin interleave
# (cp_kv_cache_interleave_size == 1): it reconstructs each token's owning rank
# from its global position. ATOM documents the same constraint and disables its
# interleave_size knob whenever speculative decode is on.
#
# vLLM will silently violate that. adjust_dcp_kv_cache_interleave_size() above
# OVERRIDES cp_kv_cache_interleave_size to local_block_size whenever a KV
# connector is configured -- that is KV offload as well as P/D disaggregation --
# and it reports the change at info_once, not warning. Worse,
# validate_block_size() then DELIBERATELY SKIPS the DCP interleave compatibility
# assert in exactly that case ("Skip DCP interleave-size compatibility when a KV
# connector is configured"). So both the override and the check that would have
# caught it are disabled together, and the override happens at WORKER init,
# after config validation has already passed.
#
# The failure mode is silent wrong logits, not a crash: attention reads the
# right number of tokens from the wrong ranks. Nothing in the sweep would show
# it except accuracy, and the long-ctx bench does not measure accuracy.
#
# We are only safe today by accident -- offload is off for PERFORMANCE reasons
# on this bench shape (memory k3-offload-harmful-on-longctx-bench-shape), not
# for correctness ones. Turning offload on for the agentic corpus, which is a
# stated goal, would walk straight into this.
#
# So assert at the override site, which is where the value actually becomes
# wrong, rather than at config time where it still looks fine.
Q_ANCHOR = (
    "                dcp_size,\n"
    "                interleave,\n"
    "                local_block_size,\n"
    "            )\n"
)
Q_PATCH = (
    "                dcp_size,\n"
    "                interleave,\n"
    "                local_block_size,\n"
    "            )\n"
    "\n"
    "        # K3-DCP-INTERLEAVE-GUARD: the qlen>1 verify cprr MLA kernel assumes\n"
    "        # TOKEN-level interleave. The override above (triggered by ANY KV\n"
    "        # connector, offload included) silently breaks that, and\n"
    "        # validate_block_size() skips its compatibility assert in the same\n"
    "        # case. Silent wrong logits, so fail loudly instead.\n"
    "        if (\n"
    "            self.num_speculative_tokens\n"
    "            and self.parallel_config.cp_kv_cache_interleave_size != 1\n"
    "        ):\n"
    "            raise ValueError(\n"
    "                'DCP + speculative decode requires '\n"
    "                'cp_kv_cache_interleave_size == 1 (token-level round robin); '\n"
    "                'got %d. A KV connector (offload or P/D) pins it to the '\n"
    "                'local block size, which the qlen>1 verify cprr MLA kernel '\n"
    "                'cannot honour -- it would read the right token count from '\n"
    "                'the wrong ranks and silently corrupt logits. Disable the KV '\n"
    "                'connector or run without speculative decode.'\n"
    "                % self.parallel_config.cp_kv_cache_interleave_size\n"
    "            )\n"
)


# --- hunk R: the reduce-scratch split cap must be an invariant, not a habit --
#
# Hunk P makes reduce_partial_map's size depend on max_split_per_batch being
# passed at EVERY build. That is a whole-program invariant with nothing holding
# it up: `self._mla_max_split_per_batch = 32` is a bare constant (line ~447)
# and the three call sites agree only because we edited them. A fourth added
# later writes up to 7.7x the sized entries, and an undersized reduce buffer
# FAULTS THE GPU rather than raising -- a fresh fault address every run, i.e.
# maximally hard to attribute back to here.
#
# Cheap always-on half: assert the kwarg is present and matches, right before
# the build. Pure Python, no sync, safe inside cudagraph capture.
#
# Expensive opt-in half: K3_DCP_CHECK_REDUCE=1 syncs after the build and checks
# the entries actually written against the buffer's real length. That is the
# check that would catch a NEW call site, and the one to reach for if a DCP
# memory fault ever reappears. Off by default -- it syncs, so it cannot run
# under capture or in steady state.
R_ANCHOR = (
    "                    dtype_kv=self._mla_kv_dtype,\n"
    "                    **cprr_kwargs,\n"
    "                )\n"
)
R_PATCH = (
    "                    dtype_kv=self._mla_kv_dtype,\n"
    "                    **cprr_kwargs,\n"
    "                )\n"
    "                # K3-DCP-SPLITCAP-GUARD: reduce_partial_map was SIZED with\n"
    "                # this cap in effect (hunk P2), so every DCP build must pass\n"
    "                # the same value or the buffer is undersized and the kernel\n"
    "                # faults the GPU instead of raising.\n"
    "                assert (\n"
    "                    self.dcp_world_size <= 1\n"
    "                    or cprr_kwargs.get('max_split_per_batch')\n"
    "                    == self._mla_max_split_per_batch\n"
    "                ), (\n"
    "                    'DCP build must pass max_split_per_batch=%r to match the '\n"
    "                    'reduce_partial_map sizing; got %r'\n"
    "                    % (\n"
    "                        self._mla_max_split_per_batch,\n"
    "                        cprr_kwargs.get('max_split_per_batch'),\n"
    "                    )\n"
    "                )\n"
    "                import os as _osrc\n"
    "                if _osrc.environ.get('K3_DCP_CHECK_REDUCE') and (\n"
    "                    not torch.cuda.is_current_stream_capturing()\n"
    "                ):\n"
    "                    torch.cuda.synchronize()\n"
    "                    _used = int(self._mla_reduce_indptr.max().item())\n"
    "                    _cap = int(self._mla_reduce_partial_map.numel())\n"
    "                    assert _used <= _cap, (\n"
    "                        'reduce_partial_map OVERFLOW: %d entries written into '\n"
    "                        'a buffer of %d. A build path is not passing the split '\n"
    "                        'cap -- see hunk P3.' % (_used, _cap)\n"
    "                    )\n"
)


# --- hunk S: declare the LSE log base explicitly -----------------------------
#
# The cross-rank merge exponentiates the per-rank LSEs, so the log base is not
# cosmetic: base-e values fed to an exp2 merge are wrong by a factor of ln 2 in
# the exponent. That is a silent temperature-like error, not a crash.
#
# aiter's asm MLA emits natural-log LSE, our direct-a2a HIP kernel converts with
# K_LOG2E_F before using exp2f, and the plumbing is correct end to end -- but
# only by INHERITANCE: MLACommonImpl defaults lse_base_on_e = True
# (v1/attention/backend.py:804) and AiterMLAImpl never says so. Every other
# backend states it outright (tokenspeed False, flashinfer_mla True,
# flashinfer_mla_sparse False), so ours is the odd one out, and a future change
# to the base default would silently flip our merge.
S_ANCHOR = (
    "class AiterMLAImpl(MLACommonImpl[AiterMLAMetadata]):\n"
    "    # Under DCP the base decode path all-gathers query heads and needs each\n"
)
S_PATCH = (
    "class AiterMLAImpl(MLACommonImpl[AiterMLAMetadata]):\n"
    "    # K3-DCP-LSE-BASE: aiter's asm MLA emits NATURAL-LOG lse. Stated\n"
    "    # explicitly rather than inherited from MLACommonImpl's default so a\n"
    "    # change to that default cannot silently flip the cross-rank merge --\n"
    "    # the direct-a2a HIP kernel converts with K_LOG2E_F on the strength of\n"
    "    # this flag, and getting it wrong is a silent ln(2) error, not a crash.\n"
    "    lse_base_on_e: bool = True\n"
    "    # Under DCP the base decode path all-gathers query heads and needs each\n"
)


SITES = (
    # K3-DCP-ATOM. Must come FIRST: it anchors on text produced by hunk D.
    ("rocm_aiter_mla.py hunk L (no draft collapse)", MLA_PATH, L_ANCHOR, L_PATCH),
    ("models/config.py hunk E (a2a combine default)", MODELS_CFG, E_ANCHOR, E_PATCH),
    # K3-DCP-FULLGRAPH. Independent of the above; env-gated, default-off.
    ("platforms/rocm.py hunk J (FULL cudagraph opt-out)", ROCM_PLAT, J_ANCHOR, J_PATCH),
    # K3-DCP-GKV-PERSIST. Stacks on hunk D; the alloc must precede the build use,
    # but the anchors are disjoint so either order applies cleanly.
    ("rocm_aiter_mla.py hunk K1 (g_kv_indptr buffer)", MLA_PATH, K_ALLOC_ANCHOR, K_ALLOC_PATCH),
    ("rocm_aiter_mla.py hunk K2 (fill in place)", MLA_PATH, K_BUILD_ANCHOR, K_BUILD_PATCH),
    # K3-DCP-SPECCAP-BARRIER. Independent of every other hunk, but REQUIRED
    # once hunk L lands: without it every DCP boot GPU-faults.
    ("dflash/speculator.py hunk N (capture barrier)", SPEC_PATH, N_ANCHOR, N_PATCH),
    # K3-DCP-REDUCE-SCRATCH. P1/P2/P3 are one fix in three places and must land
    # together -- P1 or P2 alone is inert, and P2 without P3 UNDERSIZES the
    # buffer on the fallback build path, which faults the GPU.
    ("aiter/ops/attention.py hunk P1 (tight bound wins)", AITER_ATTN, P1_ANCHOR, P1_PATCH),
    ("rocm_aiter_mla.py hunk P2 (cap the sizing call)", MLA_PATH, P2_ANCHOR, P2_PATCH),
    ("rocm_aiter_mla.py hunk P3 (cap EVERY DCP build)", MLA_PATH, P3_ANCHOR, P3_PATCH),
    # Guards. These change no behaviour on a correct config; they convert three
    # silent-wrongness modes into loud failures. R must follow P3 (it anchors on
    # text P3 leaves in place, and asserts the invariant P3 establishes).
    ("config/vllm.py hunk Q (interleave guard)", VLLM_CFG, Q_ANCHOR, Q_PATCH),
    ("rocm_aiter_mla.py hunk R (split-cap guard)", MLA_PATH, R_ANCHOR, R_PATCH),
    ("rocm_aiter_mla.py hunk S (declare lse base)", MLA_PATH, S_ANCHOR, S_PATCH),
    # K3-DCP-QLENTRIM: DISABLED -- refuted by measurement, see the hunk M note.
    # ("rocm_aiter_mla.py hunk M (verify-qlen capacity trim)", MLA_PATH, M_QLEN_ANCHOR, M_QLEN_PATCH),
)


def apply() -> int:
    # Hunk D first: it is the file the other three hunks assume exists in
    # its DCP form, and it is the one that can fail on upstream drift.
    rc = apply_mla_diff()
    if rc:
        return rc
    for label, path, anchor, patch in SITES:
        if not os.path.exists(path):
            print("  !! missing %s" % path)
            if path == AITER_ATTN:
                print("     hunk P1 patches the TRANSPLANTED aiter tree, which is")
                print("     chain step 0b and must exist before this script runs:")
                print("       docker cp <donor>:/opt/aiter-local /tmp/aiter-local")
                print("       docker cp /tmp/aiter-local <target>:/opt/")
                print("     Refusing rather than applying P2/P3 without P1 -- that")
                print("     combination undersizes nothing but leaves the 9.35 GiB")
                print("     scratch in place, i.e. a silent no-fix.")
            return 1
        src = open(path).read()
        if patch in src:
            print("  already applied: %s" % label)
            continue
        n = src.count(anchor)
        if n != 1:
            print("  !! anchor for %s matched %dx, expected 1" % (label, n))
            return 1
        open(path, "w").write(src.replace(anchor, patch, 1))
        print("  applied: %s" % label)
    return 0


def revert() -> int:
    for label, path, anchor, patch in SITES:
        src = open(path).read()
        if patch not in src:
            print("  not applied: %s" % label)
            continue
        open(path, "w").write(src.replace(patch, anchor, 1))
        print("  reverted: %s" % label)
    # Hunk D last, mirroring apply(): the small hunks come out first so a
    # refusal here leaves them already backed out, not stranded on top.
    return revert_mla_diff()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()
    sys.exit(revert() if args.revert else apply())
