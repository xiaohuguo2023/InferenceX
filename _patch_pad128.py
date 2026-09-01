#!/usr/bin/env python3
"""Replace the DCP head-FOLD verify path with NV-native PAD-128.

dcp_heads = num_heads*dcp_world_size = 96 is not a native cprr count
{16,32,64,128}. The head-fold (F=3 pseudo-reqs of nhead=32) is memory-optimal
but requires a post-kernel permute/reshape/.contiguous() un-fold copy that RACES
the symm-mem a2a combine -> needs a per-layer host sync that costs ~5x conc-1 ITL
and (MEASURED) does NOT amortize (flat ~175 ms through conc-24).

NV avoids the race structurally: cutlass pads the query to a native head count
(reserve_query_head_storage / q_pad_num_heads=128), runs ONE fixed-head kernel,
and slices the real heads back off the output as a *view* (`out[:, :H]`) with NO
.contiguous(). One producer kernel -> one consumer op -> naturally ordered -> no
sync. `_pad128_vs_fold.py` proved aiter's fp8-asm cprr kernel is BIT-EXACT under
this padding (cos=1.0, max|d|=0 across ranks 0/3/7, req/ctx combos).

This patch switches our path to the same approach:
  (1) builder: _num_attention_heads := smallest native cprr >= dcp_heads (128),
      drop the fold sizing/metadata -> do_fold is always False, so persistent
      cprr metadata is built at 128 heads (the else-branch already handles it).
  (2) impl forward_mqa: pad q 96->128 (tile+slice, a PRE-kernel copy that does
      not race), run the native kernel at 128, slice o/lse back to dcp_heads as a
      VIEW (no .contiguous()). No K3_DCP_SYNC needed.

The head-fold code (forward_mqa fold branch, _build_fold_pseudo_metadata, the
_fold_* buffers) is left in place but is now dead (fold_factor stays 1 ->
do_fold False -> fold_qo_indptr None), so the diff stays reviewable.
"""

F = ("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/"
     "mla/rocm_aiter_mla.py")
src = open(F).read()

# ---- Edit 1: builder __init__ head/fold setup -> pad-to-native ----
OLD1 = '''        self._dcp_fold_factor = 1
        self._dcp_fold_heads = 0
        _NATIVE_CPRR = (16, 32, 64, 128)
        if self.dcp_world_size > 1 and self._decode_num_heads not in _NATIVE_CPRR:
            for _nat in (64, 32, 16):
                if (
                    self._decode_num_heads > _nat
                    and self._decode_num_heads % _nat == 0
                ):
                    self._dcp_fold_heads = _nat
                    self._dcp_fold_factor = self._decode_num_heads // _nat
                    break'''

NEW1 = '''        self._dcp_fold_factor = 1
        self._dcp_fold_heads = 0
        # DCP pad-to-native (NV cutlass reserve_query_head_storage style): pad
        # the non-native dcp_heads (96) UP to the smallest native cprr count
        # (128) and run ONE native kernel; the real dcp_heads are sliced back
        # off the output as a VIEW in forward_mqa. Unlike the head-fold this
        # produces NO post-kernel un-fold copy, so the symm-mem a2a combine has
        # no race and needs NO per-layer host sync. (fold_* left unused.)
        self._dcp_pad_heads = 0
        _NATIVE_CPRR = (16, 32, 64, 128)
        if self.dcp_world_size > 1 and self._decode_num_heads not in _NATIVE_CPRR:
            for _nat in _NATIVE_CPRR:
                if _nat >= self._decode_num_heads:
                    self._dcp_pad_heads = _nat
                    break
            assert self._dcp_pad_heads, (
                f"no native cprr count >= {self._decode_num_heads}"
            )
            # Size persistent cprr metadata + run the kernel at the padded count.
            self._num_attention_heads = self._dcp_pad_heads'''

assert src.count(OLD1) == 1, f"Edit1: expected 1, got {src.count(OLD1)}"
src = src.replace(OLD1, NEW1)

# ---- Edit 2: builder __init__ metadata sizing -> drop fold-max branch ----
OLD2 = '''        _mla_meta = _mla_meta_info(max_num_reqs, self._num_attention_heads)
        if self._dcp_fold_factor > 1:
            # The qlen>=3 verify path builds metadata for max_num_reqs*F
            # pseudo-requests at nhead=fold_heads; size buffers to the max of
            # both the native (qlen==1) and the folded shapes.
            _mla_meta_fold = _mla_meta_info(
                max_num_reqs * self._dcp_fold_factor, self._dcp_fold_heads
            )
            _mla_meta = tuple(
                (max(a[0], b[0]), a[1])
                for a, b in zip(_mla_meta, _mla_meta_fold)
            )'''

NEW2 = '''        # _num_attention_heads is already the padded native cprr count under
        # DCP (pad-128), so this sizes the persistent metadata correctly for
        # both qlen==1 decode and qlen>=3 verify. No fold sizing needed.
        _mla_meta = _mla_meta_info(max_num_reqs, self._num_attention_heads)'''

assert src.count(OLD2) == 1, f"Edit2: expected 1, got {src.count(OLD2)}"
src = src.replace(OLD2, NEW2)

# ---- Edit 3a: forward_mqa head count -> pad q 96->128 ----
OLD3A = '''        dcp_heads = self.num_heads * self.dcp_world_size
        mla_padded_q = AiterMLAHelper.get_mla_padded_q(dcp_heads, q)
        mla_num_heads = AiterMLAHelper.get_actual_mla_num_heads(dcp_heads)'''

NEW3A = '''        dcp_heads = self.num_heads * self.dcp_world_size
        # DCP pad-to-native: pad the non-native dcp_heads UP to the smallest
        # native cprr count so ONE native kernel runs. dcp_world_size read LIVE
        # (the replicated draft group forces it to 1 post-construction).
        pad_heads = dcp_heads
        if self.dcp_world_size > 1 and dcp_heads not in (16, 32, 64, 128):
            for _nat in (16, 32, 64, 128):
                if _nat >= dcp_heads:
                    pad_heads = _nat
                    break
        if pad_heads != dcp_heads:
            # Tile-and-slice pad 96->128 (a PRE-kernel copy; does not race the
            # combine). MLA heads attend independently over the shared KV, so
            # the padding heads cannot affect heads [0:dcp_heads]; they are
            # sliced back off the output. Bit-exact vs fold (_pad128_vs_fold.py).
            _reps = -(-pad_heads // dcp_heads)  # ceil(pad/dcp)
            mla_padded_q = q.repeat(1, _reps, 1)[:, :pad_heads, :].contiguous()
            mla_num_heads = pad_heads
        else:
            mla_padded_q = AiterMLAHelper.get_mla_padded_q(dcp_heads, q)
            mla_num_heads = AiterMLAHelper.get_actual_mla_num_heads(dcp_heads)'''

assert src.count(OLD3A) == 1, f"Edit3a: expected 1, got {src.count(OLD3A)}"
src = src.replace(OLD3A, NEW3A)

# ---- Edit 3b: forward_mqa unpad -> slice real heads as a VIEW ----
#
# Two anchor variants exist. The original was written against the fold file
# before `.contiguous()` was added to the un-fold return path; the file shipped
# in k3-nightly4/5-test has it on BOTH o_out and lse_out (and wraps the lse
# call). Re-anchored 2026-08-25 to accept either, and the else-branch below is
# emitted to match whichever anchor matched -- adding or dropping a
# `.contiguous()` on the *fold* path is a separate question from pad-128 and is
# not this patch's call to make. (Whether the *pad* slice needs one is exactly
# what _patch_padview_contig.py exists to answer, so it stays a bare view here.)
# The pad branch is identical in both; only the else (fold) branch differs.
PAD_BRANCH = '''        if pad_heads != dcp_heads:
            # Slice the real heads back off as a non-contiguous VIEW (NV cutlass
            # `out[:, :H]` style). NO .contiguous(): a post-kernel copy is
            # exactly what races the symm-mem a2a combine. The combine op reads
            # strided partial_output correctly (the same path NV's padded
            # cutlass decode feeds), so no host sync is required.
            o_out = o[:, :dcp_heads, :]
        else:
'''

VARIANTS = (
    # (anchor, else-branch o_out, else-branch lse_out)
    ('''        o_out = AiterMLAHelper.get_mla_unpadded_o(dcp_heads, o).contiguous()
        if final_lse is not None:
            lse_out = AiterMLAHelper.get_mla_unpadded_lse(
                dcp_heads, final_lse
            ).contiguous()
            return o_out, lse_out
        return o_out, None''',
     "            o_out = AiterMLAHelper.get_mla_unpadded_o("
     "dcp_heads, o).contiguous()",
     "                lse_out = AiterMLAHelper.get_mla_unpadded_lse(\n"
     "                    dcp_heads, final_lse\n"
     "                ).contiguous()"),
    ('''        o_out = AiterMLAHelper.get_mla_unpadded_o(dcp_heads, o)
        if final_lse is not None:
            lse_out = AiterMLAHelper.get_mla_unpadded_lse(dcp_heads, final_lse)
            return o_out, lse_out
        return o_out, None''',
     "            o_out = AiterMLAHelper.get_mla_unpadded_o(dcp_heads, o)",
     "                lse_out = AiterMLAHelper.get_mla_unpadded_lse("
     "dcp_heads, final_lse)"),
)

for OLD3B, _o_else, _lse_else in VARIANTS:
    if src.count(OLD3B) == 1:
        break
else:
    raise AssertionError("Edit3b: neither un-fold anchor variant matched")

NEW3B = (
    PAD_BRANCH
    + _o_else + "\n"
    + "        if final_lse is not None:\n"
    + "            if pad_heads != dcp_heads:\n"
    + "                lse_out = final_lse[:, :dcp_heads]\n"
    + "            else:\n"
    + _lse_else + "\n"
    + "            return o_out, lse_out\n"
    + "        return o_out, None"
)

src = src.replace(OLD3B, NEW3B)

open(F, "w").write(src)
print("PATCHED rocm_aiter_mla.py -> DCP pad-128 (fold path now dead/unused)")
