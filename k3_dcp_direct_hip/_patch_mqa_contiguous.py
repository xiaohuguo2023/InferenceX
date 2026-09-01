#!/usr/bin/env python3
"""Force the non-fold DCP return of forward_mqa to be contiguous.

The qlen>=3 fold path already .contiguous()-es o_out/lse_out, but the non-fold
return builds them from get_mla_unpadded_o/lse, which return STRIDED head-slices
(o[:, ::step, :]). Those feed the direct DCP a2a combine, whose kernel reinterprets
partial_output as uint4* assuming heads are contiguous within a token -> OOB
"Memory access fault by GPU" at warmup. Contiguizing here matches the fold path
and is a K3-recipe fix (the strided slice is aiter-fold plumbing, not upstream).
Idempotent; .k3bak backup.
"""
import sys

F = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/rocm_aiter_mla.py"
src = open(F).read()

old = """        o_out = AiterMLAHelper.get_mla_unpadded_o(dcp_heads, o)
        if final_lse is not None:
            lse_out = AiterMLAHelper.get_mla_unpadded_lse(dcp_heads, final_lse)
            return o_out, lse_out
        return o_out, None"""

new = """        o_out = AiterMLAHelper.get_mla_unpadded_o(dcp_heads, o).contiguous()
        if final_lse is not None:
            lse_out = AiterMLAHelper.get_mla_unpadded_lse(
                dcp_heads, final_lse
            ).contiguous()
            return o_out, lse_out
        return o_out, None"""

if new in src:
    print("ALREADY PATCHED")
    sys.exit(0)
assert src.count(old) == 1, f"anchor not unique/absent (count={src.count(old)})"
open(F + ".k3bak_mqa", "w").write(src)
open(F, "w").write(src.replace(old, new, 1))
print("PATCHED (backup -> .k3bak_mqa)")
