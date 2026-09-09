#!/usr/bin/env python3
"""Pure-aiter A/B: CP round-robin MLA decode (`_lse_cprr_ps`) vs the non-CP
`_ps` kernel at an IDENTICAL per-rank shape.

No vLLM, no model weights, no distributed init, no out-of-tree patches -- only
aiter and one gfx950 GPU. Written so the measurement can be reproduced without
any of our tree.

What is held equal between the two arms
---------------------------------------
    batch            1
    max_qo_len       T   (default 15)
    num_heads        H   (default 128, the count the cprr kernel runs at)
    per-rank KV rows S   (default 27318)
    dtype            fp8 q / fp8 kv, page_size 1

CP-ps is context-parallel over KV, not query-parallel: every cp_rank still runs
the full max_qo_len, and Q tiling does not depend on CP. What CP changes is that
KV is round-robin sharded, so a rank reads ~S_global/W rows and the kernel
evaluates causal in global position space g(j) = j*W + r. So the honest
comparison is CP at S rows/rank against non-CP at the SAME S rows -- which is
what this does. The CP arm is handed a global indptr of S*W so its round-robin
mapping is exercised, while its local shard stays S.

Usage:
    python3 _aiter_cprr_repro.py [--heads 128] [--qlen 15] [--slocal 27318]
                                 [--world 8] [--iters 50]
"""
import argparse
import time

import torch

import aiter
from aiter import dtypes as adt
from aiter.ops.attention import get_mla_metadata_info_v1, get_mla_metadata_v1

KV_LORA = 512
ROPE = 64
D = KV_LORA + ROPE  # 576


def build(args, cp: bool):
    dev = torch.device("cuda")
    B, T, H, S = args.batch, args.qlen, args.heads, args.slocal
    W = args.world if cp else 1

    q = torch.randn(B * T, H, D, device=dev, dtype=torch.bfloat16).to(adt.fp8)
    # page_size 1: one KV row per page. Only this rank's shard is allocated.
    kv = torch.randn(B * S, 1, 1, D, device=dev, dtype=torch.bfloat16).to(adt.fp8)
    o = torch.empty(B * T, H, KV_LORA, device=dev, dtype=torch.bfloat16)

    qo_indptr = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=dev)
    kv_indptr = torch.arange(0, (B + 1) * S, S, dtype=torch.int32, device=dev)
    kv_indices = torch.arange(B * S, dtype=torch.int32, device=dev)
    kv_last_page_len = torch.ones(B, dtype=torch.int32, device=dev)

    # GLOBAL page indptr: the cprr kernel resolves each local row j to global
    # position j*W + r before applying the causal window.
    g_kv_indptr = (torch.arange(0, (B + 1) * S * W, S * W, dtype=torch.int32,
                                device=dev) if cp else None)

    # Split config must be IDENTICAL on both arms, and num_kv_splits must match
    # max_split_per_batch -- this is what op_tests/test_mla_persistent_round_robin.py
    # does. Setting max_split_per_batch on the CP arm only (or leaving
    # num_kv_splits at 1) starves one arm of context-dimension parallelism and
    # the comparison measures the split config, not the kernel.
    meta_kw = dict(intra_batch_mode=False, max_split_per_batch=args.max_split)
    sizes = get_mla_metadata_info_v1(
        B, T, H, q.dtype, kv.dtype, is_sparse=False, fast_mode=True,
        num_kv_splits=args.max_split, **meta_kw)
    bufs = [torch.empty(sz, dtype=dt, device=dev) for (sz, dt) in sizes]
    work_meta, work_indptr, work_info, red_indptr, red_final, red_partial = bufs

    cprr_kw = dict(meta_kw)
    if cp:
        cprr_kw["is_cp_round_robin"] = True
    get_mla_metadata_v1(
        qo_indptr, kv_indptr, kv_last_page_len, H, 1,
        not cp,                      # is_causal: cprr does the mask itself
        work_meta, work_info, work_indptr, red_indptr, red_final, red_partial,
        page_size=1, kv_granularity=16, max_seqlen_qo=T, uni_seqlen_qo=T,
        fast_mode=True, dtype_q=q.dtype, dtype_kv=kv.dtype, **cprr_kw)

    call_kw = dict(
        q_scale=torch.tensor(1.0, device=dev),
        kv_scale=torch.tensor(1.0, device=dev),
        work_meta_data=work_meta, work_indptr=work_indptr, work_info_set=work_info,
        reduce_indptr=red_indptr, reduce_final_map=red_final,
        reduce_partial_map=red_partial,
        num_kv_splits=args.max_split, intra_batch_mode=False, return_lse=True,
    )
    if cp:
        call_kw.update(g_kv_indptr=g_kv_indptr, cp_world_size=W, cp_rank=0)

    call_kw["sm_scale"] = D ** -0.5

    def run():
        aiter.mla.mla_decode_fwd(
            q, kv.view(-1, 1, 1, D), o,
            qo_indptr, kv_indptr, kv_indices, kv_last_page_len, T, **call_kw)

    return run


def bench(fn, iters):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=128)
    ap.add_argument("--qlen", type=int, default=15)
    ap.add_argument("--slocal", type=int, default=27318)
    ap.add_argument("--world", type=int, default=8)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--splits", type=int, default=1)
    ap.add_argument("--max-split", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    B, T, H, S = args.batch, args.qlen, args.heads, args.slocal
    flop = 2.0 * B * T * H * S * (D + KV_LORA)
    print("batch=%d max_qo_len=%d heads=%d kv_rows_per_rank=%d  (%.2f GFLOP)\n"
          % (B, T, H, S, flop / 1e9))
    print("%-28s %10s %12s" % ("kernel", "us/call", "TFLOP/s"))
    print("-" * 52)
    res = {}
    for label, cp in (("non-CP  _ps", False), ("CP      _lse_cprr_ps", True)):
        try:
            us = bench(build(args, cp), args.iters)
        except Exception as exc:
            print("%-28s FAILED: %s" % (label, str(exc)[:60]))
            continue
        res[label] = us
        print("%-28s %10.1f %12.0f" % (label, us, flop / (us * 1e-6) / 1e12))
    if len(res) == 2:
        a, b = res["non-CP  _ps"], res["CP      _lse_cprr_ps"]
        print("\nCP / non-CP = %.2fx" % (b / a))


if __name__ == "__main__":
    main()
