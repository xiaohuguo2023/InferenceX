#!/usr/bin/env python3
"""Numerically validate the native-HIP direct_dcp_a2a_lse_reduce port.

Run: torchrun --standalone --nproc_per_node=8 _test_direct_a2a.py

Loads the ported .so (registers torch.ops._C.direct_dcp_a2a_lse_reduce), then
drives vLLM's OWN DirectDCPA2AWorkspace.lse_reduce (the exact production combine
path) across 8 ranks and compares its output to a reference LSE-weighted merge
computed with torch collectives in fp32. Green => the op is numerically correct
AND the vLLM wiring (symm-mem peer-ptr alloc + call) works on ROCm.

DCP semantics: after the query all-gather each rank holds partial attention over
its OWN KV shard for ALL total_heads = W*heads_per_rank query heads. The combine
gives each rank its OWN heads_per_rank heads, softmax-merged across all W shards.
"""
import os
import torch
import torch.distributed as dist

RANK = int(os.environ["RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD = int(os.environ["WORLD_SIZE"])

SO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  "dcp_direct_a2a_lse_reduce.so")
T = 5                 # tokens (nspec2 verify: ~1+2*2)
HPR = 12              # heads_per_rank (K3: 12)
HDIM = 512            # head_dim (K3 MLA v-dim)
MAX_NT = 16
LOG2E = 1.4426950408889634


def log(m):
    print(f"[rank{RANK}] {m}", flush=True)


def reference(partial_output, partial_lse, is_lse_base_on_e):
    """fp32 reference: gather all shards, softmax-merge my heads."""
    total_heads = WORLD * HPR
    go = torch.empty((WORLD, T, total_heads, HDIM), dtype=partial_output.dtype,
                     device=partial_output.device)
    gl = torch.empty((WORLD, T, total_heads), dtype=torch.float32,
                     device=partial_lse.device)
    dist.all_gather_into_tensor(go, partial_output.contiguous())
    dist.all_gather_into_tensor(gl, partial_lse.contiguous())
    my = slice(RANK * HPR, (RANK + 1) * HPR)
    lse = gl[:, :, my].clone()                       # [W,T,HPR]
    lse[torch.isnan(lse) | (lse == float("inf"))] = float("-inf")
    if is_lse_base_on_e:
        lse = lse * LOG2E
    lse_max = lse.amax(dim=0, keepdim=True)
    lse_max = torch.where(torch.isinf(lse_max), torch.zeros_like(lse_max),
                          lse_max)
    w = torch.exp2(lse - lse_max)                    # [W,T,HPR]
    denom = w.sum(dim=0, keepdim=True)
    w = torch.where(denom > 0, w / denom, torch.zeros_like(w))
    out = go[:, :, my, :].float() * w.unsqueeze(-1)  # [W,T,HPR,HDIM]
    return out.sum(dim=0)                             # [T,HPR,HDIM]


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def main():
    torch.cuda.set_device(LOCAL_RANK)
    dev = torch.device("cuda", LOCAL_RANK)
    dist.init_process_group(backend="nccl")
    torch.ops.load_library(SO)
    assert hasattr(torch.ops._C, "direct_dcp_a2a_lse_reduce"), "op not registered"
    if RANK == 0:
        log(f"loaded {SO}; op registered OK")

    # ops/dcp_utils.py was renamed to ops/dcp.py in the dev1046 nightly;
    # accept either so this test runs against both images.
    try:
        from vllm.v1.attention.ops.dcp import DirectDCPA2AWorkspace
    except ImportError:
        from vllm.v1.attention.ops.dcp_utils import DirectDCPA2AWorkspace
    pg = dist.distributed_c10d._get_default_group()
    ws = DirectDCPA2AWorkspace(pg, dev, MAX_NT, HPR, HDIM,
                               dtype=torch.bfloat16, num_ubatches=1)
    dist.barrier()

    total_heads = WORLD * HPR
    fails = 0
    for is_e in (True, False):
        for trial in range(3):
            torch.manual_seed(1000 * trial + (1 if is_e else 0))  # same on all ranks
            # distinct per-rank partials (rank-dependent seed offset)
            g = torch.Generator(device=dev).manual_seed(
                7919 * RANK + 13 * trial + (1 if is_e else 0))
            partial_output = torch.randn((T, total_heads, HDIM), generator=g,
                                         dtype=torch.float32, device=dev
                                         ).to(torch.bfloat16).contiguous()
            partial_lse = (torch.randn((T, total_heads), generator=g,
                                       dtype=torch.float32, device=dev) * 3.0
                           ).contiguous()
            ref = reference(partial_output, partial_lse, is_e)
            dist.barrier()
            out = ws.lse_reduce(partial_output, partial_lse, is_e)
            torch.cuda.synchronize()
            c = cos(out, ref)
            md = (out.float() - ref).abs().max().item()
            ok = c > 0.999 and md < 5e-2
            fails += 0 if ok else 1
            if RANK == 0 or not ok:
                log(f"is_lse_base_on_e={is_e} trial={trial} "
                    f"cos={c:.6f} max|d|={md:.3e} finite={torch.isfinite(out).all().item()} "
                    f"{'OK' if ok else 'FAIL'}")

    ft = torch.tensor([fails], dtype=torch.int64, device=dev)
    dist.all_reduce(ft, op=dist.ReduceOp.SUM)
    if RANK == 0:
        tot = int(ft.item())
        log(f"VERDICT: {'PASS' if tot == 0 else 'FAIL'} "
            f"({tot} failing rank-trials across {WORLD} ranks)")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
