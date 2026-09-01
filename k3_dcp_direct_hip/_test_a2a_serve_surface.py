#!/usr/bin/env python3
"""Exercise the direct DCP A2A operator's serve-only surface without a model.

Run inside the container:

  K3_DCP_BOUNDS=1 torchrun --standalone --nproc-per-node=8 \
      _test_a2a_serve_surface.py

This covers the differences omitted by the small eager tests:
  * the production 384-token symmetric workspace;
  * valid and deliberately stale seq_lens/query_start_loc metadata;
  * the FULL-graph capture/replay path;
  * many back-to-back replays without per-call synchronization.

Use an external timeout. A failure must terminate the test, never trap the GPU:

  timeout --signal=TERM --kill-after=10s 300s torchrun ...
"""

import os
import sys
import time

import torch
import torch.distributed as dist


RANK = int(os.environ["RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD = int(os.environ["WORLD_SIZE"])

SO = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "dcp_direct_a2a_lse_reduce.so",
)
HPR = int(os.environ.get("HPR", "12"))
HDIM = int(os.environ.get("HDIM", "512"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "384"))
SIZES = tuple(
    int(value)
    for value in os.environ.get(
        "SIZES", "1,2,3,5,8,12,16,24,32,48,64,96,128,192,256,384"
    ).split(",")
)
GRAPH_ITERS = int(os.environ.get("GRAPH_ITERS", "100"))
LOG2E = 1.4426950408889634


def log(message: str) -> None:
    print(f"[rank{RANK}] {message}", flush=True)


def make_inputs(
    device: torch.device, num_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(
        7919 * RANK + 101 * num_tokens
    )
    total_heads = WORLD * HPR
    partial_output = torch.randn(
        (num_tokens, total_heads, HDIM),
        generator=generator,
        dtype=torch.float32,
        device=device,
    ).to(torch.bfloat16)
    partial_lse = (
        torch.randn(
            (num_tokens, total_heads),
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        * 3.0
    )
    return partial_output.contiguous(), partial_lse.contiguous()


def make_metadata(
    device: torch.device, num_tokens: int, stale_start: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    num_seqs = min(4, num_tokens)
    lengths = [num_tokens // num_seqs] * num_seqs
    for index in range(num_tokens % num_seqs):
        lengths[index] += 1
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    if stale_start:
        # Reproduce the warmup hazard: token 0 sorts before qsl[0], making the
        # old kernel's find_sequence() return -1 and read seq_lens[-1].
        boundaries = [value + 1 for value in boundaries]
    seq_lens = torch.full(
        (num_seqs,), 128, dtype=torch.int32, device=device
    )
    query_start_loc = torch.tensor(
        boundaries, dtype=torch.int32, device=device
    )
    return seq_lens, query_start_loc


def reference(
    partial_output: torch.Tensor, partial_lse: torch.Tensor
) -> torch.Tensor:
    num_tokens = partial_output.shape[0]
    total_heads = WORLD * HPR
    gathered_output = torch.empty(
        (WORLD, num_tokens, total_heads, HDIM),
        dtype=partial_output.dtype,
        device=partial_output.device,
    )
    gathered_lse = torch.empty(
        (WORLD, num_tokens, total_heads),
        dtype=torch.float32,
        device=partial_lse.device,
    )
    dist.all_gather_into_tensor(gathered_output, partial_output)
    dist.all_gather_into_tensor(gathered_lse, partial_lse)
    my_heads = slice(RANK * HPR, (RANK + 1) * HPR)
    lse = gathered_lse[:, :, my_heads] * LOG2E
    lse_max = lse.amax(dim=0, keepdim=True)
    weights = torch.exp2(lse - lse_max)
    weights /= weights.sum(dim=0, keepdim=True)
    return (
        gathered_output[:, :, my_heads, :].float()
        * weights.unsqueeze(-1)
    ).sum(dim=0)


def check(name: str, actual: torch.Tensor, expected: torch.Tensor) -> bool:
    actual_f = actual.float()
    expected_f = expected.float()
    cosine = torch.nn.functional.cosine_similarity(
        actual_f.flatten(), expected_f.flatten(), dim=0
    ).item()
    max_diff = (actual_f - expected_f).abs().max().item()
    ok = cosine > 0.999 and max_diff < 5e-2
    if RANK == 0 or not ok:
        log(
            f"{name}: cos={cosine:.6f} max|d|={max_diff:.3e} "
            f"{'OK' if ok else 'FAIL'}"
        )
    return ok


def main() -> int:
    if max(SIZES) > MAX_TOKENS:
        raise ValueError("SIZES cannot exceed MAX_TOKENS")

    torch.cuda.set_device(LOCAL_RANK)
    device = torch.device("cuda", LOCAL_RANK)
    dist.init_process_group("nccl", device_id=device)
    torch.ops.load_library(SO)

    # ops/dcp_utils.py was renamed to ops/dcp.py in the dev1046 nightly;
    # accept either so this test runs against both images.
    try:
        from vllm.v1.attention.ops.dcp import DirectDCPA2AWorkspace
    except ImportError:
        from vllm.v1.attention.ops.dcp_utils import DirectDCPA2AWorkspace

    process_group = dist.distributed_c10d._get_default_group()
    workspace = DirectDCPA2AWorkspace(
        process_group,
        device,
        MAX_TOKENS,
        HPR,
        HDIM,
        dtype=torch.bfloat16,
        num_ubatches=1,
    )
    dist.barrier()

    failures = 0
    if RANK == 0:
        log(f"metadata matrix: sizes={SIZES}, capacity={MAX_TOKENS}")
    for num_tokens in SIZES:
        partial_output, partial_lse = make_inputs(device, num_tokens)
        expected = reference(partial_output, partial_lse)
        for stale_start in (False, True):
            seq_lens, query_start_loc = make_metadata(
                device, num_tokens, stale_start
            )
            dist.barrier()
            actual = workspace.lse_reduce(
                partial_output,
                partial_lse,
                True,
                seq_lens,
                query_start_loc,
            )
            torch.cuda.synchronize()
            failures += not check(
                f"T={num_tokens} metadata={'stale' if stale_start else 'valid'}",
                actual,
                expected,
            )

    # Capture the production envelope and replay it back-to-back. All ranks
    # capture and replay the same collective count, but no replay is host-synced.
    num_tokens = MAX_TOKENS
    partial_output, partial_lse = make_inputs(device, num_tokens)
    expected = reference(partial_output, partial_lse)
    seq_lens, query_start_loc = make_metadata(device, num_tokens, False)
    for _ in range(3):
        graph_output = workspace.lse_reduce(
            partial_output, partial_lse, True, seq_lens, query_start_loc
        )
    torch.cuda.synchronize()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = workspace.lse_reduce(
            partial_output, partial_lse, True, seq_lens, query_start_loc
        )
    dist.barrier()
    started = time.monotonic()
    for _ in range(GRAPH_ITERS):
        graph.replay()
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    failures += not check(
        f"graph T={num_tokens} replays={GRAPH_ITERS}",
        graph_output,
        expected,
    )
    if RANK == 0:
        log(f"graph replay elapsed={elapsed:.3f}s")

    failure_tensor = torch.tensor(
        [failures], dtype=torch.int64, device=device
    )
    dist.all_reduce(failure_tensor)
    total_failures = int(failure_tensor.item())
    if RANK == 0:
        log(
            f"VERDICT: {'PASS' if total_failures == 0 else 'FAIL'} "
            f"({total_failures} rank-cases failed)"
        )
    dist.destroy_process_group()
    return 0 if total_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
