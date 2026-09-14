## Purpose

**Fixes a boot blocker for decode context parallelism (DCP) combined with speculative decoding: the DFlash/DSpark speculator faults during CUDA graph capture.**

When the DFlash/DSpark draft is **DCP-sharded**, the graph captured by `DFlashSpeculator.capture()` contains a context-parallel collective. Graph capture is only safe when every rank enters it aligned — a rank still finishing earlier work can have its collective recorded into a peer's graph, which faults. Nothing currently quiesces or aligns the ranks before `capture()` runs.

Symptom: every rank dies in `Capturing model for DSpark speculator...` with
`Memory access fault by GPU node-N ... Reason: Unknown`, at a **fresh fault address each time**, with a clean teardown.

**It is a race, not an out-of-bounds access.** The decisive evidence is that booting the same configuration with `AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1` **succeeds**: a configuration that only fails when kernels overlap is a race by construction. The logs show ranks entering `capture()` staggered by ~1 s.

### Why this has not been reported before

The precondition is a **sharded** draft. The replicated draft of #51705 ran at `dcp=1`, so its captured graph contained no collective and there was nothing to race. That changed with **#55472** (merged 2026-09-08), which rebuilds the draft's `ParallelConfig` from the target's and therefore preserves `decode_context_parallel_size` — so a `--decode-context-parallel-size N` run with a DSpark draft now gets a draft at `dcp=N`. The window in which this is reachable upstream is about a week old.

**Reproduction:** multi-rank, `decode_context_parallel_size > 1`, DSpark draft, CUDA graphs enabled. It is a startup race, so it is timing-dependent rather than deterministic; serializing kernel launches masks it.

This adds that alignment, and only where it is needed:

```python
if self.vllm_config.parallel_config.decode_context_parallel_size > 1:
    torch.accelerator.synchronize()
    get_dcp_group().barrier()
```

Three things worth calling out:

- **Gated on DCP.** With `decode_context_parallel_size == 1` the draft is not sharded, the captured graph holds no collective, and a barrier would additionally require an initialised process group on the single-GPU path. The gate reads the *draft's* config (`self.vllm_config` is the draft-adjusted one), so it tracks whatever the draft actually runs.
- **`GroupCoordinator.barrier()`, not `torch.distributed.barrier()`.** This is a correctness point, not style. The latter is an NCCL barrier, which — per its own docstring in `parallel_state.py` — "is internally a broadcast operation with secretly created GPU tensors. It is easy to mess up the current device." Doing that immediately before graph capture is exactly the wrong thing; `GroupCoordinator.barrier()` uses the CPU group instead. A test pins the choice so a later simplification cannot quietly undo it.
- **Boot-time only**, once per speculator. No steady-state cost.

### Relationship to #56723

#56723 touches the same file — it adds `draft_parallel_config()` and rewires `__init__`. The two are **independent and compose**: it does not touch `capture()`, and it refines the draft's effective DCP, which is exactly the value this gate reads. With PCP on, #56723 collapses the draft's DCP to 1 and this barrier then correctly skips, because the draft is no longer sharded. With PCP off (`pcp <= 1`) it returns the config unchanged and the barrier still applies.

There is no logic overlap; whichever lands second needs a one-line rebase of the import block.

## Test Plan

```bash
pytest tests/v1/spec_decode/test_dflash_capture_dcp_barrier.py

# no regression in the neighbouring speculator tests
pytest tests/v1/spec_decode/test_dflash_prepare_inputs.py \
       tests/v1/spec_decode/test_dflash_causality.py
```

**These are unit tests, not a regression test, and that is a deliberate limitation.** The bug is a multi-rank boot-time race; reproducing it needs several ranks and a real capture, which no unit test can do. They instead pin the three properties that prevent it:

1. under DCP, the device is quiesced and the ranks aligned, and both happen **before** capture begins (asserted on call order, not just call count);
2. with DCP off, neither happens — so the single-GPU path pays nothing and needs no process group;
3. the barrier goes through the DCP `GroupCoordinator` and **not** `torch.distributed.barrier()`.

The speculator is constructed with `object.__new__` and only the attributes `capture()` dereferences, so no draft model is loaded and the tests run anywhere.

## Test Result

```
tests/v1/spec_decode/test_dflash_capture_dcp_barrier.py  3 passed
tests/v1/spec_decode/test_dflash_prepare_inputs.py
tests/v1/spec_decode/test_dflash_causality.py           16 passed
```

**The tests have power, they are not merely green.** Removing the two barrier lines from the installed copy turns the ordering test red:

```
with the barrier removed:  1 failed, 2 passed
with the barrier present:  3 passed
```

`pre-commit run --files <changed files>` passes in full, including `mypy` 3.10–3.13, `typos`, `check-spdx-header` and `check-torch-cuda-call` — the last of which is why the synchronize goes through `torch.accelerator` rather than `torch.cuda` (RFC #30679).

Measured on 8x MI355X (gfx950, ROCm 7.2.3) with Kimi-K3 + DSpark at TP8/DCP8, both attention groups sharded (`target num_heads=12 dcp=8 decode_num_heads=96`, `draft num_heads=8 dcp=8 decode_num_heads=64`): every boot died at speculator capture; with the barrier the server reaches `Application startup complete` in 280 s with **no serialization penalty**, and a full concurrency sweep then ran 9/9 `rc=0`.

To be precise about what was measured: that stack carries a local port that makes the DSpark draft DCP-sharded, which is what put us on this path first. The upstream reachability argument is the #55472 one above — a stock DCP + DSpark run now produces a sharded draft by the same mechanism — rather than a claim that we have reproduced it on an unmodified tree.
