## Purpose

**Fixes a boot blocker for decode context parallelism (DCP) combined with speculative decoding: the DFlash/DSpark speculator faults during CUDA graph capture.**

Under DCP the draft is sharded, so the graph captured by `DFlashSpeculator.capture()` contains a context-parallel collective. Graph capture is only safe when every rank enters it aligned — a rank still finishing earlier work can have its collective recorded into a peer's graph, which faults at capture time. Nothing currently quiesces or aligns the ranks before `capture()` runs.

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

Observed downstream on 8x MI355X (gfx950, ROCm 7.2.3) with Kimi-K3 + DSpark at TP8/DCP8: without this change the speculator faults during capture at boot; with it, capture completes and the server reaches `Application startup complete`.
