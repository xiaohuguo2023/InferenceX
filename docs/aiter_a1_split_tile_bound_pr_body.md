# PR body — aiter: make `max_split_per_batch` shrink the MLA decode scratch

Branch: `xguo/mla-cap-aware-scratch-sizing` (1 commit, +349/−2, 3 files)

**Title:**

```
[Bugfix][MLA] Make max_split_per_batch shrink the decode scratch buffer
```

---

## Motivation

**`max_split_per_batch` has no effect on the buffer size it exists to reduce.**

On one rank, MLA decode can cut that rank's local KV into chunks and run them on different CUs (the metadata kernel's `num_clusters` / `num_cu`). `max_split_per_batch` caps how many of those extra KV fragments one batch is allowed. The fp32 `reduce_partial_map` / `logits` buffer is the scratch to merge those CU partials back.

DCP is separate: each GPU already holds only 1/`dcp_world_size` of the sequence. Ranks do not share this split-K buffer. DCP only makes the per-GPU allocation worse because that GPU's decode sees gathered query heads (`nheads × dcp_world_size`) and still split-Ks its local KV across its own CUs.

An inference framework — in our case vLLM's ROCm MLA attention backend — cannot allocate that scratch lazily, because the buffers have to exist before a cudagraph is captured. So it uses aiter's ask-allocate-fill interface:

1. `get_mla_metadata_info_v1(...)` returns the buffer sizes needed for a given shape;
2. the framework allocates exactly those buffers;
3. `get_mla_metadata_v1(...)` builds the schedule into them.

`max_split_per_batch` is an argument to both, and it means *"no request may be given more than N splits"*. Fewer splits means fewer partials, so passing it should return a smaller size from step 1.

It does not. **Step 1 returns the same size whether you pass `max_split_per_batch=1`, `=256`, or `=-1` (no limit).** The value is computed, then discarded by a `max()` that always prefers the no-limit estimate.

That is the whole bug. It matters because step 1's answer is what gets reserved.

### What this buys

We hit this on Kimi-K3 MLA decode (TP8, DCP8, the fp8 asm path).

`reduce_partial_map` drops from 4785 tiles to 1216, so the fp32 `logits` scratch drops from **9.35 GiB to 2.38 GiB** — **6.97 GiB per GPU**.

It was a crash, not just waste. At batch 48 every rank died with `torch.OutOfMemoryError: Tried to allocate 9.35 GiB ... 6.24 GiB is free`. The 6.97 GiB we get back is also more than the FULL-decode cudagraph pools need, so the shape went from not fitting to fitting with room left over.

It is not a speedup. No kernel changes and no maths changes — only the buffer size.

## Technical Details

`get_mla_metadata_info_v1` works out two estimates of how many partials can appear — one that ignores the cap, one that respects it — and keeps the **larger** of the two:

```python
if max_split_per_batch > 0:
    per_tile_cap = min(max_splits, max_split_per_batch * batch_size)
    max_split_tiles = max(max_split_tiles, tile_cnt + per_tile_cap)
    #                 ^^^ the cap-aware estimate is the smaller one,
    #                     so max() throws it away every time
```

The cap-aware estimate is by construction the smaller one, so `max()` discards it. **Changing `max` to `min` is the entire fix.**

### This finishes #3855

https://github.com/ROCm/aiter/pull/3855 (merged 2026-06-22) corrected **the same expression for the same reason** — the split budget is *global*, not per-tile — changing `tile_cnt * per_tile_cap` to `tile_cnt + per_tile_cap`. That fixed a worst case where `batch_size >> cu_num` collapsed to `tile_cnt * cu_num` (512 × 256 = 131072) and `mla_decode_fwd` sized its fp32 `logits` from `reduce_partial_map.size(0)`, giving ~32 GiB and an OOM at cudagraph capture.

It left the outer `max()`, so a supplied cap is still ignored. This is the remaining half.

### Effect on the allocation

`reduce_partial_map` entries, gfx950, `nhead=128`, `qo_len=4`:

| batch | uncapped | cap=1 | cap=256 |
|---:|---:|---:|---:|
| 1 | 1024 | **5** | **260** |
| 8 | 1052 | 40 | 288 |
| 64 | 1276 | 320 | 512 |
| 512 | 2040 | 2040 | 2040 |

The `batch=512` row is unchanged by design. `per_tile_cap` is `min(max_splits, cap * batch_size)`, so once `cap * batch_size` exceeds `max_splits` no cap can constrain the schedule — and the sizing must not depend on whether a dummy cap was passed. The cap helps at low batch, which is where the allocation was most over-sized relative to what the kernel can reach.

### Why the smaller allocation is safe

Steps 1 and 3 must be given the **same** cap, so the schedule is built under the limit the sizing assumed. Consistency between the two is the requirement, not any particular value — and the fill test drives both from one variable for exactly that reason.

**Direction of risk.** Capping step 1 but not step 3 would under-size. The reverse is harmless: the sizing simply stays on the loose estimate. Code that passes no cap is unaffected — the branch is gated on `max_split_per_batch > 0`, which defaults to `-1`.

### Not addressed here

- `max_work` / `work_info_set` are still sized from the uncapped estimate. They are int32 rather than the fp32 `logits` path, but the fill numbers below show `work_info_set` is the tighter of the two buffers (2301/3068 at batch 512), so it is the better follow-up target.
- `intra_batch_mode` sizes `reduce_partial_map` as `tile_cnt * num_kv_splits` and ignores the cap entirely.
- Coverage for `fast_mode=False` and `is_sparse=True`.

## Test Plan

```bash
pytest op_tests/test_mla_metadata_split_cap.py       # sizing arithmetic, no GPU work
pytest op_tests/test_mla_metadata_split_cap_fill.py  # runs the planner, checks it fits
```

Two files, because one cannot do the other's job.

**`test_mla_metadata_split_cap.py`** pins the arithmetic: a cap never enlarges the sizing, a tight cap shrinks it, a non-constraining cap changes nothing, relaxing the cap is monotonic, and `max_split_per_batch <= 0` is byte-identical to today. Sizes are derived rather than hardcoded — they track the CU count, so a literal from one part fails on another.

**`test_mla_metadata_split_cap_fill.py`** is the one that matters, because a sizing formula can otherwise only be checked against itself. It allocates exactly what the sizing returns, runs the planner with the same cap, and asserts:

- `reduce_indptr[-1] <= reduce_partial_map.numel()` — the partials fit;
- `work_indptr[-1] <= work_info_set.size(0)` — the work entries fit;
- `reduce_indptr` starts at 0 and is non-decreasing;
- an exactly-sized allocation produces the same populated prefix as a 4×-sized one. HIP does not reliably trap an out-of-bounds write, so an extent check alone could miss an overrun.

Two shapes are **vacuous** and are excluded, with that asserted rather than assumed: `cap=1` writes zero partials at any batch (the cap forbids the extra splits that produce them), and large batch with uniform KV never splits (batch alone supplies the parallelism). `0 <= bound` passes for any bound, so every row is checked to be non-empty. `cap=1` is still covered by a separate case asserting it does not *overflow* — vacuous for tightness, not for safety, and it is the smallest allocation this change produces.

## Test Result

```
op_tests/test_mla_metadata_split_cap.py        9 passed
op_tests/test_mla_metadata_split_cap_fill.py  19 passed
```

Measured fills against the bound, printed by the fill test so the numbers land in CI rather than in a comment:

| batch | cap | partials | work |
|---:|---:|---:|---:|
| 1 | 256 | **256 / 260** | 256 / 1024 |
| 1 | −1 | 256 / 1024 | 256 / 1024 |
| 8 | 256 | 256 / 288 | 256 / 1052 |
| 512 | 256 | 506 / 2040 | 2301 / 3068 |
| 512 | −1 | 506 / 2040 | 2301 / 3068 |

**`batch=1 / cap=256` filling 256 into 260** is the evidence that shrinking the bound is safe on the path this change affects — four entries of headroom, so the capped bound is nearly exact rather than merely smaller.

The **batch 512** rows are why the fast-mode estimate is kept there. An earlier revision raised it to `tile_cnt + max_splits` (2040 → 2304), on the argument that fast-mode does not grow with `tile_cnt` and could undersize. The planner does not come close to either bound — at most ~506 partials across jittered KV and skewed one-long/many-short shapes — so that would have cost ~13% for no measured benefit.

**The tests have power, they are not merely green.** Three mutations, each caught:

```
max() alone (current behaviour):          4 failed
this bound minus 8:                       4 failed   <- caught only by the fill test
the sum bound applied inside the branch:  1 failed   <- a dummy cap would change the size at 512
correct:                                 28 passed
```

The second is the case an arithmetic-only suite cannot see: eight entries is inside the four-entry headroom at `batch=1`, so a consistent-but-too-small formula would pass every sizing assertion and overflow on device.

`op_tests/test_metadata.py` fails identically before and after (a pytest signature mismatch — the file is an argparse CLI script), so it is pre-existing and unrelated.

## Submission Checklist

- [ ] Look over the contributing guidelines at https://github.com/ROCm/TheRock/blob/main/GOVERNANCE.md#pull-requests.
