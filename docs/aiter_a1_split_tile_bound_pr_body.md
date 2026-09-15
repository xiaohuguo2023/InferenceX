# PR body — aiter: tighter split-tile bound when a cap is supplied

Branch: `xguo/mla-tighter-split-tile-bound` (1 commit, +349/−2, 3 files)

**Title:**

```
[MLA] Take the tighter split-tile bound when a cap is supplied
```

---

## Purpose

**`max_split_per_batch` currently has no effect on the metadata allocation.**

`get_mla_metadata_info_v1` sizes the reduce scratch from a `fast_mode` estimate that assumes an unbounded per-batch split budget. When a caller supplies `max_split_per_batch`, `tile_cnt + per_tile_cap` is the cap-aware bound — but the two are combined with `max()`:

```python
if max_split_per_batch > 0:
    per_tile_cap = min(max_splits, max_split_per_batch * batch_size)
    max_split_tiles = max(max_split_tiles, tile_cnt + per_tile_cap)   # loose estimate always wins
```

so the cap is inert. Taking the `min` is what makes it mean something. The change is one operator.

### This finishes #3855

#3855 (merged 2026-06-22) corrected **the same expression for the same reason** — the split budget is *global*, not per-tile — changing `tile_cnt * per_tile_cap` to `tile_cnt + per_tile_cap`. That fixed a worst case where `batch_size >> cu_num` collapsed to `tile_cnt * cu_num` (512 × 256 = 131072) and `mla_decode_fwd` sized its fp32 `logits` from `reduce_partial_map.size(0)`, giving ~32 GiB and an OOM at cudagraph capture.

It left the outer `max()`, so a supplied cap is still ignored. This is the remaining half.

### Effect on the allocation

`reduce_partial_map` entries, gfx950, `nhead=128`, `qo_len=4`:

| batch | uncapped | cap=1 | cap=256 |
|---:|---:|---:|---:|
| 1 | 1024 | **5** | **260** |
| 8 | 1052 | 40 | 288 |
| 64 | 1276 | 320 | 512 |
| 512 | 2040 | 2040 | 2040 |

The `batch=512` row is unchanged by design: `per_tile_cap` is `min(max_splits, cap * batch_size)`, so once `cap * batch_size` exceeds `max_splits` no cap can constrain the schedule, and the sizing must not depend on whether a dummy cap was passed. The cap helps at low batch, which is where the allocation was most over-sized relative to what the kernel can reach.

Downstream, this takes vLLM's DCP MLA reduce scratch from **9.35 GiB to 2.38 GiB**, which is what lets FULL cudagraphs fit under decode context parallelism.

### Why the smaller bound is safe

The tighter bound is only valid because **the caller passes the same `max_split_per_batch` to `get_mla_metadata_v1` at build time**, so the schedule is built under the cap the sizing assumed. Consistency between the two calls is the requirement, not any particular value — and the fill test below drives both with one cap for exactly that reason.

**Direction of risk.** Passing a cap to the sizing call but not to the build call would under-size. The reverse is harmless: the sizing then stays on the loose estimate. Callers that pass no cap are unaffected — the branch is gated on `max_split_per_batch > 0`, which defaults to `-1`.

## Test Plan

```bash
pytest op_tests/test_mla_metadata_split_cap.py       # sizing arithmetic, no GPU work
pytest op_tests/test_mla_metadata_split_cap_fill.py  # runs the planner, checks it fits
```

Two files, because one of them cannot do the other's job.

**`test_mla_metadata_split_cap.py`** pins the arithmetic: a cap never enlarges the sizing, a tight cap shrinks it, a non-constraining cap changes nothing, relaxing the cap is monotonic, and `max_split_per_batch <= 0` is byte-identical to today. Sizes are derived rather than hardcoded — they track the CU count, so a literal from one part fails on another.

**`test_mla_metadata_split_cap_fill.py`** is the one that matters. A sizing formula can only be checked against itself; it cannot say whether the bound is one the planner reaches, or whether a cap sizes *too* tightly. So this allocates exactly what the sizing returns, runs `get_mla_metadata_v1` with the same cap, and asserts `reduce_indptr[-1] <= reduce_partial_map.numel()` and `work_indptr[-1] <= work_info_set.size(0)`, that `reduce_indptr` starts at 0 and is non-decreasing, and — since HIP does not reliably trap an out-of-bounds write — that an exactly-sized allocation produces the same populated prefix as a 4×-sized one.

Two shapes are excluded as **vacuous**, and this is asserted rather than assumed: `cap=1` writes zero partials at any batch (the cap forbids the extra splits that produce them), and large batch with uniform KV never splits (batch alone supplies the parallelism). `0 <= bound` passes for any bound, so every row is checked to be non-empty. `cap=1` is still covered by a separate row that asserts it does not *overflow* — vacuous for tightness, not for safety, and it is the smallest allocation this change produces.

## Test Result

```
op_tests/test_mla_metadata_split_cap.py        9 passed
op_tests/test_mla_metadata_split_cap_fill.py  19 passed
```

Measured fills against the bound (printed by the fill test, so the numbers are in CI rather than in a comment):

| batch | cap | partials | work |
|---:|---:|---:|---:|
| 1 | 256 | **256 / 260** | 256 / 1024 |
| 1 | −1 | 256 / 1024 | 256 / 1024 |
| 8 | 256 | 256 / 288 | 256 / 1052 |
| 512 | 256 | 506 / 2040 | 2301 / 3068 |
| 512 | −1 | 506 / 2040 | 2301 / 3068 |

**`batch=1 / cap=256` filling 256 into 260** is the evidence that shrinking the bound is safe on the path this change affects — four entries of headroom, so the capped bound is nearly exact rather than merely smaller.

The **batch 512** rows are why the fast-mode estimate is kept there. An earlier revision raised it to `tile_cnt + max_splits` (2040 → 2304) on the argument that fast-mode does not grow with `tile_cnt` and could undersize. The planner does not come close to either bound — at most ~506 partials across jittered KV and skewed one-long/many-short shapes — so that would have cost ~13% for no measured benefit.

**The tests have power, they are not merely green.** Three mutations, each caught:

```
max() alone (current behaviour):          4 failed
this bound minus 8:                       4 failed   <- caught only by the fill test
the sum bound applied inside the branch:  1 failed   <- a dummy cap would change the size at 512
correct:                                 28 passed
```

The second is the case an arithmetic-only suite cannot see: eight entries is inside the four-entry headroom at `batch=1`, so a consistent-but-too-small formula would pass every sizing assertion and overflow on device.

`op_tests/test_metadata.py` fails identically before and after (a pytest signature mismatch — the file is an argparse CLI script), so it is pre-existing and unrelated.

### Not addressed here

- `max_work` / `work_info_set` are still sized from the uncapped estimate. They are int32 and not the fp32 `logits` path, but the fill numbers show `work_info_set` is the tighter of the two buffers (2301/3068 at batch 512), so it is the better follow-up target.
- `intra_batch_mode` sizes `reduce_partial_map` as `tile_cnt * num_kv_splits` and ignores the cap entirely.
- Coverage for `fast_mode=False` and `is_sparse=True`.
