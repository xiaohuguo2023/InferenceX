# PR body — aiter: tighter split-tile bound when a cap is supplied

Branch: `xguo/mla-tighter-split-tile-bound` (1 commit, +110/−1, 2 files)

**Title:**

```
[MLA] Take the tighter split-tile bound when a cap is supplied
```

---

## Purpose

**`max_split_per_batch` currently has no effect on the metadata allocation.**

`get_mla_metadata_info_v1` sizes the reduce scratch from a `fast_mode` estimate that assumes an unbounded per-batch split budget. When a caller supplies `max_split_per_batch`, `tile_cnt + per_tile_cap` is a **strictly tighter** bound — but the two are combined with `max()`:

```python
if max_split_per_batch > 0:
    per_tile_cap = min(max_splits, max_split_per_batch * batch_size)
    max_split_tiles = max(max_split_tiles, tile_cnt + per_tile_cap)   # loose estimate always wins
```

so the loose estimate always wins and the cap is inert. Taking the `min` is what makes a supplied cap mean something.

### This finishes #3855

#3855 (merged 2026-06-22) corrected **the same expression for the same reason** — the split budget is *global*, not per-tile — changing `tile_cnt * per_tile_cap` to `tile_cnt + per_tile_cap`. That fixed a worst case where `batch_size >> cu_num` collapsed to `tile_cnt * cu_num` (e.g. 512 × 256 = 131072) and `mla_decode_fwd` sized its fp32 `logits` from `reduce_partial_map.size(0)`, giving ~32 GiB and an OOM at cudagraph capture.

It left the outer `max()` in place, so the cap is still ignored. This is the remaining half of that fix.

### Measured effect

`reduce_partial_map` entries on gfx950, `nhead=128`, `qo_len=4`:

| batch | uncapped | cap=1 | cap=256 |
|---:|---:|---:|---:|
| 1 | 1024 | **5** | **260** |
| 8 | 1052 | 40 | 288 |
| 64 | 1276 | 320 | 512 |
| 512 | 2040 | 2040 | 2040 |

At batch 1 with a CU-count cap of 256 that is **1024 → 260 entries (3.94x)**.

Downstream in vLLM's DCP MLA path the same change takes the reduce scratch from **9.35 GiB to 2.38 GiB (3.93x)**, which is what lets FULL cudagraphs fit under decode context parallelism. The two measurements are independent and agree to 0.3%.

The `batch=512` row is included deliberately: `per_tile_cap` is `min(max_splits, cap * batch_size)`, so once `cap * batch_size` exceeds `max_splits` the bound stops depending on the cap and the sizing is unchanged. The cap helps at low batch, which is where the allocation was most over-sized relative to what the kernel can reach.

### Why it is safe

The tighter bound is only valid because **the caller passes the same `max_split_per_batch` to `get_mla_metadata_v1` at build time**, so the schedule is built under the cap the sizing assumed. Consistency between the two calls is the actual requirement, not any particular value.

Validated over **1030 `(batch, qlen, ragged-kv)` shapes** on gfx950, both `cprr` and non-`cprr` paths: **0 violations, worst actual/bound 0.998**.

That 0.998 is deliberately reported rather than rounded — the bound is tight, and a reviewer should see how tight. It is an upper bound on what the kernel can schedule under the cap, not an empirical high-water mark, so tightness is expected; the sweep is there to confirm the derivation rather than to discover the constant.

**Direction of risk, if the invariant were ever broken:** passing a cap to the sizing call but not to the build call would under-size. The reverse — passing it to the build but not the sizing — is harmless, since the sizing then stays on the loose estimate. Callers that do not pass a cap at all are unaffected: the whole branch is gated on `max_split_per_batch > 0`, which defaults to `-1`.

## Test Plan

```bash
pytest op_tests/test_mla_metadata_split_cap.py
```

Pure sizing queries — no kernel is launched and no GPU work is done. Ten tests pin:

1. a cap never *enlarges* the allocation, across `batch ∈ {1, 8, 64, 512}`;
2. a tight cap actually *shrinks* it, across `batch ∈ {1, 8, 64}`;
3. the bound **saturates** at large batch (512) and the cap correctly stops mattering — documenting the limit rather than pretending it does not exist;
4. monotonicity: relaxing the cap never shrinks the bound;
5. `max_split_per_batch <= 0` is unchanged, so no-cap callers see exactly the historical value.

## Test Result

```
op_tests/test_mla_metadata_split_cap.py   10 passed
```

**The tests have power, they are not merely green.** Restoring `max()` turns half of them red:

```
with max() (current behaviour):  5 failed, 5 passed
with min() (this change):       10 passed
```

`op_tests/test_metadata.py` fails identically before and after this change (a pytest signature mismatch — the file is an argparse CLI script), so it is pre-existing and unrelated.
