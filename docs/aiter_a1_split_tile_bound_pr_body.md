# PR body: aiter #5559

**FILED: [ROCm/aiter#5559](https://github.com/ROCm/aiter/pull/5559)**, branch
`xguo/mla-tighter-split-tile-bound`. Title:

```
[Bugfix][MLA] Fix reduce_partial_map over-allocation when max_split_per_batch is set
```

Suggested reviewers by blame: **@ruanjm** (wrote both changed lines, #3391 and
#3459) and **honglie** (wrote #3855).

---

## Motivation

On one rank, MLA decode can cut that rank's local KV into chunks and run them on
different CUs (the metadata kernel's `num_clusters` / `num_cu`).
`max_split_per_batch` caps how many of those extra KV fragments one batch is
allowed. The fp32 `reduce_partial_map` / `logits` buffer is the scratch to merge
those CU partials back.

DCP is separate: each GPU already holds only 1/`dcp_world_size` of the sequence.
Ranks do not share this split-K buffer. DCP only makes the per-GPU allocation
worse because that GPU's decode sees gathered query heads
(`nheads x dcp_world_size`) and still split-Ks its local KV across its own CUs.

An inference framework (in our case vLLM's ROCm MLA attention backend) cannot
allocate that scratch lazily, because the buffers have to exist before a
cudagraph is captured. So it uses aiter's ask-allocate-fill interface: ask
`get_mla_metadata_info_v1` for the sizes, allocate exactly those, then let
`get_mla_metadata_v1` build the schedule into them. Whatever step one returns is
what gets reserved.

So passing `max_split_per_batch` should return a smaller size. It does not:
sizing returns the same buffer whether you pass `1`, `256`, or `-1`.
`get_mla_metadata_info_v1` computes the cap-aware bound, then discards it with a
`max()` against the uncapped estimate. This is the other half of #3855, which
fixed the same expression but left the `max()`.

For Kimi-K3 MLA decode (TP8, DCP8, fp8 asm) that is **9.35 GiB to 2.38 GiB** of
fp32 scratch, 6.97 GiB per GPU. Before the fix a batch-48 decode died on every
rank with `torch.OutOfMemoryError: Tried to allocate 9.35 GiB`.

## Technical Details

`max()` becomes `min()`. Three constraints on where it applies:

- **fast_mode only.** With `fast_mode=False` and `intra_batch_mode=False` the
  planner is `get_mla_metadata_v1_1` (`metadata.cu:149`), which takes no cap.
- **Raw KV batch count**, not the sparse-expanded one (`v1_2_device.cuh:860`).
- **Scaled by `qk_batch_ratio`** when the head count is not natively served, via
  the same gate the planner uses (`v1_2_device.cuh:910-928`).

Callers passing no cap are unaffected: the branch needs
`max_split_per_batch > 0`, default `-1`.

`reduce_partial_map` entries, gfx950, `nhead=128`, `qo_len=4`:

| batch | uncapped | cap=1 | cap=256 |
|---:|---:|---:|---:|
| 1 | 1024 | 5 | 260 |
| 64 | 1276 | 320 | 512 |

Not addressed: `max_work` / `work_info_set` are still sized from the uncapped
estimate, and `intra_batch_mode` ignores the cap entirely.

## Test Plan

```bash
pytest op_tests/test_mla_metadata_split_cap.py op_tests/test_mla_metadata_split_cap_fill.py

# also run standalone, which is how aiter CI invokes op_tests files
python3 op_tests/test_mla_metadata_split_cap.py
python3 op_tests/test_mla_metadata_split_cap_fill.py
```

Two files because a sizing formula can otherwise only be checked against itself.
The first pins the arithmetic and the native-support gate across gfx950, gfx942
and gfx1250. The second allocates exactly what sizing returns, runs the planner
into it with the same cap and dtypes, and checks the populated extents fit.

## Test Result

```
op_tests/test_mla_metadata_split_cap.py       31 passed
op_tests/test_mla_metadata_split_cap_fill.py  27 passed
```

Measured fills against the bound, printed by the fill test so the numbers land
in CI:

| shape | partials / bound |
|---|---:|
| batch 1, cap 256 | 256 / 260 |
| batch 8, cap 256, qlen 1 | 262 / 263 |
| batch 512, cap 256 | 506 / 2040 |
| nhead 48 (folded), cap 16 | 48 / 60 |

The qlen=1 row is the tight one, a single entry of headroom.

## Submission Checklist

- [ ] Look over the contributing guidelines at https://github.com/ROCm/TheRock/blob/main/GOVERNANCE.md#pull-requests.
