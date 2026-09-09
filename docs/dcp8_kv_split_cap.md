# RETRACTED: there is no `mla_*_cprr_ps` kernel bug

**Status: the original claim in this document was wrong and has been withdrawn.**
An earlier revision reported that aiter's CP round-robin MLA decode kernel was 6.1x slower
than the non-CP kernel at identical shape. That was a measurement artifact in our own
harness. With the split configuration made symmetric, the two kernels measure **1.00x**
(241.6 vs 241.7 us).

The real defect was **ours**: a hard-coded KV-split cap applied to the DCP path only.
Fixing it recovered 71% of the DCP8-vs-DCP1 deficit at concurrency 1.

## What actually went wrong

`patches/k3-dcp8/vllm/rocm_aiter_mla.patch` introduced (stock vLLM has neither symbol):

```python
self._mla_max_split_per_batch = 32
mla_num_kv_splits = (self._mla_max_split_per_batch if self.dcp_world_size > 1 else 0)
```

So **DCP8 was pinned to 32 KV splits while DCP1 passed 0 and let aiter choose**. These are
persistent (`_ps`) kernels, for which aiter's auto-selection (`get_meta_param`) is skipped
entirely -- the split schedule comes from the metadata built with `max_split_per_batch`,
whose upstream default is `-1` (uncapped). The asymmetry was invisible in any single-arm
measurement and is not something upstream would ever produce.

## Why splits dominate at low concurrency

MLA decode is lopsided: the query side is `T*H` rows (15 x 128 at conc 1) while KV is tens
of thousands of rows. `num_kv_splits` partitions the KV sequence; each split computes a
partial output + LSE and `mla_reduce_v1` merges them. Without splitting, parallelism is
roughly `batch * heads` -- at concurrency 1 the batch is **1**, so most of a 256-CU GPU
idles. Splits are the only way to manufacture parallelism along the long axis.

Measured on gfx950 (batch 1, `max_qo_len` 15, 128 heads, 27,318 KV rows/rank):

| splits | us | TFLOP/s |
|---:|---:|---:|
| 1 | 13,734 | 8 |
| 8 | 1,747 | 65 |
| **32 (what DCP8 used)** | **456** | 250 |
| 64 | 242 | 471 |
| 128 | 141 | 812 |
| **256 (= CU count)** | **103** | **1097** |
| 320 / 384 / 512 | 103-106 | saturated |

It saturates at the CU count -- one split per CU -- and going higher is harmless but
pointless.

## `max_split_per_batch` is a cap, not a mandate

It only binds when batch is small. Same shape, S=19,000 rows/rank:

| splits | batch 1 | batch 8 | batch 14 |
|---:|---:|---:|---:|
| 32 | 326 us | 420 | 696 |
| 64 | 177 | 422 | 708 |
| 128 | 109 | 421 | 697 |
| 256 | **85** | 418 | 699 |

At batch >= 8 the split count makes **no difference at all** -- `batch * heads` already
fills the GPU and aiter picks fewer splits than the cap on its own. So a high cap is safe
everywhere and only helps at low concurrency. A lower cap (e.g. 128) would cost 27% at
concurrency 1 and buy nothing elsewhere.

## The fix

`self._mla_max_split_per_batch` now defaults to the CU count
(`torch.cuda.get_device_properties(0).multi_processor_count`), overridable with
`K3_MLA_MAX_SPLIT`. Sizing (`get_mla_metadata_info_v1`) and runtime
(`get_mla_metadata_v1` + the decode call) all read the same attribute, so they stay
consistent at any value -- that consistency, not the specific number, is what the
`reduce_partial_map` sizing requires.

End to end, concurrency 1, DCP8 + DSpark K=7 + DRAM offload, 3600 s agentic trace:

| | splits 32 | splits 256 | vs DCP1 |
|---|---:|---:|---:|
| ITL p50 | 7.49 | **7.29** | 6.99 |
| ITL p90 | 9.43 | **8.18** | 7.66 |
| interactivity p90 | 106.00 | **122.22** | 130.51 |
| GPU prefix hit | 94.6% | 95.0% | 95.3% |

**ITL p90 -13.3%, interactivity +15.3%**, closing 71% of the DCP8-vs-DCP1 deficit
(1.77 -> 0.52 ms/token). No OOM and no LMCache faults despite the larger partial buffers
alongside the 32 GiB KV pin, so the conservative 32 was buying nothing.

Per ISL bucket, DCP8 is now at parity with DCP1 for 93% of traffic (>= 64K: within
0.2-5.6%); the residual aggregate gap sits in a 6-request 32-64K bucket.

## Methodology lessons (this is why the original claim was wrong)

1. **Make every knob symmetric across arms.** The bad measurement set
   `max_split_per_batch` on the CP arm only and left `num_kv_splits=1` on both, so it was
   comparing split schedules, not kernels. `op_tests/test_mla_persistent_round_robin.py`
   sets `num_kv_splits == max_split_per_batch` on all three calls -- follow it.
2. **Compare like with like on LSE.** An intermediate revision compared the CP `_lse_`
   kernel against the non-CP kernel *without* LSE, inflating the ratio.
3. **Report one timing method.** Wall-clock (`perf_counter` + `cuda.synchronize`) includes
   launch overhead and the reduce pass; rocprofv3 kernel traces do not. Mixing them
   changed the ratio by ~12%.
4. **A reproducer that needs our patches invites "your patch is the variable."**
   `_aiter_cprr_repro.py` is pure aiter + one GPU, no vLLM and no out-of-tree changes.

Reproducer:

```bash
python3 _aiter_cprr_repro.py --qlen 15 --max-split 256   # CP vs non-CP, 1.00x
python3 _aiter_cprr_repro.py --qlen 15 --max-split 32    # the starved config
python3 _aiter_cprr_repro.py --batch 8 --max-split 32    # cap does not bind at batch 8
```
