## Purpose

**On ROCm under decode context parallelism, a legal `prefix_cache_retention_interval` is rejected, and the user is forced to checkpoint 8x more coarsely than the cache can use.**

### Background

Sliding-window and Mamba (linear-attention) groups cannot prefix-cache densely, so vLLM keeps periodic *checkpoints*, and a prefix-cache hit can only resume from one. `prefix_cache_retention_interval` sets how far apart those checkpoints sit.

A checkpoint is only useful where a hit can actually land, so the interval has to be a multiple of the hit granularity. That is what the validator is for.

### The two block sizes

vLLM has two, and they are computed as opposites (`vllm/v1/core/kv_cache_utils.py`):

| | definition | property |
|---|---|---|
| `scheduler_block_size` | `math.lcm(*group_block_sizes)` | the **coarsest** common boundary — a block boundary in *every* group at once |
| `hash_block_size` | `math.gcd(*hashing_sizes)` | the **finest** common step — a valid boundary in every group, but not in all of them simultaneously |

Which one a hit is reported at depends on whether fine-grained partial hash hits are enabled. vLLM already encodes this, in `_cache_hit_alignment_tokens`:

```python
return (
    self.hash_block_size
    if self.enable_partial_hash_hits
    else self.scheduler_block_size
)
```

The validator, however, checks `retention_interval % scheduler_block_size` unconditionally.

### Why DCP exposes it

Without DCP the two values are typically equal and nothing is visibly wrong. DCP scales the full-attention group's block by `dcp_world_size`, which **raises the LCM and leaves the GCD alone**.

On a model with a 1536-token linear-attention block at DCP=8, the scaled full-attention block is 12288:

```
scheduler_block_size = lcm(1536, 12288) = 12288
hash_block_size      = gcd(1536, 12288) =  1536
```

So hits land every **1536** tokens, while the validator demands a multiple of **12288**. A user asking to checkpoint exactly where hits land is refused, and has to go 8x coarser — quantising every hit to a span the cache never reports, which costs prefix-cache hit rate.

### How it got this way

The validator was correct when written. `_validate_prefix_cache_retention_interval` arrived in **#43447** (2026-06-04), when `scheduler_block_size` was the only hit granularity there was.

**#46384** (2026-07-12) then added partial prefix-cache hits for hybrid models, introducing a *second*, finer granularity along with `enable_partial_hash_hits` and `_cache_hit_alignment_tokens` to express which one applies. The validator was not updated to ask.

So this is not a wrong check; it is a check left behind by a later change, and it only becomes visible when something pulls the two granularities apart. DCP is that something.

### The fix

`_cache_hit_alignment_tokens` is already the right answer, but it cannot simply be substituted into the base validator: it depends on `enable_partial_hash_hits`, which only the concrete coordinator resolves, after the base constructor has run.

So the check is split:

- the **base validator** keeps the checks that do not need alignment (non-negative; the "no sliding-window or Mamba group" rejection);
- a new **`_validate_retention_alignment`** runs once the coordinator has settled the granularity.

`HybridKVCacheCoordinator` passes the alignment it just computed. `UnitaryKVCacheCoordinator` never enables partial hash hits, so it passes `scheduler_block_size`.

The unitary path deliberately does **not** call `_cache_hit_alignment_tokens`: that property is defined on `HybridKVCacheCoordinator`, so reaching for it there raises `AttributeError` during construction — a boot failure for every single-group model. An earlier revision of this change did exactly that, so a test now pins which class owns the property.

### Relationship to #52527

**#52527** ("[Metrics] Report shared-prefix tokens lost to a missing sparse-retention checkpoint") is open and touches the same function, so whichever lands second will need a small rebase around `_validate_prefix_cache_retention_interval`.

The two are complementary rather than competing: #52527 *measures* prefix tokens lost to a missing retention checkpoint, and this PR removes one cause of those misses — a validator that forces checkpoints 8x sparser than the cache can use. Landing both gives you the metric and one fewer reason for it to fire.

Two other open PRs touch this file in unrelated regions: **#53479** (Mamba align) in the constructors, and **#50457** (all-sliding DFlash drafter) in `allocate_new_blocks`. Neither overlaps the lines changed here.

### Scope

**Scoped to ROCm**, which is the only platform where this has been tested. DCP is not backend-specific — `flashmla`, `flashmla_sparse`, `flashinfer_mla_sparse` and `flashattn_mla` all carry DCP support — so other platforms may well have the same latent mismatch. We have no way to exercise them, so we are not changing their behaviour on an untested claim.

**Off ROCm the original `scheduler_block_size` check is kept, not deferred.** This is the subtle part of gating: the alignment check now runs in a new place, so simply gating that new call would leave other platforms with *no* alignment validation at all — weaker than the status quo, not safer. Instead the base validator keeps its original check verbatim for them, and only ROCm takes the deferred, granularity-aware path. There is a test asserting the original rejection still fires off ROCm.

Within ROCm, the change only alters which value the interval is compared against; it does not change what a valid interval *does*.

If maintainers would like this widened once someone can test it on CUDA, the gate is a single condition in each of the two validators.

## Test Plan

```bash
pytest tests/v1/core/test_retention_interval_alignment.py

# regression check on the neighbouring coordinator/utils suite
pytest tests/v1/core/test_kv_cache_utils.py
```

The new tests are CPU-only and call the validators directly. They pin:

1. an interval equal to the hit alignment is accepted — the DCP case that used to be rejected;
2. an interval that cannot land on a hit boundary is still rejected;
3. `None` (dense) and `0` (latest boundary only) skip the check, since neither describes a spacing;
4. a multiple of the alignment is accepted;
5. **the regression itself** — 1536 accepted against `hash_block_size`, rejected against the DCP-scaled `scheduler_block_size`, which is the before/after in one test;
6. the base validator still rejects a negative interval;
7. the base validator no longer checks alignment, so the DCP case can reach the deferred check at all;
8. `_cache_hit_alignment_tokens` belongs to the hybrid coordinator and not the unitary one — the `AttributeError` guard described above;
9. **off ROCm, the original `scheduler_block_size` rejection still fires**, a scheduler-aligned interval is still accepted, and the deferred check is inert — i.e. behaviour there is byte-for-byte what it is today.

## Test Result

```
tests/v1/core/test_retention_interval_alignment.py   12 passed
```

The platform is patched rather than detected, so both branches are exercised on any runner.

**The tests have power, they are not merely green.** Two independent mutations each turn the expected tests red:

```
drop the off-ROCm fallback:     1 failed, 11 passed   (other platforms lose validation)
drop the ROCm alignment check:  2 failed, 10 passed   (the fix itself)
both present:                  12 passed
```

The first mutation is the one worth noting: it catches the failure mode that gating naively would have introduced.

No new failures in `tests/v1/core/test_kv_cache_utils.py`.

`pre-commit run --files <changed files>` passes in full, including `ruff`, `mypy` 3.10-3.13, `typos` and `check-spdx-header`.
