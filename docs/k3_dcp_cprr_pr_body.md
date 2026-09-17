## Purpose

**Makes the AITER ASM MLA decode path reachable when decode context parallelism (DCP) is combined with speculative decoding on ROCm. Today it is not reachable at all in that configuration.**

#51705 added DCP multi-token verify through the *segmented* route: it expands a verify group into one single-query row per token and carries the causal window in per-row lengths, running the Triton `mla_decode_fwd` kernel. Because a spec-decode step always has `qlen = 1 + num_speculative_tokens > 1`, that route is taken for **all** of decode, so turning on DCP together with spec decode silently costs you the entire AITER ASM MLA path.

This PR adds a second, **additive** route that keeps those steps on the ASM path: the AITER round-robin (`cprr`) decode. It keeps the verify group as one row of `qlen` queries and reconstructs each token's **global** position in-kernel from a global per-request page indptr (`g_kv_indptr`) plus `cp_rank`/`cp_world_size`, so causality is applied inside the kernel rather than materialised on the host.

|  | segmented (existing) | asm cprr (this PR) |
|---|---|---|
| batch shape | N rows × 1 query | 1 row × N queries |
| causality | per-row lengths, host-built | in-kernel, from global positions |
| kernel | Triton `mla_decode_fwd` | AITER fp8 ASM `..._cprr_ps` |
| KV reads | ~1 per verify token | once, amortised over the block |

The route is selected per KV cache group by `VLLM_ROCM_AITER_MLA_DCP_VERIFY` (registered in `vllm/envs.py`): `asm` (default) or `segmented`, optionally restricted to listed DCP-gathered head counts, e.g. `segmented:64` keeps a 64-head draft on Triton while a 96-head target stays on ASM. A speculative target and its draft gather different head counts, so they can be routed independently.

### Where the route is reachable

Four conditions, so it is enabled only where it works and is needed:

- **gfx950.** The `cprr` kernels are built only for it: `hsa/gfx942/mla` and `hsa/gfx1250/mla` carry no `cprr` rows, so elsewhere the lookup would fail at the first verify step instead of falling back.
- **Multi-token decode** (`speculative_config is not None`). Single-token decode is already served. Enabling the route there would still override the attention head count to the padded native width on every step, for no benefit.
- **`decode_context_parallel_size > 1`**, and **`cp_kv_cache_interleave_size == 1`**, the same restriction the segmented gate carries.

Within that, the head-count rules: the ASM kernel exists only at the native DCP-gathered counts `(16, 32, 64, 128)`; other counts pad up to the next native one (96 to 128), and a count above the largest native is refused at build time rather than silently leaving the ASM path.

### Causality

`causal` is forwarded to the kernel. AITER defaults it to `True`, so a non-causal group (a DSpark draft attends its whole block) would otherwise get the masked kernel and see only tokens up to its own index. The non-causal `cprr` kernels exist (`msk0_lse_cprr`); they have to be asked for.

`qlen == 1` is already correct on the plain kernel, since a decode row sees every local token.

**`qlen == 2` raises.** There is no `cprr` kernel below 3, and the plain kernel applies causality on *local* indices over a round-robin shard, which is silently wrong rather than an error. The constructor rejects a configured qlen of 2, but `max_qo_len` is a per-batch value: the scheduler can clamp a running spec request to 2 under a larger configured threshold, so the per-step path checks it too.

### Route selection is fail-fast

The head-count filter rejects empty entries (`segmented:,`, `asm:64,`) and non-positive counts (`asm:0`). Both previously parsed to a filter that matches nothing, which applies the route to *every* group: the opposite of what was asked, silently.

This PR also plumbs `max_split_per_batch` through the DCP MLA metadata, set to the device CU count. MLA decode is lopsided: a handful of query rows against tens of thousands of KV rows, so at low batch the only parallelism available is splitting the KV axis. Measured on gfx950 at batch 1, qlen 15, 128 heads, 27,318 KV rows/rank:

| cap | 32 | 64 | 128 | 256 (= CU count) | 320 / 384 / 512 |
|---|---|---|---|---|---|
| decode | 456 µs | 242 µs | 141 µs | **103 µs** | 103–106 µs |

This is a maximum, not a fixed number of splits, so setting it high costs nothing when it does not bind. At batch 8 and 14 the timing does not change between 32 and 256 splits, because the batch already fills the device.

The number itself matters less than using the same one everywhere. `reduce_partial_map` is allocated with the cap applied, so `get_mla_metadata_info_v1` and `get_mla_metadata_v1` both have to see it. They read the same attribute, so they cannot disagree.

### Notes for reviewers

- **Additive.** None of the five `_asm_dcp_verify_*` symbols exist upstream; the segmented route is untouched and stays reachable via the env var. With `decode_context_parallel_size == 1` nothing here is reachable at all.
- **No AITER changes required.** Verified by reverting our local AITER patches and re-running the whole suite, with identical results (see Test Result). We have a separate AITER change that tightens the reduce-scratch bound when a cap is supplied; it is a pure memory optimisation, is inert without a caller that passes a cap to the *sizing* call, and can land in either order. Until it does, this PR sizes that scratch generously.
- **This is an enablement change, not a tuning change**, which is why the headline is a capability rather than a percentage. Because the route is selected at runtime, `VLLM_ROCM_AITER_MLA_DCP_VERIFY=asm` vs `segmented` compares the two routes on an identical build, model and workload, so reviewers can reproduce the comparison on their own hardware. We are not quoting an end-to-end percentage yet because we want a same-config A/B we can fully stand behind rather than a number carried over from a differently-configured run; we will follow up with one.

## Test Plan

```bash
# new: routing / selection (pure CPU, runs anywhere)
pytest tests/v1/attention/test_rocm_aiter_mla_dcp_cprr.py

# new: numerics (real metadata builder + real ASM kernel; needs gfx950 + AITER)
pytest tests/v1/attention/test_rocm_aiter_mla_dcp_cprr_numerics.py

# regression: the pre-existing AITER MLA suite
pytest tests/v1/attention/test_rocm_aiter_mla_fp8_decode_routing.py \
       tests/v1/attention/test_rocm_aiter_mla_mtp_split.py
```

`test_rocm_aiter_mla_dcp_cprr.py` (31 tests, CPU) pins the *decisions*: every accepted form of the env var plus both error paths, head-count selection including the 96 → 128 pad and the unservable case, per-group routing, the interleave restriction, and the `qlen` floor.

`test_rocm_aiter_mla_dcp_cprr_numerics.py` (4 tests, GPU) is the correctness gate. It drives the real `AiterMLAMetadataBuilder.build()` and the real ASM kernel, simulating the DCP ranks one after another in a single process. That is sound because this path is collective-free: the query all-gather happens in the layer, strictly before `forward_mqa`, so `cp_rank` only selects which residue class of global positions the shard holds. It asserts:

- each shard against an exact torch reference over the global positions it holds, at 128 gathered heads (native) **and** 96 (padded to 128);
- that the LSE merge of all shards reproduces full-context attention;
- that `asm_decode_num_heads` is set, so a silent fall-out to the segmented or Triton path fails loudly instead of passing quietly;
- **positive control**: handing shard `r`'s pages to the kernel while telling it that it is rank `r+1` must blow the error up. Without this a green run would prove nothing, because a kernel that ignored global positions still produces plausible-looking output.

No model weights are downloaded: the MLA dims come from the DeepSeek-R1 *config* (already used by the existing MLA tests) with only the head count overridden, and the parallel groups are single-process stubs.

`test_rocm_aiter_mla_mtp_split.py` fakes the builder and the decode metadata with `SimpleNamespace`, so the fields this PR adds to the real classes are added there too, set to the route-off values. Those cases therefore keep exercising the segmented/persistent path they were written for.

## Test Result

All on gfx950 (MI355X), ROCm 7.2.3.

| file | collected | needs |
|---|---:|---|
| `test_rocm_aiter_mla_dcp_cprr.py` | 39 | nothing, pure CPU |
| `test_rocm_aiter_mla_dcp_cprr_numerics.py` | 4 | AITER on gfx950 |
| `test_rocm_aiter_mla_mtp_split.py` | 59 | nothing, pure CPU |

No regressions in the pre-existing AITER MLA suite.

Forcing `VLLM_ROCM_AITER_MLA_DCP_VERIFY=segmented` turns the GPU tests red, so they are genuinely pinned to the route under test.

**What the tests do not cover.** The numerics file is the only coverage of the builder and decode paths, and it skips without gfx950, so on a runner without one a change to the per-step qlen gate, the `cprr` kwargs, or the `causal` forward fails nothing. It also fixes `QLEN = 5`, `causal=True` and `non_causal_multi_token_decode=False`, so qlen 1, qlen 2 and the non-causal path are not exercised anywhere. The CPU files cover the env parsing, the reachability gate and the head-count mapping.

**Numerics.** Per-shard maximum relative error against the exact torch reference, 8 DCP shards, fp8 KV:

```
shard 0..7:  1.32e-02 … 1.87e-02
merged vs full-context reference: 1.18e-02
```

against an fp8-e4m3 error floor of ~6.2e-2 (3 mantissa bits). Gates are relative, because with fp8 the error floor scales with output magnitude and an absolute gate would just be a magnitude gate.

**Independent of our AITER changes.** We reverted both Python hunks of our local AITER patch and re-ran the whole suite:

| AITER | result |
|---|---|
| patched (ours) | 540 passed, 0 failed |
| **stock** | **540 passed, 0 failed** |

Identical, so upstream CI will go green on stock AITER. We also instrumented the stock `get_block_n_fp8[...]` lookup, which lacks the keys our local patch adds and would raise `KeyError`, and confirmed it is **never reached** on the cprr path, i.e. it is not a hidden dependency of this PR.
