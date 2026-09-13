# Upstreaming the K3 DCP work: PR plan

Goal: stop carrying 894 lines of local vLLM patches. Every nightly bump currently
costs a rebase (this is what produced `kda.patch.eed1f3d0` and the per-image patch
fallback), and upstream is actively editing the same files — they have already
natively absorbed `dcp_a2a_pack_mask` and most of `kda.patch`, so the risk is
silent divergence rather than merely rebase work.

Baseline for all of this: pristine `vllm/vllm-openai-rocm:nightly-2671fedf…`
(vLLM `0.29.1rc1.dev9+g2671fedfc`, ROCm 7.2.3), container `k3-2671`.

## Independence — measured, not assumed

Each patch was applied **alone** to a pristine tree and the touched module
imported. All seven pass, and each touches a different file in a different
subsystem:

| patch | file | applies alone | imports alone |
|---|---|---|---|
| `speculator` | `v1/worker/gpu/spec_decode/dflash/speculator.py` | YES | OK |
| `speculative_draft_dcp` | `config/speculative.py` | YES | OK |
| `scheduler` | `kv_connector/v1/offloading/scheduler.py` | YES | OK |
| `config` | `model_executor/models/config.py` | YES | OK |
| `retention_alignment` | `v1/core/kv_cache_coordinator.py` | YES | OK |
| `cp_common` | `v1/attention/ops/cp_common.py` | YES | OK |
| `rocm_aiter_mla` | `v1/attention/backends/mla/rocm_aiter_mla.py` | YES | OK |

`dcp_a2a_pack_mask` (133 lines) is **already upstream** — drop it locally.

## What is actually in the big patch

`rocm_aiter_mla.patch` is 20 hunks / 379 added lines, and it is **not** 379 lines
of tangled divergence:

| lines | share | theme |
|---:|---:|---|
| 338 | 89.2% | asm **cprr** DCP verify route |
| 18 | 4.7% | global kv indptr (`g_kv_indptr`) |
| 12 | 3.2% | head padding / `num_heads` |
| 9 | 2.4% | `max_split_per_batch` plumbing |
| 2 | 0.5% | fp8 dtype routing |

It is **additive** to upstream. #51705 (merged 2026-08-31) shipped the
*segmented* DCP verify route; all five of our `_asm_dcp_verify_*` symbols are
absent upstream. The two routes make opposite design choices:

|  | segmented (upstream) | asm cprr (ours) |
|---|---|---|
| batch shape | 8 rows x 1 query | 1 row x 8 queries |
| causality | per-row lengths, host-built | in-kernel, global positions + `g_kv_indptr` |
| interleave | `cp_interleave == 1` only | round-robin |
| kernel | **Triton** `mla_decode_fwd` | aiter fp8 asm `..._cprr_ps` |
| KV reads | ~1 per verify token (~8x) | once, amortised over 8 queries |

The motivation for the PR is not "we prefer ours": under DCP+DSpark every step
has `qlen = 1 + nspec > 1`, so **stock upstream runs the entire DCP decode on
Triton**. Adding the asm route is what makes the asm MLA path reachable at all
under DCP + spec decode.

## PR order

Ordered by (independence x ease of review x standalone value). Each is a separate
PR; none depends on another landing first.

### PR 1 — `speculator`: use the per-group cp_size (17 lines)
Bug: passed the global `cp_size` instead of `cp_sizes[gid]`, so acceptance was
computed against the wrong context-parallel size. Fixing it took AL to 2.5-2.8.
**Tests:** pure-CPU unit. Build speculator state with two KV groups at different
`cp_sizes`; assert the per-group value reaches the acceptance path. No GPU.

### PR 2 — `speculative_draft_dcp`: draft inherits DCP config (20 lines)
Bug: `create_draft_parallel_config` drops `decode_context_parallel_size`,
`dcp_comm_backend`, `cp_kv_cache_interleave_size`, so the DSpark draft silently
loses DCP. This is a **boot blocker** for DCP8+DSpark.
**Tests:** pure-CPU unit. Construct a `ParallelConfig` with DCP set, call
`create_draft_parallel_config`, assert every DCP field survives. No GPU.

### PR 3 — `scheduler`: offloading `sliding_window_size` (29 lines)
Note this is **not DCP** — it is the KV-offloading connector. Separable and can
go independently of the whole DCP story.
**Tests:** unit on the connector's block accounting with a sliding window set.

### PR 4 — `config`: DCP defaults + direct-DCP env (34 lines)
**Tests:** unit — defaults resolve correctly with and without
`VLLM_USE_DIRECT_DCP_A2A`; invalid combinations raise.

### PR 5 — `max_split_per_batch` plumbing (9 lines, carved out of `rocm_aiter_mla`)
Highest value-per-line of the set: measured **ITL p90 9.43 -> 8.18, intvty p90
106.0 -> 122.2** at conc-1, because upstream's uncapped default starved DCP8 at
low batch (32->456us, 64->242, 128->141, 256->103 at batch 1).
**Direction matters and makes this safe alone:** vLLM passing
`max_split_per_batch` when aiter lacks the tight bound is harmless (aiter sizes
generously). The reverse — aiter tight bound without vLLM passing it — overflows
reduce scratch ~7.7x and faults. So the vLLM PR may land first; the aiter PR
must not.
**Tests:** unit asserting sizing (`get_mla_metadata_info_v1`) and runtime
(`get_mla_metadata_v1`) receive the **same** value — consistency is the actual
requirement, not the number.

### PR 6 — `retention_alignment` (87 lines)
Prefix-cache block alignment under DCP.
**Tests:** unit on the block-size alignment math across
`hash_block_size`/`scheduler_block_size`/`dcp_world_size` combinations.

### PR 7 — `cp_common`: symm_mem teardown (164 lines)
Fixes the SRCU exit deadlock (symm_mem -> refcount-7 orphans -> exiting ranks
wedge in the global mmu_notifier SRCU).
**Tests:** hardest of the set — it is a teardown race. Unit-test the workspace
lifecycle and the atexit registration; a true regression test needs a multi-rank
teardown and may have to be a CI job rather than a pytest.

### PR 8 — asm cprr DCP verify route (338 lines + `g_kv_indptr`/head-pad/fp8, 32)
The feature. Additive alongside upstream's segmented route, selected by
`VLLM_ROCM_AITER_MLA_DCP_VERIFY` (which also takes a per-KV-group form,
`segmented:64`, so a DSpark target and draft can be routed independently — a
bisection tool upstream has no equivalent for).
**Tests: written and green** — 35 passed, 0 skipped, on gfx950 in `k3-2671`.
Two files, both destined for `tests/v1/attention/`:

`test_rocm_aiter_mla_dcp_cprr.py` (31 tests, pure CPU, runs in any CI job) pins
the *decisions*: `_parse_dcp_verify_env` in every accepted form plus both error
paths (bare trailing `:`, unknown route); `_asm_dcp_verify_heads` incl. the
96->128 pad and the unservable case; per-KV-group routing (`segmented:64` =
draft segmented, target still asm); the `cp_interleave == 1` restriction; and
the `_MIN_CPRR_QLEN > 2` gate.

`test_rocm_aiter_mla_dcp_cprr_numerics.py` (4 tests, **real GPU**) is the actual
correctness gate. It drives the **real** `AiterMLAMetadataBuilder.build()` and
the **real** asm kernel, simulating the 8 DCP ranks one after another on one
GPU — legitimate because `forward_mqa` is collective-free (the query all-gather
happens in the layer, before it), so `cp_rank` is just a number feeding the
causal-mask maths. It asserts:
- every shard matches an exact torch reference over the global positions it
  holds, at 128 gathered heads (native) **and** 96 (the K3 pad to 128);
- the LSE merge of all 8 shards reproduces full-context attention;
- `md.decode.asm_decode_num_heads` is set, so a silent fall-out to
  segmented/Triton fails loudly rather than passing quietly;
- **positive control:** handing shard r's pages to the kernel while telling it
  it is rank r+1 must blow the error up. Without this a green run proves
  nothing, because a kernel ignoring global positions still produces
  plausible-looking output.

No model weights are needed: the MLA dims come from DeepSeek-R1's *config*
(the model upstream's own MLA tests already use) with only the head count
overridden, and the parallel groups are single-process stubs.

Verified to have power, not just to be green: forcing
`VLLM_ROCM_AITER_MLA_DCP_VERIFY=segmented` turns all 4 red.

### Separate repo — ROCm/aiter
`0001-k3-dcp8-code.patch` (113 lines): fp8 MLA block-N lookup keys, the tight
split-tile bound (`max`->`min`, reclaims MLA reduce scratch 9.35 -> 2.38 GiB),
and a split-K guard for ASM a16w16 under cudagraph replay.
**Must land after vLLM PR 5** (see the direction argument above).
`0002-k3-tuned-gemm-csv.patch` (1618 lines) is tuned CSV data, likely not
upstreamable as-is. `0003-flydsl-032-aux-attr.patch` is a local compat shim for
flydsl 0.3.2 and should NOT be upstreamed — see the version trap below.

## Known traps

- **`flydsl.__version__` lies.** A `docker run --rm` probe reported 0.3.2 on an
  image whose container reports 0.3.0. Use
  `md5sum .../flydsl/_mlir/dialects/_rocdl_ops_gen.py`:
  `dd8c8c930e41` = 0.3.2 (keep `0003`), `d6814906b7fd` = 0.3.0 (revert `0003`).
- Develop every PR against a **pristine** tree. Our patched tree differs from
  upstream in exactly the regions these PRs touch; working from it risks
  upstreaming unrelated DCP code by accident.
- Upstream's #51705 shipped four test files — that is the bar. PR 8 now clears
  it (see above); the other PRs still ship no test and must before they go out.
- **A DCP test must patch `get_dcp_group` in three places.** Both
  `rocm_aiter_mla` and `mla_attention` bind the symbol at module import, so
  patching `vllm.distributed.parallel_state` alone leaves the builder reporting
  `dcp_world_size == 1` — the cprr route is then skipped and the test passes
  while exercising nothing.
- **A DCP test also needs a `speculative_config`.** Without one the batch
  reorder threshold stays at 1, a qlen>1 row is classified as a PREFILL, and
  the decode path under test is never reached.
- `seq_lens` is the GLOBAL context length; `dcp_local_seq_lens` is what the rank
  holds. Passing the global length as the local one walks the kernel off the end
  of the page table and faults the GPU.
