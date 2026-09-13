# PR description — vLLM: AITER ASM round-robin DCP verify

Branch: `xguo/rocm-mla-dcp-cprr-verify` (4 commits, +1077/−9, 5 files)

**Title:**

```
[ROCm][MLA] Add an AITER ASM round-robin decode route for DCP multi-token verify
```

---

## Purpose

**Makes the AITER ASM MLA decode path reachable when decode context parallelism
(DCP) is combined with speculative decoding on ROCm. Today it is not reachable
at all in that configuration.**

Today, when DCP is combined with speculative decoding on ROCm, **every** verify
step leaves the AITER ASM MLA path.

`#51705` added DCP multi-token verify via the *segmented* route, which expands a
verify group into one single-query row per token and carries the causal window
in per-row lengths. That route runs the Triton `mla_decode_fwd` kernel. Because
a spec-decode step always has `qlen = 1 + num_speculative_tokens > 1`, the
segmented route is taken for the whole of decode — so on a DCP + spec-decode
configuration the tuned ASM MLA kernels are unreachable.

This PR adds a second, *additive* route that keeps those steps on the ASM path:
the AITER round-robin (`cprr`) decode. It keeps the verify group as one row of
`qlen` queries and reconstructs each token's **global** position in-kernel from
a global per-request page indptr (`g_kv_indptr`) plus `cp_rank`/`cp_world_size`,
so causality is applied inside the kernel rather than materialised on the host.

The two routes differ only in *how* causality is expressed:

|  | segmented (existing) | asm cprr (this PR) |
|---|---|---|
| batch shape | N rows × 1 query | 1 row × N queries |
| causality | per-row lengths, host-built | in-kernel, from global positions |
| kernel | Triton `mla_decode_fwd` | AITER fp8 ASM `..._cprr_ps` |
| KV reads | ~1 per verify token | once, amortised over the block |

Selection is per KV cache group via `VLLM_ROCM_AITER_MLA_DCP_VERIFY`
(registered in `vllm/envs.py`):

- `asm` (default) / `segmented` — pick a route for every group;
- `segmented:64` — apply the route to only the listed DCP-gathered head counts.
  The per-group form exists because a speculative target and its draft gather
  different head counts, so they can be routed (and bisected) independently.

Guard rails, all covered by tests:

- the ASM kernel only exists at the native DCP-gathered head counts
  `(16, 32, 64, 128)`; other counts pad up to the next native one (96 → 128) and
  a count above the largest native is refused at build time rather than silently
  falling out of the ASM path;
- `cp_kv_cache_interleave_size == 1` only, the same restriction the segmented
  gate carries;
- `qlen >= _MIN_CPRR_QLEN` (3). `qlen == 1` is already correct on the plain
  kernel (a decode row sees every local token); `qlen == 2` is refused at boot
  rather than silently degrading, because such a row can still be causally
  truncated.

This PR also plumbs `max_split_per_batch` through the DCP MLA metadata, set to
the device CU count. MLA decode is lopsided — a handful of query rows against
tens of thousands of KV rows — so at low batch the only parallelism available is
splitting the KV axis. Measured on gfx950 at batch 1, qlen 15, 128 heads, 27,318
KV rows/rank, varying the cap:

| cap | 32 | 64 | 128 | 256 (= CU count) | 320 / 384 / 512 |
|---|---|---|---|---|---|
| decode | 456 µs | 242 µs | 141 µs | **103 µs** | 103–106 µs |

It is a **cap, not a mandate**, so it only binds at low batch: at batch 8 and 14
the split count changes nothing (418–422 µs / 696–708 µs across 32…256), because
`batch × heads` already fills the device. The requirement is that the *same*
value reaches both the sizing call (`get_mla_metadata_info_v1`) and the runtime
build (`get_mla_metadata_v1`); consistency is what matters, not the number, and
there is a test asserting exactly that.

### Notes for reviewers

- **Additive.** None of the five `_asm_dcp_verify_*` symbols exist upstream; the
  segmented route is untouched and remains reachable via the env var. With
  `decode_context_parallel_size == 1` nothing in this PR is reachable at all.
- **No AITER changes are required.** Verified by reverting our local AITER
  patches and re-running the full suite — identical results (see Test Result).
  We have a separate AITER change that tightens the reduce-scratch bound when a
  cap is supplied; it is a pure memory optimisation, is inert without a caller
  that passes a cap to the *sizing* call, and can land in either order. Until it
  does, this PR sizes that scratch generously.
- **This is an enablement change, not a tuning change.** That is why the
  headline is a capability rather than a percentage. On ROCm today, turning on
  DCP together with speculative decoding silently costs you the entire AITER ASM
  MLA path: every step has `qlen > 1`, so every step takes the Triton segmented
  route, and none of the ASM MLA work applies. This PR is what makes that
  configuration able to use the ASM kernels at all. The relevant comparison is
  therefore "ASM path reachable or not", and the supporting evidence is the
  KV-read structure above plus the kernel-level and numerical results below.
- **The head-to-head is one environment variable.** Because the route is
  selected at runtime, `VLLM_ROCM_AITER_MLA_DCP_VERIFY=asm` vs `segmented`
  compares the two routes on an identical build, model and workload, with no
  rebuild and no other variable moving. Reviewers can reproduce that on their
  own hardware. We are not quoting an end-to-end percentage in this description
  yet because we want a same-config A/B we can fully stand behind rather than a
  number carried over from a differently-configured run; we will follow up with
  one.

---

## Test Plan

Two new files, plus the field additions the existing stubs need.

```bash
# new: routing / selection (pure CPU, runs anywhere)
pytest tests/v1/attention/test_rocm_aiter_mla_dcp_cprr.py

# new: numerics (real metadata builder + real ASM kernel, needs gfx950 + AITER)
pytest tests/v1/attention/test_rocm_aiter_mla_dcp_cprr_numerics.py

# regression: the pre-existing AITER MLA suite
pytest tests/v1/attention/test_rocm_aiter_mla_fp8_decode_routing.py \
       tests/v1/attention/test_rocm_aiter_mla_mtp_split.py
```

`test_rocm_aiter_mla_dcp_cprr.py` (31 tests, CPU) pins the *decisions*: every
accepted form of the env var plus both error paths, head-count selection
including the 96 → 128 pad and the unservable case, per-group routing, the
interleave restriction, and the `qlen` floor.

`test_rocm_aiter_mla_dcp_cprr_numerics.py` (4 tests, GPU) is the correctness
gate. It drives the real `AiterMLAMetadataBuilder.build()` and the real ASM
kernel, simulating the DCP ranks one after another in a single process. That is
sound because this path is collective-free — the query all-gather happens in the
layer, strictly before `forward_mqa` — so `cp_rank` only selects which residue
class of global positions the shard holds. It asserts:

- each shard against an exact torch reference over the global positions it
  holds, at 128 gathered heads (native) **and** 96 (padded to 128);
- that the LSE merge of all shards reproduces full-context attention;
- that `asm_decode_num_heads` is set, so a silent fall-out to the segmented or
  Triton path fails loudly instead of passing quietly;
- **positive control** — handing shard `r`'s pages to the kernel while telling
  it that it is rank `r+1` must blow the error up. Without this a green run
  would prove nothing, because a kernel that ignored global positions still
  produces plausible-looking output.

No model weights are downloaded: the MLA dims come from the DeepSeek-R1 *config*
(already used by the existing MLA tests) with only the head count overridden,
and the parallel groups are single-process stubs.

`test_rocm_aiter_mla_mtp_split.py` fakes the builder and the decode metadata
with `SimpleNamespace`, so the fields this PR adds to the real classes are added
there too, set to the route-off values. Those cases therefore keep exercising
the segmented/persistent path they were written for.

---

## Test Result

All on gfx950 (MI355X), ROCm 7.2.3.

| suite | before this PR | with this PR |
|---|---|---|
| existing AITER MLA tests | 505 passed | 505 passed |
| new tests | — | 35 passed |
| **total** | **505 passed, 0 failed** | **540 passed, 0 failed** |

**No regressions.** Running the pre-existing suite against the branch is also
what caught a real bug during development: `_build_decode` originally built
`g_kv_indptr` whenever `dcp_world_size > 1`, but only the ASM route allocates
the backing buffer, so a DCP run on the *segmented* route hit an assert. That is
fixed here (gated on the route), and `test_dcp_fp8_verify_build_uses_segmented`
covers it.

**The new tests have power, they are not merely green.** Forcing
`VLLM_ROCM_AITER_MLA_DCP_VERIFY=segmented` turns all 4 GPU tests red, so they
are genuinely pinned to the route under test:

```
4 failed
```

**Numerics.** Per-shard maximum relative error against the exact torch
reference, 8 DCP shards, fp8 KV:

```
shard 0..7:  1.32e-02 … 1.87e-02
merged vs full-context reference: 1.18e-02
```

against an fp8-e4m3 error floor of ~6.2e-2 (3 mantissa bits). Gates are
relative, because with fp8 the error floor scales with output magnitude and an
absolute gate would just be a magnitude gate.

**Independent of our AITER changes.** We reverted both Python hunks of our local
AITER patch and re-ran the whole suite:

| AITER | result |
|---|---|
| patched (ours) | 540 passed, 0 failed |
| **stock** | **540 passed, 0 failed** |

Identical, so upstream CI will go green on stock AITER. We also instrumented the
stock `get_block_n_fp8[...]` lookup — which lacks the keys our local patch adds
and would raise `KeyError` — and confirmed it is **never reached** on the cprr
path, i.e. it is not a hidden dependency of this PR.
