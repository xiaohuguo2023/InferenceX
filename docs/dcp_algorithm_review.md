# Decode Context Parallelism on Kimi-K3 / MI355X — algorithm review

A systematic read of the DCP decode path across three codebases, the deltas we
added for AMD, and the defects that read turned up.

**Sources, exact versions.** All line numbers below were read on 2026-08-31 and
are true for these trees:

| tree | version | how referenced below |
|---|---|---|
| vLLM (container `k3-1dc464d`) | `nightly-1dc464d42681d22f38caf1fdc1eb632dc4421c45`, `0.28.x` | `vllm/…` |
| ATOM | `0b4f1ddb` ("Settle a step's shape once…") | `atom/…` |
| aiter (transplanted) | `/opt/aiter-local` | `aiter/…` |
| our patch chain | `_port_dcp_nightly_ba07e4a4.py` + 3 satellite patches | "ours" |

Everything here is static analysis. No GPU was used — the box was handed to
colleagues for the duration.

---

## Part 1 — The algorithm

### 1.1 Why MLA forces DCP to exist at all

Tensor parallelism shards attention by **KV head**. That works while TP width
≤ number of KV heads. MLA has, effectively, **one** latent KV "head" per token:
a single `kv_lora_rank`-wide vector plus a rope tail. So at TP8 every rank must
hold **the entire KV cache**, replicated eight times. KV capacity stops scaling
with GPU count, and since long-context decode is DRAM-read-bound on that cache,
so does throughput.

Decode Context Parallelism shards the KV cache along the **sequence** axis
instead of the head axis. Each of the 8 ranks stores 1/8 of the tokens, computes
attention against only its own shard, and the shards are then merged into the
mathematically exact global result. The merge is the whole trick, and it is
cheap because attention is a softmax-weighted average whose partial results
compose exactly (§1.3).

This is the same observation the Helix Parallelism paper starts from — see
Part 4.

### 1.2 KV placement: round-robin token interleave

Token `pos` lives on rank `(pos // S) % W`, where `W = dcp_world_size` and
`S = cp_kv_cache_interleave_size`. We run `S = 1`, i.e. plain token-level
round-robin. Its rank-local index is

```
local_index(pos) = (pos // (S*W)) * S + (pos % S)        # S=1 → pos // W
```

and the physical slot is `block_table[pos // (block_size*W)] * block_size +
local_index % block_size`.

**All three trees agree on this arithmetic**, which is the single most important
consistency check in the whole system — the target attention, the draft, and the
KV-write kernels must all place a token identically or the cache is silently
corrupt:

| tree | symbol | location |
|---|---|---|
| vLLM | `cp_local_slot` | `vllm/v1/worker/gpu/cp_utils.py:65-82` |
| ATOM | `_dcp_local_slots` / `dcp_local_index` | `atom/plugin/vllm/dspark_dcp_patch.py:68`, `atom/model_ops/dcp_ops.py:719` |
| ours | inherited from vLLM (unpatched) | — |

Round-robin is what makes the shards **balanced by construction**: with
`S=1` every rank gets `⌈len/W⌉` or `⌊len/W⌋` tokens regardless of sequence
length. This matters for Part 4.

Per-rank lengths come from `get_dcp_local_seq_lens` (`dcp_ops.py:676`), which is
the closed form of the above rather than a scan.

### 1.3 The merge: partial attention + log-sum-exp correction

Each rank `r` computes attention over only its shard and emits two things: the
partial output `O_r` and the log-sum-exp `L_r` of that shard's scores. The exact
global result is recovered as

```
L*    = logsumexp_r(L_r)                       # global normalizer
w_r   = exp(L_r - L*)                          # rank r's share of the softmax mass
O     = Σ_r  w_r · O_r                         # Σ_r w_r = 1
```

This is exact, not an approximation — it is the same rescaling flash-attention
uses across its own tiles, lifted across ranks.

Three numerical details, all three trees handle them:

1. **Subtract the max before exponentiating** (`lse_max`), else `exp` overflows.
2. **A rank owning zero tokens** for a row emits `L_r = -inf` → `w_r = 0`, but
   its `O_r` is `0/0 = NaN`. `NaN × 0 = NaN` would poison the sum, so the
   zero-weight contribution must be **skipped, not multiplied**.
3. **Log base.** Hardware has `exp2`, not `exp`. Implementations either carry
   base-2 LSE throughout or convert once. Getting this wrong is a silent
   temperature error, not a crash — see §5.1 for how we verified ours.

### 1.4 Two ways to do the collective

**Route A — AllGather LSE, correct locally, ReduceScatter output.**
`cp_lse_ag_out_rs` (`vllm/v1/attention/ops/dcp.py:275`). Every rank gathers all
`W` LSEs (small: `[N,B,H]` floats), scales its own output by `w_r`
(`_correct_attn_cp_out_kernel`, `dcp.py:69`), then a ReduceScatter sums them.
Two collectives, but the second is over the full output tensor.

**Route B — one all-to-all carrying output+LSE packed together, combine locally.**
`dcp_a2a_lse_reduce` (`dcp.py:704`). Each rank sends every peer only the head
slice that peer owns, with the LSE packed into extra columns of the same buffer
(`_dcp_a2a_lse_pack_dim`, `dcp.py:410`). One collective. Then
`_dcp_a2a_unpack_combine_kernel` (`dcp.py:504`) does the full merge in one pass.

**We run Route B**, via a direct peer-to-peer workspace rather than RCCL — §3.3.

---

## Part 2 — The three implementations side by side

| concern | vLLM upstream | ATOM | ours |
|---|---|---|---|
| interleave math | `cp_utils.cp_local_slot` | `dcp_ops.dcp_local_index` (+ inverse `dcp_global_pos`) | vLLM's, unpatched |
| LSE merge kernel | `_correct_attn_cp_out_kernel`, base-2 **and** base-e via `IS_BASE_E` | same kernel, **base-e only**, no fp32 upcast on load | vLLM's + our HIP port |
| a2a combine | Triton `_dcp_a2a_unpack_combine_kernel` | `cp_lse_a2a` (`dcp_ops.py:434`) | **native HIP kernel** (§3.3) |
| draft under DCP | DCP-aware **in the kernel** (`speculator.py:515-600`) | monkeypatched on top (`dspark_dcp_patch.py`) | vLLM's; our port is pure deletion |
| FULL cudagraph under DCP | blanket downgrade to PIECEWISE (`platforms/rocm.py:910`) | surgical un-downgrade (`rocm_dcp_full_graph_patch.py`) | env opt-out — **worse, see §5.2** |
| head count for the cprr kernel | n/a | n/a | **pad 96→128** (§3.2) |
| sparse-MLA top-k under DCP | — | full candidate exchange (`dcp_ops.py:757-1130`) | not applicable |

Two structural observations:

**ATOM's DCP work is largely already upstream.** Their
`apply_vllm_dspark_dcp_input_patch` rewrites the draft's `query_slot_mapping`
and `context_slot_mapping` after the fact, because vLLM's kernel used to compute
unsharded `block_id*block_size + pos%block_size`. On our image that is obsolete:
`_prepare_dflash_inputs_kernel` now takes `cp_rank`/`CP_SIZE`/`CP_INTERLEAVE`
directly. This confirms that our ATOM draft port being "pure deletion" was the
right call.

**And upstream is now stricter than ATOM.** vLLM's version additionally
(a) refuses to write draft KV into physical block 0 when an evicted
sliding-window position maps there, and (b) deliberately initializes the *full*
`[0, num_ctx)` span even though only `[0, num_valid_ctx)` is loaded, so **a
replayed CUDA graph cannot observe a stale slot from an earlier batch**. Point
(b) only became load-bearing for us when we turned FULL graphs on.

**ATOM carries a large capability we do not need.** Roughly 400 lines of
`dcp_ops.py` implement sparse-MLA top-k candidate exchange across DCP ranks
(`dcp_pack_topk_candidates`, `dcp_decode_candidate_exchange`,
`triton_filter_and_convert_dcp_index`). That is for their sparse attention
backend. Our path is dense MLA; ignore it when diffing.

---

## Part 3 — What we changed, and why it is AMD-specific

### 3.1 The MLA reduce scratch tight bound (6.97 GiB/rank reclaimed)

aiter sizes the fp32 partial-reduce scratch from a loose `fast_mode` estimate.
At our serving shape (batch 64, qlen 15, heads padded to 128) that asked for
`4785 × 8 × 128 × 512 × 4 B = 9.35 GiB`. But the metadata kernel's split budget
is **global** — `min(cu_num, max_split_per_batch × batch_size)`
(`csrc/kernels/mla/metadata/v1_2_device.cuh:560-562`) — so the true bound is
`tile_cnt + per_tile_cap` = 960 + 256 = **1216**, a 3.9× overestimate.

aiter combined the two with `max()`, which can only *raise*, so the tight bound
could never win. Three defects had to be fixed together:

1. aiter's `max()` → `min()` (`aiter/ops/attention.py:~1257`).
2. vLLM passed the cap as `num_kv_splits=`, which that helper ignores; the
   tight bound keys off `max_split_per_batch=`.
3. vLLM passed the cap only on the round-robin build, not the non-cprr
   fallback — measured **7.7× overflow** on that path.

Validated over 1030 shapes (0 violations, worst actual/bound 0.998) and
end-to-end with a 4096-entry canary tail (960 cases, 0 canary smashes, worst
fill exactly 1.000). Scratch 9.35 → 2.38 GiB.

**Why this is DCP-specific:** without DCP, heads pad to 16 and the same buffer
is 0.16 GiB. DCP pads to 128 — a 58× difference. Nobody upstream feels this.

**What it unblocked:** FULL cudagraph capture under DCP fits (22.47 → 18.29 GiB,
zero runtime OOM), which is what actually made DCP fast (§4.2).

### 3.2 Pad 96 heads → 128 instead of folding

`dcp_heads = num_heads × W = 12 × 8 = 96` is not a native cprr head count
(`{16,32,64,128}`). The obvious fix — fold into 3 pseudo-requests of 32 heads —
is memory-optimal but needs a post-kernel permute/reshape/`.contiguous()`
un-fold that **races the a2a combine**, forcing a per-layer host sync that cost
~5× conc-1 ITL and did not amortize (flat ~175 ms through conc-24).

We pad the query 96→128 instead: one pre-kernel copy (which does not race), one
fixed-head kernel, and the real heads sliced back off as a **view** with no
`.contiguous()`. One producer, one consumer, naturally ordered, no sync.
`_pad128_vs_fold.py` proved aiter's fp8-asm cprr kernel is **bit-exact** under
this padding (cos = 1.0, max|Δ| = 0).

This mirrors what CUTLASS does on NV via `reserve_query_head_storage` /
`q_pad_num_heads=128`; we are adopting a structural fix, not inventing one.

### 3.3 A native HIP direct-a2a combine

`k3_dcp_direct_hip/dcp_direct_a2a_lse_reduce_hip.hip` is a hand port of vLLM's
CUDA `dcp_direct_a2a_lse_reduce.cu` to ROCm/gfx950. It registers
`torch.ops._C.direct_dcp_a2a_lse_reduce` for the CUDA (== HIP) dispatch key, so
vLLM's `MLADCPManager` direct-combine path works on ROCm.

The point is **no host sync**: all four kernels are enqueued on the current
stream, and cross-rank ordering is done on-GPU via an epoch + system-scope
signal handshake. Measured cos 0.999999 against the reference.

Its `EVENT` ms/call is also our sharpest box-health probe: 0.08 ms healthy vs
1680 ms on a degraded driver.

Note the deliberate choice at `dcp_direct_a2a_lse_reduce_hip.hip:200-207`: on a
peer-wait timeout it **returns rather than trapping**. A GPU-side trap faults
the queue and poisons the driver box-wide (subsequent `ncclCommInitRank` goes
15 s → 235 s, and a `docker restart` does not cure it because the state is in
the kernel). Returning lets the run die through the normal RCCL-watchdog path.

### 3.4 A barrier before speculator capture

Under the ATOM-style sharded draft, the draft's MLA runs a **CP collective
inside its CUDA-graph capture**. Ranks reach `DFlashSpeculator.capture()`
staggered ~1 s apart, so a rank can begin capturing a collective its peers have
not entered — a rank-ordering race that surfaces as a hard GPU fault with a
fresh address every run.

Discriminating evidence: booting with `AMD_SERIALIZE_KERNEL=3
HIP_LAUNCH_BLOCKING=1` succeeds reliably. A configuration that only fails when
kernels overlap is an ordering problem, not an out-of-bounds access.

Fix is `sync → dist.barrier → sync` at the top of `capture()`. Boot-time only;
measured boot-to-ready unchanged at ~280 s.

### 3.5 Un-downgrading FULL cudagraphs

`vllm/platforms/rocm.py:910` rewrites `FULL_AND_PIECEWISE → PIECEWISE` whenever
`decode_context_parallel_size > 1`, with no backend check. On the asm-MLA + a2a
path that guard is measurably wrong. Ours is an env opt-out; ATOM's is better,
and we should adopt theirs — §5.2.

---

## Part 4 — Comparison with Helix Parallelism (arXiv 2507.07120)

Helix (NVIDIA, Jul 2025) starts from exactly the premise in §1.1: "When TP width
exceeds the number of KV heads, it leads to inefficient KV duplication, limits
parallelism, and constrains batch size." Its proposal is to apply **KV
parallelism during attention**, then **reuse the same GPUs for TP (dense) or
TP×EP (MoE) during FFN**, with a "lightweight communication step" to keep
attention exact, plus **HOP-B** to hide that communication via batchwise
overlap. Claims 1.5× TTL and 32× batch for DeepSeek-R1 on Blackwell.

**Mapping onto what we run:**

| Helix concept | our equivalent | status |
|---|---|---|
| KV Parallelism during attention | `dcp=8` token round-robin (§1.2) | have it |
| reuse same GPUs for TP×EP in FFN | TP8 + EP on the same 8 ranks | have it |
| "lightweight communication step" | the LSE merge (§1.3) via direct a2a (§3.3) | have it, in native HIP |
| HOP-B batchwise comm/compute overlap | **nothing** | **the one real gap** |

So the architecture Helix proposes is the architecture we are already running.
The paper is best read as independent validation of the design, not as a source
of new levers. Their "32× larger batches at the same latency" is the same
mechanism as our measured 6.6× KV-token capacity at a fixed 32 GiB pin (which
moved the prefix-cache cliff from conc-4 to conc-16).

**On HOP-B specifically, temper expectations.** We measured that DCP's conc-1
penalty was **CUDA-graph dispatch, not the collective**: ITL went 76.2 → 13.25 ms
from enabling FULL graphs alone, with the collective unchanged. And at conc-1
there is no batch to split, so HOP-B is a no-op there by construction. Where it
could pay is conc-16+, which is currently bound by prefix-cache behaviour
(§5.3), not by decode comms. HOP-B is worth revisiting only after that is fixed.

**Two things Helix does not model that dominate our system:** speculative
decoding (our `qlen>1` verify is what closes ATOM's QREP and `interleave_size>1`
optimizations to us), and a hybrid KDA+MLA cache (§5.3). Neither appears in the
paper.

**Caveat on sourcing.** Only the abstract was read cleanly; PDF text extraction
was unreliable (it described a backward pass, meaningless for a decode paper),
so nothing above depends on the paper's body. If HOP-B becomes a real candidate,
read the PDF properly first.

---

## Part 5 — Defects and risks

Ranked by expected cost. Each says what evidence exists and how to settle it.

### 5.1 Verified clean (recorded so nobody re-litigates them)

- **LSE log base.** `rocm_aiter_mla.py` never declares `lse_base_on_e`, so it
  inherits `True` from `vllm/attention/backend.py:804`. That is **correct**:
  aiter is base-e (`aiter/ops/triton/attention/mla_decode.py:591` uses `tl.exp`;
  `flydsl/kernels/mla_reduce.py:180` computes `exp2(x·log2 e) ≡ exp(x)`). Our
  HIP kernel takes `is_lse_base_on_e` and converts with `value *= K_LOG2E_F`
  before `exp2f` (`…_hip.hip:243-249`), and the flag is plumbed from
  `DirectDCPA2AWorkspace.lse_reduce` (`dcp.py:866-895`). Correct end to end.
  *But it works by inheritance* — see 5.4.
- **Empty-shard NaN guard.** Present in all three: vLLM Triton
  (`partial = tl.where(weight == 0.0, 0.0, partial)`), ATOM (explicit comment at
  `dcp_ops.py:~180`), and our HIP kernel, which is strictest — it masks at the
  *source* (`empty_kv ? -k_pos_inf() : …`) *and* skips zero-weight sources.
- **Draft KV addressing under DCP.** Upstream kernel is DCP-aware and handles
  the null-block and stale-slot cases ATOM's monkeypatch does not.
- **ATOM's dcp-mask trick is side-effect free.** `check_and_update_config` in
  `rocm.py` touches DCP in exactly one branch and makes no `super()` call, and
  `platforms/interface.py` has no DCP references at all.

### 5.2 Adopt ATOM's cudagraph un-downgrade (low risk, removes a footgun)

Ours is `K3_DCP_ALLOW_FULL_CUDAGRAPH=1`, default off — so the *good* config is
the one you have to remember to ask for, and every launcher that forgets
silently runs 5.75× slower at conc-1. ATOM's version
(`rocm_dcp_full_graph_patch.py`) instead keys off whether full graphs were
actually requested, and correctly still downgrades when
`prefill_context_parallel_size > 1`. Verified safe above. Adopt it and drop the
env var.

**Also delete hunk J's caveat** ("wins 5.5× step time but currently costs
acceptance"). Measured false: AL 2.38 FULL vs 2.37 PIECEWISE. Leaving a stale
warning in the tree is how a future session talks itself out of the right
setting.

### 5.3 Prefix-cache granularity is 12,288 tokens under DCP8 — intrinsic, costs 2.25 pp

Both serve logs print:

> `Setting attention block size to 1536 tokens to ensure that attention page size is >= mamba page size.`

That is the KDA page-alignment bump (K3 interleaves KDA recurrent layers with
MLA): `interface.py:895-905` sets
`attn_block_size = kernel_block_alignment_size × cdiv(mamba_page_size, kernel_block_alignment_size × attn_page_size_1_token)`,
pinned by `assert attn_page_size >= mamba_page_size`. Then
`resolve_dcp_kv_block_size` (`kv_cache_utils.py:651-658`) multiplies the
*attention* group's span by `W` and leaves the MambaSpec alone.

**Mechanism corrected 2026-08-31.** An earlier draft of this section blamed
`mamba_cache_mode == "none"` forcing the hash unit to the scheduler block size.
That branch never fires for us: `MambaModelConfig.verify_and_update_config`
(`models/config.py:649`) auto-sets `align` whenever prefix caching is on, and
the serve log confirms it (`config.py:650`). The hash unit is
`gcd(group_block_sizes) = gcd(12288, 1536) = 1536` in **both** arms. What
actually differs is the attention *group block size* — 1,536 non-DCP vs 12,288
DCP8 — and that is the cache-**hit** granularity.

Net effect, and it predicts the measured numbers exactly:

| arm | attn group block | blocks in the 63,911-tok prefix | predicted cache% | measured |
|---|---:|---|---:|---:|
| non-DCP | 1,536 | 41 × 1536 = 62,976 | 92.49% | **92.5** |
| DCP8 | 1536 × 8 = **12,288** | 5 × 12,288 = 61,440 | 90.24% | **90.2** |

Both match to rounding. **So the low-concurrency cache miss is pure granularity
loss, not eviction** — which retires the "MTP shared-prefix tail-drop"
hypothesis, and means the ~2,471 tokens dropped under DCP are recoverable by
configuration rather than by a kernel change.

**The `--use-replayssm --mamba-cache-mode align` lever is DEAD** (killed
2026-08-31, before spending box time). `align` is already on — vLLM sets it for
us — so there is nothing to turn on, and neither it nor `prefix_match_unit` nor
`--block-size 128` moves the *attention group* block size, which is where the 8×
coarsening lives. The coarsening is intrinsic to DCP8.

**Consequence: this is not the conc-16 cliff.** 2.25 pp cannot explain
90.2% → 15.0%. The cliff has no configuration explanation on the table and has
to be settled by measurement (Appendix item 5).

**But the 2.25 pp figure is MICROBENCH-SPECIFIC — do not quote it for agentic.**
Measured 2026-08-31 on the IX agentic benchmark (`aiperf --scenario
inferencex-agentx-mvp`, cc-traces), the same granularity costs **~20 pp**:

| workload | non-DCP cache% | DCP8 cache% | loss |
|---|---:|---:|---:|
| pool-of-1, one 63.9k shared prefix | 92.49 | 90.24 | **2.25 pp** |
| IX agentic cc-traces, conc-1 | 95.5 | 78.4 | **17.1 pp** |
| IX agentic cc-traces, conc-4 | 93.6 | 73.2 | **20.4 pp** |

One shared prefix amortises the coarsening over every request; agentic traffic
is many varied-length trajectories, so *every* request tails into a partial
block and loses up to 12,287 tokens instead of up to 1,535. That is a 3× TTFT
regression on the workload the IX story is actually told on, which makes this
the **single biggest DCP lever for agentic** — not a footnote. Checked against
the obvious confound: at comparable prompt lengths (DCP 131.0k tok/req → 73.2%
vs baseline 147.4k → 92.8%) the gap holds, and baseline cache% is flat 93–96%
across lengths. See memory `k3-dcp8-ix-agentic-result`.

Worth noting ATOM sidesteps this entirely by shipping K3 with
`--no-enable-prefix-caching`, on the stated grounds that "KDA recurrent state
cannot be reconstructed from the paged MLA cache alone." We run prefix caching
on and get correct GSM8K, so vLLM is evidently handling the reconstruction —
`use_replayssm` is the machinery for doing it *well*.

*Untested and cheap:* ATOM also pins `--block-size 128` for K3. Given the
1536-token auto-bump above, it is not obvious this changes anything for us;
worth one A/B rather than adoption on faith.

### 5.4 The tight bound is an unguarded whole-program invariant

After §3.1, the reduce buffer is sized on the assumption that **every** call
site passes `max_split_per_batch`. We fixed the two that exist. A third path
added later that omits it writes up to 7.7× the sized entries — and an
undersized reduce buffer **faults the GPU rather than raising**, so the failure
mode is a memory access fault with a fresh address, i.e. maximally hard to
attribute.

`self._mla_max_split_per_batch = 32` is a bare constant at
`rocm_aiter_mla.py:447` with nothing tying it to the sizing call.

Recommend: a cheap host-side assert at build time that the cap actually in
effect equals the one used for sizing. Cost is one Python comparison per build;
it converts a GPU fault into a message.

### 5.5 Two fixes exist only inside a container — FIXED 2026-08-31

The tight-bound patch (§3.1) and the speculator barrier (§3.4) *were* live edits
in `k3-1dc464d` only. Neither was in `_port_dcp_nightly_ba07e4a4.py`, so any
rebuild silently reverted both — reintroducing, respectively, a capture-time OOM
and a boot-time GPU fault.

Both are now hunks in the port script (N, P1/P2/P3). One ordering constraint
came out of this: P1 patches `/opt/aiter-local/aiter/ops/attention.py`, which is
the **hand-transplanted** aiter tree that shadows the pip install — so the
transplant moved from step 4 to step **0b** of the chain, ahead of this script.
Patching the pip copy instead would be a silent no-op, and applying P2/P3
without P1 is a silent no-fix, so `apply()` now aborts with the `docker cp`
commands if that tree is missing.

Fold both into the port script as anchor-guarded `SITES` entries. This is the
highest-value item that costs no GPU time.

### 5.6 `lse_base_on_e` is correct by inheritance, not by declaration

§5.1 establishes the value is right. But `rocm_aiter_mla.py` gets it from a base
class default, while every *other* MLA backend states it explicitly
(`tokenspeed_mla.py:159` False, `flashinfer_mla.py:215` True,
`flashinfer_mla_sparse.py:306` False). If the base default is ever flipped, or
the backend is pointed at a base-2 kernel, the result is a **silent temperature
error in attention** — no crash, no NaN, just quietly wrong logits, and
`exp2` vs `exp` is a factor of `ln 2` in the softmax exponent.

One line — `lse_base_on_e: bool = True` in the ROCm aiter impl — removes the
whole class of failure.

### 5.7 DCP + KV offload silently breaks the interleave invariant

ATOM documents `interleave_size > 1` as **incompatible with speculative
decoding** — "the qlen>1 verify cprr MLA kernel assumes token-level interleave."
We run `S = 1`, so we are fine today. But `S` is not always ours to choose.

`VllmConfig.adjust_dcp_kv_cache_interleave_size` (`config/vllm.py:2741-2785`)
**silently overrides** `cp_kv_cache_interleave_size` to `local_block_size`
whenever

```
decode_context_parallel_size > 1
  and kv_transfer_config is not None
  and kv_transfer_config.kv_connector is not None
```

— that is, **whenever a KV connector is active**, which covers P/D
disaggregation *and* KV offload. It is emitted at `info_once`, not a warning.

Worse, `validate_block_size` (`config/vllm.py:2786+`) then **deliberately skips**
the DCP interleave-size compatibility check in exactly that case
("Skip DCP interleave-size compatibility when a KV connector is configured"). So
the one guard that might have caught it is disabled precisely when the override
fires.

Consequence for us: **DCP8 + KV offload + DSpark would run with `S = local
block_size` instead of 1**, against a cprr verify kernel that assumes token-level
interleave. That is silent numerical corruption in the draft verify path — no
crash, no NaN, just degraded acceptance and wrong logits. Nothing in the logs
says so beyond one `info_once`.

We have not hit it because we keep offload off for this benchmark, and that was
for a *performance* reason (offload made high-conc TTFT 2.8–7× worse on the
pool-of-1 shape). This is an independent **correctness** reason, and it bites on
the agentic multi-prefix corpus where offload is actually wanted.

Action: add an explicit assert that `cp_kv_cache_interleave_size == 1` whenever
`num_speculative_tokens > 0 and dcp_size > 1`, placed *after*
`adjust_dcp_kv_cache_interleave_size` runs. Cheap, and it converts a silent
accuracy regression into a refusal to boot. Verify against a real offload+DCP
boot before trusting either direction.

### 5.8 Lower-priority

- **ATOM's LSE kernel drops the fp32 upcast** on the LSE load that vLLM keeps
  (`.to(tl.float32)`). Irrelevant while LSE is fp32, latent if it ever is not.
  Not our bug; noted for when we diff against them again.
- ~~**The in-code comment justifying pad-over-fold** (§3.2) cites latency and a
  symm-mem race. The durable reason is cprr-mask correctness.~~ **WITHDRAWN
  2026-08-31.** There is no cprr-mask correctness argument: `_pad128_vs_fold.py`
  runs fold-32 and pad-128 on the *same* fp8 q, the same KV shard and the same
  round-robin CP metadata, and they agree **bit-exactly** (cos = 1.0,
  max|Δ| = 0). Fold-32 masks correctly. The reason to prefer pad-128 really is
  the one `_patch_pad128.py` already documents — the un-fold `.contiguous()`
  races the a2a combine and forces a per-layer host sync that costs ~5× conc-1
  ITL and does not amortize (flat ~175 ms through conc-24). That docstring is
  measured and accurate; leave it alone.

---

## Appendix — what to do next, in order

1. ~~Fold §5.5's two patches into the port script.~~ **DONE 2026-08-31** —
   `_port_dcp_nightly_ba07e4a4.py` now carries hunks N (speculator barrier) and
   P1/P2/P3 (reduce-scratch tight bound). 12 anchor-guarded SITES; apply →
   revert → apply verified byte-identical on all 6 files.
2. Adopt ATOM's cudagraph un-downgrade; delete the stale caveat (§5.2). No GPU.
   *Deliberately deferred* until item 5 gives us conc-16+ data to judge it on.
3. ~~Add the three asserts/declarations.~~ **DONE 2026-08-31** — hunks Q (§5.7
   interleave vs spec decode, raised at the *override* site in
   `adjust_dcp_kv_cache_interleave_size`, not at config time), R (§5.4 split cap,
   plus an opt-in `K3_DCP_CHECK_REDUCE` deep check) and S (§5.6 `lse_base_on_e`
   declared on `AiterMLAImpl`). Verified not to false-trip on our config.
4. ~~Test `--use-replayssm --mamba-cache-mode align`.~~ **DEAD** — `align` is
   already on and the lever does not exist (§5.3). Do not spend box time on it.
5. **Measure conc-16/24/32/48 under FULL graphs.** Now the top item: it is the
   single measurement standing between "DCP works" and "DCP performs", the only
   remaining way to explain the conc-16 cliff, and the precondition for judging
   whether HOP-B (Part 4) or item 2 are worth any effort at all.
6. **Attack the DCP prefix-cache granularity for agentic** (§5.3). Worth ~20 pp
   of cache hit rate and ~3× TTFT on the IX agentic benchmark — the biggest DCP
   lever we have found. It needs a real change (finer hit granularity under a
   coarse attention group), not a flag. Run a non-DCP 900 s agentic control with
   the same seed first, to close the duration confound formally.
7. Tune the missing GEMM shapes (1,592 fallback lines, 5 shapes). Start with
   **N=2880 / K=7168 — zero rows in the merged CSV** — then fill the M gaps for
   N=7168/K=1792, N=7168/K=1024, N=1536/K=1536, N=20480/K=7168.
