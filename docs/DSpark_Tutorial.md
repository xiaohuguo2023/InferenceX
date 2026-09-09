# DSpark, explained from zero

A beginner's tutorial to **DSpark** speculative decoding: what problem it solves,
how it works, a picture of the data flow, and exactly where it lives in the vLLM
source tree. No prior knowledge of speculative decoding assumed.

Grounded in the code running in this repo's container
(`vllm/vllm-openai-rocm:nightly-cb8104839c`, vLLM 0.26.1rc1.dev306) and our own
measured Kimi-K3 numbers.

---

## 1. First, why is text generation slow?

A large language model (LLM) generates text **one token at a time**. To produce
token #101 it must first have produced token #100, because #100 is fed back in as
input. So generating 350 tokens means **350 sequential forward passes** through
the whole model.

Here is the painful part: each of those forward passes is **memory-bound**, not
compute-bound. To decode a single token the GPU has to read *all* the model's
weights (for Kimi-K3, hundreds of GB across 8 GPUs) out of memory — just to
multiply them against one skinny vector. The GPU's math units sit mostly idle
waiting for memory. You paid to load the entire model and used ~1% of the
hardware's arithmetic capacity.

```
Normal decoding (one token per forward pass):

 step 1        step 2        step 3        step 4
[read 300GB]  [read 300GB]  [read 300GB]  [read 300GB]
   → tok A       → tok B       → tok C       → tok D

4 tokens  =  4 full weight reads  =  slow
```

**Key insight:** a forward pass that produces 1 token and a forward pass that
*checks* 4 tokens cost almost the same, because both are dominated by that one
giant weight read. If we could somehow *check 4 tokens per forward pass*, we'd go
~4× faster — for free.

That "somehow" is **speculative decoding**.

---

## 2. Speculative decoding in one picture

The idea: use a **cheap, fast "draft" model** to *guess* the next few tokens, then
let the **expensive "target" model verify all the guesses in a single forward
pass**. Guesses that match what the target would have produced are kept; the first
wrong guess and everything after it is thrown away.

```
                 ┌─────────────────────────────────────────┐
   draft model   │  guesses:  "the"  "cat"  "sat"  "on"     │   cheap, fast
   (small/fast)  └─────────────────────────────────────────┘
                              │  feed all 4 guesses at once
                              ▼
                 ┌─────────────────────────────────────────┐
   target model  │  verify in ONE forward pass:             │   expensive, but
   (big/slow)    │   "the" ✓  "cat" ✓  "sat" ✓  "on" ✗→"a"  │   only ONE pass
                 └─────────────────────────────────────────┘
                    accepted: "the cat sat"  + 1 fixed token "a"
                    → 4 tokens produced in 1 big pass instead of 4
```

Two properties that matter:

1. **It is loss-less.** The output is *mathematically identical* to what the
   target model would have produced alone. The draft only *proposes*; the target
   still decides. A bad draft just means fewer guesses accepted (less speedup),
   never a wrong answer. (This is why our GSM8K score is unchanged: **0.9689 with
   spec vs 0.9682 baseline**.)

2. **The win is "acceptance length"** — the average number of tokens you get out
   per target forward pass. If the draft is good, you accept many guesses and go
   fast; if it's bad, you fall back toward one-token-at-a-time.

The whole game is: **make a draft that is cheap to run AND accurate enough that
the target accepts a lot of its guesses.** Different speculative-decoding methods
are just different answers to "how do we build that draft?" DSpark is one answer.

---

## 3. What makes DSpark *DSpark*

Most classic speculative decoding builds the draft **autoregressively**: the draft
model itself generates guess #1, feeds it back to generate guess #2, and so on.
That means the *draft* now has the same sequential bottleneck we were trying to
escape — just on a smaller model. To draft 4 tokens you run the draft model 4
times.

**DSpark drafts a whole block of N tokens in essentially one parallel pass, then
adds the token-to-token dependency back in with a tiny, cheap "Markov head."**

The source docstring says it directly
(`v1/worker/gpu/spec_decode/dspark/speculator.py`):

> *"DSpark drafts a block of `num_speculative_tokens` tokens in one parallel pass
> (reusing the DFlash machinery: context-KV precompute + a query-block forward),
> then injects intra-block dependency with a lightweight sequential Markov head."*

So DSpark has **two stages** every draft step:

### Stage A — the parallel "backbone" pass (the expensive-ish part, done once)

The draft model looks at the target model's internal state (its hidden states)
and, in **one forward pass**, produces a hidden vector for *each* of the N draft
positions at the same time. This is "semi-autoregressive": all N positions are
computed together, in parallel, rather than one after another.

To make this possible DSpark reuses the **DFlash** machinery (DSpark is literally
a subclass of `DFlashSpeculator`):
- **context-KV precompute**: it writes the prompt's key/value cache entries once,
  up front, so the parallel block pass can attend to the context.
- **a query-block forward**: N "query" slots per request go through the model
  together.

### Stage B — the sequential Markov head (the cheap part, fixes up dependencies)

A pure parallel pass has a blind spot: position 3's guess doesn't know what
position 2 guessed. Language is sequential ("New York ___" wants "City", but only
if you know the previous word was "York"). DSpark fixes this **without** re-running
the big backbone. It runs a tiny head left-to-right:

```
prev = anchor token
for i in 0..N-1:
    bias   = markov_bias(markov_embed(prev))   # tiny lookup + small matmul
    logits = backbone_logits[i] + bias         # nudge stage-A logits by what we just picked
    tok_i  = sample(logits)
    prev   = tok_i                             # feed forward to the next step
```

This is the loop in `_sample_sequential()`. `markov_embed` / `markov_bias` are
tiny (an embedding + a small linear) — nothing like a full transformer layer — so
running them N times is negligible. You get *sequential-quality* drafts at
*parallel* cost.

```
DSpark one draft step (produces N guesses):

  target hidden states
        │
        ▼
  ┌──────────────────────────────┐
  │ Stage A: parallel backbone    │   ONE forward pass
  │ forward over N query slots     │   → hidden[0..N-1]
  └──────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────┐
  │ Stage B: sequential Markov     │   N tiny steps (embed+bias)
  │ h[0]→tok0→h[1]+bias→tok1→...   │   cheap, adds dependency
  └──────────────────────────────┘
        │
        ▼
   N draft tokens  ──────────────►  target verifies all N in its next pass
```

---

## 4. The full loop, end to end

Here's how one full generation step looks with DSpark turned on. "N" =
`num_speculative_tokens` (we benchmarked N=2 and N=7).

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. TARGET decode step                                                     │
│    The big model runs its normal forward pass. It produces:               │
│      • 1 real "bonus" token (as always), AND                              │
│      • its internal hidden states (free by-product)                       │
└────────────────────────────────────────────────────────────────────────┘
                              │  hidden states
                              ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. DSpark DRAFT (the two stages from §3)                                  │
│    Stage A parallel backbone  →  Stage B Markov head                      │
│    Output: N guessed tokens for this request                              │
└────────────────────────────────────────────────────────────────────────┘
                              │  N draft tokens
                              ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. TARGET VERIFY (next target pass)                                       │
│    Feed [bonus token + N drafts] to the target in ONE forward pass.       │
│    Rejection sampler compares each draft to the target's own distribution:│
│      • accept the longest correct prefix                                  │
│      • throw away from the first mismatch onward                          │
│      • the target's own next token replaces the first wrong guess         │
└────────────────────────────────────────────────────────────────────────┘
                              │  accepted tokens + 1 corrected token
                              ▼
                    loop back to step 1
```

Because step 3's verification is **one** target forward pass but yields **multiple
accepted tokens**, you amortize that giant weight-read over several tokens. That's
the speedup.

### The numbers from our Kimi-K3 run make this concrete

For **N=2**, over the whole benchmark we measured (from the server's
`spec_decode_*` metrics):

```
drafts made          = 200,562
draft tokens made    = 401,124   (= 200,562 × 2, the two guesses per draft)
draft tokens accepted = 264,118

acceptance rate       = 264,118 / 401,124 = 65.8%
mean acceptance length = (accepted + drafts) / drafts
                       = (264,118 + 200,562) / 200,562 = 2.32 tokens / target pass
```

"Mean acceptance length 2.32" means: **on average each expensive target forward
pass produced 2.32 tokens instead of 1.** At low concurrency (where decode is
purely memory-bound, the regime this helps most) that showed up directly as:

| metric        | baseline (no spec) | DSpark N=2 | speedup |
|---------------|-------------------:|-----------:|--------:|
| ITL @ conc 1  | 22.8 ms            | 10.5 ms    | 2.17×   |
| tok/s @ conc 1| 42.5               | 88.9       | 2.09×   |

The ~2.3× acceptance length turned almost 1:1 into a ~2.1× latency/throughput win.
(At *high* concurrency the GPU is already busy across many requests, so there's
less idle time to reclaim — spec decoding helps least there. That's expected.)

---

## 5. Where it lives in the source code

vLLM keeps the **orchestration** (the how-to-draft loop, CUDA graphs, sampling) in
the worker's `spec_decode/` tree, and the **model** (the actual draft network
weights and layers) in `model_executor/models/`.

```
vllm/v1/worker/gpu/spec_decode/
├── speculator.py                     ← generic base: DraftModelSpeculator
├── dflash/
│   ├── speculator.py                 ← DFlashSpeculator: the parallel-block engine
│   │                                   • propose()  (the top-level draft step, §4)
│   │                                   • precompute_and_store_context_kv call
│   │                                   • _run_model() = Stage-A backbone forward
│   │                                   • requires_non_causal flag (see §6)
│   ├── cudagraph.py                  ← DFlashCudaGraphManager (FULL graph capture)
│   └── utils.py                      ← prepare_dflash_inputs (builds the N query slots)
└── dspark/
    ├── speculator.py                 ← DSparkSpeculator(DFlashSpeculator)
    │                                   • _sample_sequential() = Stage-B Markov head
    │                                   • _generate_draft() = Stage A then Stage B
    │                                   • sample_from_anchor / num_query_per_req
    └── utils.py                      ← load_dspark_model() (wires draft to target:
                                         shares embed_tokens & lm_head, sets
                                         non-causal + draft quant/kv config)

vllm/model_executor/models/
├── qwen3_dflash.py                   ← dflash_has_any_non_causal(),
│                                       _dflash_layer_causal()  (the causal override)
└── (Kimi-K3 draft network)          ← K3DSparkForCausalLM: the actual draft model
    vllm/models/kimi_k3/nvidia/dspark_mla.py
                                      • combine_hidden_states()  (mean-pool target
                                        aux hidden states → draft hidden_size)
                                      • precompute_and_store_context_kv()
                                      • compute_draft_logits()   (Stage-A logits)
                                      • markov_embed() / markov_bias()  (Stage-B head)
                                      • map_draft_to_target()    (reduced-vocab remap)

vllm/transformers_utils/configs/k3_dspark.py
                                      ← K3DSparkConfig: draft_vocab_size,
                                        target_layer_ids, num_target_layers, etc.
```

### The single most important function to read

`DFlashSpeculator.propose()` in `dflash/speculator.py` (around line 303) is the
whole §4 loop in code. Skim it in this order:

1. `combine_hidden_states(...)` — take the target's hidden states, squeeze them
   into the draft's width.
2. `prepare_dflash_inputs(...)` — lay out the N query slots per request (the
   "anchor + noise" positions DSpark predicts at).
3. `precompute_and_store_context_kv(...)` — write context K/V (runs **eagerly**,
   outside the CUDA graph, because the context length varies per step).
4. `run_fullgraph(...)` **or** `_generate_draft(...)` — the captured draft step:
   Stage A backbone forward + Stage B Markov sampling.
5. `return self.draft_tokens[:num_reqs]` — the N guesses handed back for the
   target to verify.

And `DSparkSpeculator._sample_sequential()` in `dspark/speculator.py` is Stage B —
the little left-to-right Markov loop quoted in §3.

### How you turn it on (the serve config)

DSpark is selected entirely through `--speculative-config` (see
`_serve_k3_bench_spec.sh`):

```json
{
  "model": "Inferact/Kimi-K3-DSpark",   // the draft checkpoint
  "num_speculative_tokens": 2,           // N: how many tokens to guess per step
  "method": "dspark",                    // picks DSparkSpeculator
  "draft_sample_method": "probabilistic",// how the draft samples (vs greedy)
  "rejection_sample_method": "block"     // how the target accepts/rejects the block
}
```

- `method: "dspark"` is what routes vLLM to `DSparkSpeculator`.
- `draft_sample_method: "probabilistic"` → the draft samples with Gumbel noise so
  its distribution matches the target's for correct rejection sampling (the
  `gumbel_sample(...)` calls in `_sample_sequential`).
- `rejection_sample_method: "block"` → verify the drafted block against the
  target and accept the longest matching prefix (the rejection sampler in
  `spec_decode/rejection_sampler.py`).

---

## 6. The one wrinkle we hit on AMD: causal vs non-causal

This is specific to *our* hardware bring-up, but it's a great window into how the
method works.

DSpark's parallel block drafting is *designed* to use **non-causal attention**
inside the draft block: when guessing the block of N tokens together, each draft
position is allowed to "look at" the others (that's part of how a parallel pass
approximates a sequential one). The flag is computed by
`dflash_has_any_non_causal(...)` in `qwen3_dflash.py` and threaded through
`load_dspark_model()` as `use_non_causal`.

On this ROCm nightly, **non-causal attention wasn't runnable** for K3:
- the fp8 path (`ROCM_AITER_MLA`) doesn't implement non-causal at all;
- the bf16 non-causal path (`TRITON_MLA`) hit a GPU kernel fault on gfx950.

So we **forced the draft to causal** (`dflash_config.causal=true` in the draft
config → `_dflash_layer_causal()` makes every draft layer causal →
`use_non_causal=False`). That let the draft run on the fast fp8 `ROCM_AITER_MLA`
path with CUDA graphs. Forcing causal is a slight *quality* concession (the draft
loses some intra-block context), but:
- it is **still loss-less** — the target verifies everything, so correctness is
  unchanged (GSM8K 0.9689 confirms it);
- the **only** cost is possibly a bit lower acceptance length — and we still
  measured **2.32** for N=2, i.e. drafts are being accepted well.

The last blocker was unrelated to attention: with cudagraphs on, the aiter a16w16
**split-K GEMM** deadlocked on replay (a semaphore that was never re-zeroed inside
the captured graph). **aiter PR #4494** fixes that (a fresh zeroed semaphore under
graph capture). With that patch applied, the whole DSpark path runs with CUDA
graphs and no eager fallback. See `docs/…/dspark-blocked-rocm-nightly` memory for
the blow-by-blow.

---

## 7. TL;DR

- LLM decoding is slow because each token needs a full, memory-bound pass over the
  model. A pass that *verifies* K tokens costs about the same as one that makes 1.
- **Speculative decoding**: a cheap draft guesses several tokens; the target
  verifies them all in one pass; correct guesses are kept. **Loss-less.**
- **DSpark**'s trick: draft a whole block of N tokens in **one parallel pass**
  (reusing DFlash's context-KV precompute + query-block forward), then restore
  token-to-token dependency with a **tiny sequential Markov head** — so you get
  sequential-quality guesses at near-parallel cost.
- The payoff metric is **acceptance length**. We measured **2.32** for N=2 on
  Kimi-K3, which became a **~2.1× decode speedup at low concurrency**, with GSM8K
  accuracy unchanged.
- Code: engine in `v1/worker/gpu/spec_decode/{dflash,dspark}/`, draft network in
  `models/kimi_k3/nvidia/dspark_mla.py`. Start reading at
  `DFlashSpeculator.propose()` and `DSparkSpeculator._sample_sequential()`.
