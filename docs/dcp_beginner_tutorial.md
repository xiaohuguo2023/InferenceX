# DCP (Decode Context Parallelism) — from zero

A beginner-friendly walkthrough, grounded in this project's Kimi-K3 fp8-asm DSpark
setup on MI355X (TP8, gfx950, `ROCM_AITER_MLA`, dcp-size=8).

---

## 1. The problem: attention has to look at the whole past

When an LLM generates text, it produces one token at a time. To produce the next
token, the model runs **attention**: the new token "looks back" at *every* previous
token to decide what's relevant.

To avoid recomputing the past every step, we cache it. For each past token we store
two vectors: a **K** ("key") and a **V** ("value"). This store is the **KV cache**.

The KV cache grows with context length:

```
1,000 tokens of context  → 1,000 K/V pairs cached
100,000 tokens           → 100,000 K/V pairs cached
```

For long context (say 68,000 tokens, like our K3 benchmark), the KV cache becomes
**huge** — often bigger than the model weights. Two pains follow:

1. **It doesn't fit** on one GPU comfortably.
2. **Reading it is slow** — every single decode step must stream the *entire* KV
   cache through the GPU. Decode is **memory-bandwidth-bound**: the GPU spends its
   time reading K/V, not doing math.

## 2. The idea: split the KV cache across GPUs

We already use several GPUs. In our K3 setup there are **8 GPUs** (TP8 = "tensor
parallel, 8 ways"). What if each GPU holds only **1/8 of the KV cache**?

```
GPU0: tokens      0 … 8,499     (KV shard 0)
GPU1: tokens  8,500 … 16,999    (KV shard 1)
...
GPU7: tokens 59,500 … 67,999    (KV shard 7)
```

Now each GPU only reads 1/8 of the cache per step → attention over the past is ~8×
faster, and the cache fits with room to spare.

Splitting the **context (the sequence of tokens) across GPUs** is called **Context
Parallelism (CP)**. Doing it specifically during the **decode** phase
(one-token-at-a-time generation) is **Decode Context Parallelism — DCP**.

> Why "decode" specifically? Prefill (processing the initial prompt) and decode have
> very different performance shapes. Prefill is compute-heavy; decode is
> bandwidth-heavy and latency-sensitive. DCP is the optimization aimed at the decode
> phase.

## 3. The catch: attention needs the *whole* context, but each GPU has only a slice

Here's the tension. The new token must attend to **all 68,000 past tokens**. But
GPU3 only has tokens 25,500–33,999. So GPU3 can only compute a **partial** attention
result — "what the answer would be if only my slice existed."

So the flow is:

1. **Every GPU gets the new query.** (The query is small — just the current token's Q
   vector. We copy it to all 8 GPUs. This is the "query all-gather.")
2. **Each GPU computes partial attention over its own KV shard.** GPU0 attends to
   tokens 0–8,499, GPU3 to 25,500–33,999, etc. Each produces a partial output.
3. **We merge the 8 partials into the one true answer.** ← this is the hard,
   interesting part, and it's what our current work is about.

## 4. How do you merge partial attentions correctly? (the LSE trick)

You can't just average the 8 partial outputs. Attention is a **softmax-weighted**
average, and the weights depend on scores from the *whole* context. Each GPU computed
softmax over *only its slice*, so its weights are "locally normalized" — wrong
globally.

The fix is a beautiful piece of math called **online softmax** (a.k.a. log-sum-exp
merging). Each GPU returns two things per query head:

- its **partial output** `O_i` (the locally-weighted value sum), and
- its **LSE** `L_i` = `log(sum of exp(scores))` over its slice — a single number that
  records "how much attention mass my slice held."

To merge, you combine partials weighted by their LSE:

```
weight_i  = exp(L_i - L_max)              # how important shard i is, globally
combined  = Σ_i (weight_i · O_i) / Σ_i weight_i
```

Intuition: **the LSE tells you how much each shard "mattered."** A shard whose tokens
were highly relevant has a big LSE and dominates the merge; an irrelevant shard has a
small LSE and barely contributes. This produces *exactly* the same answer as if one
GPU had done attention over all 68,000 tokens — no approximation.

> This is why you keep seeing `lse` everywhere in our code and logs (`lse=(1,96)`,
> `is_lse_base_on_e`, `lse_reduce`). The whole DCP correctness question reduces to:
> *did every GPU report a correct LSE, and did we merge them right?*

## 5. The communication step: "all-to-all" combine

There's one more wrinkle. After step 2, each GPU has partials for **all 96 query
heads** (K3 has 96 attention heads), but only over *its* KV slice. For the final
answer, we decide **GPU_k owns the final result for heads [k·12 … k·12+11]** (96 heads
/ 8 GPUs = 12 heads each).

So every GPU needs to:
- **send** its partials for heads owned by other GPUs → to those GPUs, and
- **receive** the partials for its own 12 heads → from all other GPUs.

Everyone sends to everyone → an **all-to-all** exchange. Then each GPU does the LSE
merge (step 4) for its own 12 heads. That combined function — *exchange head-slices,
then LSE-merge* — is the operation our whole project centers on:
`direct_dcp_a2a_lse_reduce`.

Picture it (each GPU keeps only its diagonal block after the exchange):

```
         heads owned by →   G0    G1   ...  G7
 G0's partials     :       [keep][→G1] ... [→G7]
 G1's partials     :       [→G0][keep] ... [→G7]
  ...
 G7's partials     :       [→G0][→G1] ... [keep]
                            ↓     ↓          ↓
 after exchange+merge:    G0 has  G1 has    G7 has
                          final   final     final
                          heads   heads     heads
                          0-11    12-23     84-95
```

## 6. Sync-free: why we're not just using the normal network library

The easy way to do all-to-all is to call a communication library (RCCL/NCCL — the GPU
equivalent of MPI). It works, but it's **expensive at decode time**: it uses a
separate "communication stream," needs the CPU to coordinate, and forces the GPU to
stop-and-wait (a "host sync") every single step. In our measurements that overhead was
a **~175 ms per-user tax** — brutal for a latency-sensitive decode loop.

NVIDIA's trick (which we're porting to AMD/ROCm) is a **sync-free direct combine**:

- Each GPU writes its head-slices **directly into its peers' memory** (the GPUs share
  "symmetric memory" — each can see the others' buffers).
- Instead of the CPU coordinating "everyone done? ok, proceed," the GPUs coordinate
  **among themselves, on-chip**, using a tiny handshake:
  - an **epoch counter** (which round are we on), and
  - a **signal flag** each GPU raises when its writes are done (`release`), which peers
    wait on before reading (`acquire`).

No CPU, no separate comm stream, no per-step stop-and-wait. That's what "sync-free"
means, and it's the point of the whole exercise.

The handshake in one breath:
```
1. bump epoch
2. write my head-slices into each peer's buffer   (dispatch)
3. raise my "done" flag to every peer             (signal / release)
4. spin until every peer raised THEIR flag, then  (wait / acquire)
   do the LSE merge for my 12 heads
```

Steps 3–4 are two one-line special instructions (a **release store** and an **acquire
load**). On NVIDIA they're written in PTX assembly; the port's job is to express the
same thing on AMD gfx950 (`__scoped_atomic_store_n` / `__scoped_atomic_load_n`) — that's
the bulk of the PR we've drafted.

## 7. Mini-glossary

| Term | Beginner meaning |
|---|---|
| **KV cache** | Saved K/V vectors for all past tokens, so we don't recompute them |
| **Decode** | Generating one token at a time (vs. "prefill" = digesting the prompt) |
| **CP / DCP** | Splitting the token sequence across GPUs; DCP = doing it during decode |
| **Shard** | One GPU's slice of the KV cache |
| **Partial attention** | The attention answer using only one shard |
| **LSE (log-sum-exp)** | One number per head saying "how much attention mass my shard held" — the key to merging partials exactly |
| **All-to-all** | Everyone sends to everyone; here, exchanging head-slices |
| **Symmetric memory** | GPUs can directly read/write each other's buffers |
| **Host sync** | CPU makes the GPU stop and wait — what we're trying to eliminate |
| **Epoch / signal** | On-GPU counter + flag the GPUs use to coordinate without the CPU |
| **TP8** | 8 GPUs working together on one model |

---

## 8. This project's concrete DCP shapes (for onboarding)

- **Target MLA layer** (the big attention that does verify): `non_causal=False`,
  per-rank `num_heads=12`, `dcp=8`, `decode_num_heads=96` (folded to nhead32, F=3),
  `head_dim=512` (= `kv_lora_rank`). Partial into the op = `[T, 96, 512]`; combined
  output = `[T, 12, 512]`.
- **Draft MLA layer**: `non_causal=True`, `dcp=1` (no CP combine — the draft is
  replicated, per vLLM #51705's replicated-draft pin).
- **DCP forces PIECEWISE cudagraph on ROCm** (`rocm.py`). Fine for correctness;
  matters for the perf arm (production wants `FULL_AND_PIECEWISE`).
- Enable path: `VLLM_USE_DIRECT_DCP_A2A=1` + env-gated
  `torch.ops.load_library(K3_DCP_A2A_SO)` registers
  `torch.ops._C.direct_dcp_a2a_lse_reduce`; `_init_combine` selects the direct
  workspace and logs "Using direct symmetric-memory DCP A2A for MLA."
