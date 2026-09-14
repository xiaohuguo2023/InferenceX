# conc-1: separating draft depth from credited AL

## Why

A run at depth 3 / AL 3.75 measured **intvty p90 160.5** against **136** at
depth 7 / AL 3.84 on the same machine — **+18%**. We cannot submit that
configuration (AgentX fixes the permitted (depth, AL) pairs, and depth 3 / AL
3.75 is not one of them), so this is **diagnostic, not a lever**.

It is still worth measuring, for two reasons:

1. **Our cost model predicts 10–15%, not 18%.** MLA attention is 14.1% of the
   step and dense GEMM 30%, and the GEMMs do not shrink with query length at
   these sizes (already in the skinny dead zone). If the real number is 18%, the
   qlen-sensitive fraction of the step is larger than our trace decomposition
   says — and that changes what we optimise next.
2. **We got this wrong once already** by moving depth and AL together and
   attributing the result to depth. Three arms fix that properly.

## The arms

Same machine, same image, same patch stack, same corpus, conc-1, 3600 s. The
**only** things that differ are the two named variables.

| arm | depth | AL | tag | what it is |
|---|---|---|---|---|
| **A** | 7 | 3.84 | `dal_k7_al384` | the permitted conc-1 config — baseline |
| **B** | 3 | 3.75 | `dal_k3_al375` | the +18% arm (not submittable; diagnostic) |
| **C** | 3 | 3.00 | `dal_k3_al300` | the permitted conc-8+ pairing |

**The decomposition this buys:**

| contrast | isolates | expected |
|---|---|---|
| A → B | **depth**, with AL essentially held (−2.5%) | the +18%, if it is real |
| B → C | **credited AL alone**, depth fixed at 3 | AL 3.75 → 3.00 is −20% tokens/step; throughput should fall, interactivity should barely move |
| A → C | both moved | what our original (mistaken) test measured |

B → C is the arm that did not exist before. Without it, depth and AL cannot be
told apart — which is exactly how we concluded "K=3 is worse" when the AL drop
was doing the damage.

## Commands

Run on the machine that produced 136 / 160.5, one at a time (each needs the
whole box).

```bash
# Arm A -- baseline, permitted conc-1 config
CONC=1 TAG=dal_k7_al384 \
  SPEC_NUM_TOKENS_OVERRIDE=7 SYNTHETIC_ACCEPT_LEN_OVERRIDE=3.84 \
  bash benchmarks/single_node/agentic/repro/k3_mi355x_agentic_repro.sh

# Arm B -- depth 3, AL held high
CONC=1 TAG=dal_k3_al375 \
  SPEC_NUM_TOKENS_OVERRIDE=3 SYNTHETIC_ACCEPT_LEN_OVERRIDE=3.75 \
  bash benchmarks/single_node/agentic/repro/k3_mi355x_agentic_repro.sh

# Arm C -- depth 3, permitted AL
CONC=1 TAG=dal_k3_al300 \
  SPEC_NUM_TOKENS_OVERRIDE=3 SYNTHETIC_ACCEPT_LEN_OVERRIDE=3.00 \
  bash benchmarks/single_node/agentic/repro/k3_mi355x_agentic_repro.sh
```

`MAX_CUDAGRAPH_CAPTURE_SIZE` is derived from the depth (`recipe:594`), so
capture sizes follow automatically — do not set them by hand.

Check `LMCACHE_VERSION` before launching. The `nightly-rocm` index is not
immutable; dev89, dev105 and dev134 have all been withdrawn, and a stale pin
aborts the run ~8 min in, *after* patching, leaving a half-patched tree that
breaks the next launch. Current pin: `0.5.6.dev3+rocm7.2`.

## Reporting

```bash
python3 _ab_route_report.py dal_k7_al384 dal_k3_al375 dal_k3_al300
```

Report **both** IX metrics for every arm — this is the number missing from the
original comparison:

| | intvty p90 | tok/s/GPU | n | ISL med | OSL med |
|---|---|---|---|---|---|

Use `frITL = full_decode_duration / OSL`, **not** aiperf's
`inter_token_latency`. They differ ~9%, and mixing them has already inverted the
sign of one verdict.

**The contamination check is not optional.** IX replays a fixed trace, so if
`n` / ISL med / OSL med differ between arms, the arms drew different work and
the delta is an artifact. `_ab_route_report.py` prints this automatically and
flags it.

## What each outcome would mean

- **A→B ≈ +18% and B→C shows interactivity flat with throughput down.**
  Confirms the story: depth drives interactivity, AL drives throughput. Our
  cost model understates the qlen-sensitive fraction and needs revisiting.
- **A→B well under 18%.** Something other than depth was in the original
  comparison — different image, offload setting, or duration. Find it before
  trusting any of these numbers.
- **B→C moves interactivity a lot.** Then credited AL affects step *timing*, not
  just token accounting, and the synthetic-acceptance mechanism needs
  understanding before any of this is used.

## What this does not change

Nothing here is submittable — the permitted pairs still bind. The output is a
better model of where conc-1 time goes, which decides whether we keep pushing on
glue/launch count (our current Stage 2 plan, ~1.0–1.1 ms/iter) or shift toward
the qlen-linear attention cost.
