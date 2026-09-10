# B300 conc-1 decode trace — what to capture, and what we will do with it

Purpose: decide whether the **40% conc-1 decode gap** (B300 15.62 ms/step vs MI355X
26.32 ms/step, AL identical at 3.84) is **structural** (they launch fewer kernels) or
**per-kernel efficiency** (same structure, faster kernels). Those imply completely
different roadmaps, and nothing we can measure on our side separates them.

## The one number that decides it

**Kernels per decode step.**

| if B300 shows | interpretation | what we do next |
|---|---|---|
| **~1,730 kernels/step** (at our 9.0 us each) | **structural** — ~40% fewer launches | fusion / cross-layer grouping. Per-kernel tuning is already exhausted and cannot close it |
| **~2,900 kernels/step** (at ~5.4 us each) | **efficiency** — same structure, faster kernels | their kernel stack (TOKENSPEED_MLA, FlashKDA, cutlass_dsl, FlashInfer AR) is better per call; port or match individual kernels |

Our reference, measured 2026-09-10 on the ROCm 7.2.3 image: **2,913 kernels/step,
26,350 us/step**, of which dense bf16 GEMM is 28.7% over 788 calls. See
`k3-conc1-is-kernel-count-bound-not-bandwidth`.

## Capture spec

Agreed with SL:

- **ISL ~100K, ideally exactly 99,757** — matches the MI355X trace we already hold, so it is
  a like-for-like diff with no extra work on our side. 64–128K is also where the relative gap
  is widest (1.66x) and where the most paired samples sit (61 of 120).
- **Steady-state decode-only window**, OSL >= 512, profile mid-decode. Do **not** include
  prefill: it is a different kernel mix and we already know our prefill *beats* theirs above
  64K.
- **All 8 ranks.** One rank is not enough — busy-wait collectives charge spin time as "busy",
  so a single rank cannot distinguish real work from waiting. This is the trap that produced
  a bogus "rank7 laggard" finding on our side.
- **Per-category split**: dense GEMM / MoE expert / MoE route / all-reduce / MLA / KDA /
  elementwise / norm-quant. We produce this ourselves from the raw trace — just send traces.

### Additions SL's list does not cover

**1. The decode-step count for the profiled window. Without this the capture is unusable** —
"kernels per step" cannot be computed from a trace alone. Two independent ways; please do
both, they are nearly free:

- The torch profiler emits `ProfilerStep#` markers, which our parser reads directly
  (`_build_step_index` -> `num_decode_iters`). Leave them on (they are on by default).
- Belt-and-braces: `curl -s localhost:<port>/metrics | grep spec_decode_num_drafts`
  **immediately before and immediately after** the profile window. The delta is the exact
  step count. (This is how we derived 26.32 ms/step without a profiler at all.)

**2. `record_shapes: true`.** Gives per-GEMM shapes, which lets us compare kernel-for-kernel
against our 788 dense calls rather than only in aggregate. Cheap.

**3. `with_stack: false`.** We do not need python frames for kernel counts, and stacks inflate
the trace several-fold.

**4. The `server.log` from the same run**, so we can confirm which backends were selected in
the profiled process (their earlier run chose `TOKENSPEED_MLA`, `TRTLLM_RAGGED` MLA prefill,
`FlashKDA`, `FLASHINFER_TRTLLM_MXFP4_MXFP8` MoE, FlashInfer all-reduce). Config drift between
the benchmarked run and the profiled run would invalidate the comparison.

### Suggested invocation

```
--profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/k3prof",
                    "torch_profiler_with_stack":false,"torch_profiler_record_shapes":true}'
```
then, once warm and in steady-state decode:
```
curl -s localhost:$PORT/metrics | grep spec_decode_num_drafts   # BEFORE
curl -X POST localhost:$PORT/start_profile
sleep 3
curl -X POST localhost:$PORT/stop_profile
curl -s localhost:$PORT/metrics | grep spec_decode_num_drafts   # AFTER
```
Deliverable: the 8 `dp0_pp0_tp*_*.pt.trace.json.gz` files (~25 MB gz each), the two metrics
lines, and `server.log`.

## How we will process it

`trace_compare_k3.py` is already cross-platform and already does the side-by-side, dividing
DECODE by each config's own `num_decode_iters`:

```bash
python3 trace_compare_k3.py \
  --tp-dirs /dev/shm/prof_dcp8_ep1 <their_dir> \
  --tp-labels mi355x_r72 b300 \
  --output-csv /tmp/cmp.csv --output-md /tmp/cmp.md
```

**Caveat on absolute times.** The torch profiler inflates decode substantially (it is
launch-heavy at conc-1) and widens collective arrival gaps. So compare **counts and ratios**,
not absolute microseconds. Kernel *count* is the robust metric precisely because profiler
overhead does not change it — which is the main reason it is the number we want.

**Known categoriser gap.** `CATEGORY_PATTERNS` has generic NVIDIA entries (CUTLASS MXFP4,
NVJet, cuBLASLt, Lamport AR, SymmMem AR, SM100 FMHA) but **none of B300's actual K3 stack**:
`TOKENSPEED_MLA`, `TRTLLM_RAGGED`, `FlashKDA`, FlashInfer MoE/AR, `cutlass_dsl`. Those will
initially land in **"Other"** — which is fine and self-correcting: the tool lists uncategorised
kernels individually **by name**, so the first run tells us exactly which patterns to add, and
adding them is a five-minute edit. Do not treat a large "Other" on the first pass as a result.
