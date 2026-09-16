# Findings log

## 2026-08-16 — Phase 0: decompose the 35 GiB KV reserve (measure-first)

**Boot:** K3 DSpark nspec=7, TP8, gfx950, no-pin (`KV_CACHE_MEMORY=none`) +
`VLLM_KV_RUNTIME_RESERVE_GIB=35`, mandated config (GPU_MEM=0.95, seqs=64, MNBT=16384,
FULL_AND_PIECEWISE). Serve on :8890. Method: boot-log memory summary + 1 Hz `rocm-smi`
VRAM sampling around a real 68K prefill (token-id driver, no aiperf) and a conc-24 load.

### Per-rank boot decomposition (from `gpu_worker.py:862`, TP0 representative; ±0.4 GiB across 8 ranks)
| component | GiB |
|---|---|
| physical card | 287.98 |
| requested (0.95×) | 273.59 |
| consumed (weights + non-torch) | 200.78 |
| profile_run peak_activation | 5.5 |
| **CUDA graph pool** (`capture_model()` → "Graph capturing … took") | **11.18** |
| KV allocated (reserve=35 applied) | 32.3 |

Sizing math verified exactly: `available_kv = 273.59 − (200.78+5.5) − 0[V2 graph stub] − 35[reserve] = 32.3` ✓.
V2 `profile_cudagraph_memory()` returns 0, so the reserve is literally standing in for the
missing graph accounting — **plus more**.

### Runtime VRAM (1 Hz sampler)
- idle post-capture, pre-prefill: **253.4–254.8 GiB** (~33 GiB free)
- single real 68K prefill (0 cached, 4.76 s): peak **+5.64 GiB** — **matches** profile_run's 5.5 GiB estimate.
  The +5.64 GiB is **persistent** (does not free after the request) → off-torch backend scratch
  allocated on first real prefill, invisible to `profile_run` (which frees its dummy before KV sizing).
- conc-24 (24 distinct resident 68K seqs, 64 decode steps w/ spec verify, 109 s): peak **267.2 GiB**,
  **min free 20.8 GiB**. Peak − pre-prefill-idle = **12.38 GiB** (= 5.64 persistent scratch + 6.74 conc-scaled activation).

### The 35 GiB reserve decomposition (per rank)
| bucket | GiB | % | who accounts it |
|---|---|---|---|
| CUDA graph pool | 11.18 | 32% | **PR1** (V2 `profile_cudagraph_memory` stub → real value) |
| persistent first-prefill scratch | 5.64 | 16% | **PR2** (backend workspace declaration / init-alloc) |
| high-conc activation under-count + margin | 18.18 | 52% | **PR3** (generic high-conc/mixed `MemoryProfileShape`) |

### Verdicts (this is what Phase 0 was for)
1. **The plan's hypothesis is REFUTED.** The graph pool is only **32%** of the reserve, not "most of it."
   The majority (68%) is genuine activation/scratch that `profile_run`'s single-prefill dummy under-counts.
2. **PR1 alone does NOT free any KV.** Accounting the 11.18 GiB graph pool while preserving the same real
   headroom just means `reserve' = 35 − 11.18 = 23.82` → KV stays **32.3 GiB**. PR1 is a
   correctness/transparency change (move a precisely-measurable quantity out of a magic number), not an
   immediate perf win. Verified numerically.
3. **Growing KV requires PR2 + PR3**, each validated at conc-48, because the under-count is real
   (explains why KV=56 OOMed at conc-48 — memory `k3-kv-pin-vram-ceiling`: vLLM's own "fit requested"=56.13
   suggestion omits the ~5.6 scratch + ~14 high-conc activation ≈ 20 GiB, so safe KV ≈ 36, ≈ the current 32.3).
4. **The reserve=35 is well-calibrated, not wildly over-provisioned** (~4 GiB true margin at conc-48).

### Implication for the user's fork (immediate perf vs long-term project)
This is a **long-term recorded project**: PR1 (transparency) → PR2 (scratch) → PR3 (high-conc profile shape),
each generic + upstreamable + validated at conc-48. There is **no quick reserve→0 flip** and no free KV from
PR1 alone. The only immediate lever would be re-tuning the constant down by the measured conc-48 slack
(~10 GiB), but that keeps a magic number, which is what we're trying to remove.

Artifacts: `_phase0/{vram_sampler.sh,prefill68k.py,vram.tsv,markers.txt}`, `serve_k3_bench_spec7.log`.

### DECISION (2026-08-16, user): STOP — record as project, no code change now
Two facts settle it: (1) the 35 GiB is genuinely needed — measured under-count = graph 11.18 +
scratch 5.64 + hi-conc activation 18.18 = **exactly 35**, so KV is **invariant** under any faithful
re-accounting (move a term out of the reserve, drop the reserve by the same amount → KV stays 32.3).
No KV to free on this box. (2) V2 has **none** of V1's throwaway-graph-profiling seam
(`cudagraph_dispatcher`/`_warmup_and_capture`/`CUDAGraphWrapper` pool-swap/`graph_capture`); it uses an
opaque `cudagraph_manager.capture()`. A faithful pre-sizing graph profile = reimplement throwaway
capture+teardown against V2, and V1's own `shutdown()` warns delayed ROCm graph destruction surfaces
HSA faults on next startup. So the "faithful profiling pass" is a **correctness/portability** change
(compute ~35 instead of hardcoding, to be right on other models/concurrencies) with **zero perf payoff
here** and real boot-fragility risk. **Retain `VLLM_KV_RUNTIME_RESERVE_GIB=35`.** PR1/PR2/PR3 (tasks
#88/#89/#90) stay recorded as the upstream project, gated on a future model/config change or an upstream
appetite for the V2 profiling reimplementation — not being implemented now.

## 2026-08-17 — Session: Codegen
- Extended the common eager fused all-reduce + RMSNorm helper to dispatch to AITER on ROCm, allowing K3 DSpark's existing unreduced row-parallel path to use the fused primitive without `torch.compile`.

## 2026-09-11 — Session: Codegen
- Comms vs B300 are structural (408 vs 147 launches), not per-kernel. First impl: wire AITER into eager `fused_allreduce_rms_norm` and use it on the K3 ROCm latent-MoE tail (the 92 CustomAR 1-stage calls). Same-stream, graph-safe. Did not raise the 2-stage cutoff, did not re-enable direct DCP, did not add aux-stream overlap (graph-capture hazard). Measure next: 1-stage CustomAR count should drop from 92 toward 0 and a fused AR+RMS kernel should appear at ~92/step.

## 2026-09-11 — Session: Codegen (ISL 99,757 A/B)
- Profiler-off conc-1 A/B of the fused latent AR+RMS vs the live unfused `k3-r72` baseline. Same recipe (DCP8, EP1, DSpark K=7, AL 3.84, LMCache DRAM). User content tokenized to exactly 99,757; chat template made `usage.prompt_tokens=99845` on both arms. OSL 512. `ms/step = 1000 * Δt / Δspec_decode_num_drafts`. Raw JSON: `/dev/shm/ab_isl99757/{A,B_first,B2}.json`.
- Like-for-like (prefix-cache hot): **A 25.17 ms/step** (window, 103 drafts) vs **B2 25.29 ms/step** (102 drafts). E2E 27.54 vs 27.44. Interval median 25.89 vs 25.90. Neutral — within ~0.1 ms, far below the hoped 0.5–1 ms. Arm B log confirmed `fused AR+RMSNorm on the latent`. Next lever is still deleting collectives (latent+shared tail fusion / GEMM-AR), not gluing RMS onto the same CustomAR.

## 2026-09-11 — Session: Codegen (packed CustomAR latent tail)
- Software stand-in for NVIDIA `KimiK3LatentMoETailOp`: decode-only pack of unreduced `[latent | shared]`, one CustomAR, RMS on the latent slice, full up-proj GEMM. Env `VLLM_ROCM_K3_PACKED_LATENT_TAIL`. Prefill stays sharded two-AR. Cannot fuse AR+RMS on the packed tensor. CPU identity-AR parity passed for M=1,5,8,16. Serve log confirmed both `fused AR+RMSNorm` (prefill) and `packed CustomAR` (decode).
- Like-for-like ISL 99,757 prefix-cache hot vs fused B2: **C2 26.74 ms/step** (window, 107 drafts) vs **B2 25.29**. E2E 29.24 vs 27.44. Interval median 26.00 vs 25.90. **Regression ~1.5 ms/step.** Cause: packed 168 KB is CustomAR 2-stage (lost 1-stage latent AR) plus full up-proj weight traffic on every rank. Default flipped to 0. Next lever is still a sharded GEMM + cheap second collective (Lamport/multicast), not a bigger all-reduce. Raw JSON: `/dev/shm/ab_isl99757/{C,C2}.json`.

## 2026-09-13 — Session: Codegen (AMD router/down-projection overlap)
- Ported NVIDIA K3's `_maybe_overlap_router_and_down_proj` schedule to AMD and validated full TP8/DCP8 graph capture plus profiler-off conc-1 at fixed content ISL 99,757. Four hot repeats averaged **25.37 ms/step** and **27.53 ms/step E2E**, versus fused baseline B2 **25.29** and **27.44**: neutral/slightly worse (~0.1 ms, within noise). Candidate was reverted; raw JSON: `/dev/shm/ab_isl99757/D{2,3,4,5}_router_overlap.json`.

## 2026-09-13 — Session: Codegen (TP8 shared-expert/routed-MoE overlap)
- A targeted #52033 A/B first showed that lifting only the ROCm TP profitability gate is insufficient for K3: `is_multistream_safe=False` because MXFP4 activation quantization is not represented by `moe_quant_config.quant_dtype`, so the trace remained serialized. The diagnostic K3 arm also bypassed this generic metadata check; this is safe for K3 latent MoE because shared experts read the original 7168-wide hidden states while routed experts consume a distinct 3584-wide down-projection. FULL_DECODE_ONLY HIP-graph capture and repeated 512-token generation completed without faults.
- Trace proof at M=8: shared GEMM/SiTU/GEMM ran on stream 4 while grouped-top-k/sort/routed experts ran on stream 1. Mean shared work was **1.49 ms/step**, with **1.16 ms/step (77.7%) directly overlapped**. Concurrent inflation was grouped-top-k **+0.229 ms**, sort **+0.032 ms**, routed GEMM1 **+0.075 ms**, routed GEMM2 **+0.118 ms** per step. More importantly, collective waiting became highly asymmetric: seven ranks accumulated ~8.0 ms/step in CustomAR kernels while one rank stayed at 1.49 ms, versus a balanced 1.52–1.84 ms serialized range.
- Matched profiler-off fixed-ISL B-A-B: pooled hot serialized baselines **25.607 ms/step** (`n=23`, median 25.517, SD 0.318); forced overlap **26.079 ms/step** (`n=11`, median 26.007, SD 0.207). Delta **+0.471 ms/step / +1.84% regression** (normal-approx 95% CI +0.293 to +0.650 ms). The ~1.16 ms hidden shared work is more than consumed by routing inflation, stream scheduling, and especially cross-rank collective wait. Keep ROCm TP8 overlap disabled. Raw JSON: `/dev/shm/ab_isl99757_shared/{SB*,SC*,SD*,SE*}.json`; traces: `/dev/shm/k3_shared_overlap_trace{,2}/`. The server was stopped and the installed baseline file restored.

## 2026-09-13 — Session: Codegen (MI355X CU scheduling feasibility)
- MI355X/gfx950 reports **256 active CUs (32 per XCD × 8 XCDs)**. ROCm 7.2.3 exposes queue-level CU affinity through `hipExtStreamCreateWithCUMask`; a 32-CU mask was created, read back with `hipExtStreamGetCUMask`, wrapped by `torch.cuda.ExternalStream`, and successfully used for a BF16 GEMM. This can cap the auxiliary shared branch while leaving the routed/default stream unrestricted, but it is not a true disjoint partition unless both branches move to complementary masked streams. Masks must preserve gfx950's contiguous CU-pair rule and be balanced across the eight XCDs.
- Per-stream CU masking is **not directly graph-safe in the current stack**: `hipExtStreamCreateWithCUMask` creates a blocking stream (flags 0) and has no nonblocking/priority variant. Raw HIP capture on that stream works in isolation, but PyTorch capture with normal default-stream dependencies fails with `hipErrorStreamCaptureImplicit`. `ROC_GLOBAL_CU_MASK` preserves ordinary graph-compatible streams but constrains the whole process, including communication and auxiliary queues, so it cannot partition K3's two branches within one TP rank.
- NVIDIA K3 MegaMoE (#53556 / DeepGEMM #416) uses a persistent heterogeneous task scheduler, not a fixed SM split. AITER's existing gfx950 FHMoE is not directly reusable: it assumes one common model width/output, SiLU, MXFP4 routed plus FP8 shared weights, and a DSV4-specific `(H=7168, I=384, E=385, topk=7)` contract. K3 needs separate `[M,3584]` routed and `[M,7168]` shared inputs/outputs, SiTU, MXFP4 routed experts, and BF16 shared weights.
- Decision gate: use a 16/32/64-CU-capped shared stream only in an eager microbenchmark to test whether CU starvation causes routed inflation and CustomAR skew; do not wire it into production graph capture. A graph-compatible low-priority ordinary stream is the smaller end-to-end scheduling experiment. If either confirms starvation, build a K3-specific persistent AITER scheduler with separate routed/shared task descriptors and buffers; dynamic task stealing is preferable to fixed in-kernel CU quotas.

## 2026-09-13 — Session: Codegen (K3 latent FHMoE kernel)
- Implemented the K3-specific FlyDSL heterogeneous two-stage scheduler (MXFP4 routed H=3584/I=384 plus BF16 shared H=7168/I=768, separate outputs). On MI355X, E=896/top-k=16 passed repeated M=1 and M=8 correctness runs at 0.147/0.149 ms; routed rel-L2 was 0.667%/0.655% versus standalone AITER (BF16 atomic-order difference), and shared rel-L2 was 0.408%/0.417%. Bounding the launch grid by the maximum active experts fixed a stage-2 fault caused by hundreds of inactive blocks. All 48 FHMoE contract/regression tests pass.
- The 317k-token IX hang was preceded by LMCache dying after expanding pinned DRAM to ~764 GiB; fixed the repro script so `TOTAL_CPU_DRAM_GB` can actually override its 803 GiB reference-box default.
- The corrected eager TP8/DCP8 run completed 3600 s with no collective hang: 100 successful measured requests, 0.99% request error rate, median ISL 121,996, median OSL 354.5, 5,193.55 input tok/s, 23.11 output tok/s, and 652 total tok/s/GPU. CUDA-graph replay remains disabled because the prototype latent FHMoE path faults on its first M=8 replay after long prefill.

## 2026-09-14 — Session: Codegen (K3 latent FHMoE graph safety)
- Root-caused the first M=8 replay fault to vLLM skipping the latent path during its eager graph warmup, which forced FlyDSL compilation and persistent-workspace allocation into capture. The adapter now warms each layer once outside capture while retaining the normal prefill fallback; the rebased exact-shape MI355X graph test also passes after changing replay to 128 distinct routes and churning 128 MiB of unrelated GPU memory (0.189 ms, routed/shared rel-L2 0.645%/0.389%). Full TP8 long-prefill validation is pending restaging after a host reboot left only 16/96 K3 weight shards in the volatile cache.
- Full graph-enabled TP8/DCP8 conc-1 IX validation completed after sizing stage-1 intermediate storage by padded sorted rows and retaining persistent routing buffers: 185/186 measured requests succeeded, no GPU/graph faults, 1,232 total tok/s/GPU, and 131.10 tok/s/user p90 full-response interactivity.
- M=8 stage-1 profiling found the latent heterogeneous GEMM dominant at 115 us; raising `persist_m` from 1 to 4 reduced it to 81 us and repeated graph latency from 0.178 ms to 0.157 ms (11.8%) with unchanged correctness. A 120 s TP8/DCP8 graph smoke passed without faults at 134.54 tok/s/user p90 full-response interactivity.
- Removed five redundant per-replay routing/sort metadata initialization launches by initializing graph-stable workspace tails once; the dynamic-routing graph correctness test passes at 0.139 ms. Shared stage-2 output zeroing remains required because that branch accumulates atomically.
- Replaced the wide-expert one-shot sorter's serial E>block prefix extension with chunked all-wave scans. At K3 M=8/E=897/topk=17, sorting fell from 11.68 to 7.93 us (32%), and full latent FHMoE measured 0.137 ms with graph replay and Opus/CK parity. A matched 120 s TP8/DCP8 smoke completed with 0/9 measured errors; its 117.28 tok/s/user average was neutral versus 118.04 baseline, while p90 was too noisy at nine samples (122.71 vs 134.54).

## 2026-09-15 — Session: Codegen (K3 latent FHMoE M-scaling gate)
- The conc-1 recipe-flag 3600 s run finished at **128.64 tok/s/user p90 interactivity** and 1380 tok/s/chip (196/208 measured, 0.51% errors, AL 3.8416, 80 `afp8_wfp4` dispatches, ISL med 161,001 / OSL med 654 all on pin). That is **−2.1% versus the 131.35 pin**, so the split-stream latent FHMoE is neutral-to-negative end to end despite winning ~3% on its own kernel (119.7 → 115.7 us).
- Tested the "larger M will pay off" hypothesis at op level before spending an e2e arm, because `_K3_DECODE_M = 8` in `aiter/latent_fhmoe_vllm.py` hard-gates the vLLM adapter: at conc-2 (M=16) and conc-4 (M=32) the shape check fails and both arms would have silently run the conventional path. New harness `op_tests/k3_latent_fhmoe_m_sweep.py`, one process per M (FlyDSL specializes runtime integers process-locally).
- Initial reference-only result (superseded below): M=8 FHMoE ~142 us vs `_routed_aiter_reference` + shared ~161 us (**1.13x**); M=16 ~177 vs ~163 us (**0.93x**). This did **not** justify a design conclusion: the reference used flydsl stage1/stage2 at `persist_m=1`, not vLLM's tuned production fused-MoE path, and M=16 had not received a production-routed or overlap-aware implementation.
- Corrected graph-replay comparison against `aiter.fused_moe` with K3 A8W4/SiTU plus the shared branch: sequential production surrogate was **91.5 us at M=8 / 134.6 us at M=16**. Reusing the tuned production routed path inside FHMoE and overlapping the shared branch measured **79.8 us / 122.7 us** after reducing shared stage1 to `TILE_N=32, num_waves=1`. This is a real **1.15x M=8 / 1.10x M=16 local win**. M=8 and M=16 graph correctness passed (routed rel-L2 0.673%/0.668%, shared 0.389%/0.399%), and alternating M=8/M=16 graph replay in one process completed safely.
- The vLLM adapter now accepts M in {8,16}, tracks eager preparation per M, and completed full TP8/DCP1 conc-2 graph capture on all ranks. A matched 120 s conc-2 A/B had identical 13-request workload (ISL med 105,652, OSL med 774, AL 3.8394, zero measured errors): conventional **166.40 tok/s/user, frITL p90 6.268 ms** versus split-stream FHMoE **159.71 tok/s/user, 6.497 ms**. The local 10% MoE win therefore does not yet survive TP8 execution; the +0.229 ms ITL regression is consistent with the previously observed separate-stream/collective-arrival interaction, not evidence that heterogeneous scheduling itself lacks value.
- Next direction: retain the production routed kernels and shared N32/W1 geometry, but replace coarse separate-stream overlap with a persistent/ticketed heterogeneous scheduler (or otherwise preserve TP collective arrival order). Do not interpret the earlier M=16 reference crossover as a final FHMoE scalability result.
- Follow-up on the existing single-grid `latent_heterogeneous` scheduler: rewired it behind `AITER_K3_LATENT_INTEGRATED=1`. Default M=8 graph correctness passes, but replay is **135.6 us**, slower than both hybrid (79.8 us) and sequential production surrogate (91.5 us). A geometry sweep found an apparent 102.9 us result (`S1 N32/P1`, `S2 N256/P1`), but the strengthened correctness test rejected it at **48.7% routed rel-L2**. M=16 default integrated execution either produced zero routed output in the E=1 case or GPU-faulted for E=896. The integrated path is therefore guarded to M=8 and remains experimental/default-off.
- The audit clarified that this existing persistent loop is static expert-id dispatch, not the fine-grained ticket/readiness scheduler needed to preserve production routed kernels. A real next implementation must dynamically schedule heterogeneous work (or defer synchronization into the latent tail); simply restoring the old integrated host wiring does not solve the performance or M>8 correctness problem.
