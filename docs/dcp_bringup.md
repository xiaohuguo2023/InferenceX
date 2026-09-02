# Kimi-K3 FP4 + DSpark + DCP8 bring-up (MI355X, TP8)

Everything needed to reproduce our DCP8 arm on another MI355X box and run the
InferenceX agentic benchmark against it. Companion doc:
[`dcp_algorithm_review.md`](dcp_algorithm_review.md).

## 1. Image

```
vllm/vllm-openai-rocm:nightly-1dc464d42681d22f38caf1fdc1eb632dc4421c45
```

vLLM `0.28.1rc1.dev108+g1dc464d42`. On this image the old fp8-asm recipe patch
chain (`_patch_fp8asm.py`, `_patch_fp8_prefill.py`, `_patch_ps_metadata16.py`,
`_patch_skip_k3_fp8_ps.py`, `_patch_wvsplitk.py`) is **native** — do not look for
those scripts, they were removed at the 0.27 bump.

Nightly images set `ENTRYPOINT vllm`, so override it:

```bash
docker run -d --name k3-dcp --entrypoint sleep \
  --device=/dev/kfd --device=/dev/dri --network=host --ipc=host --pid=host \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /path/to/InferenceX-dspv4:/workspace -v /dev/shm:/dev/shm \
  vllm/vllm-openai-rocm:nightly-1dc464d42681d22f38caf1fdc1eb632dc4421c45 infinity
```

## 2. `/opt/aiter-local` (mandatory, before any patching)

Stock aiter ships the `cprr` MLA asm kernels for bf16 only, so the fp8 DCP decode
path SIGABRTs in `get_heuristic_kernel_mla` without our local build. It is a
**binary artifact and is not in this repo.** Transplant it from a container built
against the same torch/HIP/triton rather than rebuilding:

```bash
docker cp <donor-ctr>:/opt/aiter-local /tmp/aiter-local
docker cp /tmp/aiter-local k3-dcp:/opt/
docker exec k3-dcp python3 -c "import aiter; print(aiter.__file__)"   # -> /opt/aiter-local/...
```

It also carries `aiter/configs/merged_bf16_tuned_gemm.csv`, which the serve script
wires up via `AITER_CONFIG_GEMM_BF16`. Without it the conc-1 GEMMs fall back to
untuned configs.

## 3. Patch chain, in order — gate on each script's own success print

```bash
V=/usr/local/lib/python3.12/dist-packages/vllm
cd /workspace
python3 _patch_draft_causal.py                              # checkpoint, NOT vLLM — see 3a
python3 _port_dcp_nightly_ba07e4a4.py                       # hunks A-D: DCP MLA plumbing
python3 _patch_pad128.py                                    # pad 96 heads -> 128 (not fold)
python3 k3_dcp_direct_hip/_patch_dcp_skip_multicast_probe.py
```

All four are anchor-guarded and idempotent; each takes `--revert`.

### 3a. Force the draft causal — the one patch that lives outside vLLM

The DSpark draft checkpoint ships with **no** `dflash_config`. vLLM resolves
per-layer causality in `models/qwen3_dflash.py::_dflash_layer_causal` in this
order — `config.is_causal`, then `dflash_config["causal"]`, then
`layer_types[i] == "sliding_attention"`. This checkpoint has none of the three,
so every layer resolves **non-causal**, `dflash_has_any_non_causal()` returns
True, and the draft is routed off the fp8 asm path. `_serve_k3_dcp_test.sh`
refuses to launch in that state:

```
!! draft must be forced causal before serve
```

`_patch_draft_causal.py` writes `"dflash_config": {"causal": true}` into
`/dev/shm/hf-cache/models--Inferact--Kimi-K3-DSpark/snapshots/*/config.json`.
That path is a symlink into the HF `blobs/` store, so the script **replaces the
symlink with a real file** rather than writing through it — writing through
would corrupt the shared content-addressed blob.

**This is a checkpoint edit, so it does not survive a `/dev/shm` wipe or a
re-download**, and re-downloads are routine here (`/dev/shm` has no fstab entry
and resets on reboot). Nothing in the vLLM patch chain restores it. Re-run the
script after any re-download; `--check` exits non-zero if it is missing, which
is the cheap thing to put in front of a batch run. A silent re-download is
exactly what killed the 2026-09-01 DCP A/B, 71 s in.

### 3b. `K3-DCP-DRAFT-REPL` (optional — this is the A/B knob)

`_patch_dcp_draft_repl.diff` makes the DSpark **draft** KV group run replicated
(cp=1) while the target stays DCP8-sharded. Upstream's invariant (vLLM #52188 /
#53598) is that the draft group keeps the process DCP size, so this is a knowing
divergence. Apply it to compare:

```bash
cd $V && patch -p1 < /workspace/_patch_dcp_draft_repl.diff        # apply
cd $V && patch -p1 -R < /workspace/_patch_dcp_draft_repl.diff     # revert
find $V -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
grep -rl K3-DCP-DRAFT-REPL $V --include=*.py | grep -v __pycache__ | wc -l   # -> 9
```

Touches 9 files / 12 marker lines, in two halves.

*KV management* — where the draft's blocks live: `v1/kv_cache_interface.py` (spec
runs at cp=1), `v1/worker/gpu/block_table.py` (per-group `cp_sizes` + Triton
kernel), `v1/worker/gpu/model_runner.py` (builds `cp_sizes_per_group`),
`v1/worker/gpu/spec_decode/dflash/speculator.py` (`cp_sizes[gid]`, not the global
`cp_size`), `v1/worker/gpu/attn_utils.py` (per-group `dcp_local_seq_lens`), plus
`v1/core/{kv_cache_coordinator,kv_cache_utils,single_type_kv_cache_manager}.py` on
the lookup/allocation paths.

*Attention* — how the draft **reads** those blocks:
`model_executor/layers/attention/mla_attention.py`, 2 sites. Both halves are
required; see the defect note below for what happens with only the first.

The diff is generated against the *unpatched* 1dc464d files; if the image moves,
regenerate rather than force-applying.

#### What this patch does and does not change

Every hunk is gated on `non_causal_multi_token_decode`, which is set at exactly
one site (`models/kimi_k3/nvidia/dspark_mla.py:81`) and marks the DSpark draft
group. **The target attention group is untouched in both arms.**

**This patch does move the prefix-cache rate, decisively — measured 2026-09-02.**
A full 9-point A/B on the long-context sweep, same image, same serve config, same
`GPU KV cache size: 12,685,809 tokens` in both arms:

| conc | 1 | 2 | 4 | 8 | 12 | 16 | 24 | 32 | 48 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cache% sharded (upstream) | 90.2 | 90.2 | 90.2 | 90.2 | 90.2 | **15.0** | 0.7 | 0.5 | 0.4 |
| cache% repl (this patch)  | 90.2 | 90.2 | 90.2 | 90.2 | 90.2 | **90.2** | 90.2 | 89.7 | 89.5 |

The upstream arm falls off the documented conc-12 -> 16 cliff and never recovers;
the patched arm holds ~90% to conc-48. KV capacity is identical in both, so this
is block *granularity*, not capacity.

An earlier source-level argument in this doc claimed the patch could not matter,
on the grounds that `MLAAttentionSpec` subclasses `FullAttentionSpec` so the
target group is DCP-scaled in both arms and
`HybridKVCacheCoordinator.find_longest_cache_hit` takes the **min across groups**,
flooring the hit at the target's 12,288-token span. **That argument was wrong** --
the measurement above supersedes it. Something downstream of the per-group block
size carries the effect; the exact mechanism is still unpinned, but the effect is
no longer in question.

#### Fixed defect: the KV-management half alone collapses acceptance length

Worth reading even though it is fixed -- it is the third instance of the same
class of bug, and the next per-group CP change will hit it again.

The 2026-09-02 A/B measured **AL 1.03 in the patched arm at every concurrency**,
against 2.37-2.41 upstream, with position-0 draft acceptance of 2.5% vs 67.7%.
That is a broken draft, not low acceptance -- a draft that merely disagreed would
still get token 0 right most of the time.

Root cause: the patch made the draft group replicated everywhere its **blocks**
are managed, but the **attention** code still read the global DCP size:

- `mla_attention.py:2274` -- the metadata builder does
  `self.dcp_world_size = get_dcp_group().world_size`, i.e. 8, for *every* group.
  It already reads `non_causal_multi_token_decode` off the group's spec 27 lines
  earlier, and never used it here.
- `mla_attention.py:3159` -- the attention impl does the same from
  `parallel_config.decode_context_parallel_size`.

So the draft attended with round-robin CP indexing and a cross-rank LSE combine
over a block table in which **every rank already holds the full KV**. Each rank
read a different wrong eighth and the combine summed them.

Tell-tale: `rocm_aiter_mla.py:1977-1986` already carried comments reading *"the
replicated non-causal draft group has its `impl.dcp_world_size` forced to 1 by the
layer AFTER construction"*. That forcing site had never been written -- the
backend was patched for a mechanism that did not exist.

Fix (the `mla_attention.py` half of the diff, both gated on
`non_causal_multi_token_decode`):

- builder `__init__`: force `self.dcp_world_size = 1` **before** the DCP-derived
  quantities below it (`dcp_virtual_block_size`, the chunked-prefill workspace
  split, the decode `seq_lens`/`max_seq_len` rescale).
- layer `__init__`, immediately after `self.impl = impl_cls(...)`: force
  `impl.dcp_world_size = 1`, `dcp_rank = 0`, and reset the cached
  `_decode_num_heads`. `rocm_aiter_mla` reads `dcp_world_size` live for
  `dcp_heads`, so only the cached values need resetting.

Verified 2026-09-02 03:50 on a conc-4 / 20-request probe: **AL 1.03 -> 2.12**,
position-0 acceptance 2.5% -> **68.2%** (sharded reference 67.7%), and the
histogram decays geometrically instead of sitting flat in noise:

| | pos-0 | 1 | 2 | 3 | 4 | 5 | 6 | AL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| broken | 3,670 | 326 | 138 | 118 | 113 | 2 | 0 | 1.03 |
| fixed | 4,043 | 2,327 | 223 | 40 | 17 | 9 | 8 | 2.12 |

Read 2.12 as "the mechanism is fixed", not as a parity number: it is a 20-request
probe with warmup inside the numerator, against sweep points that ran 900 s. Full
parity vs the sharded arm's 2.37-2.41 is unmeasured.

**The AL/ITL/throughput half of the 2026-09-02 A/B predates this fix and is void;
the cache% half stands, because prefix-cache accounting does not depend on the
draft reading the right KV.** A first attempt that only swapped the
`dcp_local_seq_lens` tensor per group (the `attn_utils.py` hunk) was measured
ineffective on GPU -- it is correct and kept, but it is not the fix, because the
builder rescales `seq_lens` and `max_seq_len` from `dcp_world_size` regardless of
which tensor it was handed.

**Generalisable rule:** when adding a per-group CP degree, audit every consumer of
the *global* `dcp_size`/`cp_size` in both the KV-management path **and** the
attention path. Five sites so far: block-table row width, slot-mapping write,
speculator kernel arg, attention local seq len, and the attention builder+impl
world size.

Still worth measuring on their own: decode ITL and the KV capacity cost of
replicating the draft (5 MLA layers x 576 B/token: ~2,880 B/tok/rank replicated vs
~360 sharded).

#### Pre-check before spending a GPU-hour

Resolve the disagreement above by measuring the geometry, not by running an
hour-long benchmark. Under **each** arm, on a booted serve, record the per-group
block sizes and the cache counters:

```bash
docker exec k3-dcp bash -lc '
  curl -s http://127.0.0.1:8890/metrics |
    grep -E "prefix_cache_(queries|hits)_total|gpu_cache_usage"'
grep -iE "GPU KV cache size|Maximum concurrency|block_size|scheduler_block_size" \
  /tmp/serve_k3_dcp7.log
```

Read three numbers per arm: the **target** group's effective block span, the
**scheduler** block size (the LCM across groups), and the prefix-cache hit ratio.
If the target span is 12,288 in both arms but the hit ratio still moves, the
scheduler LCM is the mechanism — which would also mean the effect is real and the
agentic run is worth doing. If nothing moves, the question is closed for free.

## 4. Serve

```bash
docker exec k3-dcp bash -lc 'cd /workspace &&
  K3_DCP_ALLOW_FULL_CUDAGRAPH=1 NUM_SPEC=7 PORT=8890 \
  GPU_MEM=0.95 MAX_NUM_SEQS=64 MNBT=16384 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
  bash _serve_k3_dcp_test.sh'
```

Defaults: `DCP_SIZE=8`, `DCP_BACKEND=a2a`, `CP_INTERLEAVE=1`. **Keep `a2a`** — the
`symm_mem` backend builds an 8x8x3 cross-process dma-buf mesh it never frees, which
strands refcount-7 buffers and wedges the whole box on teardown.

The script preflights the box before loading weights (`_gpu_preflight.py` for H2D,
`_rccl_init_probe.py` for an 8-rank collective). A degraded box turns weight
loading into ~500 s/shard, which reads as a hang and has repeatedly been
misdiagnosed as DCP. Set `SKIP_PREFLIGHT=1` / `SKIP_COLLECTIVE_CHECK=1` to bypass.

## 5. Benchmarks

Long-context microbench (ISL 68,089 / OSL 350, conc 48->1):

```bash
docker exec k3-dcp bash -lc 'cd /workspace &&
  AIPERF=/workspace/.aiperf_v012_dev193/bin/aiperf ROOT=/workspace/k3_dcp8 PORT=8890 \
  bash _dspark_longctx_bench.sh'
python3 _dspark_perf_diag.py /workspace/k3_dcp8 --tp 8
```

### InferenceX agentic

The canonical definition of this benchmark is **not** a hand-written aiperf
command. It is `benchmarks/benchmark_lib.sh` (`build_agentic_replay_cmd`, around
line 3246) driven by the recipe block in `configs/amd-master.yaml`. Read those
first; the flag set below is derived from them and will drift.

**Use the pinned aiperf.** CI builds it from the `utils/aiperf` submodule with
`uv`, editable, on Python 3.11 — the "version" of the benchmark *is* that
submodule commit. Currently `754356e9` = `agentx-v1.0.5` (package version
`0.12.0`). Do not use the ad-hoc `/workspace/.aiperf_*` venvs.

```bash
git submodule update --init utils/aiperf     # -> agentx-v1.0.5
docker exec k3-dcp bash -lc '
  uv venv --python 3.11 /tmp/aiperf-venv &&
  uv pip install --python /tmp/aiperf-venv/bin/python -e /workspace/utils/aiperf'
```

**Do not use `_run_agentic_dspark.sh` for a DCP run** — it tears down with
`kill -9`, which is exactly what strands the dma-bufs. Drive aiperf directly
against the already-warm serve from §4, once per concurrency.

```bash
CONC=1
ROOT=/workspace/k3_dcp8_ns7_ixci_c$CONC

docker exec k3-dcp bash -lc "cd /workspace &&
  export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800 \
         AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800 \
         AIPERF_UI_REALTIME_METRICS_ENABLED=true \
         AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0 &&
  /tmp/aiperf-venv/bin/aiperf profile --scenario inferencex-agentx-mvp \
    --public-dataset semianalysis_cc_traces_weka_062126 \
    --url http://localhost:8890 --endpoint /v1/chat/completions \
    --endpoint-type chat --streaming \
    --model moonshotai/Kimi-K3 --tokenizer moonshotai/Kimi-K3 \
    --tokenizer-trust-remote-code \
    --concurrency $CONC --benchmark-duration 3600 \
    --stats-interval 30 --random-seed 42 \
    --failed-request-threshold 0.10 \
    --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
    --warmup-requests-per-lane 10 --warmup-grace-period 1800 \
    --trace-idle-gap-cap-seconds 300 --use-server-token-count \
    --no-gpu-telemetry --max-context-length \$MAX_MODEL_LEN \
    --num-dataset-entries 393 --slice-duration 1.0 \
    --output-artifact-dir $ROOT/aiperf_artifacts"
```

Rules that decide whether the run counts:

* **`--random-seed 42`**, not 0. **`--benchmark-duration 3600`** (the
  `benchmark-tmpl.yml` default), not 900. **`--warmup-grace-period 1800`**, not 600.
* **Never pass `--unsafe-override` on a run you intend to report.** CI adds it only
  when `duration < 900` or `AIPERF_UNSAFE_OVERRIDE=true`, and it stamps
  `submission_valid: false`.
* Since `agentx-v1.0.4`/`v1.0.5` the scenario enforces a post-run coverage gate:
  TTFT **or** ITL observations must span ≥95% of the profiling phase, else the run
  exits non-zero with `insufficient_profile_metric_coverage`. Both of those
  releases exist specifically to stop low-concurrency runs failing spuriously, so
  conc-1/conc-4 arms need `v1.0.5` to be judged fairly.
* Pass the **HF model id** as `--tokenizer`, not a `/dev/shm` snapshot path.
  Snapshot hashes change on every re-download.
* The scenario silently rewrites four settings — `timing_mode=agentic_replay`,
  `extra_inputs.ignore_eos=true`, `--cache-bust=first_turn_prefix`,
  `--system-idle-gap-cap-seconds=10.0`. Expect them in the log; not errors.
* Results: `$ROOT/aiperf_artifacts/profile_export_console.txt`. The exact CLI of any
  past run is preserved under `"cli_command"` in `profile_export_aiperf.json`,
  alongside `"aiperf_version"` — check both before trusting a comparison.

Recipe shape, from `configs/amd-master.yaml`
(`kimik3-fp4-mi355x-vllm-agentic-mtp`, same image as §1):

| | |
|---|---|
| conc 1 | `kv-offloading: none`, `dram-utilization: 0.60` |
| conc 4, 8, 10, 12, 14 | `kv-offloading: dram` via LMCache `0.5.5.dev60+rocm7.2` |

The vLLM recipe carries **no `dcp-size`** at any concurrency. The companion ATOM
recipe (`kimik3-fp4-mi355x-atom-agentic-mtp`) uses `dcp-size: 8` only from
**conc-8 upward**, on the stated reasoning that decode there is KV-bandwidth-bound
over 100k+ contexts. So conc-1/conc-4 is outside the regime anyone claims DCP helps
— worth knowing before reading a DCP conc-1 TTFT regression as a defect.

Agentic AL is ~1.4–1.6 for both arms. Do **not** compare it to the ~2.4 of the
long-context microbench — that workload is `ignore_eos` synthetic.

### Accuracy gate

`bash _gsm8k_k3.sh`, in **block** acceptance mode.

## 6. Teardown — SIGTERM only, never `-9`

```bash
docker exec k3-dcp bash -lc '
  for p in $(pgrep -f "vllm|multiprocessing.spawn|multiprocessing-fork|EngineCore|VllmWorker"); do
    c=$(cat /proc/$p/comm 2>/dev/null); case "$c" in bash|sh|pgrep|docker|"") continue;; esac
    kill -TERM "$p" 2>/dev/null
  done'
```

Wrap every GPU run in `timeout --signal=TERM --kill-after=600s <secs> ...` —
**600 s, not 30 s**: the ordered teardown synchronises and barriers across 8 ranks
and then waits for KFD restore work to drain; 30 s guarantees a SIGKILL mid-teardown.

Damage meter: `awk '/^amdgpu /{print $3}' /proc/modules`. Record it before every
run. If it does not return to baseline after the processes are gone, dma-bufs were
stranded — that does **not** self-clear and it blocks `modprobe -r` (the leak
blocks its own remedy). Report the number rather than escalating.

## 7. Known result on this configuration

> **Caveat: the recorded numbers below are not a valid AgentX submission.** They
> were produced with aiperf **0.8.0**, which predates the scenario validity gate
> entirely — `profile_export_aiperf.json` carries no `submission_valid` field at
> all. They also used `--random-seed 0` (canonical is 42), `--benchmark-duration
> 900` (canonical is 3600), `--warmup-grace-period 600` (canonical is 1800), and
> passed `--unsafe-override` unnecessarily. The non-DCP baseline they are compared
> against ran the canonical 3600 s. Treat the direction as informative and re-run
> both arms under §5 before reporting anything.


DCP8 + DSpark nspec-7 runs the IX agentic benchmark cleanly, at throughput parity
(+3.6% output tok/s at conc-4), with acceptance slightly *better* than the non-DCP
baseline. The cost is TTFT, and the cause is prefix-cache hit rate — 73-78% vs
93-96% — because the DCP attention group's block size is 8x coarser, so every
request tails into a partial 12,288-token block instead of a 1,536-token one. On
the pool-of-1 long-context microbench that same effect is worth only ~2.25 pp; on
varied-length agentic traces it is ~20 pp. That gap is the single biggest DCP lever
for the agentic story.

Two candidate levers, neither yet demonstrated on the agentic corpus:

* Upstream's replay-boundary retention work — vLLM #51295, #50897, #53917, all
  unmerged as of 2026-09-01. #53598's own PR body reports 15.60% actual hits
  without retention against 71.90% with it.
* `K3-DCP-DRAFT-REPL`, whose applicability here is **open** — see §3b. It swings
  the long-context conc-16 cache rate by 75 pp, but the source read predicts no
  effect on the target group's granularity. Settle that with the §3b pre-check
  before spending agentic GPU-hours on it.

### How to evaluate `K3-DCP-DRAFT-REPL`

Use the long-context sweep (§5), not the agentic benchmark: fixed shape, pool of 1,
deterministic, so a small delta is resolvable. Read acceptance length, decode ITL,
and each arm's "GPU KV cache size" for the capacity cost.

Reserve the agentic benchmark for confirming that a win transfers, and then run it
where the recipe actually puts DCP — **conc-8 and up, with LMCache DRAM offload** —
with at least two repeats per arm. A single conc-1 agentic point cannot decide this:
it hands the replicated draft all of its benefit and none of its cost, because the
8x draft KV only bites when capacity is tight.
