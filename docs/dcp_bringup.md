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
python3 _port_dcp_nightly_ba07e4a4.py                       # hunks A-D: DCP MLA plumbing
python3 _patch_pad128.py                                    # pad 96 heads -> 128 (not fold)
python3 k3_dcp_direct_hip/_patch_dcp_skip_multicast_probe.py
```

All three are anchor-guarded and idempotent; each takes `--revert`.

### 3b. `K3-DCP-DRAFT-REPL` (optional — this is the A/B knob)

`_patch_dcp_draft_repl.diff` makes the DSpark **draft** KV group run replicated
(cp=1) while the target stays DCP8-sharded. Upstream's invariant (vLLM #52188 /
#53598) is that the draft group keeps the process DCP size, so this is a knowing
divergence — it exists because DCP's 8x-scaled block granularity (1,536 -> 12,288
tokens) is what costs the agentic prefix-cache hit rate. Apply it to compare:

```bash
cd $V && patch -p1 < /workspace/_patch_dcp_draft_repl.diff        # apply
cd $V && patch -p1 -R < /workspace/_patch_dcp_draft_repl.diff     # revert
find $V -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
grep -rl K3-DCP-DRAFT-REPL $V --include=*.py | grep -v __pycache__ | wc -l   # -> 6
```

Touches 6 files / 9 marker lines: `v1/kv_cache_interface.py` (spec runs at cp=1),
`v1/worker/gpu/block_table.py` (per-group `cp_sizes` + Triton kernel),
`v1/worker/gpu/model_runner.py` (builds `cp_sizes_per_group`),
`v1/worker/gpu/spec_decode/dflash/speculator.py` (`cp_sizes[gid]`, not the global
`cp_size`), plus `v1/core/{kv_cache_coordinator,kv_cache_utils,
single_type_kv_cache_manager}.py` on the lookup/allocation paths.

The diff is generated against the *unpatched* 1dc464d files; if the image moves,
regenerate rather than force-applying.

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

**Do not use `_run_agentic_dspark.sh` for a DCP run** — it tears down with
`kill -9`, which is exactly what strands the dma-bufs. Drive aiperf directly
against the already-warm serve from §4, once per concurrency.

Use **aiperf 0.8.0** (`/workspace/.aiperf_venv`). 0.12.0 dropped
`--warmup-requests-per-lane`, and the `.aiperf_v1_0_1` / `.aiperf_818c3a5a` venvs
have broken interpreters. Note this is a *different* aiperf from the long-context
sweep above, which pins `v012_dev193`.

```bash
CONC=1                                  # we ran 1 and 4
ROOT=/workspace/k3_dcp8_ns7_ixci_c$CONC
TOK=$(ls -d /dev/shm/hf-cache/models--moonshotai--Kimi-K3/snapshots/*/ | head -1)

docker exec k3-dcp bash -lc "cd /workspace &&
  /workspace/.aiperf_venv/bin/aiperf profile \
    --scenario 'inferencex-agentx-mvp' \
    --public-dataset 'semianalysis_cc_traces_weka_062126' \
    --url 'http://localhost:8890' --endpoint '/v1/chat/completions' \
    --endpoint-type 'chat' --streaming --model 'Kimi-K3' \
    --concurrency $CONC --benchmark-duration 900 \
    --unsafe-override --stats-interval 30 --random-seed 0 \
    --failed-request-threshold 0.1 \
    --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
    --warmup-requests-per-lane 10 --warmup-grace-period 600 \
    --trace-idle-gap-cap-seconds 300 --use-server-token-count \
    --tokenizer '$TOK' --tokenizer-trust-remote-code \
    --no-gpu-telemetry --num-dataset-entries 393 --slice-duration 1.0 \
    --output-artifact-dir '$ROOT/aiperf_artifacts'"
```

Gotchas:

* The scenario **rejects `--benchmark-duration` below 900 s.** Budget ~25 min per
  point including warmup.
* **Resolve the tokenizer snapshot at run time**, as above. Older scripts hardcode
  `9f62e4e9…`, which no longer exists after a `/dev/shm` wipe; the live one at the
  time of our run was `a590ce09…`.
* The scenario silently rewrites four flags — `timing_mode=agentic_replay`,
  `extra_inputs.ignore_eos=true`, `--cache-bust=first_turn_prefix`,
  `--system-idle-gap-cap-seconds=10.0`. Expect them in the log; they are not errors.
* Read results from `$ROOT/aiperf_artifacts/profile_export_console.txt`; the exact
  CLI of any past run is preserved under `"cli_command"` in
  `profile_export_aiperf.json`.
* Compare against a **non-DCP control of the same duration and seed**. Our recorded
  baseline ran 3600 s against DCP's 900 s, so per-request prompt lengths differ;
  the conclusion survived a length-matched slice, but the control was never run
  formally.
* Agentic AL is ~1.4–1.6 for both arms. Do **not** compare it to the ~2.4 of the
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

DCP8 + DSpark nspec-7 runs the IX agentic benchmark cleanly, at throughput parity
(+3.6% output tok/s at conc-4), with acceptance slightly *better* than the non-DCP
baseline. The cost is TTFT, and the cause is prefix-cache hit rate — 73-78% vs
93-96% — because the DCP attention group's block size is 8x coarser, so every
request tails into a partial 12,288-token block instead of a 1,536-token one. On
the pool-of-1 long-context microbench that same effect is worth only ~2.25 pp; on
varied-length agentic traces it is ~20 pp. That gap is the single biggest DCP lever
for the agentic story, and it is what `K3-DCP-DRAFT-REPL` targets.
