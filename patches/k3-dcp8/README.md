# Kimi-K3 DCP8 + DSpark repro bundle (asm fp8 MLA path)

Everything needed to reproduce the green DCP8 + DSpark configuration on a second MI355X
box. Five vLLM patches, two aiter patches, and the tuned-GEMM CSV they depend on.

Measured on this configuration: full long-context sweep at concurrency 48→1, 9/9 `rc=0`,
acceptance length 2.39–2.43 at every point, ITL within ±7% of the non-DCP baseline.

## Pins

| | |
|---|---|
| vLLM image | `vllm/vllm-openai-rocm:nightly-73029d42441321b631779db3475031f5ec26dd6c` |
| vLLM version | `0.28.1rc1.dev278+g73029d424` |
| aiter base commit | `55dbc4f475da26c23cdaf73ce6ed38342a2d7f83` |

The nightly image's `ENTRYPOINT` is `vllm`, so start the container with
`--entrypoint sleep` or it will try to serve immediately.

## 1. vLLM patches

All five apply with `-p1` from the installed vllm package root. Order does not matter —
they touch five different files.

```bash
cd /usr/local/lib/python3.12/dist-packages/vllm
for f in scheduler config cp_common speculator rocm_aiter_mla; do
    patch -p1 < /path/to/patches/k3-dcp8/vllm/$f.patch
done
```

Confirm each landed (a *failing* reverse dry-run means not applied):

```bash
for f in scheduler config cp_common speculator rocm_aiter_mla; do
    printf "%-18s " "$f"
    patch --dry-run -R -p1 < /path/to/patches/k3-dcp8/vllm/$f.patch >/dev/null 2>&1 \
        && echo applied || echo NOT-APPLIED
done
```

| patch | target | what it does |
|---|---|---|
| `scheduler.patch` | `distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` | stops the full-attention EAGLE group from vetoing prefixes of ≤1 chunk (the unconditional `num_hit_chunks -= 1` is only correct for the sliding-window path) |
| `config.patch` | `model_executor/models/config.py` | defaults Kimi-K3 to the `a2a` DCP combine |
| `cp_common.patch` | `v1/attention/ops/cp_common.py` | ordered symmetric-memory teardown, plus skipping the NVLS multicast probe on ROCm |
| `speculator.patch` | `v1/worker/gpu/spec_decode/dflash/speculator.py` | syncs and barriers ranks before speculator cudagraph capture |
| `rocm_aiter_mla.patch` | `v1/attention/backends/mla/rocm_aiter_mla.py` | the asm round-robin-CP route for DCP multi-token verify, the 96→128 native-tile pad, and the split-cap plumbing |

`rocm_aiter_mla.patch` is the substantial one (~+352/−9, 20 hunks). Without it, stock vLLM
runs the entire DCP decode on Triton, which is the route we rejected on measured
performance.

There is no `rocm.patch` here on purpose. Upstream removed the blanket DCP→PIECEWISE
cudagraph downgrade, so DCP with FULL cudagraphs is stock behaviour now.

## 2. aiter

The `.cu` change means a **source rebuild is required** — installing a wheel will not pick
it up.

```bash
cd /path/to/aiter
git checkout 55dbc4f475da26c23cdaf73ce6ed38342a2d7f83
git apply /path/to/patches/k3-dcp8/aiter/0001-k3-dcp8-code.patch
git apply /path/to/patches/k3-dcp8/aiter/0002-k3-tuned-gemm-csv.patch

export PREBUILD_KERNELS=0
export AITER_USE_SYSTEM_TRITON=1
pip uninstall -y aiter amd-aiter
rm -rf ~/.aiter
pip install -r requirements.txt
python setup.py develop
```

Build **inside the target container** so aiter compiles against that container's torch and
ROCm.

| patch | files | size | what it does |
|---|---|---|---|
| `0001-k3-dcp8-code.patch` | `aiter/mla.py`, `aiter/ops/attention.py`, `csrc/py_itfs_cu/asm_gemm_a16w16.cu` | +50/−4 | fp8 MLA block-N lookup keys; the tight split-tile bound (`max`→`min`) that reclaims MLA reduce scratch from 9.35 GiB to 2.38 GiB; and a split-K guard for the ASM a16w16 kernels under cudagraph replay |
| `0002-k3-tuned-gemm-csv.patch` | 6 CSVs under `aiter/configs/model_configs/` | +765/−694 | tuned GEMM rows, closing 371 conc-1 tuned-config misses |

`merged_bf16_tuned_gemm.csv` in this directory is the folded CSV that
`AITER_CONFIG_GEMM_BF16` must point at. Copy it to
`aiter/configs/merged_bf16_tuned_gemm.csv` (it is untracked upstream, not produced by
either patch).

**The tight split-tile bound and `rocm_aiter_mla.patch` are a matched pair.** The bound is
only sound because vLLM passes the same `max_split_per_batch` at metadata-build time. Apply
one without the other and the non-cprr path overflows its reduce scratch by ~7.7×, which
faults the GPU rather than raising.

## 3. Environment and serve flags

```bash
export AITER_CONFIG_GEMM_BF16=/path/to/aiter/configs/merged_bf16_tuned_gemm.csv
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_AITER_MLA_DCP_VERIFY=asm   # default; the production route
```

`VLLM_ROCM_AITER_MLA_DCP_VERIFY` also accepts `segmented`, `segmented:64` and `asm:64` for
routing the draft and target independently. Those are **diagnostic bisects only** — `asm`
is the path everything is tuned against.

DCP serve flags:

```
--decode-context-parallel-size 8
--dcp-comm-backend a2a
--cp-kv-cache-interleave-size 1
```

`a2a` is deliberate: it is RCCL-only, needs no symmetric memory, and leaves no orphaned
dma-bufs behind.

Mandated benchmark config (do not vary these when comparing against our numbers):
`GPU_MEM=0.95`, `MAX_NUM_SEQS=64`, `MNBT=16384`, `CUDAGRAPH_MODE=FULL_AND_PIECEWISE`, KV
cache pinned at 32 GiB. `MAX_NUM_SEQS` must be ≥ the top concurrency or the high points
silently cap.

## 4. Verifying the bundle landed

Cheapest end-to-end check is acceptance length at concurrency 1 — it should be ~2.4. An AL
near 1.0 means the draft is proposing garbage and something in the DCP path is wrong.

Accuracy gate is GSM8K in **block** mode (expect ~0.963–0.969). Note that GSM8K alone
cannot catch a broken draft: it scored 0.9674 in a run where AL had collapsed to 1.03,
because a perfect target simply rejects every bad proposal. Always check AL too.

## Teardown — read this before killing anything

Shut the server down with a **graceful SIGTERM to the explicit pid**. Never SIGKILL, and
never `pkill -9 python3`.

A hard kill strands exported symmetric-memory buffers at refcount 7 with live eviction
fences. That blocks KFD's delayed restore, wedges the exiting rank inside the *global*
`mmu_notifier` SRCU section, and queues every other exiting process on the box behind it —
including other users'. `pkill "vllm serve"` does not free VRAM. If VRAM stays pinned,
`docker restart` on the container reaps the zombies.

On this class of box, GPU reset and reboot do not recover it.
