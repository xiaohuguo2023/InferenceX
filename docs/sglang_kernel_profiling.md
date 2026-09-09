# SGLang Kernel Profiling on MI355X (how it works, and how to see decode kernels)

Verified against the DeepSeek-V4 SGLang image
`lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260706` (sglang
`0.5.14.dev20260706+g80decc78ec`), source at
`/sgl-workspace/sglang/python/sglang`.

## TL;DR

- SGLang has **no `--profiler-config`-style flag** like vLLM. Profiling is
  **env var + runtime HTTP endpoints**, plus one capture-time serve flag.
- There are **two** mechanisms, and they capture different things:
  1. **Runtime stage profiler** — `SGLANG_TORCH_PROFILER_DIR` + `/start_profile`
     (this is what the `PROFILE=1` / `--profile` benchmark path drives). Dumps
     per-stage **prefill (EXTEND)** and **decode** traces. **Decode kernels are
     hidden** because decode runs under a full CUDA graph, so the trace only
     shows the graph *replay* (memcpy/launch).
  2. **CUDA-graph capture profiler** — serve flag `--enable-profile-cuda-graph`
     (+ opt-in env `SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE=1`). Profiles the
     decode graph **during capture** (eager, `record_shapes=True`), so every
     decode kernel is visible by name + input shape + CUDA time. **This is the
     way to get the decode kernel breakdown.**

## Mechanism A — runtime stage profiler (the `job.slurm` `PROFILE` path)

Flow:

1. `SGLANG_TORCH_PROFILER_DIR=<dir>` tells the server where to dump traces
   (default `/tmp`). Read in `sglang/profiler.py:18`, `srt/environ.py:270`,
   `srt/managers/scheduler_components/profiler_manager.py:122`.
2. `PROFILE=1` makes `run_benchmark_serving` add `--profile`; the benchmark
   client then calls the server endpoints **`/start_profile`** and
   **`/stop_profile`** (`srt/entrypoints/http_server.py:1050`, `:1061`).
3. `ProfileManager.configure` (`srt/utils/profile_utils.py`) profiles **by
   stage**: `profile_stages=["prefill","decode"]`, `activities=["CPU","GPU"]`,
   with `record_shapes`, `with_stack`, and `num_steps` options carried on the
   `ProfileReq` body.

Output: per-rank, per-stage chrome traces, e.g.
`…-TP-0-DP-0-EXTEND.trace.json.gz` (prefill) and `…-TP-0-DP-0-DECODE.trace.json.gz`
(decode), plus a `merged-*.trace.json.gz`.

**Gotcha (why decode kernels are missing):** SGLang serves decode as a single
**full CUDA graph** (`full_cuda_graph_backend` / `decode_cuda_graph_runner`).
The torch profiler correlates at the HIP-API layer, where a graph replay is one
`hipGraphLaunch`; it does not expand the replayed graph into per-kernel events.
So a DECODE-stage trace is dominated by memcpy/launch (observed: ~83% memcpy,
~18 visible kernels totaling ~0.1 ms — only the out-of-graph sampling/indexing
ops). The prefill/EXTEND trace is more informative but, under DP-attention, is
often dominated by NCCL waits.

This maps to the `job.slurm` passthrough:

```sh
# job.slurm (passthrough; default empty = no-op)
PROFILE="${PROFILE:-}"
SGLANG_TORCH_PROFILER_DIR="${SGLANG_TORCH_PROFILER_DIR:-}"
# ... docker run ...
    -e PROFILE=$PROFILE
    -e SGLANG_TORCH_PROFILER_DIR=$SGLANG_TORCH_PROFILER_DIR
```

i.e. `job.slurm` wires up **Mechanism A** only. Point
`SGLANG_TORCH_PROFILER_DIR` at a shared/NFS path so prefill+decode traces land
in one host-visible location.

## Mechanism B — CUDA-graph capture profiler (get the decode kernels)

- Serve flag **`--enable-profile-cuda-graph`** (`srt/server_args.py:1438`,
  "Enable profiling of cuda graph capture") sets the gate
  `enable_profile_cuda_graph` (`decode_cuda_graph_runner.py:207`).
- During decode graph capture the runner wraps capture with
  `torch.profiler.profile(activities=[CPU, CUDA], record_shapes=True)`
  (`decode_cuda_graph_runner.py:479`, gate at `:678`, capture at `:695`).
- Capture runs **eagerly**, so each decode kernel is a real dispatch →
  captured by name + input shape + CUDA time. This is representative: the
  captured graph is exactly what replays at steady state.
- After capture, `_post_process_after_profile` (`:487`):
  - always logs a **top-10 kernels by CUDA time (grouped by input shape)**
    table to the server log;
  - if env **`SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE=1`** (`environ.py:258`),
    also exports a full chrome trace to
    `<SGLANG_TORCH_PROFILER_DIR>/graph_capture_profile/cuda_graph_capture-<runner>-TP-<rank>.json.gz`
    (`profile_utils.py:32`, `export_cuda_graph_capture_trace`).

This is the SGLang-native answer to the decode-kernel-visibility problem — no
eager *serving* (which would distort steady-state timings) and no `rocprofv3`
attach (which returns status 19 on the already-spawned workers).

### Recipe (decode kernel breakdown for DeepSeek-V4)

Add the flag + env to the serve launch (e.g. in `xguo-sgl`):

```sh
SGLANG_TORCH_PROFILER_DIR=/workspace/sgl_traces \
SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE=1 \
sglang serve <model> ... --enable-profile-cuda-graph
```

- Fires during **startup graph capture** — no benchmark / `--profile` needed for
  the capture trace.
- Decode kernels (with shapes + CUDA time) appear in:
  - the server log's top-10 table, and
  - `sgl_traces/graph_capture_profile/cuda_graph_capture-*-TP-*.json.gz`
    (one per rank) for offline per-kernel analysis.
- Note: `record_shapes=True` is what makes the trace usable as a per-kernel
  shape/identity record offline.

## Which mechanism for which question

| Need | Use |
|---|---|
| Prefill vs decode **stage** time split | A (`PROFILE=1` + `SGLANG_TORCH_PROFILER_DIR`) |
| Prefill kernel breakdown | A (EXTEND trace) |
| **Decode kernel breakdown** (names/shapes/time) | **B** (`--enable-profile-cuda-graph` + `SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE=1`) |
| Cross-check with dispatch-level counters | `rocprofv3 --kernel-trace` launching the server under it (attach is unreliable) |

## Source references (sglang 0.5.14)

- `sglang/profiler.py:18` — `PROFILER_DIR = os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp")`
- `srt/environ.py:258` — `SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE = EnvBool(False)`
- `srt/environ.py:270` — `SGLANG_TORCH_PROFILER_DIR = EnvStr("/tmp")`
- `srt/entrypoints/http_server.py:1050/1061` — `/start_profile`, `/stop_profile`
- `srt/utils/profile_utils.py:32` — `export_cuda_graph_capture_trace(...)`
- `srt/server_args.py:1438` — `--enable-profile-cuda-graph`
- `srt/model_executor/runner/decode_cuda_graph_runner.py:479/507/678` — capture-time profiler wrap + export
