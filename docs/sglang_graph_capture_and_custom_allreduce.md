# SGLang CUDA-graph capture & the `cross_device_reduce_2stage` custom all-reduce

Verified against the `~/work/sglang` checkout (paths below). Answers two things:
1. how SGLang captures CUDA graphs, and
2. whether AITER/SGLang's `cross_device_reduce_2stage` custom all-reduce can be
   captured inside a CUDA graph.

## TL;DR

- **Yes — `cross_device_reduce_2stage` is CUDA-graph-capturable, and SGLang
  captures it by design.** It is one of the two all-reduce paths explicitly
  *enabled in graph mode* (`custom allreduce` and `quick allreduce`); plain
  `torch.distributed` all-reduce is the one that is **disabled** in graph mode.
- It is capturable because the kernel does **all cross-GPU synchronization
  on-device** (atomic signal flags in peer IPC memory + spin-wait), uses
  **pre-registered fixed IPC buffers**, and does **no host-side work / no
  allocation** during the call. The only graph-specific step is
  `register_graph_buffers()` after capture, which records the buffer addresses
  used inside the graph so replay is valid on every rank.

## 1. How SGLang does CUDA-graph capture

**Runners.** `model_executor/runner/{decode,prefill}_cuda_graph_runner.py` +
`base_cuda_graph_runner.py`. At startup, for each batch size in the capture list
(`cuda_graph_config.py`; e.g. decode bs `[1,2,4,8,…,max_bs]`) the runner:
1. allocates **static input buffers** once (`cuda_graph_buffer_registry.py`) —
   `input_ids`, `positions`, `out_cache_loc`, attention metadata, etc.;
2. runs a few **warmup** iterations on a side stream;
3. **captures** the whole model forward for that bs into a HIP graph under the
   `get_parallel().graph_capture()` context (`torch.cuda.graph`-style capture);
4. stores one graph per bs.

**Replay (decode).** Each decode step pads the running batch up to the nearest
captured bs bucket, copies the live tensors into the static buffers, and issues a
single `hipGraphLaunch` that replays the entire forward. (This is why a torch
profiler sees *one* graph launch and not the individual decode kernels — see
`docs/sglang_kernel_profiling.md`; use `--enable-profile-cuda-graph` to see them.)

**The capture context** (`distributed/parallel_state.py:559` `graph_capture`)
enters, on the capture stream, the collective-communicator capture contexts:
`ca_comm.capture()` (custom all-reduce), and `change_state(enable=True)` for
PyNccl / PyMscclpp. It documents which all-reduce is legal in each mode:

```
allreduce \ Mode   |  Eager  |  Graph
-------------------+---------+--------
quick allreduce    | enabled | enabled
custom allreduce   | enabled | enabled   <-- cross_device_reduce_2stage
PyNccl             | disabled| enabled
PyMscclpp          | disabled| enabled
TorchSymmMem       | disabled| enabled
torch.distributed  | enabled | disabled  <-- NOT captured
```

So in graph mode the TP all-reduce is served by the **custom** (or quick)
all-reduce, never by `torch.distributed`.

## 2. How `cross_device_reduce_2stage` is implemented

Kernel: `sgl-kernel/csrc/allreduce/custom_all_reduce_hip.cuh` (AITER ships an
equivalent `aiter::cross_device_reduce_2stage`; both share this design). It is a
**two-stage ring/peer all-reduce over IPC/XGMI peer memory**: stage 1 =
reduce-scatter (each rank reduces its shard by reading all peers' buffers),
stage 2 = all-gather (each rank broadcasts its reduced shard).

Cross-GPU synchronization is a **fully on-device barrier** (`start_sync`/
`end_sync`, lines ~156-223): each block holds a per-rank `Signal{ uint32 _flag[] }`
in *peer* memory; a rank atomically writes an incrementing flag into every peer's
signal (`__atomic_store_n(..., __MEMORY_SCOPE_SYSTEM)`) and spin-waits until all
peers' flags reach the expected value. No `__syncthreads`-to-host, no CUDA
events, no stream sync, no `cudaMalloc` — the whole handshake is GPU-side.

Buffers are **pre-registered IPC memory** set up once at init
(`custom_all_reduce.py` `create_shared_buffer`, `register_buffer`,
`meta_ptrs`/`rank_data`) — the kernel only ever touches these fixed peer pointers.

## 3. Why that makes it graph-capturable

A CUDA/HIP graph can only record operations that are (a) pure stream work with
(b) no host synchronization and (c) stable pointers across replays. The custom
all-reduce satisfies all three:

- **No host sync / no allocation** → the call is a single kernel launch on the
  capture stream; nothing forces a host round-trip that would abort capture.
  (Standard `torch.distributed`/RCCL collectives do host-side or non-graph-safe
  work, which is why the table disables them in graph mode.)
- **On-device flag barrier** → the inter-GPU handshake happens *inside* the
  replayed kernel every time, so a replay re-synchronizes correctly without any
  captured host op.
- **Fixed IPC buffers + `register_graph_buffers()`** → the graph replays with the
  *same* buffer addresses it captured. `custom_all_reduce.py:182 capture()` sets
  `_IS_CAPTURING=True` and, on exit, calls `register_graph_buffers()`
  (`:234`), which fetches the graph buffer IPC handles
  (`get_graph_buffer_ipc_meta`) and exchanges them across ranks
  (`register_graph_buffers`) so every peer can resolve the pointers used inside
  the graph. This is the one extra step a graph needs that eager doesn't.

**Capture-time routing** (`parallel_state.py`): when
`ca_comm._IS_CAPTURING and torch.cuda.is_current_stream_capturing()`, the
all-reduce / reduce-scatter / all-gather take the **`registered=True`** path
(pre-registered IPC buffers) — e.g. `reduce_scatter(input, output, registered=True)`
(`:1002`) and `custom_all_reduce`/`all_gather_reg` during capture. Eager falls
back to the `unreg` path. `should_custom_ar()` (`:262`) bounds it to
16B-aligned, weak-contiguous, supported-topology tensors under `max_size`.

## Reference: Qwen (dense TP) vs DeepSeek-V4

- **Qwen3** (`models/qwen3*.py`) is the clean case: after attention and MLP it
  calls `tensor_model_parallel_all_reduce` → `ca_comm` → `cross_device_reduce_2stage`,
  and that all-reduce is captured *inside* the decode graph (one launch replays
  attention+MLP+AR together).
- **DeepSeek-V4** uses **DP-attention** (`attn_tp = tp/dp = 1`), so the per-layer
  TP all-reduce is largely replaced by DP gather/reduce-scatter across the DP
  group; the custom AR/reduce-scatter still follows the same
  `_IS_CAPTURING → registered` graph path (`:1000-1010`, `all_gather_reg`).

## Bottom line
`cross_device_reduce_2stage` (custom/quick all-reduce, sgl-kernel or AITER) **is
captured by CUDA graph in SGLang** — it is purpose-built for it via an on-device
flag barrier, pre-registered IPC buffers, and the post-capture
`register_graph_buffers()` handshake. The all-reduce that *cannot* be captured is
plain `torch.distributed`, which SGLang disables in graph mode.

## Source references
- `distributed/parallel_state.py:559` — `graph_capture()` + eager/graph AR table
- `distributed/parallel_state.py:1000-1010` — capture-time `registered=True` routing
- `distributed/device_communicators/custom_all_reduce.py:182` — `capture()` ctx
- `custom_all_reduce.py:234` — `register_graph_buffers()`
- `custom_all_reduce.py:262` — `should_custom_ar()`
- `sgl-kernel/csrc/allreduce/custom_all_reduce_hip.cuh:34,156,197` — `Signal`, `start_sync`, `end_sync`
- `model_executor/runner/decode_cuda_graph_runner.py`, `cuda_graph_buffer_registry.py`
