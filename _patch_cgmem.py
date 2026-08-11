#!/usr/bin/env python3
"""Make vLLM's CUDA graph memory estimate match what capture actually takes.

profile_cudagraph_memory() captured only the first two graphs per mode and
extrapolated. Within a profiling pass the second graph reuses the first one's
pool memory, so its delta measures ~0 and the per-graph term collapses to the
1 MiB floor; at capture time each graph pins its own per-shape scratch instead.
Measured on Kimi-K3 (gfx950, TP8): 1.29 GiB estimated against 10.69 GiB
captured, so gpu_worker sized the KV cache with ~9.4 GiB that did not exist and
the server ran ~3.4 points above the requested utilization.

Profiles every descriptor and takes one measurement spanning all modes. A
single span handles both the negative per-mode samples (FULL measured
-498 MiB) and whatever the modes overlay in the shared pool, without the
overcounting that clamping each mode before summing would introduce.

The span starts at function entry so it also covers the attention backends and
metadata builders that standing up the profiling KV cache initializes. Those
are rebuilt for the real KV cache and are budgeted nowhere else, since memory
profiling ran before any of them existed.

Also skips profiling entirely when VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS is
off, rather than capturing every graph and discarding the result.

Mirrors the upstream change in /home/xiaohugu/work/vllm.

CGMEM_WARMUP_REUSE=0 also drops the capture-side warmup reuse, which is what
the upstream PR ships: profiling cleanup tears down the KV cache and attention
state before the real capture rebuilds them, so skipping the second warmup
assumes no backend lazily initializes there, and that is unvalidated off ROCm.
Reuse keeps the real capture from re-allocating ~1.6 GiB of warmup scratch
after the KV cache is already sized, so the two settings are not interchangeable
on a node running near its utilization ceiling. Default 1 for continuity with
the arms measured before the split.
"""

import os
import sys
from pathlib import Path

WARMUP_REUSE = os.environ.get("CGMEM_WARMUP_REUSE", "1") == "1"

PKG = Path("/usr/local/lib/python3.12/dist-packages/vllm")
TARGET = PKG / "v1/worker/gpu_model_runner.py"
WORKER = PKG / "v1/worker/gpu_worker.py"
MARKER = "PATCH(cudagraph-mem-estimate)"

OLD_INIT = """        # Cudagraph dispatcher for runtime cudagraph dispatching.
        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)"""

NEW_INIT = """        # Cudagraph dispatcher for runtime cudagraph dispatching.
        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)

        # Descriptors already warmed and captured by memory profiling, so that
        # the real capture can skip warming them a second time.
        self._profiled_capture_descs: set[tuple[CUDAGraphMode, BatchDescriptor]] = set()

        # Memory that memory profiling allocated and did not give back. The
        # real capture reuses it, so it belongs to any comparison of the
        # estimate against what the graphs actually cost.
        self.cudagraph_profiling_retained_memory = 0"""

OLD_ENTRY = """    def profile_cudagraph_memory(self) -> int:
        with set_current_vllm_config(self.vllm_config):
            self._init_minimal_kv_cache_for_profiling()"""

NEW_ENTRY = """    def profile_cudagraph_memory(self) -> int:
        # Baseline for the estimate. Everything this function allocates from
        # here on is memory the steady state has to live with, whether it is
        # the capture pool or the setup the pool needs.
        torch.accelerator.synchronize()
        torch.accelerator.empty_cache()
        free_before_profiling = torch.accelerator.get_memory_info()[0]

        with set_current_vllm_config(self.vllm_config):
            self._init_minimal_kv_cache_for_profiling()"""

OLD_DECLS = """        shared_memory_estimate = {}
        per_graph_estimate = {}
        encoder_memory_estimate = 0"""

NEW_DECLS = """        # PATCH(cudagraph-mem-estimate)
        decoder_memory_estimate = 0
        encoder_memory_estimate = 0
        profiled_descs: set[tuple[CUDAGraphMode, BatchDescriptor]] = set()"""

OLD_LOOP = """                for mode, descs in capture_descs:
                    profile_descs = descs[:2]
                    mem_samples: list[int] = []

                    for i, desc in enumerate(profile_descs):
                        mem_before = torch.accelerator.get_memory_info()[0]
                        self._warmup_and_capture(
                            desc,
                            cudagraph_runtime_mode=mode,
                            profile_seq_lens=(
                                min(
                                    self.max_model_len,
                                    self.max_num_tokens // desc.num_tokens,
                                )
                                if mode == CUDAGraphMode.FULL and i == 0
                                else None
                            ),
                        )
                        torch.accelerator.synchronize()
                        free_after = torch.accelerator.get_memory_info()[0]
                        mem_samples.append(mem_before - free_after)

                    first_capture = mem_samples[0]
                    # Use at least 1 MiB per graph for driver overhead
                    per_graph = max(
                        mem_samples[1] if len(mem_samples) > 1 else 0, 1 << 20
                    )

                    shared_memory_estimate[mode] = first_capture
                    per_graph_estimate[mode] = per_graph * (len(descs) - 1)

                    logger.debug(
                        "Estimated %s CUDA graph memory: "
                        "%.2f MiB first-capture + (%d-1) \u00d7 %.2f MiB per-graph",
                        mode.name,
                        first_capture / (1 << 20),
                        len(descs),
                        per_graph / (1 << 20),
                    )"""

NEW_LOOP = """                # Capture every descriptor instead of extrapolating from the
                # first two. Pool growth across capture sizes is not linear, and
                # a graph captured during profiling can reuse the memory of the
                # one before it, so a two-sample extrapolation can report a
                # small fraction of the pool the real capture goes on to build.
                # Under-reporting here is not conservative: the shortfall is
                # handed to the KV cache, which then pushes total usage past
                # gpu_memory_utilization.
                for mode, descs in capture_descs:
                    mode_mem_before = torch.accelerator.get_memory_info()[0]

                    for i, desc in enumerate(descs):
                        self._warmup_and_capture(
                            desc,
                            cudagraph_runtime_mode=mode,
                            profile_seq_lens=(
                                min(
                                    self.max_model_len,
                                    self.max_num_tokens // desc.num_tokens,
                                )
                                if mode == CUDAGraphMode.FULL and i == 0
                                else None
                            ),
                        )
                        profiled_descs.add((mode, desc))

                    torch.accelerator.synchronize()
                    # Diagnostic only, and deliberately unclamped: a negative
                    # value is a useful signal that this mode ran mostly out of
                    # memory the allocator already held.
                    logger.debug(
                        "Estimated %s CUDA graph memory: %.2f MiB for %d graphs",
                        mode.name,
                        (mode_mem_before - torch.accelerator.get_memory_info()[0])
                        / (1 << 20),
                        len(descs),
                    )

                # Measure the modes together rather than summing them. They
                # capture back to back into one pool, so a mode can measure
                # negative when the allocator releases more than that mode took,
                # and clamping each mode before summing would then overcount.
                # One span across all of them also subsumes whatever the modes
                # overlay in the shared pool.
                #
                # The span starts at function entry, not at the capture loop.
                # Standing up the profiling KV cache also initializes the
                # attention backends and metadata builders, whose scratch is
                # sized by the capture shapes and is rebuilt for the real KV
                # cache. Nothing else budgets it: memory profiling ran before
                # any of it existed. It does count the profiling KV cache,
                # which is deliberately minimal and errs toward reserving
                # slightly too much rather than too little.
                decoder_free_after = torch.accelerator.get_memory_info()[0]
                decoder_memory_estimate = max(
                    free_before_profiling - decoder_free_after, 0
                )"""

OLD_TOTAL = """        # FULL and PIECEWISE graphs share the global pool at runtime and are
        # never replayed concurrently, so the pool overlays their memory.
        # Take the max to avoid double-counting the overlap.
        decoder_estimate = max(shared_memory_estimate.values(), default=0) + sum(
            per_graph_estimate.values()
        )"""

NEW_TOTAL = """        # Every descriptor here has now been warmed and captured once, so the
        # real capture can capture them without warming them again. Recorded
        # only on success: a partial pass leaves shapes unwarmed.
        self._profiled_capture_descs = profiled_descs

        # Cleanup above discards the graphs and empties the cache, but scratch
        # the captured shapes allocated stays live and the real capture reuses
        # it. Without this, that memory looks like it was never spent.
        self.cudagraph_profiling_retained_memory = max(
            free_before_profiling - torch.accelerator.get_memory_info()[0], 0
        )

        decoder_estimate = decoder_memory_estimate"""

OLD_WORKER_GATE = """        cudagraph_memory_estimate = 0
        if (
            current_platform.is_cuda_alike()
            and self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            cudagraph_memory_estimate = self.model_runner.profile_cudagraph_memory()

        # Respect the opt-in flag as originally designed.
        cudagraph_memory_estimate_applied = (
            cudagraph_memory_estimate
            if envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS
            else 0
        )

        self.total_consumed = profile_result.total_consumed
        self.peak_activation_memory = (
            profile_result.transient_peak_headroom + cudagraph_memory_estimate_applied
        )"""

NEW_WORKER_GATE = """        # Profiling captures every graph, so it is not free. Skip it entirely
        # when the estimate would only be discarded.
        cudagraph_memory_estimate = 0
        if (
            envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS
            and current_platform.is_cuda_alike()
            and self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            cudagraph_memory_estimate = self.model_runner.profile_cudagraph_memory()

        # memory_profiling() measures one model forward, but async scheduling
        # keeps max_concurrent_batches forwards in flight. Reserve one profiled
        # transient peak for every additional in-flight batch. CUDA graph
        # memory is persistent/shared and must not be multiplied.
        concurrent_batch_headroom = profile_result.transient_peak_headroom * (
            self.vllm_config.max_concurrent_batches - 1
        )
        if concurrent_batch_headroom:
            logger.info(
                "Reserving %s GiB activation headroom for %d additional "
                "in-flight batch(es) under async scheduling.",
                format_gib(concurrent_batch_headroom),
                self.vllm_config.max_concurrent_batches - 1,
            )

        self.total_consumed = profile_result.total_consumed
        self.peak_activation_memory = (
            profile_result.transient_peak_headroom
            + concurrent_batch_headroom
            + cudagraph_memory_estimate
        )"""

OLD_WORKER_BUDGET = """        self.available_kv_cache_memory_bytes = (
            self.requested_memory
            - profile_result.non_kv_cache_memory
            - cudagraph_memory_estimate_applied
        )"""

NEW_WORKER_BUDGET = """        self.available_kv_cache_memory_bytes = (
            self.requested_memory
            - profile_result.non_kv_cache_memory
            - cudagraph_memory_estimate
            - concurrent_batch_headroom
        )"""

OLD_WORKER_ADVISORY = """        if cudagraph_memory_estimate > 0:
            total_mem = self.init_snapshot.total_memory
            current_util = self.cache_config.gpu_memory_utilization
            cg_util_delta = cudagraph_memory_estimate / total_mem
            if envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:
                equiv_util = round(current_util - cg_util_delta, 4)
                suggested_util = min(
                    round(current_util + cg_util_delta, 4),
                    1.0,
                )
                logger.info(
                    "CUDA graph memory profiling is enabled (default since "
                    "v0.21.0). The current --gpu-memory-utilization=%.4f is "
                    "equivalent to --gpu-memory-utilization=%.4f without "
                    "CUDA graph memory profiling. To maintain the same "
                    "effective KV cache size as before, increase "
                    "--gpu-memory-utilization to %.4f. To disable, set "
                    "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0.",
                    current_util,
                    equiv_util,
                    suggested_util,
                )
            else:
                suggested_util = min(
                    round(current_util + cg_util_delta, 4),
                    1.0,
                )
                logger.warning(
                    "CUDA graph memory profiling is disabled "
                    "(VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0). "
                    "Without it, CUDA graph memory is not accounted for "
                    "during KV cache allocation, which may require lowering "
                    "--gpu-memory-utilization to avoid OOM. Consider "
                    "re-enabling it (the default as of v0.21.0) and increasing "
                    "--gpu-memory-utilization from %.4f to %.4f.",
                    current_util,
                    suggested_util,
                )"""

NEW_WORKER_ADVISORY = """        if not envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:
            # No estimate to quote a utilization against, because profiling was
            # skipped rather than measured and thrown away.
            logger.warning(
                "CUDA graph memory profiling is disabled "
                "(VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0). "
                "Without it, CUDA graph memory is not accounted for "
                "during KV cache allocation, which may require lowering "
                "--gpu-memory-utilization to avoid OOM. Consider "
                "re-enabling it (the default as of v0.21.0)."
            )
        elif cudagraph_memory_estimate > 0:
            total_mem = self.init_snapshot.total_memory
            current_util = self.cache_config.gpu_memory_utilization
            cg_util_delta = cudagraph_memory_estimate / total_mem
            equiv_util = round(current_util - cg_util_delta, 4)
            suggested_util = min(
                round(current_util + cg_util_delta, 4),
                1.0,
            )
            logger.info(
                "CUDA graph memory profiling is enabled (default since "
                "v0.21.0). The current --gpu-memory-utilization=%.4f is "
                "equivalent to --gpu-memory-utilization=%.4f without "
                "CUDA graph memory profiling. To maintain the same "
                "effective KV cache size as before, increase "
                "--gpu-memory-utilization to %.4f. To disable, set "
                "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0.",
                current_util,
                equiv_util,
                suggested_util,
            )"""

OLD_WORKER_LOG = """            GiB = lambda b: round(b / GiB_bytes, 2)
            diff = abs(cuda_graph_memory_bytes - self.cudagraph_memory_estimate)
            logger.info(
                "CUDA graph pool memory: %s GiB (actual), %s GiB (estimated), "
                "difference: %s GiB (%.1f%%).",
                GiB(cuda_graph_memory_bytes),
                GiB(self.cudagraph_memory_estimate),
                GiB(diff),
                100 * diff / max(cuda_graph_memory_bytes, 1),
            )"""

NEW_WORKER_LOG = """            GiB = lambda b: round(b / GiB_bytes, 2)
            retained = self.model_runner.cudagraph_profiling_retained_memory
            actual = cuda_graph_memory_bytes + retained
            diff = abs(actual - self.cudagraph_memory_estimate)
            logger.info(
                "CUDA graph pool memory: %s GiB (actual: %s GiB captured + "
                "%s GiB retained by profiling), %s GiB (estimated), "
                "difference: %s GiB (%.1f%%).",
                GiB(actual),
                GiB(cuda_graph_memory_bytes),
                GiB(retained),
                GiB(self.cudagraph_memory_estimate),
                GiB(diff),
                100 * diff / max(actual, 1),
            )"""

OLD_CAPTURE = """            self._warmup_and_capture(
                batch_desc,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                allow_microbatching=allow_microbatching,
                profiler=profiler,
            )
            torch.accelerator.synchronize()"""

NEW_CAPTURE = """            # Memory profiling already warmed and captured this shape, so the
            # warmup run here would be redundant. It also allocates scratch
            # that the profiling pass already left behind, so reusing it keeps
            # that memory out of the capture pool. Microbatched graphs are a
            # different capture that profiling does not exercise, so they keep
            # their warmup.
            already_warmed = (
                not allow_microbatching
                and (cudagraph_runtime_mode, batch_desc) in self._profiled_capture_descs
            )
            self._warmup_and_capture(
                batch_desc,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                allow_microbatching=allow_microbatching,
                profiler=profiler,
                num_warmups=0 if already_warmed else None,
            )
            torch.accelerator.synchronize()"""

REUSE_ONLY_FRAGMENTS = [
    (
        "NEW_INIT",
        """
        # Descriptors already warmed and captured by memory profiling, so that
        # the real capture can skip warming them a second time.
        self._profiled_capture_descs: set[tuple[CUDAGraphMode, BatchDescriptor]] = set()
""",
    ),
    (
        "NEW_DECLS",
        "\n        profiled_descs: set[tuple[CUDAGraphMode, BatchDescriptor]] = set()",
    ),
    ("NEW_LOOP", "\n                        profiled_descs.add((mode, desc))"),
    (
        "NEW_TOTAL",
        """        # Every descriptor here has now been warmed and captured once, so the
        # real capture can capture them without warming them again. Recorded
        # only on success: a partial pass leaves shapes unwarmed.
        self._profiled_capture_descs = profiled_descs

""",
    ),
]

if not WARMUP_REUSE:
    # Strip the bookkeeping the reuse depends on, so the patched file is the
    # shipped code rather than the shipped code plus dead state.
    _globals = globals()
    for _name, _fragment in REUSE_ONLY_FRAGMENTS:
        if _fragment not in _globals[_name]:
            print(
                f"ERROR: {_name} no longer contains its warmup-reuse fragment; "
                "CGMEM_WARMUP_REUSE=0 would silently leave it in.",
                file=sys.stderr,
            )
            sys.exit(1)
        _globals[_name] = _globals[_name].replace(_fragment, "", 1)

REPLACEMENTS = [
    ("init", OLD_INIT, NEW_INIT),
    ("entry measurement", OLD_ENTRY, NEW_ENTRY),
    ("declarations", OLD_DECLS, NEW_DECLS),
    ("profiling loop", OLD_LOOP, NEW_LOOP),
    ("total", OLD_TOTAL, NEW_TOTAL),
]
if WARMUP_REUSE:
    REPLACEMENTS.append(("capture warmup reuse", OLD_CAPTURE, NEW_CAPTURE))

WORKER_REPLACEMENTS = [
    ("worker profiling gate", OLD_WORKER_GATE, NEW_WORKER_GATE),
    ("worker kv budget", OLD_WORKER_BUDGET, NEW_WORKER_BUDGET),
    ("worker advisory log", OLD_WORKER_ADVISORY, NEW_WORKER_ADVISORY),
    ("worker comparison log", OLD_WORKER_LOG, NEW_WORKER_LOG),
]


def _apply(path: Path, replacements: list[tuple[str, str, str]]) -> str | None:
    src = path.read_text()
    for label, old, new in replacements:
        if src.count(old) != 1:
            print(
                f"ERROR: {label} block matched {src.count(old)} times, expected 1. "
                "The installed vLLM differs from the version this patch targets.",
                file=sys.stderr,
            )
            return None
        src = src.replace(old, new)
    compile(src, str(path), "exec")
    return src


def main() -> int:
    for path in (TARGET, WORKER):
        if not path.exists():
            print(f"ERROR: {path} not found", file=sys.stderr)
            return 1

    if MARKER in TARGET.read_text():
        print("[patch] cudagraph-mem-estimate already applied")
        return 0

    runner_src = _apply(TARGET, REPLACEMENTS)
    worker_src = _apply(WORKER, WORKER_REPLACEMENTS)
    if runner_src is None or worker_src is None:
        return 1

    for path, text in ((TARGET, runner_src), (WORKER, worker_src)):
        backup = path.with_suffix(".py.cgmem_bak")
        if not backup.exists():
            backup.write_text(path.read_text())
        path.write_text(text)

    print(
        "[patch] cudagraph-mem-estimate applied (backups: *.py.cgmem_bak) "
        f"warmup_reuse={int(WARMUP_REUSE)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
