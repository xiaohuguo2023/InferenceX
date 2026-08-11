#!/usr/bin/env python3
"""Container-only debug instrumentation: attribute CUDA graph memory by call site.

Records torch allocator history across profile_cudagraph_memory and runtime,
and dumps snapshots at three points on device 0:

  cgmem_snap_after_profiling.pickle -- what the profiling pass left behind
                                       (today we only see it as one number)
  cgmem_snap_after_capture.pickle   -- steady state after the real capture
  runtime_mem_snap_low_free.pickle  -- first execute_model entry below
                                       VLLM_RUNTIME_SNAPSHOT_FREE_GIB

Gated on VLLM_CGMEM_SNAPSHOT=1. Applies on top of _patch_cgmem.py. Not upstream.
"""

import os
import shutil
import sys

VLLM = "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker"
TARGET = os.path.join(VLLM, "gpu_model_runner.py")

START_ANCHOR = """        torch.accelerator.synchronize()
        torch.accelerator.empty_cache()
        free_before_profiling = torch.accelerator.get_memory_info()[0]"""

START_NEW = """        torch.accelerator.synchronize()
        torch.accelerator.empty_cache()
        # SNAPSHOT(debug)
        self._cgmem_snap = os.environ.get("VLLM_CGMEM_SNAPSHOT") == "1" and (
            torch.cuda.current_device() == 0
        )
        if self._cgmem_snap:
            torch.cuda.memory._record_memory_history(
                enabled="all", context="all", stacks="python", max_entries=300000
            )
        free_before_profiling = torch.accelerator.get_memory_info()[0]"""

END_ANCHOR = """        self.cudagraph_profiling_retained_memory = max(
            free_before_profiling - torch.accelerator.get_memory_info()[0], 0
        )"""

END_NEW = """        self.cudagraph_profiling_retained_memory = max(
            free_before_profiling - torch.accelerator.get_memory_info()[0], 0
        )
        # SNAPSHOT(debug)
        if getattr(self, "_cgmem_snap", False):
            torch.cuda.memory._dump_snapshot(
                "/workspace/cgmem_snap_after_profiling%s.pickle"
                % os.environ.get("VLLM_CGMEM_SNAPSHOT_TAG", "")
            )
            logger.info("SNAPSHOT: dumped post-profiling allocator snapshot")"""

CAP_ANCHOR = """            "Graph capturing finished in %.0f secs, took %.2f GiB",
            elapsed_time,
            cuda_graph_size / (1 << 30),
        )
        return cuda_graph_size"""

CAP_NEW = """            "Graph capturing finished in %.0f secs, took %.2f GiB",
            elapsed_time,
            cuda_graph_size / (1 << 30),
        )
        # SNAPSHOT(debug)
        if getattr(self, "_cgmem_snap", False):
            torch.cuda.memory._dump_snapshot(
                "/workspace/cgmem_snap_after_capture%s.pickle"
                % os.environ.get("VLLM_CGMEM_SNAPSHOT_TAG", "")
            )
            self._runtime_mem_snap_dumped = False
            logger.info("SNAPSHOT: dumped post-capture allocator snapshot")
        return cuda_graph_size"""

EXEC_ANCHOR = """        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        with ("""

EXEC_NEW = """        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens

        # SNAPSHOT(debug): capture live allocations left by the previous
        # asynchronous in-flight batch before this batch launches. No device
        # synchronization here: synchronizing would remove the overlap being
        # diagnosed. The threshold is explicitly supplied by the debug run.
        if (
            getattr(self, "_cgmem_snap", False)
            and not getattr(self, "_runtime_mem_snap_dumped", False)
        ):
            threshold_gib = float(
                os.environ.get("VLLM_RUNTIME_SNAPSHOT_FREE_GIB", "0")
            )
            free_bytes, _ = torch.cuda.mem_get_info()
            if threshold_gib > 0 and free_bytes <= threshold_gib * (1 << 30):
                torch.cuda.memory._dump_snapshot(
                    "/workspace/runtime_mem_snap_low_free%s.pickle"
                    % os.environ.get("VLLM_CGMEM_SNAPSHOT_TAG", "")
                )
                self._runtime_mem_snap_dumped = True
                logger.info(
                    "SNAPSHOT: dumped runtime allocator snapshot at %.2f GiB "
                    "free before %d scheduled tokens",
                    free_bytes / (1 << 30),
                    num_scheduled_tokens,
                )

        with ("""

REPLACEMENTS = [
    ("snapshot start", START_ANCHOR, START_NEW),
    ("snapshot after profiling", END_ANCHOR, END_NEW),
    ("snapshot after capture", CAP_ANCHOR, CAP_NEW),
    ("runtime low-free snapshot", EXEC_ANCHOR, EXEC_NEW),
]


def main():
    src = open(TARGET, encoding="utf-8").read()
    if "SNAPSHOT(debug)" in src:
        print("[snap] already applied")
        return 0
    if "PATCH(cudagraph-mem-estimate)" not in src:
        print("[snap] ERROR: base cgmem patch missing; run _patch_cgmem.py first")
        return 1

    for name, old, new in REPLACEMENTS:
        n = src.count(old)
        if n != 1:
            print("[snap] ERROR: anchor %r found %d times, expected 1" % (name, n))
            return 1
        src = src.replace(old, new)

    if "\nimport os\n" not in src:
        src = src.replace("\nimport torch\n", "\nimport os\n\nimport torch\n", 1)

    shutil.copy2(TARGET, TARGET + ".snap_bak")
    open(TARGET, "w", encoding="utf-8").write(src)
    print("[snap] applied (backup: gpu_model_runner.py.snap_bak)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
