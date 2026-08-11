"""Root-cause fix: make aiter's a16w16 (bf16) ASM GEMM heuristic cudagraph-safe by
never selecting a split-K kernel when split-K is disabled (the default here).

Why split-K deadlocks / faults under vLLM + DSpark
--------------------------------------------------
The split-K a16w16 ASM kernels ("*_splitk_clean", cfg.splitK==1) reduce their
partial-K results through a per-(device,stream) atomic semaphore. The protocol is
"the last workgroup to finish does the reduction", which requires the semaphore
counter to be ZERO at launch and the stream to own it exclusively. Under vLLM
CUDA-graph replay that invariant breaks two ways:
  * replay reuses the same semaphore memory without re-zeroing it, and
  * DSpark's semi-autoregressive drafting runs draft+target on separate streams,
    so counts from different launches mix.
The reduction phase then never fires and every wave spin-waits forever (all 8
GPUs 100% / ~310W, log frozen, shm_broadcast timeouts) at seqs=64 warmup.
Confirmed via rocgdb: aiter::bf16gemm_fp32bf16_tn_64x64_splitk_clean, all waves
stuck at one PC. Eager single launches are fine, which is why it only bites
in-serve.

Two selection paths reach these kernels, so both must be closed:
  (E-csv) a tuned CSV (merged_bf16_tuned_gemm.csv) can PIN a *_splitk_clean kernel
          by name. If forced to split=1 it dereferences a null semaphore ->
          Memory access fault by GPU ... on address (nil).
  (E-auto) the untuned heuristic auto-selects a kernel and auto-derives split>=2.

The fix (Option A)
------------------
In csrc/py_itfs_cu/asm_gemm_a16w16.cu :: get_heuristic_kernel():
  E0  #include <cstdlib>                     (getenv/atoi for the toggle)
  E1  add a static kAllowSplitK toggle (env AITER_ALLOW_SPLITK, default OFF) and,
      when split-K is disabled, drop any CSV pin that names a splitK==1 kernel so
      the heuristic below falls back to a non-split (cfg.splitK==0) kernel.
  E2  in the auto-select branch, require (kAllowSplitK || cfg.splitK==0) so
      *_splitk_clean kernels are never auto-selected. A non-split pf3/bshuffle
      kernel with matching tileN/bPreshuffle always exists (all tiles are tileN
      64), so there is always a safe fallback.
  E3  gate the auto split-count enable on kAllowSplitK AND pure_tg_num<=1024
      (the semaphore workspace is 16*64=1024 tiles; split-K past that is both
      unsafe and pointless once the grid saturates every CU).
AITER_ALLOW_SPLITK=1 restores the original split-K behavior for eager-only runs.

Rebuild path (per project directive: use aiter's own build, do NOT hand-delete
the .so): after editing the source we ask aiter to drop + clear the JIT module
and re-invoke the op, which triggers a ninja recompile of
module_gemm_a16w16_asm (~5s). Pass --no-build to only edit the source.

Idempotent: safe to run repeatedly (each edit is marker-guarded).

  python3 _patch_aiter_splitk_cudagraph.py            # edit source + rebuild
  python3 _patch_aiter_splitk_cudagraph.py --no-build # edit source only
"""
import sys

CU = "/opt/aiter-local/csrc/py_itfs_cu/asm_gemm_a16w16.cu"
MODULE = "module_gemm_a16w16_asm"

# --- E0: include <cstdlib> for getenv/atoi -------------------------------------
E0_ANCHOR = "#include <optional>\n#include <hip/hip_runtime.h>"
E0_NEW = "#include <optional>\n#include <cstdlib>\n#include <hip/hip_runtime.h>"

# --- E1: kAllowSplitK toggle + drop CSV pins on splitK==1 kernels ---------------
E1_ANCHOR = """    int selectedsplitK             = 1;

    for(const auto& el : *cfgs)"""
E1_NEW = """    int selectedsplitK             = 1;

    // PATCH(splitk-cudagraph): the split-K a16w16 ASM kernels ("*_splitk_clean",
    // cfg.splitK==1) reduce partial-K results through a per-(device,stream) atomic
    // semaphore whose "last workgroup does the reduction" protocol relies on the
    // counter being zero at launch. Under vLLM CUDA-graph replay + multi-stream
    // DSpark drafting that invariant is violated (stale counters are never reset
    // on replay / counts mix across streams) and the reduction phase never fires
    // -> all waves spin forever (GPU 100%, ~310W) at seqs=64 warmup. Eager single
    // launches are fine, which is why this only bites in-serve. Disable auto
    // split-K by default; AITER_ALLOW_SPLITK=1 re-enables it for eager-only runs.
    static const bool kAllowSplitK = [] {
        const char* e = getenv("AITER_ALLOW_SPLITK");
        return e && atoi(e) != 0;
    }();

    // PATCH(splitk-cudagraph-csv): a tuned CSV can pin a *_splitk_clean kernel
    // (cfg.splitK==1). Those REQUIRE the split-K semaphore path and crash with a
    // null-pointer memory-access-fault if forced to split=1. When split-K is
    // disabled, drop such a pin so the heuristic below falls back to a non-split
    // (cfg.splitK==0) kernel. Non-clean tuned pins are left untouched.
    if(kernelName && !kAllowSplitK)
    {
        for(const auto& el : *cfgs)
        {
            if(el.first == (arch_id + kernelName))
            {
                if(el.second.splitK == 1)
                    kernelName = nullptr;
                break;
            }
        }
    }
    for(const auto& el : *cfgs)"""

# --- E2: exclude splitK==1 kernels from auto-selection -------------------------
E2_ANCHOR = """        // auto select splitk or kernel
        if(N % cfg.tileN == 0 && cfg.bPreshuffle == (bpreshuffle ? 1 : 0) &&
           (add_bias == 0 || cfg.bias == 1))
        {"""
E2_NEW = """        // auto select splitk or kernel
        // PATCH(splitk-cudagraph): when split-K is disabled, exclude *_splitk_clean
        // kernels (cfg.splitK==1) from selection -- they cannot run without the
        // semaphore-reduction path. A non-split (cfg.splitK==0) kernel with matching
        // tileN/bPreshuffle always exists (pf3/bshuffle variants).
        if(N % cfg.tileN == 0 && cfg.bPreshuffle == (bpreshuffle ? 1 : 0) &&
           (add_bias == 0 || cfg.bias == 1) && (kAllowSplitK || cfg.splitK == 0))
        {"""

# --- E3: gate the auto split-count enable on kAllowSplitK + grid size ----------
E3_ANCHOR = "            if(cfg.splitK == 1 && K / cfg.subK >= 2) // kernel and Kdim support splitk"
E3_NEW = (
    "            if(kAllowSplitK && cfg.splitK == 1 && K / cfg.subK >= 2 &&\n"
    "               pure_tg_num <= 1024) // PATCH(splitk-grid-guard): semaphore "
    "workspace is 16*64=1024 tiles, so never SplitK a larger grid (deadlock) — and "
    "SplitK is useless once the grid saturates all CUs"
)

EDITS = [
    ("E0 include <cstdlib>", E0_ANCHOR, E0_NEW, "#include <cstdlib>"),
    ("E1 kAllowSplitK + CSV pin drop", E1_ANCHOR, E1_NEW, "PATCH(splitk-cudagraph-csv)"),
    ("E2 auto-select excludes splitk_clean", E2_ANCHOR, E2_NEW, "kAllowSplitK || cfg.splitK == 0"),
    ("E3 auto split gate", E3_ANCHOR, E3_NEW, "PATCH(splitk-grid-guard)"),
]


def apply_source_edits():
    s = open(CU).read()
    changed = False
    for name, anchor, new, marker in EDITS:
        if marker in s:
            print(f"  [skip] {name} (already applied)")
            continue
        assert anchor in s, f"anchor for {name} not found -- aiter source layout changed"
        s = s.replace(anchor, new, 1)
        changed = True
        print(f"  [ok]   {name}")
    if changed:
        open(CU, "w").write(s)
        print(f"applied edits to {CU}")
    else:
        print(f"{CU} already fully patched")
    return changed


def rebuild():
    """Rebuild module_gemm_a16w16_asm through aiter's own JIT (no manual .so delete)."""
    import torch
    import aiter
    from aiter.jit import core

    core.rm_module(MODULE)
    core.clear_build(MODULE)
    # Invoke the op on a small bf16 shape to trigger a ninja recompile from source.
    m, n, k = 128, 128, 256
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    out = aiter.gemm_a16w16_asm(a, b)
    torch.cuda.synchronize()
    print(f"rebuilt {MODULE}; op ran, out.shape={tuple(out.shape)} dtype={out.dtype}")


if __name__ == "__main__":
    no_build = "--no-build" in sys.argv[1:]
    apply_source_edits()
    if no_build:
        print("--no-build: skipped rebuild; module_gemm_a16w16_asm will build on next op call")
    else:
        rebuild()
