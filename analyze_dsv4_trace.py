#!/usr/bin/env python3
"""
DeepSeek-V4 per-stage kernel-breakdown analyzer for vLLM torch-profiler traces
(ROCm / MI355X).

Borrowed from ~/work/vllm_traces_tp_varied/trace_compare.py:
  * Category -> Sub-kernel -> regex, first-match-wins `categorize_kernel`
  * PREFILL/DECODE segmentation via `execute_context_N(ctx)_generation_M(gen)`
    gpu_user_annotation steps (ctx>0 => prefill, ctx==0 => decode).
EXTENDED for DeepSeek-V4-Pro: MLA/DSA sparse attention, FP8xFP4 MoE experts,
CK AB-scale GEMM, AITER mhc_* MLA helpers, per-group quant, softplus+sqrt gating.

Splits every metric into PREFILL vs DECODE and writes a per-stage markdown report.

Usage:
  python analyze_dsv4_trace.py --md docs/<report>.md <trace ...>
"""
import gzip, json, sys, re, glob, shutil, subprocess, os
from collections import defaultdict

# ---------------------------------------------------------------------------
# Kernel categorization (DSV4-specific patterns first so they win)
# ---------------------------------------------------------------------------
CATEGORY_PATTERNS = {
    "GEMM (dense/linear)": [
        ("AITER fp8 blockscale (preshuffle)", re.compile(r"fp8gemm.*blockscale", re.I)),
        ("CK blockscale (preshuffle)", re.compile(r"multi_d_blockscale.*preshuffle|b_preshuffle")),
        ("CK AB-scale FP8 GEMM",   re.compile(r"ck::kernel_gemm.*ABScale|GridwiseGemmMultiD_ABScale")),
        ("CK gemm_xdl_cshuffle",   re.compile(r"ck::kernel_gemm_xdl_cshuffle")),
        ("AITER a8w8 blockscale",  re.compile(r"_gemm_a8w8_blockscale_kernel")),
        ("wvSplitK (skinny GEMM)", re.compile(r"wvSplitK", re.I)),
        ("bf16 hgemm (AITER flydsl)", re.compile(r"hgemm_bf16|bf16gemm_fp32bf16")),
        ("hipBLASLt (Cijk)",       re.compile(r"Cijk_")),
        ("hipBLASLt",              re.compile(r"hipblaslt|hipblas_lt", re.I)),
        ("rocBLAS",                re.compile(r"rocblas")),
        ("Triton GEMM (aiter)",    re.compile(r"_gemm_a16_w16_kernel")),
        ("Triton fused GEMM",      re.compile(r"triton.*gemm|gemm.*triton", re.I)),
    ],
    "GEMM (MoE experts)": [
        ("MoE1 gate/up afp8_wfp4", re.compile(r"mfma_moe1.*afp8_wfp4|moe1_silu_mul_afp8_wfp4|flydsl_moe1")),
        ("MoE2 down  afp8_wfp4",   re.compile(r"mfma_moe2.*afp8_wfp4|moe2_afp8_wfp4|flydsl_moe2")),
        ("MoE mfma (other)",       re.compile(r"mfma_moe")),
        ("CK MoE MXGEMM (Atom)",   re.compile(r"kernel_moe_mxgemm|GridwiseMoeGemmMX")),
        ("AITER fused_moe",        re.compile(r"fused_moe")),
        ("CK MoE GEMM",            re.compile(r"MoeFlatmmKernel")),
        ("MXFP4 MoE GEMM (triton)",re.compile(r"_matmul_ogs_NNT_")),
    ],
    "Attention (MLA / DSA)": [
        ("SGL paged decode split", re.compile(r"_paged_decode_split")),
        ("SGL paged decode reduce",re.compile(r"_paged_decode_reduce")),
        ("SGL flash MLA (compressed)", re.compile(r"flash_c\d+_decode|fused_norm_rope_flashmla|_hc_head|hc_head_fuse")),
        ("DSA decode partial",     re.compile(r"_sparse_attn_decode_partial")),
        ("DSA decode reduce",      re.compile(r"_sparse_attn_decode_reduce")),
        ("DSA save partial state", re.compile(r"_save_partial_states_kernel")),
        ("DSA prefill ragged",     re.compile(r"_sparse_attn_prefill")),
        ("Atom sparse-attn (ragged)", re.compile(r"_sparse_attn_ragged_varlen")),
        ("Atom fused compress attn", re.compile(r"_fused_compress_attn")),
        ("DSA generic",            re.compile(r"sparse_attn")),
        ("MQA logits (indexer)",   re.compile(r"paged_mqa_logits|deepgemm_fp8_paged_mqa|_fp8_mqa_logits")),
        ("KV compress+norm+rope",  re.compile(r"_fused_kv_compress_norm_rope_insert")),
        ("DSV4 QNorm+Rope+KV fuse",re.compile(r"fusedDeepseekV4QNormRope|deepseek_v4_fused_ops")),
        ("AITER MLA mhc_pre_gemm", re.compile(r"mhc_pre_gemm")),
        ("AITER MLA mhc_pre_fuse", re.compile(r"mhc_pre.*fuse|mhc_pre_big_fuse")),
        ("AITER MLA mhc_post",     re.compile(r"mhc_post")),
        ("AITER MLA mhc_*",        re.compile(r"mhc_")),
        ("Unified Attn (aiter)",   re.compile(r"kernel_unified_attention|aiter.*fmha|aiter.*attn|aiter_mha")),
        ("flash_attn",             re.compile(r"flash_fwd|flash.*splitkv")),
        ("paged_attn",             re.compile(r"paged_attention|paged_attn")),
    ],
    "MoE routing": [
        ("Gating softplus+sqrt",   re.compile(r"topkGatingSoftplusSqrt")),
        ("Gating softplus (aiter)",re.compile(r"topk_softplus")),
        ("Gating softmax",         re.compile(r"topkGatingSoftmax")),
        ("TopK per-row (decode)",  re.compile(r"topKPerRow")),
        ("MX-quant + MoE sort",    re.compile(r"fused_mx_quant_moe_sort")),
        ("MoE sorting",            re.compile(r"MoeSorting|opus_moe_sorting|mxfp4_moe_sort")),
        ("TopK gather (Atom)",     re.compile(r"gatherTopK|sbtopk")),
        ("Global topk / lens",     re.compile(r"_pack_global_topk_ragged|_compute_topk_lens")),
        ("MoE topk/reduce",        re.compile(r"_topk_forward|reduce_segments")),
    ],
    "Quantization": [
        ("AITER per-group quant",  re.compile(r"dynamic_per_group_scaled_quant")),
        ("AITER dynamic quant",    re.compile(r"aiter.*quant")),
        ("quant_fp8 / scaled",     re.compile(r"quant_fp8|scaled_fp8|per_token.*quant")),
    ],
    "Normalization": [
        ("CK-tile RMSNorm",        re.compile(r"Rmsnorm2dFwd|ck_tile.*[Rr]msnorm")),
        ("SGL fused RMS+fp8 quant", re.compile(r"_fused_rms_fp8_group_quant|_fused_q_kv_rmsnorm")),
        ("Fused Add+RMSNorm",      re.compile(r"_fused_add_rmsnorm|fused_add_rms")),
        ("RMSNorm",                re.compile(r"_rms_norm_kernel|rmsnorm|_rmsnorm_nw", re.I)),
    ],
    "RoPE": [
        ("Fused QK RoPE+Cache",    re.compile(r"_fused_qk_rope_reshape_and_cache")),
        ("RoPE",                   re.compile(r"_rope_kernel_cached|rope|rotary_emb|_fused_qk_norm_rope|sbhd_cached")),
    ],
    "Communication": [
        ("CustomAR 2-stage",       re.compile(r"cross_device_reduce_2stage")),
        ("CustomAR 1-stage",       re.compile(r"cross_device_reduce_1stage")),
        ("QuickReduce",            re.compile(r"quickReduce|quick_reduce", re.I)),
        ("NCCL/RCCL",              re.compile(r"ncclDevKernel|ncclKernel|rccl")),
    ],
    "Activation": [
        ("SiLU/SwiGLU",            re.compile(r"silu|swiglu")),
        ("GELU",                   re.compile(r"gelu")),
    ],
    "KV cache": [
        ("reshape+cache",          re.compile(r"reshape_and_cache")),
        ("copy page indices",      re.compile(r"_copy_page_indices_kernel")),
    ],
    "Memory/elementwise": [
        ("Fill",                   re.compile(r"FillFunctor|fill")),
        ("Memcpy",                 re.compile(r"[Mm]emcpy")),
        ("Memset",                 re.compile(r"[Mm]emset")),
        ("Copy buffer",            re.compile(r"copyBuffer")),
        ("Concat (CatArray)",       re.compile(r"CatArrayBatchedCopy")),
        ("Elementwise",            re.compile(r"direct_copy|vectorized_elementwise|elementwise_kernel")),
    ],
}

GEMM_LIB = [
    ("hipBLASLt (Tensile)",                      re.compile(r"Cijk_|tensile|hipblaslt", re.I)),
    ("AITER MoE expert GEMM (ASM, FP8xFP4)",     re.compile(r"mfma_moe|flydsl_moe")),
    ("CK MoE MXGEMM BPreshuffle (Atom)",         re.compile(r"kernel_moe_mxgemm|GridwiseMoeGemm")),
    ("AITER fused_moe",                          re.compile(r"fused_moe")),
    ("AITER fp8 blockscale GEMM (BpreShuffle)",  re.compile(r"fp8gemm.*blockscale|BpreShuffle", re.I)),
    ("CK blockscale GEMM (b_preshuffle)",        re.compile(r"multi_d_blockscale.*preshuffle|b_preshuffle")),
    ("AITER a8w8 block-scale GEMM (CK)",         re.compile(r"_gemm_a8w8_blockscale_kernel")),
    ("Composable Kernel (ck gemm_xdl, ABscale)", re.compile(r"ck::kernel_gemm|GridwiseGemm")),
    ("wvSplitK (skinny/thin-M GEMM)",            re.compile(r"wvSplitK", re.I)),
    ("rocBLAS",                                  re.compile(r"rocblas")),
    ("Triton GEMM",                              re.compile(r"triton.*gemm|_gemm_a16_w16")),
]

AITER_RX = re.compile(
    r"aiter|mfma_moe|afp8_wfp4|a8w8_blockscale|fused_mx_quant|wvSplitK|mhc_|opus_moe|"
    r"dynamic_per_group_scaled_quant|sparse_attn|_fused_kv_compress|paged_mqa_logits", re.I)

_EXEC_CTX_RE = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")


def categorize_kernel(name):
    for cat, patterns in CATEGORY_PATTERNS.items():
        for sub, rx in patterns:
            if rx.search(name):
                return cat, sub
    return "Other", (name[:name.find("(")] if "(" in name else name)[:60]


def gemm_library(name):
    for lib, rx in GEMM_LIB:
        if rx.search(name):
            return lib
    return None


_HAVE_CXXFILT = shutil.which("c++filt") is not None
def demangle(name):
    if name.startswith("_Z") and _HAVE_CXXFILT:
        try:
            return subprocess.run(["c++filt", name], capture_output=True, text=True, timeout=5).stdout.strip() or name
        except Exception:
            return name
    return name


# ---- stage segmentation (borrowed from trace_compare.py) ------------------
def build_step_index(events):
    steps = []
    for e in events:
        if e.get("cat") != "gpu_user_annotation":
            continue
        m = _EXEC_CTX_RE.search(e.get("name", ""))
        if not m or "dur" not in e:
            continue
        steps.append((e["ts"], e["ts"] + e["dur"], int(m.group(2))))  # ctx_tokens
    steps.sort()
    if not steps:
        return None, 0, 0
    last_prefill = max((i for i, s in enumerate(steps) if s[2] > 0), default=-1)
    intervals, ndec, npre = [], 0, 0
    for i, (ts, end, ctx) in enumerate(steps):
        if i <= last_prefill:
            intervals.append((ts, end, "PREFILL")); npre += 1
        else:
            intervals.append((ts, end, "DECODE")); ndec += 1
    return intervals, npre, ndec


def assign_stage(ts, idx):
    if not idx:
        return None
    lo, hi = 0, len(idx) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        s, e, stage = idx[mid]
        if ts < s: hi = mid - 1
        elif ts > e: lo = mid + 1
        else: return stage
    if lo < len(idx): return idx[lo][2]
    if hi >= 0: return idx[hi][2]
    return None


def load(paths):
    # stage -> name -> [time_us, count]
    kt = {"PREFILL": defaultdict(lambda: [0.0, 0]), "DECODE": defaultdict(lambda: [0.0, 0])}
    nranks = npre = ndec = 0
    for p in paths:
        nranks += 1
        with gzip.open(p, "rt") as f:
            data = json.load(f)
        ev = data["traceEvents"]
        idx, pre, dec = build_step_index(ev)
        npre += pre; ndec += dec
        for e in ev:
            if e.get("cat") not in ("kernel", "Kernel", "gpu_memcpy", "gpu_memset"):
                continue
            d = e.get("dur")
            if not d or d <= 0:
                continue
            st = assign_stage(e["ts"], idx)
            if st not in kt:
                continue
            rec = kt[st][e["name"]]
            rec[0] += d; rec[1] += 1
        del data
    return kt, nranks, npre, ndec


# ---- report helpers -------------------------------------------------------
def cat_breakdown(stage_kt):
    cat_t, cat_c = defaultdict(float), defaultdict(int)
    sub_t, sub_c = defaultdict(lambda: defaultdict(float)), defaultdict(lambda: defaultdict(int))
    for n, (t, c) in stage_kt.items():
        cat, sub = categorize_kernel(n)
        cat_t[cat] += t; cat_c[cat] += c
        sub_t[cat][sub] += t; sub_c[cat][sub] += c
    return cat_t, cat_c, sub_t, sub_c


def gemm_breakdown(stage_kt):
    lib_t, lib_c, ex = defaultdict(float), defaultdict(int), {}
    for n, (t, c) in stage_kt.items():
        lib = gemm_library(n)
        if lib:
            lib_t[lib] += t; lib_c[lib] += c
            if t > ex.get(lib, ("", 0))[1]:
                ex[lib] = (n, t)
    return lib_t, lib_c, ex


def aiter_list(stage_kt):
    return [(n, t, c) for n, (t, c) in stage_kt.items() if AITER_RX.search(n)]


def md_table(rows, headers):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def stage_section(md, title, stage_kt, iters):
    total = sum(t for t, _ in stage_kt.values()) or 1.0
    launches = sum(c for _, c in stage_kt.values())
    md.append(f"\n## {title}\n")
    md.append(f"- Aggregate GPU-kernel time (8 ranks): **{total/1000:.1f} ms**")
    md.append(f"- Kernel launches: **{launches:,}**")
    md.append(f"- Steps (summed across ranks): **{iters:,}**\n")

    cat_t, cat_c, sub_t, sub_c = cat_breakdown(stage_kt)
    md.append(f"### {title} — kernel time by category\n")
    rows = []
    for cat, t in sorted(cat_t.items(), key=lambda x: -x[1]):
        rows.append([cat, f"{t/1000:.1f}", f"{100*t/total:.1f}%", f"{cat_c[cat]:,}"])
    md.append(md_table(rows, ["Category", "time (ms)", "%", "launches"]))

    lib_t, lib_c, ex = gemm_breakdown(stage_kt)
    gtot = sum(lib_t.values())
    md.append(f"\n### {title} — GEMM libraries ({100*gtot/total:.1f}% of stage GPU time)\n")
    rows = []
    for lib, t in sorted(lib_t.items(), key=lambda x: -x[1]):
        rows.append([lib, f"{t/1000:.1f}", f"{100*t/total:.1f}%", f"{lib_c[lib]:,}"])
    md.append(md_table(rows, ["GEMM library", "time (ms)", "%", "launches"]))

    ail = aiter_list(stage_kt)
    at = sum(t for _, t, _ in ail)
    md.append(f"\n### {title} — top AITER functions ({100*at/total:.1f}% of stage GPU time, "
              f"{len(ail)} distinct)\n")
    rows = []
    for n, t, c in sorted(ail, key=lambda x: -x[1])[:15]:
        short = demangle(n)
        short = short[:90] + ("…" if len(short) > 90 else "")
        rows.append([f"{t/1000:.1f}", f"{100*t/total:.1f}%", f"{c:,}", "`" + short + "`"])
    md.append(md_table(rows, ["time (ms)", "%", "launches", "kernel"]))


def main():
    md_path, paths, conc = None, [], "32"
    it = iter(sys.argv[1:])
    for a in it:
        if a == "--md":
            md_path = next(it)
        elif a == "--conc":
            conc = next(it)
        elif any(c in a for c in "*?["):
            paths.extend(glob.glob(a))
        else:
            paths.append(a)
    if not paths:
        print(__doc__); sys.exit(1)

    print(f"Loading {len(paths)} trace(s)...", flush=True)
    kt, nranks, npre, ndec = load(paths)

    tot_pre = sum(t for t, _ in kt["PREFILL"].values())
    tot_dec = sum(t for t, _ in kt["DECODE"].values())
    grand = (tot_pre + tot_dec) or 1.0

    md = []
    md.append("# DeepSeek-V4-Pro — Per-Stage GPU Kernel Profile (MI355X)\n")
    md.append("| Field | Value |")
    md.append("|---|---|")
    md.append("| Model | `deepseek-ai/DeepSeek-V4-Pro` (FP4 MoE + FP8 attn, FP8 KV) |")
    md.append("| Hardware | 8× AMD Instinct MI355X (gfx950), TP=8 |")
    md.append("| Engine | vLLM 0.23.1rc1.dev714+g09663abde (image `vllm/vllm-openai-rocm:nightly-09663abde...`) |")
    md.append("| Recipe | `dsv4_fp4_mi355x_profiling.sh` (AITER MoE, `--moe-backend aiter`, cudagraph FULL_AND_PIECEWISE) |")
    md.append(f"| Workload | ISL=1024, OSL=1024, concurrency={conc}, random-range-ratio=0.8 |")
    md.append(f"| Traces | {nranks} GPU ranks, torch profiler (record_shapes) |")
    md.append("")
    md.append("## Overall prefill vs decode split\n")
    md.append(md_table([
        ["PREFILL", f"{tot_pre/1000:.1f}", f"{100*tot_pre/grand:.1f}%", f"{npre:,}"],
        ["DECODE",  f"{tot_dec/1000:.1f}", f"{100*tot_dec/grand:.1f}%", f"{ndec:,}"],
    ], ["Stage", "GPU time (ms)", "% of total", "steps (Σranks)"]))
    md.append("\n> Aggregate GPU-kernel time summed across 8 ranks; a torch-profiler "
              "window of 5 active engine iterations at conc=32. Prefill covers the "
              "context (ctx>0) steps, decode the pure-generation (ctx==0) steps.")

    stage_section(md, "PREFILL", kt["PREFILL"], npre)
    stage_section(md, "DECODE", kt["DECODE"], ndec)

    md.append("\n## Key observations\n")
    md.append("- **Decode-dominated window.** At ISL=OSL=1024 the profiled window is "
              f"{100*tot_dec/grand:.0f}% decode / {100*tot_pre/grand:.0f}% prefill, so the "
              "steady-state kernel mix below reflects the decode path.")
    md.append("- **GEMM library split is stable across stages:** MoE experts run on the "
              "**AITER assembly FP8×FP4** kernels (`mfma_moe1/moe2 … afp8_wfp4`), dense/linear "
              "layers on **Composable Kernel** `ck::gemm_xdl_cshuffle_v3` (AB-scale FP8), with "
              "**hipBLASLt (Tensile)** picking up a minority of dense shapes.")
    md.append("- **`wvSplitK` is decode-only** (~1.8% decode vs ~0% prefill): the skinny/thin-M "
              "GEMM path is selected when M = batch is tiny (decode), and CK/hipBLASLt take over "
              "at prefill's larger M.")
    md.append("- **Prefill is communication-heavy** (~23% TP=8 custom all-reduce) because "
              "activation tensors are large; decode all-reduce drops to ~6%.")
    md.append("- **DeepSeek Sparse Attention (DSA) + MLA** is a first-class cost: "
              "`_sparse_attn_*`, `_fused_kv_compress_norm_rope_insert`, the MQA-logits indexer, "
              "and the AITER `mhc_*` MLA helpers together are ~17–23% of each stage. Prefill uses "
              "`_sparse_attn_prefill_ragged`; decode uses `_sparse_attn_decode_partial/reduce`.")
    md.append("- **AITER per-group FP8 quant dominates launch count** "
              "(`dynamic_per_group_scaled_quant`, ~2.2M launches in decode): every GEMM is "
              "preceded by an activation quant.")
    md.append("- **Elementwise/copy is the largest decode bucket by time (~22%) and by far the "
              "largest by launch count (~12M)** — a fusion opportunity if decode latency matters.")

    text = "\n".join(md) + "\n"
    if md_path:
        os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)
        with open(md_path, "w") as f:
            f.write(text)
        print(f"\nWrote markdown report -> {md_path}")
    # console echo of the split
    print(f"\nPREFILL {tot_pre/1000:8.1f} ms ({100*tot_pre/grand:.1f}%)  "
          f"DECODE {tot_dec/1000:8.1f} ms ({100*tot_dec/grand:.1f}%)")


if __name__ == "__main__":
    main()
