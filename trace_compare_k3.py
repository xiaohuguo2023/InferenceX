#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Compare PyTorch profiler traces across TP configurations.

Parses Chrome Trace Event JSON (.pt.trace.json.gz) files produced by
vLLM's torch profiler, segments into prefill/decode stages, categorizes
GPU kernels by implementation, and produces per-rank imbalance analysis
plus TP-vs-TP comparison tables.

Usage:
    python trace_compare.py \
        --tp-dirs 4TP/vllm_traces 8TP/vllm_traces \
        --tp-labels TP4 TP8 \
        --output-csv comparison.csv
"""

import argparse
import csv
import gzip
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Kernel categorization: Category -> Sub-kernel -> regex pattern
# Order matters within a category: first match wins.
# ---------------------------------------------------------------------------
CATEGORY_PATTERNS: dict[str, list[tuple[str, re.Pattern]]] = {
    "GEMM": [
        # --- ROCm / Kimi-K3 ---
        ("AITER flydsl MoE",       re.compile(r"mfma_moe")),
        ("AITER wfp4 batched",     re.compile(r"_batched_gemm_a16wfp4")),
        ("AITER hgemm bf16",       re.compile(r"hgemm_bf16")),
        ("Triton GEMM (aiter)",    re.compile(r"_gemm_a16_w16_kernel")),
        ("CK MoE GEMM",           re.compile(r"MoeFlatmmKernel")),
        ("MXFP4 MoE GEMM",        re.compile(r"_matmul_ogs_NNT_")),
        ("MoE scatter/finalize",   re.compile(r"_finalize_matmul_scatter")),
        ("hipBLASLt (Cijk)",       re.compile(r"Cijk_")),
        ("rocBLAS splitK",         re.compile(r"wvSplitK")),
        ("hipBLASLt",              re.compile(r"hipblaslt|hipblas_lt", re.I)),
        # --- NVIDIA ---
        # B300 Kimi-K3 stack (provisional -- verify against a real trace).
        ("FlashInfer TRTLLM MoE",  re.compile(r"trtllm_fused_moe|fused_moe.*trtllm|flashinfer.*fused_moe")),
        ("CUTLASS MXFP4 GEMM",    re.compile(r"bmm_(?:Bfloat16|MxE4m3)_MxE2m1(?:Bfloat16|MxE4m3)_")),
        ("NVJet GEMM",            re.compile(r"nvjet_tst_")),
        ("cuBLASLt splitK",       re.compile(r"cublasLt::splitKreduce_kernel")),
        ("Triton template GEMM",  re.compile(r"triton_tem_fused_|triton_per_fused_.*mm")),
        # --- Cross-platform ---
        ("Triton fused GEMM",     re.compile(r"triton.*gemm|gemm.*triton", re.I)),
    ],
    "Communication": [
        # --- Cross-platform ---
        ("CustomAR 2-stage",       re.compile(r"cross_device_reduce_2stage")),
        ("CustomAR 1-stage",       re.compile(r"cross_device_reduce_1stage")),
        # --- ROCm ---
        ("QuickReduce",            re.compile(r"quickReduce|quick_reduce", re.I)),
        ("Iris two_shot AR",       re.compile(r"persistent_all_reduce_two_shot")),
        ("Iris device_barrier",    re.compile(r"_device_barrier_kernel")),
        ("Iris ReduceScatter",     re.compile(r"_reduce_scatter_kernel")),
        ("Iris AllGather",         re.compile(r"_all_gather_kernel")),
        # --- NVIDIA ---
        ("Lamport 1-shot AR",     re.compile(r"allreduce_fusion_kernel_oneshot_lamport")),
        ("Lamport quantize",      re.compile(r"quantize_with_block_size")),
        ("Lamport 2-shot AR",     re.compile(r"allreduce_fusion_kernel_twoshot")),
        ("SymmMem multimem AR",   re.compile(r"multimem_all_reduce_kernel")),
        ("SymmMem 2-shot AR",     re.compile(r"two_shot_all_reduce_kernel")),
        # --- Cross-platform ---
        ("NCCL/RCCL",              re.compile(r"ncclDevKernel|ncclKernel|nccl_|rcclGenericKernel")),
    ],
    "KDA Linear Attn": [
        # B300 Kimi-K3 stack (provisional -- verify against a real trace).
        ("FlashKDA (NV)",          re.compile(r"flash_?kda", re.I)),
        # --- Kimi-K3 Kimi-Delta-Attention (linear attention, fla-style Triton) ---
        ("KDA gated-delta",        re.compile(r"chunk_gated_delta_rule|fused_recurrent_kda|chunk_kda")),
        ("KDA GLA",                re.compile(r"chunk_gla")),
        ("KDA conv1d",             re.compile(r"causal_conv1d")),
        ("KDA gate/state",         re.compile(r"kda_gate|recompute_w_u|gather_initial_states|l2norm_fwd")),
    ],
    "Attention": [
        # --- Kimi-K3 MLA ---
        # B300 Kimi-K3 stack (provisional -- verify against a real trace).
        ("TokenSpeed MLA (NV)",    re.compile(r"tokenspeed|token_speed", re.I)),
        ("TRTLLM ragged MLA",      re.compile(r"trtllm.*ragged|ragged.*attention", re.I)),
        ("MLA gluon (decode)",     re.compile(r"_mla_gluon")),
        ("MLA merge states",       re.compile(r"merge_attn_states")),
        ("MLA attn residual",      re.compile(r"_attn_res")),
        # --- ROCm ---
        ("Unified Attn 2D",        re.compile(r"kernel_unified_attention_2d")),
        ("Unified Attn 3D",        re.compile(r"kernel_unified_attention_3d")),
        ("aiter attn",             re.compile(r"aiter.*fmha|aiter.*attn|aiter_mha")),
        ("flash_attn fwd",         re.compile(r"flash_fwd")),
        ("flash_attn splitkv",     re.compile(r"flash.*splitkv")),
        ("paged_attn",             re.compile(r"paged_attention|paged_attn")),
        # --- NVIDIA (SM100 / Blackwell) ---
        ("SM100 FMHA sliding",    re.compile(r"fmhaSm100fKernel.*SlidingOrChunked")),
        ("SM100 FMHA causal",     re.compile(r"fmhaSm100fKernel")),
    ],
    "Quantization": [
        ("dynamic per-group quant", re.compile(r"dynamic_per_group_scaled_quant|scaled_quant")),
    ],
    "Normalization": [
        # --- ROCm / Kimi-K3 ---
        ("Add+RMSNorm+quant",      re.compile(r"add_rmsnorm_quant")),
        ("Fused Add+RMSNorm",      re.compile(r"_fused_add_rmsnorm")),
        ("RMSNorm",                re.compile(r"_rms_norm_kernel|rmsnorm")),
        ("LayerNorm",              re.compile(r"layernorm|layer_norm")),
        # --- NVIDIA (Triton torch.compile) ---
        ("Triton fused RMSNorm",  re.compile(r"triton_red_fused.*rsqrt")),
    ],
    "RoPE": [
        ("Fused QK RoPE+Cache",    re.compile(r"_fused_qk_rope_reshape_and_cache")),
        ("RoPE cached",            re.compile(r"_rope_kernel_cached")),
    ],
    "MoE Routing": [
        # --- ROCm / Kimi-K3 ---
        ("AITER grouped_topk",     re.compile(r"grouped_topk")),
        ("MoE sort mxfp4",         re.compile(r"mxfp4_moe_sort|fused_mx_quant_moe_sort|opus_moe_sorting")),
        ("MoE reduction",          re.compile(r"moe_reduction")),
        ("MoE Gating (softmax)",   re.compile(r"topkGatingSoftmax")),
        ("MoE Sorting",            re.compile(r"MoeSorting")),
        ("MoE Reduce",             re.compile(r"reduce_segments")),
        ("MoE ClearWS",            re.compile(r"MoeSortingClearWorkspace")),
        ("MoE TopK",               re.compile(r"_topk_forward")),
        ("MoE Routing Compute",    re.compile(r"_combined_routing_compute")),
        ("MoE Routing Memset",     re.compile(r"_combined_routing_memset")),
        ("MoE Bitmatrix",          re.compile(r"_sum_bitmatrix_rows")),
        # --- NVIDIA ---
        ("MoE Routing (NV)",      re.compile(r"moe::dev::routing::")),
        ("MoE Finalize (NV)",     re.compile(r"moe::dev::finalize::")),
        ("Triton MoE fused",      re.compile(r"triton.*moe_forward")),
    ],
    "Activation": [
        ("SiLU/SwiGLU",            re.compile(r"silu|swiglu|silu_and_mul")),
        ("GELU",                   re.compile(r"gelu")),
    ],
    "KV Cache": [
        ("reshape+cache flash",    re.compile(r"reshape_and_cache")),
        ("copy page indices",      re.compile(r"_copy_page_indices_kernel")),
    ],
    "Memory": [
        ("Fill",                   re.compile(r"FillFunctor")),
        ("Memcpy",                 re.compile(r"[Mm]emcpy")),
        ("Memset",                 re.compile(r"[Mm]emset")),
    ],
    "Sampling": [
        ("ArgMax/Reduce",          re.compile(r"reduce_kernel.*ArgMax|ReduceOp.*ArgMax")),
        ("Softmax (logits)",       re.compile(r"cunn_SoftMaxForward")),
        ("RNG sampling",           re.compile(r"distribution_elementwise_grid_stride_kernel")),
        ("Sampling",               re.compile(r"sample|sampling", re.I)),
    ],
    "Triton Fused": [
        ("Triton fused op",        re.compile(r"triton_poi_fused_|triton_red_fused_")),
    ],
}


def categorize_kernel(name: str) -> tuple[str, str]:
    """Return (category, sub_kernel) for a GPU kernel name."""
    for cat, patterns in CATEGORY_PATTERNS.items():
        for sub_name, regex in patterns:
            if regex.search(name):
                return cat, sub_name
    return "Other", _shorten_kernel_name(name)


def _shorten_kernel_name(name: str) -> str:
    """Shorten verbose C++ template kernel names for display."""
    if len(name) > 80:
        paren = name.find("(")
        if paren > 0:
            name = name[:paren]
        if len(name) > 80:
            name = name[:77] + "..."
    return name


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class SubKernelStats:
    total_time_us: float = 0.0
    call_count: int = 0
    raw_names: set = field(default_factory=set)

    @property
    def avg_time_us(self) -> float:
        return self.total_time_us / self.call_count if self.call_count else 0.0


@dataclass
class CategoryStats:
    total_time_us: float = 0.0
    call_count: int = 0
    sub_kernels: dict[str, SubKernelStats] = field(default_factory=dict)


@dataclass
class StageStats:
    categories: dict[str, CategoryStats] = field(default_factory=dict)

    @property
    def total_time_us(self) -> float:
        return sum(c.total_time_us for c in self.categories.values())

    def add_kernel(self, cat: str, sub: str, dur: float, raw_name: str):
        if cat not in self.categories:
            self.categories[cat] = CategoryStats()
        cs = self.categories[cat]
        cs.total_time_us += dur
        cs.call_count += 1
        if sub not in cs.sub_kernels:
            cs.sub_kernels[sub] = SubKernelStats()
        sk = cs.sub_kernels[sub]
        sk.total_time_us += dur
        sk.call_count += 1
        sk.raw_names.add(raw_name[:120])


@dataclass
class RankResult:
    rank: int
    stages: dict[str, StageStats] = field(default_factory=dict)


@dataclass
class TPConfig:
    label: str
    dir_path: str
    ranks: list[RankResult] = field(default_factory=list)
    world_size: int = 0
    num_decode_iters: int = 0


# ---------------------------------------------------------------------------
# Trace parsing
# ---------------------------------------------------------------------------
def discover_traces(dir_path: str) -> list[str]:
    """Find worker trace files in a directory, sorted by name."""
    files = []
    for f in os.listdir(dir_path):
        if f.endswith(".pt.trace.json.gz") and "async_llm" not in f:
            files.append(os.path.join(dir_path, f))
    return sorted(files)


_EXEC_CTX_RE = re.compile(
    r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)"
)


def parse_single_trace(filepath: str) -> RankResult:
    """
    Parse one rank's trace file. Returns aggregated per-stage stats.
    Uses gpu_user_annotation execute_context events for GPU-side step
    boundaries, which gives accurate per-stage kernel attribution even
    with async scheduling.
    """
    print(f"  Loading {os.path.basename(filepath)} ...", end=" ", flush=True)
    with gzip.open(filepath, "rt") as f:
        data = json.load(f)
    print("done.", flush=True)

    dist_info = data.get("distributedInfo", {})
    rank = dist_info.get("rank", -1)

    events = data["traceEvents"]

    # --- Build GPU-side step boundaries from gpu_user_annotation ---
    gpu_steps = _parse_gpu_steps(events)
    step_index, num_decode_iters = _build_step_index(gpu_steps)

    # If no gpu_user_annotations, fall back to CPU-side annotations
    if step_index is None:
        cpu_steps = _parse_cpu_steps(events)
        step_index, num_decode_iters = _build_step_index(cpu_steps)

    # --- Aggregate GPU kernels by stage and category ---
    result = RankResult(rank=rank)
    result.stages["PREFILL"] = StageStats()
    result.stages["DECODE"] = StageStats()

    for e in events:
        if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        dur = e.get("dur")
        if dur is None or dur <= 0:
            continue

        ts = e["ts"]
        name = e["name"]
        cat_label = e["cat"]

        if cat_label in ("gpu_memcpy", "gpu_memset"):
            cat, sub = "Memory", "Memcpy" if "memcpy" in cat_label else "Memset"
        else:
            cat, sub = categorize_kernel(name)

        stage = _assign_stage_from_index(ts, step_index)
        if stage and stage in result.stages:
            result.stages[stage].add_kernel(cat, sub, dur, name)

    del data
    return result, num_decode_iters


@dataclass
class StepInfo:
    ts: float
    end_ts: float
    ctx_tokens: int
    gen_tokens: int

    @property
    def has_prefill(self) -> bool:
        return self.ctx_tokens > 0


def _parse_gpu_steps(events: list) -> list[StepInfo]:
    """Extract execute_context steps from gpu_user_annotation events."""
    steps = []
    for e in events:
        if e.get("cat") != "gpu_user_annotation":
            continue
        m = _EXEC_CTX_RE.search(e.get("name", ""))
        if not m or "dur" not in e:
            continue
        steps.append(StepInfo(
            ts=e["ts"],
            end_ts=e["ts"] + e["dur"],
            ctx_tokens=int(m.group(2)),
            gen_tokens=int(m.group(4)),
        ))
    steps.sort(key=lambda s: s.ts)
    return steps


def _parse_cpu_steps(events: list) -> list[StepInfo]:
    """Fallback: extract execute_context steps from CPU user_annotation."""
    steps = []
    for e in events:
        if e.get("cat") != "user_annotation":
            continue
        m = _EXEC_CTX_RE.search(e.get("name", ""))
        if not m or "dur" not in e:
            continue
        steps.append(StepInfo(
            ts=e["ts"],
            end_ts=e["ts"] + e["dur"],
            ctx_tokens=int(m.group(2)),
            gen_tokens=int(m.group(4)),
        ))
    steps.sort(key=lambda s: s.ts)
    return steps


def _build_step_index(
    steps: list[StepInfo],
) -> tuple[Optional[list[tuple[float, float, str]]], int]:
    """
    Build a sorted list of (start, end, stage) intervals for fast lookup.

    Steps with context tokens > 0 are classified as PREFILL.
    Steps with context tokens == 0 (pure decode) AFTER the last prefill
    step are classified as DECODE.
    Early decode steps interleaved with prefill steps are also included
    as PREFILL to keep the prefill window contiguous.

    Returns (step_index, num_decode_iters).
    """
    if not steps:
        return None, 0

    last_prefill_idx = -1
    for i, s in enumerate(steps):
        if s.has_prefill:
            last_prefill_idx = i

    intervals = []
    num_decode = 0

    for i, s in enumerate(steps):
        if i <= last_prefill_idx:
            intervals.append((s.ts, s.end_ts, "PREFILL"))
        else:
            intervals.append((s.ts, s.end_ts, "DECODE"))
            num_decode += 1

    return intervals, num_decode


def _assign_stage_from_index(
    ts: float, step_index: Optional[list[tuple[float, float, str]]]
) -> Optional[str]:
    """Assign a GPU kernel to PREFILL or DECODE using step intervals."""
    if not step_index:
        return None

    # Binary search for the interval containing ts
    lo, hi = 0, len(step_index) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        start, end, stage = step_index[mid]
        if ts < start:
            hi = mid - 1
        elif ts > end:
            lo = mid + 1
        else:
            return stage

    # Kernel falls between steps (gap) — assign to nearest step
    if lo < len(step_index):
        return step_index[lo][2]
    if hi >= 0:
        return step_index[hi][2]
    return None


# ---------------------------------------------------------------------------
# Cross-rank analysis
# ---------------------------------------------------------------------------
@dataclass
class CrossRankCategoryStats:
    per_rank_time: list[float]
    per_rank_count: list[int]
    sub_kernels: dict[str, list[float]]

    @property
    def mean(self) -> float:
        return statistics.mean(self.per_rank_time) if self.per_rank_time else 0

    @property
    def max_val(self) -> float:
        return max(self.per_rank_time) if self.per_rank_time else 0

    @property
    def min_val(self) -> float:
        return min(self.per_rank_time) if self.per_rank_time else 0

    @property
    def straggler_rank(self) -> int:
        return self.per_rank_time.index(self.max_val) if self.per_rank_time else -1

    @property
    def imbalance_pct(self) -> float:
        m = self.mean
        if m <= 0:
            return 0.0
        return (self.max_val - self.min_val) / m * 100


def cross_rank_analysis(
    tp_config: TPConfig, stage: str
) -> dict[str, CrossRankCategoryStats]:
    """Compute per-category cross-rank statistics for a given stage."""
    # Collect all categories across all ranks
    all_cats = set()
    all_subs: dict[str, set] = defaultdict(set)
    for rr in tp_config.ranks:
        ss = rr.stages.get(stage)
        if not ss:
            continue
        for cat, cs in ss.categories.items():
            all_cats.add(cat)
            for sub in cs.sub_kernels:
                all_subs[cat].add(sub)

    result = {}
    for cat in sorted(all_cats):
        per_rank_time = []
        per_rank_count = []
        sub_times: dict[str, list[float]] = {s: [] for s in all_subs.get(cat, set())}

        for rr in tp_config.ranks:
            ss = rr.stages.get(stage)
            cs = ss.categories.get(cat) if ss else None
            per_rank_time.append(cs.total_time_us if cs else 0.0)
            per_rank_count.append(cs.call_count if cs else 0)

            for sub in sub_times:
                sk = cs.sub_kernels.get(sub) if cs else None
                sub_times[sub].append(sk.total_time_us if sk else 0.0)

        result[cat] = CrossRankCategoryStats(
            per_rank_time=per_rank_time,
            per_rank_count=per_rank_count,
            sub_kernels=sub_times,
        )

    return result


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------
IMBALANCE_THRESHOLD = 10.0  # percent


def print_header(text: str, width: int = 78):
    print()
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def print_per_rank_table(
    tp_config: TPConfig,
    stage: str,
    cr_stats: dict[str, CrossRankCategoryStats],
    num_decode_iters: int = 0,
):
    """Print per-rank breakdown for one TP config and one stage."""
    n_ranks = tp_config.world_size
    rank_hdrs = [f"R{i}" for i in range(n_ranks)]

    divider_by = num_decode_iters if (stage == "DECODE" and num_decode_iters > 0) else 1
    unit_label = "Per-Iter Avg (μs)" if divider_by > 1 else "(μs)"

    # Header
    hdr = f"{'Category':<22}"
    for rh in rank_hdrs:
        hdr += f"  {rh:>9}"
    hdr += f"  {'Imbal%':>7}  {'':>3}"
    print(hdr)
    print("─" * len(hdr))

    total_per_rank = [0.0] * n_ranks

    # Sort categories by max-rank time descending
    sorted_cats = sorted(
        cr_stats.items(), key=lambda x: x[1].max_val, reverse=True
    )

    for cat, crs in sorted_cats:
        vals = [t / divider_by for t in crs.per_rank_time]
        imbal = crs.imbalance_pct
        flag = "⚠" if imbal > IMBALANCE_THRESHOLD else "✓"

        row = f"{cat:<22}"
        for v in vals:
            row += f"  {v:>9.1f}"
        row += f"  {imbal:>6.1f}%  {flag:>3}"
        print(row)

        # Sub-kernel detail
        sorted_subs = sorted(
            crs.sub_kernels.items(),
            key=lambda x: max(x[1]) if x[1] else 0,
            reverse=True,
        )
        for sub_name, sub_vals in sorted_subs:
            sv = [t / divider_by for t in sub_vals]
            if max(sv) < 0.5:
                continue
            row = f"  └ {sub_name:<18}"
            for v in sv:
                row += f"  {v:>9.1f}"
            print(row)

        for i, v in enumerate(vals):
            total_per_rank[i] += v

    # Total row
    print("─" * len(hdr))
    row = f"{'TOTAL':<22}"
    for v in total_per_rank:
        row += f"  {v:>9.1f}"
    mean_total = statistics.mean(total_per_rank) if total_per_rank else 0
    max_total = max(total_per_rank) if total_per_rank else 0
    imbal_total = ((max_total - min(total_per_rank)) / mean_total * 100) if mean_total > 0 else 0
    row += f"  {imbal_total:>6.1f}%"
    print(row)

    # Straggler info
    if total_per_rank:
        straggler = total_per_rank.index(max(total_per_rank))
        pct_above = (max_total - mean_total) / mean_total * 100 if mean_total > 0 else 0
        if pct_above > 1.0:
            # Find which category contributes most to the straggler
            worst_cat = ""
            worst_imbal = 0
            for cat, crs in cr_stats.items():
                if crs.imbalance_pct > worst_imbal:
                    worst_imbal = crs.imbalance_pct
                    worst_cat = cat
            print(
                f"Straggler: R{straggler} ({max_total:.1f}{unit_label}) — "
                f"{pct_above:.1f}% above mean ({mean_total:.1f})"
            )
            if worst_imbal > IMBALANCE_THRESHOLD:
                print(f"  Cause: {worst_cat} is {worst_imbal:.1f}% imbalanced")
        else:
            print(f"All ranks balanced (max {pct_above:.1f}% above mean)")

    # Comm/Compute ratio
    comm_total = cr_stats.get("Communication")
    if comm_total:
        compute_total = sum(
            crs.max_val / divider_by
            for cat, crs in cr_stats.items()
            if cat not in ("Communication", "Memory")
        )
        comm_max = comm_total.max_val / divider_by
        if compute_total > 0:
            ratio = comm_max / (comm_max + compute_total) * 100
            print(f"Comm/Total ratio (bottleneck rank): {ratio:.1f}%")


def print_tp_comparison(
    configs: list[TPConfig],
    stage: str,
    all_cr_stats: list[dict[str, CrossRankCategoryStats]],
):
    """Print side-by-side TP comparison for one stage."""
    # Collect all categories
    all_cats = set()
    for cr in all_cr_stats:
        all_cats.update(cr.keys())

    labels = [c.label for c in configs]
    dividers = []
    for c in configs:
        if stage == "DECODE" and c.num_decode_iters > 0:
            dividers.append(c.num_decode_iters)
        else:
            dividers.append(1)

    unit = "per-iter μs" if any(d > 1 for d in dividers) else "μs"

    # Header
    hdr = f"{'Category / Kernel':<30}"
    for lb in labels:
        hdr += f"  {lb + ' max':>12}"
    if len(labels) == 2:
        hdr += f"  {'Ratio':>8}  {'Scaling':>10}"
    print(hdr)
    print("─" * len(hdr))

    # Sort categories by time in first config
    sorted_cats = sorted(
        all_cats,
        key=lambda c: all_cr_stats[0][c].max_val / dividers[0] if c in all_cr_stats[0] else 0,
        reverse=True,
    )

    totals = [0.0] * len(configs)

    for cat in sorted_cats:
        vals = []
        for i, cr in enumerate(all_cr_stats):
            v = cr[cat].max_val / dividers[i] if cat in cr else 0
            vals.append(v)
            totals[i] += v

        row = f"{cat:<30}"
        for v in vals:
            row += f"  {v:>12.1f}"
        if len(vals) == 2 and vals[0] > 0:
            ratio = vals[1] / vals[0]
            scaling = _scaling_label(cat, ratio)
            row += f"  {ratio:>7.2f}x  {scaling:>10}"
        print(row)

        # Sub-kernel detail
        all_subs = set()
        for cr in all_cr_stats:
            if cat in cr:
                all_subs.update(cr[cat].sub_kernels.keys())

        sorted_subs = sorted(
            all_subs,
            key=lambda s: max(all_cr_stats[0][cat].sub_kernels.get(s, [0])) / dividers[0]
            if cat in all_cr_stats[0] else 0,
            reverse=True,
        )

        for sub in sorted_subs:
            sub_vals = []
            for i, cr in enumerate(all_cr_stats):
                if cat in cr and sub in cr[cat].sub_kernels:
                    sv = max(cr[cat].sub_kernels[sub]) / dividers[i]
                else:
                    sv = 0
                sub_vals.append(sv)

            if max(sub_vals) < 1.0:
                continue

            row = f"  └ {sub:<26}"
            for v in sub_vals:
                row += f"  {v:>12.1f}"
            if len(sub_vals) == 2 and sub_vals[0] > 0:
                r = sub_vals[1] / sub_vals[0]
                row += f"  {r:>7.2f}x"
            print(row)

    # Total
    print("─" * len(hdr))
    row = f"{'TOTAL':<30}"
    for v in totals:
        row += f"  {v:>12.1f}"
    if len(totals) == 2 and totals[0] > 0:
        ratio = totals[1] / totals[0]
        row += f"  {ratio:>7.2f}x"
    print(row)

    # Comm/Compute for each config
    for i, cfg in enumerate(configs):
        cr = all_cr_stats[i]
        comm = cr["Communication"].max_val / dividers[i] if "Communication" in cr else 0
        compute = sum(
            cr[cat].max_val / dividers[i]
            for cat in cr
            if cat not in ("Communication", "Memory")
        )
        if comm + compute > 0:
            pct = comm / (comm + compute) * 100
            print(f"  {cfg.label} comm/total: {pct:.1f}%")


def _scaling_label(cat: str, ratio: float) -> str:
    """Generate a scaling assessment label."""
    if cat == "Communication":
        if ratio < 0.6:
            return "← better"
        elif ratio < 1.2:
            return "~ same"
        else:
            return "← worse"
    else:
        if 1.8 <= ratio <= 2.2:
            return "✓ linear"
        elif ratio < 1.8:
            return "sub-linear"
        else:
            return "super-lin"


def print_summary(
    configs: list[TPConfig],
    all_stage_cr: dict[str, list[dict[str, CrossRankCategoryStats]]],
):
    """Print key findings summary."""
    print_header("SUMMARY")

    findings = _collect_findings(configs, all_stage_cr)
    # Strip markdown bold for console output
    for i, f in enumerate(findings, 1):
        print(f"  {i}. {f.replace('**', '')}")

    if not findings:
        print("  No significant findings.")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------
def write_markdown(
    filepath: str,
    configs: list[TPConfig],
    all_stage_cr: dict[str, list[dict[str, CrossRankCategoryStats]]],
):
    """Write a formatted Markdown comparison report."""
    lines: list[str] = []
    w = lines.append

    w(f"# Trace Comparison: {' vs '.join(c.label for c in configs)}")
    w("")
    w("## Configuration")
    w("")
    w("| | " + " | ".join(c.label for c in configs) + " |")
    w("|---|" + "|".join(["---"] * len(configs)) + "|")
    w("| World size | " + " | ".join(str(c.world_size) for c in configs) + " |")
    w("| Decode iterations | " + " | ".join(str(c.num_decode_iters) for c in configs) + " |")
    w("")

    for stage in ("PREFILL", "DECODE"):
        cr_list = all_stage_cr.get(stage, [])
        if not cr_list or not any(cr_list):
            continue

        w(f"## {stage}")
        w("")

        # --- Per-rank table for each TP config ---
        for ci, cfg in enumerate(configs):
            cr = cr_list[ci]
            if not cr:
                continue

            n_ranks = cfg.world_size
            divider = cfg.num_decode_iters if (stage == "DECODE" and cfg.num_decode_iters > 0) else 1
            unit = "per-iter avg us" if divider > 1 else "us"

            w(f"### {cfg.label} Per-Rank Breakdown ({unit})")
            w("")

            rank_hdrs = [f"R{i}" for i in range(n_ranks)]
            w("| Category | Sub-kernel | " + " | ".join(rank_hdrs) + " | Imbal% |")
            w("|---|---|" + "|".join(["---:"] * n_ranks) + "|---:|")

            sorted_cats = sorted(cr.items(), key=lambda x: x[1].max_val, reverse=True)

            total_per_rank = [0.0] * n_ranks

            for cat, crs in sorted_cats:
                vals = [t / divider for t in crs.per_rank_time]
                imbal = crs.imbalance_pct
                flag = " :warning:" if imbal > IMBALANCE_THRESHOLD else ""
                vals_str = " | ".join(f"{v:.1f}" for v in vals)
                w(f"| **{cat}** | *(total)* | {vals_str} | {imbal:.1f}%{flag} |")

                for i, v in enumerate(vals):
                    total_per_rank[i] += v

                sorted_subs = sorted(
                    crs.sub_kernels.items(),
                    key=lambda x: max(x[1]) if x[1] else 0,
                    reverse=True,
                )
                for sub_name, sub_vals in sorted_subs:
                    sv = [t / divider for t in sub_vals]
                    if max(sv) < 0.5:
                        continue
                    sv_str = " | ".join(f"{v:.1f}" for v in sv)
                    w(f"| | {sub_name} | {sv_str} | |")

            total_str = " | ".join(f"{v:.1f}" for v in total_per_rank)
            mean_t = statistics.mean(total_per_rank) if total_per_rank else 0
            max_t = max(total_per_rank) if total_per_rank else 0
            min_t = min(total_per_rank) if total_per_rank else 0
            imbal_t = ((max_t - min_t) / mean_t * 100) if mean_t > 0 else 0
            w(f"| **TOTAL** | | {total_str} | {imbal_t:.1f}% |")
            w("")

            # Straggler
            if total_per_rank:
                straggler = total_per_rank.index(max_t)
                pct_above = (max_t - mean_t) / mean_t * 100 if mean_t > 0 else 0
                if pct_above > 1.0:
                    worst_cat = max(cr.items(), key=lambda x: x[1].imbalance_pct)[0]
                    worst_imbal = cr[worst_cat].imbalance_pct
                    w(f"> **Straggler**: R{straggler} ({max_t:.0f} {unit}) — "
                      f"{pct_above:.1f}% above mean. "
                      f"Main cause: {worst_cat} ({worst_imbal:.0f}% imbalanced)")
                else:
                    w(f"> All ranks balanced (max {pct_above:.1f}% above mean)")

            # Comm ratio
            comm_cr = cr.get("Communication")
            if comm_cr:
                compute = sum(
                    c.max_val / divider for cat, c in cr.items()
                    if cat not in ("Communication", "Memory")
                )
                comm_max = comm_cr.max_val / divider
                if comm_max + compute > 0:
                    ratio = comm_max / (comm_max + compute) * 100
                    w(f">\n> **Comm/Total ratio** (bottleneck rank): {ratio:.1f}%")
            w("")

        # --- TP comparison table ---
        if len(configs) >= 2 and all(cr_list):
            w(f"### {' vs '.join(c.label for c in configs)} Comparison (bottleneck rank)")
            w("")

            all_cats = set()
            for cr in cr_list:
                all_cats.update(cr.keys())

            dividers = []
            for c in configs:
                if stage == "DECODE" and c.num_decode_iters > 0:
                    dividers.append(c.num_decode_iters)
                else:
                    dividers.append(1)

            labels = [c.label for c in configs]
            hdr_cols = [f"{lb} max (us)" for lb in labels]
            if len(labels) == 2:
                hdr_cols += [f"{labels[0]} %", f"{labels[1]} %", "Ratio", "Scaling"]

            w("| Category | Sub-kernel | " + " | ".join(hdr_cols) + " |")
            w("|---|---|" + "|".join(["---:"] * len(hdr_cols)) + "|")

            sorted_cats = sorted(
                all_cats,
                key=lambda c: cr_list[0][c].max_val / dividers[0] if c in cr_list[0] else 0,
                reverse=True,
            )

            # Pre-compute totals for percentage calculation
            totals = [0.0] * len(configs)
            for cat in all_cats:
                for i, cr in enumerate(cr_list):
                    v = cr[cat].max_val / dividers[i] if cat in cr else 0
                    totals[i] += v

            running_totals = [0.0] * len(configs)
            for cat in sorted_cats:
                vals = []
                for i, cr in enumerate(cr_list):
                    v = cr[cat].max_val / dividers[i] if cat in cr else 0
                    vals.append(v)
                    running_totals[i] += v

                vals_str = " | ".join(f"{v:.1f}" for v in vals)
                extra = ""
                if len(vals) == 2:
                    pct0 = vals[0] / totals[0] * 100 if totals[0] > 0 else 0
                    pct1 = vals[1] / totals[1] * 100 if totals[1] > 0 else 0
                    if vals[0] > 0:
                        ratio = vals[1] / vals[0]
                        scaling = _scaling_label(cat, ratio)
                        extra = f" | {pct0:.1f}% | {pct1:.1f}% | {ratio:.2f}x | {scaling}"
                    else:
                        extra = f" | {pct0:.1f}% | {pct1:.1f}% | new |"
                w(f"| **{cat}** | *(total)* | {vals_str}{extra} |")

                # Sub-kernels
                all_subs = set()
                for cr in cr_list:
                    if cat in cr:
                        all_subs.update(cr[cat].sub_kernels.keys())

                for sub in sorted(all_subs, key=lambda s: max(cr_list[0][cat].sub_kernels.get(s, [0])) / dividers[0] if cat in cr_list[0] else 0, reverse=True):
                    sub_vals = []
                    for i, cr in enumerate(cr_list):
                        if cat in cr and sub in cr[cat].sub_kernels:
                            sv = max(cr[cat].sub_kernels[sub]) / dividers[i]
                        else:
                            sv = 0
                        sub_vals.append(sv)
                    if max(sub_vals) < 1.0:
                        continue
                    sv_str = " | ".join(f"{v:.1f}" for v in sub_vals)
                    extra = ""
                    if len(sub_vals) == 2:
                        pct0 = sub_vals[0] / totals[0] * 100 if totals[0] > 0 else 0
                        pct1 = sub_vals[1] / totals[1] * 100 if totals[1] > 0 else 0
                        if sub_vals[0] > 0:
                            r = sub_vals[1] / sub_vals[0]
                            extra = f" | {pct0:.1f}% | {pct1:.1f}% | {r:.2f}x |"
                        else:
                            extra = f" | {pct0:.1f}% | {pct1:.1f}% | new |"
                    w(f"| | {sub} | {sv_str}{extra} |")

            # Total row
            totals_str = " | ".join(f"{v:.1f}" for v in totals)
            extra = ""
            if len(totals) == 2 and totals[0] > 0:
                extra = f" | 100% | 100% | {totals[1]/totals[0]:.2f}x |"
            w(f"| **TOTAL** | | {totals_str}{extra} |")
            w("")

            # Comm/compute per config
            for i, cfg in enumerate(configs):
                cr = cr_list[i]
                comm = cr["Communication"].max_val / dividers[i] if "Communication" in cr else 0
                compute = sum(
                    cr[cat].max_val / dividers[i] for cat in cr
                    if cat not in ("Communication", "Memory")
                )
                if comm + compute > 0:
                    pct = comm / (comm + compute) * 100
                    w(f"> {cfg.label} comm/total: **{pct:.1f}%**")
            w("")

    # --- Summary ---
    w("## Summary")
    w("")
    findings = _collect_findings(configs, all_stage_cr)
    for i, f in enumerate(findings, 1):
        w(f"{i}. {f}")
    if not findings:
        w("No significant findings.")
    w("")

    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    print(f"\nMarkdown report written to {filepath}")


def _collect_findings(
    configs: list[TPConfig],
    all_stage_cr: dict[str, list[dict[str, CrossRankCategoryStats]]],
) -> list[str]:
    """Collect key findings for the summary section."""
    findings = []
    for stage, cr_list in all_stage_cr.items():
        if len(cr_list) < 2:
            continue

        for i, (cfg, cr) in enumerate(zip(configs, cr_list)):
            for cat, crs in cr.items():
                if crs.imbalance_pct > IMBALANCE_THRESHOLD:
                    findings.append(
                        f"**{cfg.label} {stage}**: {cat} has "
                        f"{crs.imbalance_pct:.0f}% rank imbalance "
                        f"(straggler: R{crs.straggler_rank})"
                    )

        dividers = []
        for c in configs:
            if stage == "DECODE" and c.num_decode_iters > 0:
                dividers.append(c.num_decode_iters)
            else:
                dividers.append(1)

        comm_vals, compute_vals = [], []
        for ci, cr in enumerate(cr_list):
            comm = cr["Communication"].max_val / dividers[ci] if "Communication" in cr else 0
            compute = sum(
                cr[cat].max_val / dividers[ci] for cat in cr
                if cat not in ("Communication", "Memory")
            )
            comm_vals.append(comm)
            compute_vals.append(compute)

        if len(comm_vals) == 2 and comm_vals[0] > 0:
            findings.append(
                f"**{stage}**: Comm time ratio "
                f"({configs[1].label}/{configs[0].label}): "
                f"{comm_vals[1]/comm_vals[0]:.2f}x"
            )

        for ci, (cfg, cr) in enumerate(zip(configs, cr_list)):
            total = comm_vals[ci] + compute_vals[ci]
            if total > 0:
                pct = comm_vals[ci] / total * 100
                if pct > 25:
                    findings.append(
                        f"**{stage}**: {cfg.label} is communication-bound "
                        f"({pct:.0f}% of time)"
                    )
    return findings


def write_csv(
    filepath: str,
    configs: list[TPConfig],
    all_stage_cr: dict[str, list[dict[str, CrossRankCategoryStats]]],
):
    """Write detailed per-rank, per-stage, per-category data to CSV."""
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "tp_label", "stage", "rank", "category", "sub_kernel",
            "total_time_us", "call_count", "avg_time_us",
        ])

        for cfg in configs:
            for stage_name in ("PREFILL", "DECODE"):
                for rr in cfg.ranks:
                    ss = rr.stages.get(stage_name)
                    if not ss:
                        continue
                    for cat, cs in sorted(ss.categories.items()):
                        for sub, sk in sorted(
                            cs.sub_kernels.items(),
                            key=lambda x: x[1].total_time_us,
                            reverse=True,
                        ):
                            w.writerow([
                                cfg.label, stage_name, rr.rank, cat, sub,
                                f"{sk.total_time_us:.1f}",
                                sk.call_count,
                                f"{sk.avg_time_us:.1f}",
                            ])

    print(f"\nCSV written to {filepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def process_tp_config(label: str, dir_path: str) -> TPConfig:
    """Parse all rank traces for one TP configuration."""
    print(f"\nProcessing {label} from {dir_path}")
    trace_files = discover_traces(dir_path)
    if not trace_files:
        print(f"  ERROR: No trace files found in {dir_path}")
        sys.exit(1)

    print(f"  Found {len(trace_files)} worker trace(s)")

    tp = TPConfig(label=label, dir_path=dir_path, world_size=len(trace_files))
    max_decode_iters = 0

    for filepath in trace_files:
        rr, n_decode = parse_single_trace(filepath)
        tp.ranks.append(rr)
        max_decode_iters = max(max_decode_iters, n_decode)

    # Sort ranks by rank ID
    tp.ranks.sort(key=lambda r: r.rank)
    tp.num_decode_iters = max_decode_iters
    print(f"  Ranks: {[r.rank for r in tp.ranks]}, decode iterations: {max_decode_iters}")
    return tp


def main():
    parser = argparse.ArgumentParser(
        description="Compare PyTorch profiler traces across TP configurations."
    )
    parser.add_argument(
        "--tp-dirs", nargs="+", required=True,
        help="Paths to trace directories (one per TP config).",
    )
    parser.add_argument(
        "--tp-labels", nargs="+",
        help="Labels for each TP config (default: dir basename).",
    )
    parser.add_argument(
        "--output-csv", default=None,
        help="Path to write CSV output.",
    )
    parser.add_argument(
        "--output-md", default=None,
        help="Path to write Markdown report.",
    )
    args = parser.parse_args()

    if args.tp_labels and len(args.tp_labels) != len(args.tp_dirs):
        print("ERROR: --tp-labels must match --tp-dirs count.")
        sys.exit(1)

    labels = args.tp_labels or [os.path.basename(d.rstrip("/")) for d in args.tp_dirs]

    # --- Parse all configs ---
    configs = []
    for label, dir_path in zip(labels, args.tp_dirs):
        configs.append(process_tp_config(label, dir_path))

    # --- Cross-rank analysis per stage ---
    all_stage_cr: dict[str, list[dict[str, CrossRankCategoryStats]]] = {}
    stages = ["PREFILL", "DECODE"]

    for stage in stages:
        print_header(f"STAGE: {stage}")
        cr_list = []

        for cfg in configs:
            cr = cross_rank_analysis(cfg, stage)
            cr_list.append(cr)

            if not cr:
                print(f"\n  {cfg.label}: No data for {stage} stage.")
                continue

            print(f"\n{cfg.label} Per-Rank Breakdown "
                  f"({'per-iter avg' if stage == 'DECODE' and cfg.num_decode_iters > 0 else 'total'}):")
            print_per_rank_table(cfg, stage, cr, cfg.num_decode_iters)

        all_stage_cr[stage] = cr_list

        # TP comparison
        if len(configs) >= 2 and all(cr_list):
            print(f"\n{' vs '.join(c.label for c in configs)} Comparison "
                  f"(bottleneck rank, {stage}):")
            print_tp_comparison(configs, stage, cr_list)

    # --- Summary ---
    print_summary(configs, all_stage_cr)

    # --- CSV ---
    if args.output_csv:
        write_csv(args.output_csv, configs, all_stage_cr)

    # --- Markdown ---
    if args.output_md:
        write_markdown(args.output_md, configs, all_stage_cr)


if __name__ == "__main__":
    main()
