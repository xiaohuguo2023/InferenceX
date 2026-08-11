import gzip, json, sys, re
from collections import defaultdict
BACKEND = [
    ("AITER asm (hsaco .co)",        re.compile(r"fp8gemm.*blockscale|BpreShuffle|bf16gemm_fp32bf16", re.I)),
    ("AITER flydsl (DSL-gen MFMA)",  re.compile(r"mfma_moe|flydsl|hgemm_bf16")),
    ("Composable Kernel (CK)",       re.compile(r"\bck::|GridwiseGemm")),
    ("hipBLASLt (Tensile)",          re.compile(r"Cijk_")),
    ("AITER JIT C++/HIP (aiter::)",  re.compile(r"aiter|_ZN5aiter|mhc_|opus_moe_sorting|topk_softplus|dynamic_per_group_scaled_quant|fused_mx_quant_moe_sort|add_rmsnorm_quant")),
    ("SGLang gluon (Triton)",        re.compile(r"_gluon_")),
    ("SGLang Triton",                re.compile(r"_paged_decode_(split|reduce)|_fused_rms_fp8_group_quant|_fused_qk_norm_rope_store|_fused_clamp_silu_mul|apply_rotary_emb_flat|moe_reduction_kernel|_hash_topk_triton|memcpy_triton")),
    ("SGLang jit_kernel (C++/HIP)",  re.compile(r"flash_c\d+_decode|fused_q_indexer|deepseek_v4_topk_transform|fused_norm_rope|_fill_compress_tail|_hc_head|_v4_paged_decode_indices|_init_compressed_attn_metadata|plan_compress_decode")),
    ("PyTorch native (at::native)",  re.compile(r"at::native")),
    ("rocPRIM",                      re.compile(r"rocprim")),
    ("RCCL",                         re.compile(r"nccl", re.I)),
]
def be(n):
    for name, rx in BACKEND:
        if rx.search(n): return name
    return "other/unclassified"
kt=defaultdict(float)
for p in sys.argv[1:]:
    for e in json.load(gzip.open(p,"rt"))["traceEvents"]:
        if e.get("cat") in ("kernel","Kernel") and e.get("dur"): kt[e["name"]]+=e["dur"]
tot=sum(kt.values()) or 1.0
bt=defaultdict(float)
for n,t in kt.items(): bt[be(n)]+=t
print(f"total {tot/1000:.2f} ms")
for b,t in sorted(bt.items(),key=lambda x:-x[1]):
    print(f"{100*t/tot:5.1f}%  {t/1000:7.2f} ms  {b}")
