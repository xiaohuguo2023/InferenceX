import gzip, json, sys, collections, re
f=sys.argv[1]
d=json.load(gzip.open(f)); ev=d["traceEvents"]
kern=[e for e in ev if e.get("cat")=="kernel" and "dur" in e]
tot=sum(e["dur"] for e in kern)

def classify(n):
    # returns (component, library)
    if "cross_device_reduce" in n: return ("Communication (TP all-reduce)","AITER custom-AR (C++/HIP)")
    if n.startswith("Cijk_"): return ("Dense / MLA-proj GEMM (bf16)","hipBLASLt (Tensile)")
    if "mfma_moe" in n: return ("MoE expert GEMM (FP8xFP4)","AITER flydsl (MFMA asm)")
    if "moe_reduction" in n: return ("MoE combine/reduce","Triton/CUDA")
    if "opus_moe_sorting" in n or "MoeSorting" in n: return ("MoE routing/sort","AITER ck_tile")
    if "grouped_topk" in n: return ("MoE routing/sort","AITER")
    if "_mla_gluon" in n: return ("MLA attention","AITER Gluon (Triton)")
    if "fmha_fwd" in n: return ("MLA attention","AITER FMHA (asm)")
    if "_attn_res" in n: return ("MLA attention","Triton")
    if any(k in n for k in("chunk_gated_delta_rule","chunk_kda","chunk_gla","recompute_w_u","kda_gate","causal_conv1d")):
        return ("KDA linear attention","Triton (fla)")
    if "add_rmsnorm_quant" in n: return ("Norm+Quant fusion","AITER")
    if "dynamic_per_group_scaled_quant" in n or "scaled_quant" in n: return ("Quantization","AITER")
    if n.startswith("triton_"): return ("Elementwise/fusion","Triton (inductor)")
    if "copyBuffer" in n or "Memcpy" in n or "Memset" in n: return ("Memory copy","ROCr/HIP")
    if "at::native" in n or "elementwise" in n or "reduce_kernel" in n: return ("Elementwise/other","PyTorch native")
    return ("Other/uncategorized","?")

comp=collections.defaultdict(float); lib=collections.defaultdict(float)
compkern=collections.defaultdict(lambda: collections.defaultdict(float))
for e in kern:
    c,l=classify(e["name"]); comp[c]+=e["dur"]; lib[l]+=e["dur"]
    compkern[c][e["name"]]+=e["dur"]
print(f"TOTAL GPU kernel time: {tot/1e3:.1f} ms  ({len(kern)} kernels)\n")
print("=== BY COMPONENT ===")
for c,t in sorted(comp.items(),key=lambda x:-x[1]):
    print(f"{100*t/tot:6.2f}%  {t/1e3:9.1f}ms  {c}")
print("\n=== BY LIBRARY ===")
for l,t in sorted(lib.items(),key=lambda x:-x[1]):
    print(f"{100*t/tot:6.2f}%  {t/1e3:9.1f}ms  {l}")
unc=comp.get("Other/uncategorized",0)
if unc>0:
    print(f"\n=== uncategorized top kernels ({100*unc/tot:.2f}%) ===")
    for n,t in sorted(compkern["Other/uncategorized"].items(),key=lambda x:-x[1])[:10]:
        print(f"  {100*t/tot:5.2f}%  {n[:90]}")
