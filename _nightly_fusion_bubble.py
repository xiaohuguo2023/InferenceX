#!/usr/bin/env python3
"""Two analyses on ONE load of a K3 nightly rank trace:
  (A) Elementwise/glue FUSION SIZING — decompose the vectorized_elementwise +
      copyBuffer/memcpy + cast + residual glue kernels vLLM leaves separate
      (the ~20% ATOM fuses), split prefill/decode, to size the config-only ceiling.
  (B) CPU-BUBBLE / GAP — idle on the busiest GPU stream, attributed to the kernel
      after each gap (launch-bound vs compute-bound).

Usage: python3 _nightly_fusion_bubble.py <rank.pt.trace.json.gz>
"""
import gzip, json, sys, collections

f = sys.argv[1]
print(f"loading {f} ...", file=sys.stderr)
d = json.load(gzip.open(f)); ev = d["traceEvents"]

# ---- prefill/decode phase from the launching CPU op (graph => decode) ----
corr2launch = {}
for e in ev:
    n = e.get("name", "")
    if "Launch" in n or "Graph" in n or "graph" in n:
        c = e.get("args", {}).get("correlation")
        if c is not None:
            corr2launch[c] = n
def phase_of(corr):
    l = (corr2launch.get(corr) or "").lower()
    if "graph" in l:  return "decode"
    return "prefill"

kern = [e for e in ev if e.get("cat") == "kernel" and "dur" in e]
tot = sum(e["dur"] for e in kern) or 1.0

# ---- glue-kernel taxonomy (what ATOM would fuse away) ----
def glue_kind(n):
    s = n
    low = s.lower()
    if "copybuffer" in low or "Memcpy" in s or "memcpy" in low or "Memset" in s or "memset" in low:
        return "buffer_copy/memset"
    if "vectorized_elementwise" in low or "elementwise_kernel" in low or "CatArrayBatched" in s:
        return "elementwise(add/act/cast)"
    if s.startswith("triton_") and not any(k in low for k in ("gemm","attn","moe","norm")):
        return "triton_elementwise"
    if any(k in low for k in ("add_rmsnorm_quant","rmsnorm","add_rms","layer_norm")):
        return "rms_norm(+quant)"
    if any(k in low for k in ("per_token_group_quant","dynamic_per_token_scaled_quant",
                              "dynamic_per_group_scaled_quant","float8_copy","scaled_quant",
                              "fp8_quant","mx_quant")):
        return "quant/cast"
    return None  # not glue

# aggregate glue: phase -> kind -> name -> [us,count]
glue = {"prefill": collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0])),
        "decode":  collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0]))}
phase_tot = {"prefill": 0.0, "decode": 0.0}
for e in kern:
    ph = phase_of(e.get("args", {}).get("correlation"))
    phase_tot[ph] += e["dur"]
    k = glue_kind(e["name"])
    if k is None:
        continue
    a = glue[ph][k][e["name"]]; a[0] += e["dur"]; a[1] += 1

print("\n" + "=" * 78)
print("(A) ELEMENTWISE / GLUE FUSION SIZING  (config-only fusion ceiling)")
print("=" * 78)
grand_glue = 0.0
for ph in ("prefill", "decode"):
    ptot = phase_tot[ph] or 1.0
    gtot = sum(u for kind in glue[ph].values() for (u, _) in kind.values())
    grand_glue += gtot
    print(f"\n-- {ph}: phase GPU-kernel time {ptot/1e3:8.1f} ms | "
          f"glue = {gtot/1e3:7.1f} ms = {100*gtot/ptot:4.1f}% of phase --")
    for kind in sorted(glue[ph], key=lambda k: -sum(u for u, _ in glue[ph][k].values())):
        kt = sum(u for u, _ in glue[ph][kind].values())
        kc = sum(c for _, c in glue[ph][kind].values())
        print(f"   {kind:26} {kt/1e3:8.1f} ms  {100*kt/ptot:5.1f}%  ({kc} launches)")
        for name, (us, cnt) in sorted(glue[ph][kind].items(), key=lambda kv: -kv[1][0])[:4]:
            print(f"       {us/1e3:8.1f} ms  {100*us/ptot:4.1f}%  x{cnt:<5}  {name[:74]}")
print(f"\n== TOTAL glue across both phases: {grand_glue/1e3:.1f} ms = "
      f"{100*grand_glue/tot:.1f}% of all GPU-kernel time ({tot/1e3:.1f} ms) ==")
print("   (fusion ceiling ~ this %, minus the irreducible reduce/copy ATOM can't remove)")

# ---- (B) main-stream bubble / gap ----
print("\n" + "=" * 78)
print("(B) CPU-BUBBLE / GAP  (idle on busiest GPU stream)")
print("=" * 78)
kt = [e for e in kern if "ts" in e]
streams = collections.defaultdict(list)
for e in kt:
    streams[e["tid"]].append(e)
main = max(streams.values(), key=lambda L: sum(e["dur"] for e in L))
main.sort(key=lambda e: e["ts"])
span = main[-1]["ts"] + main[-1]["dur"] - main[0]["ts"]
busy = sum(e["dur"] for e in main)
print(f"main stream: {len(main)} kernels  span={span/1e3:.1f}ms  busy={busy/1e3:.1f}ms  "
      f"idle={100*(span-busy)/span:.1f}%")

def short(n):
    for k in ("cross_device_reduce", "mfma_moe1", "mfma_moe2", "flydsl_moe", "_mla_gluon",
              "fmha_fwd", "_attn_res", "mla_decode", "mla_reduce", "concat_and_cache_mla",
              "chunk_gated_delta", "chunk_kda", "chunk_gla", "causal_conv1d", "kda_gate",
              "recompute_w_u", "fused_recurrent_kda", "add_rmsnorm_quant", "grouped_topk",
              "moe_reduction", "opus_moe_sorting", "moe_sorting", "merge_attn_states",
              "Cijk_", "hgemm_bf16", "wvSplitK", "wvSplitKrc", "LLMM1", "l2norm",
              "dynamic_per_group", "copyBuffer", "vectorized_elementwise"):
        if k in n:
            return k
    return n[:32]

gap_before = collections.defaultdict(float); gap_cnt = collections.Counter()
GAPS = []
for a, b in zip(main, main[1:]):
    g = b["ts"] - (a["ts"] + a["dur"])
    if g > 2:
        gap_before[short(b["name"])] += g; gap_cnt[short(b["name"])] += 1
        GAPS.append((g, short(a["name"]), short(b["name"])))
tot_idle = sum(g for g, _, _ in GAPS)
print(f"\ntotal idle in gaps>2us: {tot_idle/1e3:.1f}ms  ({100*tot_idle/span:.1f}% of span)")
print("\n=== idle attributed to the kernel AFTER the gap (GPU waited to run this) ===")
for n, t in sorted(gap_before.items(), key=lambda x: -x[1])[:15]:
    print(f"  {t/1e3:8.1f}ms  ({gap_cnt[n]:5d} gaps)  before -> {n}")
print("\n=== biggest single gaps (idle_us : prev -> next) ===")
for g, a, b in sorted(GAPS, key=lambda x: -x[0])[:12]:
    print(f"  {g:8.0f}us   {a}  ->  {b}")
