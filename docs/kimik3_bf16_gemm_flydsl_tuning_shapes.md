# Kimi-K3 — bf16 GEMM shapes to tune with flydsl (MI355X, TP8)

K3's bf16 (a16w16) GEMMs currently run **untuned** on MI355X — `aiter.tuned_gemm.tgemm.mm` finds no
entry in `bf16_tuned_gemm.csv` and falls back to torch/default (the **8,512 "not found tuned config …
using torch/default"** warnings in the serve log). Tuning them with **flydsl `hgemm_bf16`** (a per-model
`kimik3_bf16_tuned_gemm.csv`, exactly like the shipped `kimik2_bf16_tuned_gemm.csv`) replaces those
fallbacks with fast hand-generated MFMA kernels.

Shapes captured with `AITER_TUNE_GEMM=1` → `kimik3_bf16_untuned_gemm.csv` (14 distinct (N,K), 325 rows).
Shapes are **per-rank at TP8**. K3 dims: hidden=7168, q_lora_rank=1536, kv_lora_rank=512,
qk_nope=128, qk_rope=64, v_head=128, heads=96 (12/rank), moe_intermediate=3072, dense intermediate=33792.

## The shapes (flydsl-eligible = N%64==0 and K%64==0)

| N (per-rank) | K | flydsl? | M ladder | K3 layer |
|---:|---:|:---:|---|---|
| 1536 | 7168 | ✅ | 1–64 + prefill→4096 | **q_a_proj** (hidden→q_lora 1536) |
| 2304 | 1536 | ✅ | 1–4096 | **q_b_proj** (q_lora→96×192 ÷8) |
| 3072 | 512 | ✅ | **1–7122** (per-token) | **kv_b_proj** (kv_lora→96×256 ÷8) — MLA core |
| 7168 | 1536 | ✅ | 1–4096 | **o_proj** (96×128 ÷8 →hidden) |
| 8448 | 7168 | ✅ | 1–4096 | **dense gate_up** (2×33792 ÷8) |
| 7168 | 4224 | ✅ | 1–4096 | **dense down** (33792 ÷8→hidden) |
| 20480 | 7168 | ✅ | 1–64 (decode only) | **lm_head** (vocab ÷8) |
| 3584 | 7168 | ✅ | 1–4096 | proj-from-hidden (gate/up class) |
| 2112 | 7168 | ✅ | 1–4096 | proj-from-hidden (kv_a / fused) |
| 896 | 7168 | ✅ | 1–4096 | proj-from-hidden |
| 7168 | 3584 | ✅ | 1–4096 | proj-to-hidden (down class) |
| 7168 | 768 | ✅ | 1–4096 | proj-to-hidden (down class) |
| 1536 | 128 | ✅ | 1–4096 | small proj (indexer/aux) |
| **6288** | 7168 | **❌ (N%64=16)** | 1–4096 | proj-from-hidden — **flydsl can't tile; stays hipBLASLt/asm** |

**13 of 14 are flydsl-eligible.** Only **N=6288** (16 mod 64) can't use flydsl `hgemm_bf16` (its tiling
needs N%64==0) → leave it on hipBLASLt/asm.

## M ladder to tune per shape
- **Decode / agentic conc**: 1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64 (covers conc 1–24 + serving).
- **Prefill chunks**: the captured 2772/2836/3141/4096 (bounded by max-num-batched-tokens=4096).
- **kv_b_proj (3072,512)** additionally sees **per-token M up to ~7122** (MLA attention over the batch's
  tokens) — tune a denser M grid here (it's the hottest bf16 GEMM in decode).

## How to produce `kimik3_bf16_tuned_gemm.csv` (the flydsl tuning)
1. `kimik3_bf16_untuned_gemm.csv` (already collected) is the input shape list.
2. Run the aiter bf16 GEMM tuner (gradlib / the a16w16 path) over it — it benchmarks flydsl / asm / opus /
   hipBLASLt / skinny per (M,N,K) and writes the winner. For the eligible shapes above, **flydsl `hgemm_bf16`
   should win at the small-M (decode) sizes** (same pattern as `kimik2_bf16_tuned_gemm.csv`, whose N=384
   router rows are all `libtype=flydsl`).
3. Drop the result at `aiter/configs/model_configs/kimik3_bf16_tuned_gemm.csv` (auto-merged) or
   `/tmp/aiter_configs/bf16_tuned_gemm.csv`; re-serve → `tgemm.mm` dispatches to flydsl instead of torch.

## Expected payoff
Removes the torch/default fallback on the dense/MLA-projection GEMMs (a real slice of decode — dense
GEMM was ~22% of the fp8 conc32 profile). Biggest wins on **kv_b_proj (3072,512)** and **q_a/o_proj**
at decode M (1–24), where flydsl `hgemm_bf16` beats the untuned path — complementary to the ASM
576/512 attention-decode speedup already validated.

## Refreshed shape collection — from all serve logs (2026-08-02)

Swept **every `*.log` in `~/work/InferenceX-dspv4`** for the
`not found tuned config in /tmp/aiter_configs/bf16_tuned_gemm.csv` warning
(agentic, long-context, conc32 profiling, and the a8w4/ref-config runs):
**667,744 warnings → 37,085 distinct (M,N,K) → 21 distinct (N,K) GEMMs.**

| N | K | flydsl? | Mmax obs | occ | K3 layer (best guess) |
|---:|---:|:---:|---:|---:|---|
| 1536 | 128 | ✅ | 8192 | 72464 | small proj (indexer/aux) |
| 3584 | 7168 | ✅ | 8192 | 72464 | proj-from-hidden (gate/up) |
| 896 | 7168 | ✅ | 8192 | 72136 | proj-from-hidden |
| 3072 | 512 | ✅ | **98304** | 51456 | **kv_b_proj — MLA core (per-token M)** |
| 8448 | 7168 | ✅ | 8192 | 43968 | dense gate_up |
| 7168 | 4224 | ✅ | 8192 | 43968 | dense down |
| 1536 | 7168 | ✅ | 8192 | 43968 | q_a_proj |
| 7168 | 768 | ✅ | 8192 | 43968 | proj-to-hidden |
| 7168 | 3584 | ✅ | 8192 | 43968 | proj-to-hidden (down) |
| 2304 | 1536 | ✅ | 8192 | 43968 | q_b_proj |
| 7168 | 1536 | ✅ | 8192 | 43960 | o_proj |
| 2112 | 7168 | ✅ | 8192 | 15952 | proj-from-hidden (kv_a/fused) |
| 20480 | 7168 | ✅ | 128 | 2632 | lm_head (decode only) |
| 4096 | 4096 | ✅ | 16817 | 144 | (long-ctx prefill class) |
| 7168 | 4096 | ✅ | 16817 | 144 | (long-ctx prefill class) |
| 1024 | 1536 | ✅ | 67268 | 18 | (ref-config / a8w4 aux) |
| 1024 | 4096 | ✅ | 67268 | 18 | (ref-config / a8w4 aux) |
| 4096 | 1024 | ✅ | 67268 | 18 | (ref-config / a8w4 aux) |
| 4608 | 1024 | ✅ | 67268 | 18 | (ref-config / a8w4 aux) |
| **6288** | 7168 | **❌ N%64=16** | 8192 | 72464 | proj-from-hidden — **keep hipBLASLt/asm** |
| **16160** | 7168 | **❌ N%64=32** | 1024 | 48 | **keep hipBLASLt/asm** |

**19 of 21 are flydsl-eligible.** Only **N=6288** (the single hottest GEMM, 72k occ)
and **N=16160** can't be flydsl-tiled (N%64≠0) → leave on hipBLASLt/asm.

## flydsl is NOT the target for every shape — the tuner picks per shape
The untuned CSV is only the **shape list**. Tuning does **not** mean "force flydsl": the
aiter tuner benchmarks **flydsl + hipBLASLt + asm + skinny(wvSplitK)** per (M,N,K) and
writes the fastest (`libtype` column in the tuned CSV). Expected winners on K3:
- **flydsl `hgemm_bf16`** → **small/skinny M** (decode M=1–64) **and only** N%64==0 & K%64==0.
- **hipBLASLt / asm** → **large/fat M** (prefill M≥1024), and **always** for the two
  non-eligible shapes (6288, 16160).
- So the tuned output is a **mix** — flydsl at decode-M on eligible shapes, hipBLASLt/asm
  at prefill-M and on the non-flydsl shapes.

Note the fallback today is **torch** (`using torch solution:0`), which is *worse* than a
tuned hipBLASLt — so **all 21 shapes** (incl. the 2 non-flydsl) should be tuned to escape
the torch path, not just the flydsl-eligible ones.

## Artifacts (tuning-ready)
- **`kimik3_bf16_untuned_gemm.csv`** — full log-collected set: **37,085 distinct
  (M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle)** rows (record of everything observed).
- **`kimik3_bf16_tuning_gemm.csv`** — **curated tuner input: 424 rows** = **all 21 (N,K)**
  (19 flydsl-eligible + 2 hipBLASLt/asm-only) × a decode+prefill M ladder (1–8192) +
  coarse large-M points (16k/24k/32k/49k/64k/98k, capped per shape) for the MLA
  `kv_b_proj` per-token and long-context prefill. **Feed this to the tuner.**
- Reference: `aiter/configs/model_configs/kimik2_bf16_tuned_gemm.csv` (shows the mix —
  e.g. flydsl for the router small-M rows, other libs elsewhere).
- Scripts: `_collect_untuned_gemm.py` (log sweep) + `_make_tuning_csv.py` (curation).

## To tune
Run the aiter a16w16 (bf16) GEMM tuner over `kimik3_bf16_tuning_gemm.csv` (gradlib / the
a16w16 path); it benchmarks flydsl/asm/hipBLASLt/skinny per (M,N,K) and writes the winner
per row. Drop the result at `aiter/configs/model_configs/kimik3_bf16_tuned_gemm.csv`
(auto-merged) or `/tmp/aiter_configs/bf16_tuned_gemm.csv`; re-serve → `tgemm.mm`
dispatches to the tuned kernel (flydsl or hipBLASLt/asm, per shape) instead of the
torch/default fallback (which produced all 667k warnings).
