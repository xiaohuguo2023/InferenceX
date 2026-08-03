# Kimi-K3 MI355X — MLA decode kernel options (replace TRITON_MLA with a C++ 576/512 decode)

## TL;DR
K3 fp8 decode is stuck on the slow generic **TRITON_MLA** because K3 has **12 MLA heads/rank**
(96 heads ÷ TP8), which:
1. routes vLLM's `ROCM_AITER_MLA` to the **gluon** decode (whose fp8 kernel `mla_gluon[bh16bn128]`
   is **batch_size==1** — unusable batched), and
2. is excluded from the fast **ASM 576/512 decode** (`asm_mla_decode_fwd`), which is gated to
   `num_heads % 16 == 0`.

The suggested fix — **a C++ 576/512 decode kernel** — is right, but needs two pieces, and one of
them (an **fp8** asm kernel) **does not exist yet**.

## Geometry
- K3 MLA: `kv_lora_rank=512` (ckv) + `qk_rope_head_dim=64` → head_dim **576/512** (512 value dim).
- Heads: `num_attention_heads=96` → **12 per rank at TP8**.

## Dispatch today (`vllm/v1/attention/backends/mla/rocm_aiter_mla.py`)
```
_AITER_MIN_MLA_HEADS = 16
is_valid_num_heads(n)  = n>0 and (n<16 or n%16==0)          # 12 is "valid" (n<16 branch)
use_gluon_decode(n,q)  = n<16 and max_qo_len==1              # 12 -> TRUE -> gluon
get_mla_padded_q(n,q)  = q.repeat_interleave(16//n, dim=1)   # 16//12 == 1 -> NO-OP (broken for 12!)
```
So for K3 (12 heads):
- `use_gluon_decode → True` → **gluon decode** → fp8 = `bh16bn128` → **batch=1 assert**.
- The repeat-based pad-to-16 (`repeat_interleave(16//n)`) is a **silent no-op** for 12 (`16//12==1`),
  because it only works when `16 % n == 0` (n∈{1,2,4,8}). So 12 can't reach the ASM path at all.

## What kernels actually exist
| Kernel | Path | KV dtype | batched | heads | K3 usable? |
|---|---|---|---|---|---|
| `mla_gluon[bh16bn64]` | ROCM_AITER_MLA (gluon) | **bf16** | ✔ | <16 | yes (bf16 only) |
| `mla_gluon[bh16bn128]` | ROCM_AITER_MLA (gluon) | **fp8** | **NO (batch=1)** | <16 | no (batch=1) |
| `asm_mla_decode_fwd` (`mla_dec_stage1_bf16_a16w16`) | ASM persistent | **bf16 only** | ✔ | **%16** | no (12∤16, bf16) |
| **(missing) asm fp8 mla_dec_stage1** | ASM persistent | fp8 | — | %16 | **needs to be written** |
| `TRITON_MLA` (generic Triton) | TRITON_MLA | bf16/fp8 | ✔ | any | yes (current fp8 fallback, **slow**) |

The shipped ASM decode hsaco are **`mla_dec_stage1_bf16_a16w16_subQ{16,128}_mqa{16,128}.co`** — **bf16 (a16w16)
only**. There is **no fp8 (a16w8) ASM decode kernel**.

## The fix — two parts
### Part A (kernel, aiter): write the fp8 576/512 ASM/CK decode
Add an **fp8-KV** variant of `mla_dec_stage1` (a16w8, v_head_dim=512, +64 rope), analogous to the
existing `mla_dec_stage1_bf16_a16w16`. This is the actual "C++ 576/512 decode kernel." It should take
`kv_dtype=fp8` (`kv_itemsize=1`) — the `asm_mla_decode_fwd` Python wrapper already parameterizes
`kv_dtype`, so it just needs the fp8 hsaco + kernel. Target 16-head tiles (K3 pads 12→16).

### Part B (dispatch + padding, vLLM `rocm_aiter_mla.py`): let 12 heads use the ASM path
The existing pad-to-16 is repeat-based and can't handle 12. Add **append/mask padding** (12→16 by
appending 4 zero heads; take first 12 outputs), and route single-token fp8 decode to the ASM path.

Draft:
```python
# --- new append-based padding (handles head counts that don't divide 16, e.g. 12) ---
@staticmethod
def get_mla_qpad_append(num_heads: int, q: torch.Tensor) -> torch.Tensor:
    pad = AiterMLAHelper._AITER_MIN_MLA_HEADS - num_heads          # 16-12 = 4
    if pad <= 0:
        return q
    # q: [B, num_heads, head_size] -> append `pad` zero heads on dim=1
    return torch.nn.functional.pad(q, (0, 0, 0, pad))

@staticmethod
def get_mla_opad_unappend(num_heads: int, o: torch.Tensor) -> torch.Tensor:
    return o[:, :num_heads, :]                                     # drop the padded heads

# --- dispatch: prefer the ASM path for single-token decode once an fp8 asm kernel exists ---
@staticmethod
def use_gluon_decode(num_heads: int, max_qo_len: int) -> bool:
    # Keep gluon ONLY where the asm path can't go:
    #  - multi-token (<16-head DSpark verify) -> gluon flatten (asm has no gqa<16 qseqlen>1)
    #  - (optional) bf16 <16 heads where gluon bh16bn64 is already fine
    # For single-token fp8 decode with 1-15 heads, return False so we take the
    # append-padded ASM 576/512 kernel instead of gluon's batch=1 bh16bn128.
    if max_qo_len != 1:
        return num_heads < AiterMLAHelper._AITER_MIN_MLA_HEADS
    return False   # single-token -> asm (append-padded) for all head counts
```
Then in `_forward_decode`'s ASM branch, wrap the q/o with `get_mla_qpad_append` /
`get_mla_opad_unappend`, and build the asm metadata with `get_actual_mla_num_heads(num_heads)` (=16).

**Caveats to validate when the box is free:**
- Append-padding adds 4 dummy heads → asm reads/writes 16-head tiles; decode is memory-bound so the
  4 wasted lanes are ~free (aiter's gluon docs note this). Confirm the asm kernel masks/ignores the
  padded heads (zeros) so output heads 0–11 are correct.
- fp8 asm kernel (Part A) must exist first; until then Part B can only target the **bf16** asm decode
  (marginal vs gluon bh16bn64) — the real win is Part A (fp8).

## Interim (no kernel work)
- **Keep TRITON_MLA + fp8 + no-shuffle + native context** (current working config). It's correct and
  full-context; just slower per-token than an asm kernel would be.
- Optionally route 12-head **bf16** decode through the existing bf16 asm decode via append-padding
  (Part B only) to A/B it vs gluon bh16bn64 — but bf16 KV bandwidth makes this less interesting than
  the fp8 asm kernel.

## Expected payoff
Replacing TRITON_MLA with an fp8 576/512 ASM decode should close much of the per-token gap to B300
(the kernel-efficiency gap identified in `docs/kimik3_ref_config_vs_ours.md`): same fast asm decode
family DeepSeek-V3/V4 use, now for K3's fp8 + 12-head geometry, batched, full context.

## Evidence / source
- `rocm_aiter_mla.py`: `_AITER_MIN_MLA_HEADS=16`, `use_gluon_decode`, `get_mla_padded_q`
  (repeat-based), `check_num_heads_validity`.
- `aiter_meta/csrc/cpp_itfs/mla/asm_mla_decode_fwd.py`: hsaco = `mla_dec_stage1_bf16_a16w16_*` (bf16 only).
- K3 `config.json`: `num_attention_heads=96`, `kv_lora_rank=512`, `qk_rope_head_dim=64`.
