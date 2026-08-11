#!/usr/bin/env python3
"""PR-G: tile kv_b_proj along M to cap GEMM activation peak (ROCm MLA context path).

Chunked context prefill was running kv_b_proj on up to max_num_seqs*1536 rows
(~196k), allocating ~1.3 GiB/GPU of transient activations. This writes the
same [M,H,P+V] output FMHA needs, but projects at most
VLLM_MLA_KV_B_PROJ_CHUNK rows per launch (default 4096).

Apply inside k3-benchmark after _patch_cgmem.py and _patch_moe_scratch.py:
  export VLLM_MLA_KV_B_PROJ_CHUNK=4096
  python3 /workspace/_patch_kv_b_proj_chunk.py

Restore: cp *.kv_b_proj_bak back (script creates backups on first run).
"""

from __future__ import annotations

import sys
from pathlib import Path

PKG = Path("/usr/local/lib/python3.12/dist-packages/vllm")
MLA = PKG / "model_executor/layers/attention/mla_attention.py"
AITER = PKG / "v1/attention/backends/mla/rocm_aiter_mla.py"
MARKER = "PATCH(kv-b-proj-chunk)"

HELPER = f'''
def kv_b_proj_chunk_size() -> int:
    """Row tile for chunked kv_b_proj (PR-G). 0 disables chunking."""
    return int(os.getenv("VLLM_MLA_KV_B_PROJ_CHUNK", "4096"))


def kv_b_proj_expand(
    kv_b_proj: ColumnParallelLinear,
    kv_c_normed: torch.Tensor,
    *,
    num_heads: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Up-project latent KV in M tiles to cap GEMM activation peak. {MARKER}"""
    chunk_size = kv_b_proj_chunk_size() if chunk_size is None else chunk_size
    m = kv_c_normed.shape[0]
    head_dim = qk_nope_head_dim + v_head_dim
    if chunk_size <= 0 or m <= chunk_size:
        return kv_b_proj(kv_c_normed)[0].view(-1, num_heads, head_dim)

    out = torch.empty(
        (m, num_heads, head_dim),
        dtype=kv_c_normed.dtype,
        device=kv_c_normed.device,
    )
    for start in range(0, m, chunk_size):
        end = min(start + chunk_size, m)
        out[start:end] = kv_b_proj(kv_c_normed[start:end])[0].view(
            -1, num_heads, head_dim
        )
    return out


'''

OLD_KV_B = """            kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
            )"""

NEW_KV_B = """            kv_nope = kv_b_proj_expand(
                self.kv_b_proj,
                kv_c_normed,
                num_heads=self.num_heads,
                qk_nope_head_dim=self.qk_nope_head_dim,
                v_head_dim=self.v_head_dim,
            )"""

OLD_KV_B_FORWARD = """        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )"""

NEW_KV_B_FORWARD = """        kv_nope = kv_b_proj_expand(
            self.kv_b_proj,
            kv_c_normed,
            num_heads=self.num_heads,
            qk_nope_head_dim=self.qk_nope_head_dim,
            v_head_dim=self.v_head_dim,
        )"""

OLD_AITER_IMPORT = """from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)"""

NEW_AITER_IMPORT = """from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
    kv_b_proj_expand,
)"""


def _backup(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".kv_b_proj_bak")
    if not bak.exists():
        bak.write_text(path.read_text())
        print(f"backup: {bak}")


def _ensure_import_os(text: str) -> str:
    if "import os\n" in text:
        return text
    for anchor in ("import functools\n", "import torch\n"):
        if anchor in text:
            return text.replace(anchor, anchor + "import os\n", 1)
    sys.exit(f"cannot find import anchor for os in {MLA}")


def _patch_mla(text: str) -> str:
    if MARKER in text:
        return text
    text = _ensure_import_os(text)
    anchor = "class MLAAttention(nn.Module, AttentionLayerBase):"
    if "def kv_b_proj_expand(" not in text:
        if anchor not in text:
            sys.exit(f"anchor missing in {MLA}")
        text = text.replace(anchor, HELPER + anchor, 1)
    text = text.replace(OLD_KV_B, NEW_KV_B)
    text = text.replace(OLD_KV_B_FORWARD, NEW_KV_B_FORWARD)
    if OLD_KV_B in text or OLD_KV_B_FORWARD in text:
        sys.exit("mla_attention.py: some kv_b_proj call sites were not patched")
    return text


def _patch_aiter(text: str) -> str:
    if "kv_b_proj_expand" in text and MARKER not in text:
        pass
    if "kv_b_proj_expand," not in text:
        if OLD_AITER_IMPORT not in text:
            if "kv_b_proj_expand" in text:
                return text
            sys.exit(f"aiter import block missing in {AITER}")
        text = text.replace(OLD_AITER_IMPORT, NEW_AITER_IMPORT, 1)
    text = text.replace(OLD_KV_B_FORWARD, NEW_KV_B_FORWARD)
    return text


def main() -> None:
    for path in (MLA, AITER):
        if not path.is_file():
            sys.exit(f"missing {path}")

    _backup(MLA)
    _backup(AITER)

    mla_text = _patch_mla(MLA.read_text())
    MLA.write_text(mla_text)
    print(f"patched {MLA}  kv_b_proj_expand={'kv_b_proj_expand' in mla_text}")

    aiter_text = _patch_aiter(AITER.read_text())
    AITER.write_text(aiter_text)
    print(f"patched {AITER}  kv_b_proj_expand={'kv_b_proj_expand' in aiter_text}")


if __name__ == "__main__":
    main()
