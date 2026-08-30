#!/usr/bin/env python3
"""Add DSpark verify-width keys to aiter get_block_n_fp8 (P1).

K=7 padded verify is nhead*qlen = 16*5 = 80. Nightly 6d4562c still indexes the
dict directly and has no 80/96/112 entries, so the first DSpark decode KeyErrors.

Idempotent. Marker: 80: 64 plus .get(..., 64).
"""
import os
import py_compile
import re
import shutil
from pathlib import Path

F = Path(
    os.environ.get(
        "AITER_MLA",
        "/usr/local/lib/python3.12/dist-packages/aiter/mla.py",
    )
)
text = F.read_text()
bak = F.with_suffix(".py.pre_dspark_blockn")
if not bak.exists():
    shutil.copy2(F, bak)

if "80: 64" not in text:
    text, n = re.subn(
        r"(get_block_n_fp8\s*=\s*\{)",
        r"\g<1>\n        80: 64, 96: 64, 112: 64,",
        text,
        count=1,
    )
    if n != 1:
        raise SystemExit(f"ERROR: get_block_n_fp8 dict not found once in {F}")

old = "min_block_n = get_block_n_fp8[int(nhead * max_seqlen_q)]"
new = "min_block_n = get_block_n_fp8.get(int(nhead * max_seqlen_q), 64)"
if old in text:
    text = text.replace(old, new, 1)
elif "get_block_n_fp8.get(" not in text:
    raise SystemExit(f"ERROR: get_block_n_fp8 index site missing in {F}")

F.write_text(text)
py_compile.compile(str(F), doraise=True)
print(f"PATCH(aiter-blockn-fp8): 80-key={'80: 64' in text} get={'get_block_n_fp8.get(' in text} {F}")
