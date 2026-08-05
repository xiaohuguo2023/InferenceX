#!/usr/bin/env python3
"""Shape guards for asm/opus during K3 GEMM tuning (gfx950).

Host-side opus filters (kid_rejects_shape, 4g_safe, K-parity) still let through
kernels that GPU-fault under mp_tuner on large-M prefill shapes (e.g. M=4096/32768,
N=1024). asm splitK can hit the same. Skip those backends when M (or M*N) exceeds
safe thresholds so tuning stays on flydsl / hipBLASLt / skinny.

Env (read inside patched gemm_a16w16_tune.py at runtime):
  AITER_TUNE_ASM_MAX_M   default 2048   — skip asm when M exceeds
  AITER_TUNE_ASM_MAX_MN  default 4194304 (2048*2048) — skip asm when M*N exceeds
  AITER_TUNE_OPUS_MAX_M  default 2048   — skip opus when M exceeds
  AITER_TUNE_DISABLE_ASM=1  — always skip asm tasks
  AITER_TUNE_DISABLE_OPUS=1 — always skip opus tasks

Applied from tune.sh when AITER_LIVE_MOUNT=1.
"""
from __future__ import annotations

import sys
from pathlib import Path

AITER = Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/aiter")
TUNE_PY = AITER / "csrc/gemm_a16w16/gemm_a16w16_tune.py"

ASM_MARKER = "PATCH(gemm-tune-safe): skip asm for large-M shapes"
OPUS_MARKER = "PATCH(gemm-tune-safe): skip opus for large-M shapes"


def patch_file(path: Path, marker: str, old: str, new: str) -> bool:
    text = path.read_text()
    if marker in text:
        print(f"already patched: {path.name} ({marker})")
        return False
    if old not in text:
        raise SystemExit(f"patch anchor missing in {path}: {marker}")
    path.write_text(text.replace(old, new, 1))
    print(f"patched OK: {path.name} ({marker})")
    return True


def main() -> None:
    changed = 0
    changed += patch_file(
        TUNE_PY,
        ASM_MARKER,
        """        M, N, K = info_keys[2], info_keys[3], info_keys[4]
        if (scaleAB or K % 64 != 0 or indtype != dtypes.bf16) and get_gfx() == "gfx942":
""",
        f"""        M, N, K = info_keys[2], info_keys[3], info_keys[4]
        # {ASM_MARKER}
        if os.environ.get("AITER_TUNE_DISABLE_ASM", "0") == "1":
            return []
        _asm_max_m = int(os.environ.get("AITER_TUNE_ASM_MAX_M", "2048"))
        _asm_max_mn = int(os.environ.get("AITER_TUNE_ASM_MAX_MN", "4194304"))
        if M > _asm_max_m or M * N > _asm_max_mn:
            return []
        if (scaleAB or K % 64 != 0 or indtype != dtypes.bf16) and get_gfx() == "gfx942":
""",
    )
    changed += patch_file(
        TUNE_PY,
        OPUS_MARKER,
        """        M, N, K = info_keys[2], info_keys[3], info_keys[4]
        cu_num = get_cu_num()
""",
        f"""        M, N, K = info_keys[2], info_keys[3], info_keys[4]
        # {OPUS_MARKER}
        if os.environ.get("AITER_TUNE_DISABLE_OPUS", "0") == "1":
            return []
        _opus_max_m = int(os.environ.get("AITER_TUNE_OPUS_MAX_M", "2048"))
        if M > _opus_max_m:
            return []
        cu_num = get_cu_num()
""",
    )
    print(f"done ({changed} file(s) updated)")


if __name__ == "__main__":
    main()
