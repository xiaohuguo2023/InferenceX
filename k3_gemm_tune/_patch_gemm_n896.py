#!/usr/bin/env python3
"""Idempotent patches for K3 GEMM tuning on N=896 MoE shapes (gfx950).

1. Skip asm kernels whose tileN does not divide N (avoids GPU fault on N=896).
2. Cap hipBLASLt fast-mode enumeration (avoids ~239k-task hang per shape).

Applied automatically from tune.sh when AITER_LIVE_MOUNT=1.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

AITER = Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/aiter")

ASM_MARKER = "PATCH(gemm-n896): skip asm when N % tileN != 0"
HIPB_MARKER = "PATCH(gemm-n896): cap hipBLASLt fast_mode sol count"


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
    asm_tune = AITER / "csrc/gemm_a16w16/gemm_a16w16_tune.py"
    hipb = AITER / "gradlib/gradlib/GemmTuner.py"
    changed = 0

    changed += patch_file(
        asm_tune,
        ASM_MARKER,
        """        for key in asm_kernels:
            tile_m, tile_n, _pf, splitK_flag, subK, bias_flag, bPreshuffle = key
            kernelName = asm_kernels[key][0]
""",
        f"""        for key in asm_kernels:
            tile_m, tile_n, _pf, splitK_flag, subK, bias_flag, bPreshuffle = key
            # {ASM_MARKER}
            if N % tile_n != 0:
                continue
            kernelName = asm_kernels[key][0]
""",
    )

    changed += patch_file(
        hipb,
        HIPB_MARKER,
        """        self.hipb_prefer_ratio = 0.995
        self.mp = mp
""",
        f"""        self.hipb_prefer_ratio = 0.995
        self.hipb_fast_max = int(os.environ.get("AITER_HIPBLASLT_FAST_MAX", "8192"))
        self.mp = mp
""",
    )

    hipb_text = hipb.read_text()
    if HIPB_MARKER not in hipb_text:
        raise SystemExit(f"hipBLASLt init patch failed: {hipb}")

    if "elif fast_mode and len(solutions) > self.hipb_fast_max:" not in hipb_text:
        changed += patch_file(
            hipb,
            HIPB_MARKER + " loop",
            """        solutions = self.hipb_sols
        if top_sols:
            solutions = self.hipb_top_sols
        task = []
""",
            f"""        solutions = self.hipb_sols
        if top_sols:
            solutions = self.hipb_top_sols
        elif fast_mode and len(solutions) > self.hipb_fast_max:
            print(
                f">>> hipblaslt fast_mode: capping {{len(solutions)}} -> "
                f"{{self.hipb_fast_max}} solutions (AITER_HIPBLASLT_FAST_MAX)",
                flush=True,
            )
            solutions = solutions[: self.hipb_fast_max]
        task = []
""",
        )

    print(f"done ({changed} file(s) updated)")


if __name__ == "__main__":
    main()
