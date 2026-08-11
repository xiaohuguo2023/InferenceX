#!/usr/bin/env python3
"""Make cudagraph_num_of_warmups settable, for measuring what warmups cost.

VllmConfig hard-sets cudagraph_num_of_warmups to 1 whenever cudagraphs are on,
so --compilation-config cannot change it. This patch has it read
VLLM_CG_WARMUPS instead (defaulting to the same 1), which lets a run with
VLLM_CG_WARMUPS=0 measure how much of the profiling and capture phases is
spent on the eager warmup run that precedes each capture.

Measurement aid only -- not part of the upstream fix.
"""

import sys
from pathlib import Path

TARGET = Path("/usr/local/lib/python3.12/dist-packages/vllm/config/vllm.py")
MARKER = "VLLM_CG_WARMUPS"

OLD = """            else:
                self.compilation_config.cudagraph_num_of_warmups = 1"""

NEW = """            else:
                self.compilation_config.cudagraph_num_of_warmups = int(
                    os.environ.get("VLLM_CG_WARMUPS", "1")
                )"""


def main() -> int:
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found", file=sys.stderr)
        return 1

    src = TARGET.read_text()
    if MARKER in src:
        print("[patch] warmup override already applied")
        return 0

    if src.count(OLD) != 1:
        print(
            f"ERROR: anchor matched {src.count(OLD)} times, expected 1",
            file=sys.stderr,
        )
        return 1

    backup = TARGET.with_suffix(".py.warmup_bak")
    if not backup.exists():
        backup.write_text(src)

    src = src.replace(OLD, NEW)
    compile(src, str(TARGET), "exec")
    TARGET.write_text(src)
    print(f"[patch] warmup override applied (backup: {backup})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
