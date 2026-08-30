#!/usr/bin/env python3
"""Restore 311b dense KDA/Mamba prefix-cache checkpoints on aa990.

vLLM #52216 changed CacheConfig.prefix_cache_retention_interval default from
None (dense: a checkpoint at every block boundary) to 0 (sparse: latest replay
boundary + Marconi junctions only). The MI355X recipe never pins the env, so
aa990 silently flipped. Positive values are rejected unless they are a multiple
of K3's scheduler_block_size (3_145_728), so None is the only dense setting.

Idempotent. Marker: PATCH(retention-dense).
"""
from pathlib import Path

DIST = Path("/usr/local/lib/python3.12/dist-packages")
CACHE_PY = DIST / "vllm/config/cache.py"
MARKER = "PATCH(retention-dense)"
OLD = "    return 0 if env_value is None else int(env_value)\n"
NEW = (
    "    # PATCH(retention-dense): 311b default. Unset env -> None -> dense "
    "KDA checkpoints.\n"
    "    return None if env_value is None else int(env_value)\n"
)


def main() -> None:
    text = CACHE_PY.read_text()
    if MARKER in text:
        print(f"{MARKER}: already applied in {CACHE_PY}")
        return
    if text.count(OLD) != 1:
        raise SystemExit(
            f"ERROR: expected exactly one default-0 return in {CACHE_PY}, "
            f"found {text.count(OLD)}"
        )
    CACHE_PY.write_text(text.replace(OLD, NEW, 1))
    print(f"{MARKER}: {CACHE_PY} factory now returns None when env unset")


if __name__ == "__main__":
    main()
