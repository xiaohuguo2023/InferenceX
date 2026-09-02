#!/usr/bin/env python3
"""Force the DSpark draft checkpoint causal.

The Kimi-K3 DSpark draft ships without ``dflash_config``. vLLM resolves per-layer
causality in ``qwen3_dflash._dflash_layer_causal``:

  1. ``config.is_causal``            -- absent from this checkpoint
  2. ``dflash_config["causal"]``     -- absent from this checkpoint
  3. ``layer_types[i] == "sliding_attention"`` -- ``layer_types`` is also absent,
     so this evaluates False

With all three missing every layer resolves **non-causal**, so
``dflash_has_any_non_causal()`` returns True and the draft is routed to the
non-causal backend instead of the fp8 asm path -- which is the cudagraph OOB
source. ``_serve_k3_dcp_test.sh`` refuses to launch in that state.

This is a checkpoint edit, not a vLLM patch, so it does NOT survive a /dev/shm
wipe and re-download. Re-run it after any re-download (that is what silently
broke the 2026-09-01 21:52 DCP A/B, 71 s in).

``config.json`` in an HF snapshot dir is a symlink into ``blobs/``. Writing
through it would corrupt the content-addressed blob, so this replaces the
symlink with a real file and leaves the blob byte-identical.

  python3 _patch_draft_causal.py            # apply (idempotent)
  python3 _patch_draft_causal.py --revert   # restore the symlink
  python3 _patch_draft_causal.py --check    # exit 0 iff causal is true
"""

import argparse
import glob
import json
import os
import sys

CACHE = "/dev/shm/hf-cache/models--Inferact--Kimi-K3-DSpark/snapshots"


def resolve() -> tuple[str, str]:
    snaps = sorted(glob.glob(os.path.join(CACHE, "*/")))
    if not snaps:
        sys.exit(f"!! DSpark draft not staged at {CACHE}")
    cfg = os.path.join(snaps[0].rstrip("/"), "config.json")
    if not os.path.exists(cfg):
        sys.exit(f"!! no config.json in {snaps[0]}")
    return cfg, snaps[0].rstrip("/")


def blob_for(cfg: str, snap: str) -> str | None:
    """The blob this snapshot entry points at, if it is still a symlink."""
    if not os.path.islink(cfg):
        return None
    return os.path.realpath(cfg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    cfg, snap = resolve()
    causal = json.load(open(cfg)).get("dflash_config", {}).get("causal")

    if args.check:
        print(f"dflash_config.causal = {causal}  ({cfg})")
        return 0 if causal is True else 1

    if args.revert:
        if os.path.islink(cfg):
            print("already a symlink; nothing to revert")
            return 0
        # Re-point at the blob whose name matches the on-disk content-addressed
        # store. There is exactly one config blob for this repo.
        blobs = glob.glob(
            os.path.join(os.path.dirname(os.path.dirname(snap)), "blobs", "*")
        )
        cands = []
        for b in blobs:
            try:
                d = json.load(open(b))
            except Exception:
                continue
            if isinstance(d, dict) and "num_hidden_layers" in d:
                cands.append(b)
        if len(cands) != 1:
            sys.exit(f"!! cannot identify the config blob ({len(cands)} candidates)")
        os.remove(cfg)
        os.symlink(os.path.relpath(cands[0], snap), cfg)
        print(f"reverted to symlink -> {cands[0]}")
        return 0

    if causal is True:
        print(f"already applied: dflash_config.causal = True  ({cfg})")
        return 0

    data = json.load(open(cfg))
    if "dflash_config" in data:
        sys.exit(
            f"!! dflash_config already present with causal={causal!r}; "
            "inspect before overwriting"
        )
    data["dflash_config"] = {"causal": True}

    blob = blob_for(cfg, snap)
    if blob:
        os.remove(cfg)  # drop the symlink only; the blob stays intact
    with open(cfg, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    print(f"applied: dflash_config.causal = True  ({cfg})")
    if blob:
        print(f"         blob left untouched: {blob}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
