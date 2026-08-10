#!/usr/bin/env bash
# =============================================================================
# _k3_dspark_fp8asm_apply_patches.sh
# One-command apply of the Kimi-K3 DSpark + native fp8-asm KV enablement.
# Run INSIDE the container (xguo-k3nc) on top of a WORKING baseline K3 that
# already has the SHIPPED recipe rocm_aiter_mla.py (main + #51011 + #51040 +
# #51606) installed. See docs/kimik3_dspark_fp8asm_recipe.md for the full recipe.
#
#   docker cp _k3_dspark_fp8asm_apply_patches.sh xguo-k3nc:/workspace/
#   docker exec -it xguo-k3nc bash -lc 'bash /workspace/_k3_dspark_fp8asm_apply_patches.sh'
#
# Idempotent: every step guards on its anchor / target text and takes a .bak.
# =============================================================================
set -uo pipefail
D=/usr/local/lib/python3.12/dist-packages
R="$D/vllm/v1/attention/backends/mla/rocm_aiter_mla.py"
AITER=/root/aiter/aiter/mla.py
KDA="$D/vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py"
export R AITER KDA
say(){ echo; echo "=================== $* ==================="; }

say "0/6 sanity: shipped recipe backend present"
grep -q _mtp_decode_qlen "$R" || { echo "!! recipe rocm_aiter_mla.py not installed — apply the base recipe first"; exit 1; }

say "1/6 force DSpark draft causal (dflash_config.causal=true)"
DCACHE=/dev/shm/hf-cache/models--Inferact--Kimi-K3-DSpark/snapshots
CFG="$(ls -d "$DCACHE"/*/ 2>/dev/null | head -1)config.json"
if [ -f "$CFG" ]; then
  cp -n "$CFG" "$CFG.orig.bak"
  python3 - "$CFG" <<'PY'
import json, sys
f = sys.argv[1]
c = json.load(open(f))
d = c.setdefault("dflash_config", {})
if d.get("causal") is True:
    print("  draft already forced causal")
else:
    d["causal"] = True
    json.dump(c, open(f, "w"), indent=2)
    print("  forced causal:", f)
PY
else
  echo "  !! draft config not found at $DCACHE — stage the draft or set it manually"
fi

say "2/6 aiter get_block_n_fp8 — add DSpark verify-width key (16*5=80)"
python3 - <<'PY'
import re, py_compile, shutil, os
F = os.environ.get("AITER", "/root/aiter/aiter/mla.py")
s = open(F).read()
shutil.copy2(F, F + ".pre_dspark.bak")
if "80: 64" not in s:
    s = re.sub(r"(get_block_n_fp8\s*=\s*\{)", r"\g<1>\n    80: 64, 96: 64, 112: 64,", s, count=1)
s = s.replace(
    "min_block_n = get_block_n_fp8[int(nhead * max_seqlen_q)]",
    "min_block_n = get_block_n_fp8.get(int(nhead * max_seqlen_q), 64)",
)
open(F, "w").write(s); py_compile.compile(F, doraise=True)
print("  80-key:", "80: 64" in s, " get():", "get_block_n_fp8.get(" in s)
PY

say "3/6 recipe backend — DSpark verify width (_mtp_decode_qlen = 2*num_spec+1)"
python3 - <<'PY'
import py_compile, shutil, os
F = os.environ["R"]
s = open(F).read()
shutil.copy2(F, F + ".pre_dsparkqlen.bak")
if 'speculative_config.method == "dspark"' in s:
    print("  dspark qlen branch already present")
else:
    anchor = "        else:\n            self._mtp_decode_qlen = 1"
    branch = (
        "        elif (\n"
        "            speculative_config is not None\n"
        '            and speculative_config.method == "dspark"\n'
        "            and speculative_config.num_speculative_tokens is not None\n"
        "        ):\n"
        "            # DSpark drafts a semi-autoregressive block; the target verifies\n"
        "            # 1 anchor + 2*num_spec positions.\n"
        "            self._mtp_decode_qlen = 2 * int(speculative_config.num_speculative_tokens) + 1\n"
    )
    if s.count(anchor) == 1:
        s = s.replace(anchor, branch + anchor, 1)
        open(F, "w").write(s); py_compile.compile(F, doraise=True)
        print("  dspark qlen branch added")
    else:
        print(f"  !! anchor count {s.count(anchor)} — add the dspark branch manually")
PY

say "4/6 recipe backend — broaden persistent-metadata gate"
python3 - <<'PY'
import py_compile, shutil, os
F = os.environ["R"]
s = open(F).read()
shutil.copy2(F, F + ".pre_gate.bak")
old = (
    "        use_persistent_metadata = (\n"
    "            self.num_heads >= AiterMLAHelper._AITER_MIN_MLA_HEADS\n"
    "            and max_qo_len >= 1\n"
    "            and max_qo_len <= self._mtp_decode_qlen\n"
    "        )"
)
new = (
    "        uses_asm_decode = (\n"
    "            self.num_heads >= AiterMLAHelper._AITER_MIN_MLA_HEADS\n"
    '            or _aiter_mla_small_head_mode() == "asm"\n'
    "            or max_qo_len > 1\n"
    "        )\n"
    "        use_persistent_metadata = (\n"
    "            uses_asm_decode\n"
    "            and max_qo_len >= 1\n"
    "            and max_qo_len <= self._mtp_decode_qlen\n"
    "        )"
)
if "uses_asm_decode" in s:
    print("  gate already broadened")
elif old in s:
    s = s.replace(old, new, 1); open(F, "w").write(s); py_compile.compile(F, doraise=True)
    print("  gate broadened")
else:
    print("  !! gate anchor not found — inspect use_persistent_metadata manually")
PY

say "5/6 KDA stride fix (Fangzhou-Ai/vllm PR #27; correctness)"
python3 - <<'PY'
import py_compile, shutil, os
F = os.environ["KDA"]
if not os.path.exists(F):
    print("  !! KDA file not found — skip"); raise SystemExit(0)
s = open(F).read()
shutil.copy2(F, F + ".pre_pr27.bak")
changed = False
if "stride_indices_seq: tl.constexpr," not in s:
    s = s.replace("    stride_state_token: tl.constexpr,\n",
                  "    stride_state_token: tl.constexpr,\n    stride_indices_seq: tl.constexpr,\n", 1); changed = True
s2 = s.replace("state_idx = tl.load(state_indices + i_n).to(tl.int64)",
               "state_idx = tl.load(state_indices + i_n * stride_indices_seq).to(tl.int64)")
changed |= s2 != s; s = s2
s2 = s.replace(
    "    if state_indices.ndim != 1 or state_indices.stride(0) != 1:\n"
    "        state_indices = state_indices.reshape(-1).contiguous()",
    "    if state_indices.ndim != 1:\n"
    '        raise ValueError("`state_indices` must be one-dimensional.")')
changed |= s2 != s; s = s2
if "stride_indices_seq=state_indices.stride(0)," not in s:
    s = s.replace("        stride_state_token=initial_state.stride(0),\n",
                  "        stride_state_token=initial_state.stride(0),\n"
                  "        stride_indices_seq=state_indices.stride(0),\n", 1); changed = True
open(F, "w").write(s); py_compile.compile(F, doraise=True)
print("  KDA PR#27 applied" if changed else "  KDA PR#27 already present")
PY

say "6/6 verify"
echo "aiter 80-key   = $(grep -c '80: 64' "$AITER")                (expect >=1)"
echo "aiter get()    = $(grep -c 'get_block_n_fp8.get(' "$AITER")   (expect 1)"
echo "dspark qlen    = $(grep -c 'method == \"dspark\"' "$R")        (expect >=1)"
echo "asm gate       = $(grep -c 'uses_asm_decode' "$R")            (expect >=2)"
echo "kda stride     = $(grep -c 'stride_indices_seq' "$KDA")        (expect >=3)"
python -c "import vllm.v1.attention.backends.mla.rocm_aiter_mla; print('IMPORT_OK')"
echo
echo "DONE. Serve with:"
echo "  export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm"
echo "  NUM_SPEC=2 PORT=8890 GPU_MEM=0.88 MAX_NUM_SEQS=16 bash _serve_k3_bench_spec.sh"
