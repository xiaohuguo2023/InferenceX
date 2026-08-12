# Kimi-K3 DSpark + native fp8-asm KV — reproduction recipe (MI355X, TP8)

A self-contained runbook to reproduce **DSpark speculative decoding on Kimi-K3
FP4 with a native fp8 KV cache served on the asm MLA path** — clean
`FULL_AND_PIECEWISE` cudagraphs (NO eager), on 8× MI355X (gfx950), TP8.

**Validated 2026-08-10** on container `xguo-k3nc`:
mean acceptance length **2.39** (natural text), clean capture (PIECEWISE 15/15,
FULL 8/8), aiperf sweep with **0 faults** — including at the ISL that faulted the
earlier PIECEWISE-only attempt.

> This is the **fp8-asm** variant. The all-bf16 / gluon-verify variant and the
> conceptual background live in the shipped docs — read those first, do not edit
> them:
> - `docs/DSpark_Tutorial.md` — what DSpark is and how it works.
> - `dspark.md` — the operational runbook for the **bf16** DSpark path.
> - `K3_DSpark_Benchmark_Report.md` — reference numbers.
>
> This file is the **delta** for pushing the KV cache to native fp8 on the asm
> fold. It builds on the SHIPPED recipe backend (main + #51011 + #51040 +
> #51606), NOT the non-causal nightly fork (that fork was the source of the
> cudagraph OOB — a dead end).

---

## 0. TL;DR — what makes fp8-asm work

Starting from a **working baseline K3 server** (see §1) plus the **SHIPPED recipe
`rocm_aiter_mla.py`** (main + #51011 + #51040 + #51606), fp8-asm DSpark needs
exactly **five** things:

1. **Force the DSpark draft causal** (`dflash_config.causal=true`) so the draft
   runs on the fp8 asm path (loss-less — the target verify catches mispredicts).
2. **aiter `get_block_n_fp8` table fix** — add the DSpark verify-width key
   (`nhead*max_seqlen_q = 16*5 = 80`) so the fp8 block-size lookup doesn't
   `KeyError`.
3. **DSpark verify width** in the recipe backend — `_mtp_decode_qlen` must be
   `1 + 2*num_spec` for `method="dspark"` (not the `else→1` fallthrough).
4. **Persistent-metadata gate** in the recipe backend — build asm work-meta for
   K3's 12-head rank so the qlen=5 verify uses the **persistent** asm kernel
   (the non-persistent asm decode caps at qo_len≤4 and faults at 5).
5. **`VLLM_ROCM_AITER_MLA_ASM_PADDING=asm`** at serve time — pads 12→16 heads and
   folds the qlen>1 verify onto the asm path.

Plus one correctness improvement (not the blocker): **KDA stride fix** (§4.5).

Plus one **decode-perf** fix (not a blocker, but required for a flat ITL curve):
**reroute the decode-range FlyDSL dense GEMMs to torch** (§4.6). At `M = 3×conc = 72`
(conc-24) the K3 dense BF16 GEMMs `(N,K) ∈ {(1024,7168),(384,7168),(7168,512)}` route to
aiter FlyDSL, whose per-call Python launcher + split-K semaphore path runs **eager inside
the FULL decode cudagraph** → decode is launch-bound → TP ranks desync → all-reduce spin
(conc-24 ITL p50 ~47 ms, off-trend). Fix = convert those FlyDSL rows (all `M≤192`) to a
native torch row in `merged_bf16_tuned_gemm.csv` via `_patch_flydsl_decode_to_torch.sh`.
Full analysis: `docs/kimik3_conc24_regression_allreduce.md`.

---

## 1. Prerequisites — a working baseline K3 (+ recipe backend)

fp8-asm DSpark is a delta on top of a correct **non-spec** K3 server. Get that
serving first — the full base setup is in the `k3-nightly-migration` notes and
in `dspark.md` §2. Summarized so this file stands alone:

| Component | Value |
|---|---|
| Image | `vllm/vllm-openai-rocm:nightly-cb8104839c141609d99f1254459ef3a4f1bd4263` |
| vLLM | `0.26.1rc1.dev306+gcb8104839` |
| torch | `2.11.0` |
| ROCm | `7.2.3` |
| aiter | rebuilt in-container (this session: `55dbc4f47`) **+ the §4.2 table fix** |
| GPUs | 8× MI355X, gfx950, TP8 |
| Target | `moonshotai/Kimi-K3` FP4 (a8w4 MoE) |
| Draft | `Inferact/Kimi-K3-DSpark` (forced causal, §3) |

Baseline requirements (all in the migration notes / `dspark.md` §2):

1. **Triton 3.7.0** (nightly ships 3.6.0):
   ```bash
   pip install --extra-index-url https://pypi.amd.com/triton/release/rocm-7.2.0/simple/ \
       triton==3.7.0 tabulate
   ```
2. **aiter rebuilt in-container** (carries the Opus a8w4 dispatch fix; a bare
   merge of latest-main is not enough — it warms up fine but a worker
   GPU-core-dumps a few decode steps into real load).
3. **PR #50618** in `vllm/model_executor/layers/utils.py` — `contiguous()` guard
   before `ops.wvSplitKrc` / `ops.wvSplitK`. **THE decode-crash fix.**
4. **a8w4 MoE — BOTH env flags** (or MoE silently falls back to bf16):
   `VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1` **and** `AITER_SITUV2_A8W4=1`.
5. **Do NOT set** `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` (breaks
   aiter custom all-reduce IPC → all ranks die at init).
6. **The SHIPPED recipe `rocm_aiter_mla.py`** (main + #51011 + #51040 + #51606)
   swapped into
   `/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/rocm_aiter_mla.py`.
   #51011 is the core 12-head fp8 spec-verify enabler (routes the small-head
   verify to the asm q-row-fold). #51040/#50578 add the `ASM_PADDING` env;
   #51606 the amd DSpark MLA glue.

Validate the baseline serves and passes GSM8K (~0.968) **without** spec before
continuing.

---

## 2. Container setup (host repo mounted at `/workspace`)

Use `setup_benchmark.sh` with the **DSpark nightly image** (not `kimi-k3`):

```bash
# on compute node
export K3_CTR=k3-dspark-benchmark
./setup_benchmark.sh start-dspark    # nightly-cb8104839...
./setup_benchmark.sh setup-dspark    # aiter + aiperf + triton 3.7 + draft + all patches
./setup_benchmark.sh verify-dspark-patches
./setup_benchmark.sh serve-dspark    # PORT=8890 NUM_SPEC=2 by default
```

`setup-dspark` applies **both** stacks in order:
1. InferenceX fp8-asm patches (decode #50578, prefill PR-A, PS metadata16, skip K3 fp8 PS, wvSplitK #50618)
2. DSpark enablement (`_k3_dspark_fp8asm_apply_patches.sh`: draft causal, aiter table key 80, dspark qlen, persistent gate, KDA PR#27)

Manual / legacy container (`xguo-k3nc`, no mount): copy scripts in and edit package files via `docker exec`:

```bash
docker cp _serve_k3_bench_spec.sh   xguo-k3nc:/workspace/
docker cp _bench_k3_dspark_fp8asm.sh xguo-k3nc:/workspace/
docker cp _k3_dspark_fp8asm_apply_patches.sh xguo-k3nc:/workspace/
docker exec -it xguo-k3nc bash -lc 'bash /workspace/_k3_dspark_fp8asm_apply_patches.sh'
```

All §4 snippets are meant to run **inside** the container. Every one is
idempotent (guards on the anchor / target text) and takes a `.bak` backup.

---

## 3. Force the DSpark draft causal

The draft's semi-autoregressive parallel drafting defaults to **non-causal**
attention (`use_non_causal = dflash_has_any_non_causal`). On this ROCm build the
non-causal MLA path is the source of the cudagraph OOB. Forcing the draft causal
makes `use_non_causal=False`, so the draft runs on the fp8 asm path like the
target; the target verify still catches any mispredictions, so accuracy is
preserved.

Edit the **draft** `config.json` (resolve the snapshot dir first):

```bash
DCACHE=/dev/shm/hf-cache/models--Inferact--Kimi-K3-DSpark/snapshots
CFG="$(ls -d "$DCACHE"/*/ | head -1)config.json"
cp -n "$CFG" "$CFG.orig.bak"
python3 - "$CFG" <<'PY'
import json, sys
f = sys.argv[1]
c = json.load(open(f))
d = c.setdefault("dflash_config", {})
d["causal"] = True
json.dump(c, open(f, "w"), indent=2)
print("forced causal:", f)
PY
```

> The forced-causal serve script (§5) **refuses to start** if it can't confirm
> the draft is causal — this edit is required.

---

## 4. The in-container patches

Run each block inside the container. They are ordered; all are idempotent.

> **One-command apply:** all of §3 + §4.2–§4.5 are packaged in
> `_k3_dspark_fp8asm_apply_patches.sh`. Copy it in and run it instead of the
> individual blocks below (the blocks remain here for reference / manual fixup if
> an anchor drifts):
> ```bash
> docker cp _k3_dspark_fp8asm_apply_patches.sh xguo-k3nc:/workspace/
> docker exec -it xguo-k3nc bash -lc 'bash /workspace/_k3_dspark_fp8asm_apply_patches.sh'
> ```

### 4.1 (baseline) recipe backend must already be in place

Confirm the SHIPPED recipe file is installed (§1.6). The two edits below (§4.3,
§4.4) modify **that** file. Sanity:

```bash
D=/usr/local/lib/python3.12/dist-packages
grep -c _mtp_decode_qlen "$D/vllm/v1/attention/backends/mla/rocm_aiter_mla.py"  # >0
```

### 4.2 aiter `get_block_n_fp8` — add the DSpark verify-width key

The fp8 block-size table is keyed by `nhead * max_seqlen_q`. The DSpark verify is
`nhead=16` (padded) × `qo_len=5` = **80**, which the table lacked → `KeyError:
80`. `min_block_n` only bounds `num_kv_splits` in integer arithmetic (it does NOT
select a kernel variant), so defaulting missing keys to 64 is correctness-safe.

```bash
python3 - <<'PY'
import re, py_compile
F = "/root/aiter/aiter/mla.py"
s = open(F).read()
import shutil; shutil.copy2(F, F + ".pre_dspark.bak")
# add keys 80/96/112 -> 64 to the get_block_n_fp8 dict if absent
if "80: 64" not in s:
    s = re.sub(r"(get_block_n_fp8\s*=\s*\{)", r"\g<1>\n    80: 64, 96: 64, 112: 64,", s, count=1)
# make the lookup default to 64 instead of raising
s = s.replace(
    "min_block_n = get_block_n_fp8[int(nhead * max_seqlen_q)]",
    "min_block_n = get_block_n_fp8.get(int(nhead * max_seqlen_q), 64)",
)
open(F, "w").write(s); py_compile.compile(F, doraise=True)
print("aiter get_block_n_fp8 patched; 80-key:", "80: 64" in s,
      "get():", "get_block_n_fp8.get(" in s)
PY
```

> The exact anchor text may differ slightly by aiter revision. If either
> `replace` is a no-op, open `/root/aiter/aiter/mla.py`, find the
> `get_block_n_fp8` dict and its single subscript use, add keys `80/96/112: 64`,
> and change the `[...]` lookup to `.get(..., 64)`.

### 4.3 recipe backend — DSpark verify width (`_mtp_decode_qlen`)

The recipe only handled `mtp` / `deepseek_mtp` (qlen = `num_spec+1`); `dspark`
fell through to `else → 1`. DSpark's target verifies a **wide block =
`1 + 2*num_spec`** (= 5 at num_spec=2). Add a `dspark` branch so the persistent
metadata buffers are sized for qlen=5.

```bash
python3 - <<'PY'
import py_compile, shutil
F = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/rocm_aiter_mla.py"
s = open(F).read()
shutil.copy2(F, F + ".pre_dsparkqlen.bak")
if 'speculative_config.method == "dspark"' not in s:
    # insert a dspark branch just before the generic `else:` that sets qlen = 1.
    # Anchor: the mtp branch that sets self._mtp_decode_qlen from num_speculative_tokens.
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
    assert s.count(anchor) == 1, f"anchor count {s.count(anchor)}"
    s = s.replace(anchor, branch + anchor, 1)
    open(F, "w").write(s); py_compile.compile(F, doraise=True)
    print("dspark _mtp_decode_qlen branch added")
else:
    print("dspark _mtp_decode_qlen branch already present")
PY
```

> If the anchor text differs, find where `self._mtp_decode_qlen` is set from the
> speculative config and add an `elif method=="dspark"` branch setting it to
> `2*num_speculative_tokens + 1`.

### 4.4 recipe backend — persistent-metadata gate

The gate keyed off raw `self.num_heads` (=12 at K3 TP8) `>= 16` → always False
for K3 → no `work_meta_data` built → the qlen=5 verify hit the **non-persistent**
asm decode (caps qo_len≤4) and faulted (`asm_mla.cu:907 ... qo_len>4 in
persistent mode`). Broaden the gate so asm-padded / multi-token verify builds the
persistent metadata.

```bash
python3 - <<'PY'
import py_compile, shutil
F = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/rocm_aiter_mla.py"
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
    print("gate already broadened")
elif old in s:
    s = s.replace(old, new, 1); open(F, "w").write(s); py_compile.compile(F, doraise=True)
    print("persistent-metadata gate broadened")
else:
    print("!! gate anchor not found — inspect use_persistent_metadata manually")
PY
```

> If the anchor differs: find where `use_persistent_metadata` is computed and
> replace the `self.num_heads >= _AITER_MIN_MLA_HEADS` term with
> `uses_asm_decode = num_heads>=16 OR small_head_mode=="asm" OR max_qo_len>1`.

### 4.5 KDA stride fix (correctness — Fangzhou-Ai/vllm PR #27)

Not the fp8-asm blocker, but correct for single-request strided `state_indices`
views (supersedes the earlier `.contiguous()` no-op). Make the KDA packed decode
kernel stride-aware.

```bash
python3 - <<'PY'
import py_compile, shutil
F = "/usr/local/lib/python3.12/dist-packages/vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py"
s = open(F).read()
shutil.copy2(F, F + ".pre_pr27.bak")
changed = False
# 1) kernel signature: add stride_indices_seq after stride_state_token
if "stride_indices_seq: tl.constexpr," not in s:
    s = s.replace("    stride_state_token: tl.constexpr,\n",
                  "    stride_state_token: tl.constexpr,\n    stride_indices_seq: tl.constexpr,\n", 1)
    changed = True
# 2) strided load of the state index
s2 = s.replace("state_idx = tl.load(state_indices + i_n).to(tl.int64)",
               "state_idx = tl.load(state_indices + i_n * stride_indices_seq).to(tl.int64)")
changed |= s2 != s; s = s2
# 3) wrapper: reject non-1D instead of silently reshaping
s2 = s.replace(
    "    if state_indices.ndim != 1 or state_indices.stride(0) != 1:\n"
    "        state_indices = state_indices.reshape(-1).contiguous()",
    "    if state_indices.ndim != 1:\n"
    '        raise ValueError("`state_indices` must be one-dimensional.")')
changed |= s2 != s; s = s2
# 4) launch: pass the stride (skip if nightly already ships it)
import re
if not re.search(r"stride_indices_seq=\w+\.stride\(0\)", s):
    s = s.replace("        stride_state_token=initial_state.stride(0),\n",
                  "        stride_state_token=initial_state.stride(0),\n"
                  "        stride_indices_seq=state_indices.stride(0),\n", 1)
    changed = True
if changed:
    open(F, "w").write(s)
py_compile.compile(F, doraise=True)
print("KDA PR#27 applied" if changed else "KDA PR#27 already present")
PY
```

### 4.6 FlyDSL decode GEMM reroute (decode perf — flat ITL curve)

At `M = 3×conc ≤ 192` the K3 dense BF16 GEMMs `(N,K) ∈ {(1024,7168),(384,7168),(7168,512)}`
match `libtype=="flydsl"` rows in `merged_bf16_tuned_gemm.csv`. FlyDSL's per-call Python
launcher + split-K semaphore path is not cudagraph-capturable in the decode static-buffer
FULL path, so at M=72 (conc-24) those GEMMs run **eager every decode step** → launch-bound
decode → TP rank desync → `cross_device_reduce_2stage` spin (conc-24 ITL p50 ~47 ms,
off-trend). Reroute them to a native torch row (rocBLAS/hipBLASLt Cijk, captured) — exactly
what conc-16/48 already fall back to.

```bash
# from the host repo (edits the in-container CSV; idempotent; backs up to .pre_flydsl_fix.bak)
CONTAINER=k3-dspark-benchmark bash /workspace/_patch_flydsl_decode_to_torch.sh
```

Do NOT leave `splitK` empty when hand-editing: `pandas.read_csv` turns the empty cell to NaN,
floats the whole column, and pre-existing `asm` rows then pass `splitK=3.0` to
`aiter::_gemm_a16w16_asm` (Optional[int]) → engine-core init crash. Native torch rows use
`splitK=0`. Full analysis: `docs/kimik3_conc24_regression_allreduce.md`.

### 4.7 verify the patches

```bash
D=/usr/local/lib/python3.12/dist-packages
R="$D/vllm/v1/attention/backends/mla/rocm_aiter_mla.py"
echo "aiter 80-key   = $(grep -c '80: 64' /root/aiter/aiter/mla.py)                (expect >=1)"
echo "aiter get()    = $(grep -c 'get_block_n_fp8.get(' /root/aiter/aiter/mla.py)   (expect 1)"
echo "dspark qlen    = $(grep -c 'method == \"dspark\"' "$R")                        (expect >=1)"
echo "asm gate       = $(grep -c 'uses_asm_decode' "$R")                            (expect >=2)"
echo "kda stride     = $(grep -c 'stride_indices_seq' "$D/vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py")  (expect >=3)"
CSV=/opt/aiter-local/aiter/configs/merged_bf16_tuned_gemm.csv
echo "flydsl reroute = $(awk -F, '\$11=="flydsl" && \$3<=192 && (((\$4==1024||\$4==384)&&\$5==7168)||(\$4==7168&&\$5==512))' "$CSV" | wc -l)  (expect 0)"
python -c "import vllm.v1.attention.backends.mla.rocm_aiter_mla; print('IMPORT_OK')"
```

---

## 5. Serve (forced-causal, fp8-asm, NO eager)

Use the canonical forced-causal serve script `_serve_k3_bench_spec.sh` (copy it
into `/workspace` first, §2). It hardcodes `--max-model-len 1048576`,
`--enable-prefix-caching`, `ROCM_AITER_MLA`, `--kv-cache-dtype fp8`,
`FULL_AND_PIECEWISE`, `--async-scheduling`, and both a8w4 MoE flags. You MUST
export `VLLM_ROCM_AITER_MLA_ASM_PADDING=asm` (it folds the qlen>1 verify onto asm
and pads 12→16 heads).

```bash
# inside the container
export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm
NUM_SPEC=2 PORT=8890 GPU_MEM=0.88 MAX_NUM_SEQS=16 \
  bash _serve_k3_bench_spec.sh
# waits for /health; log at /workspace/serve_k3_bench_spec2.log
```

Notes:
- **NO `--enforce-eager`.** Real perf needs cudagraphs; the patches make FULL +
  PIECEWISE capture cleanly under spec. Eager is for debug only.
- `GPU_MEM=0.88` / `MAX_NUM_SEQS=16` are the validated concessions for the 1M
  context pool at TP8. **Do not use the agentic `MAX_NUM_SEQS=2*CONC` ladder**
  for DSpark serve — a large seq cap widens the PIECEWISE capture batch and the
  fp8 asm verify kernel writes past its arena (write-to-read-only-page on all
  GPUs). `_serve_k3_bench_spec.sh` defaults to 16; always export explicitly.
- With `NUM_SPEC=2`, vLLM still derives `max_cudagraph_capture_size = 96`
  (`16 × (1+N) × 2`), so aiter GEMM lines during capture will show `M:72`,
  `M:80`, … — that is normal at the recipe cap, not evidence of a missing seq
  limit. The loaded kernel `mla_a8w8_qh16_qseqlen4_…_ps` is the **correct**
  DSpark verify kernel; `qseqlen4` is not a qlen-sizing mismatch.
- Wire tuned GEMM when available (perf-only, not a fault blocker):
  `export AITER_CONFIG_GEMM_BF16=/opt/aiter-local/aiter/configs/merged_bf16_tuned_gemm.csv`
- To free the GPUs between runs (a bare `pkill "vllm serve"` does NOT work):
  `pkill -9 -f spawn_main; pkill -9 -f VllmWorker; pkill -9 -f EngineCore; pkill -9 -f "vllm serve"`
  then confirm each GPU is back to the ~0.28 GiB idle baseline.

### Confirm the capture was clean (no eager)

```bash
LOG=/workspace/serve_k3_bench_spec2.log
grep -iE "Capturing|cudagraph|PIECEWISE|FULL" "$LOG" | tail -20
grep -i "enforce.eager\|eager mode" "$LOG"   # should be empty
```

Expect both PIECEWISE and FULL capture lines to complete (this session: PIECEWISE
15/15, FULL 8/8) and no eager-mode messages.

---

## 6. Verify acceptance + smoke test

```bash
# health
curl -sf http://localhost:8890/health && echo OK

# quick coherence check
curl -s http://localhost:8890/v1/chat/completions -H 'content-type: application/json' -d '{
  "model":"Kimi-K3",
  "messages":[{"role":"user","content":"In one paragraph, explain speculative decoding."}],
  "max_tokens":128}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["message"]["content"])'

# acceptance length (read from the serve log after some traffic)
grep -iE "acceptance|accepted|draft.accept" /workspace/serve_k3_bench_spec2.log | tail -20
```

Validated result: **mean acceptance length 2.39** on natural text
(per-position 0.788 / 0.606, avg draft accept ≈69.7%); ~2.19–2.22 on the
synthetic aiperf workload. This is above the forced-causal N=2 target of 2.32.

---

## 7. InferenceX benchmark (aiperf sweep)

```bash
# copy in, then run inside the container
PORT=8890 bash _bench_k3_dspark_fp8asm.sh
# results under /workspace/k3_dspark_fp8asm_bench/concurrency_*__requests_*/
```

Sweep = ISL 1024 / OSL 256 (ignore_eos), concurrency 1 / 8 / 16. Validated
(0 faults):

| conc | TTFT | ITL | out tok/s | tok/s/user | req/s |
|---|---|---|---|---|---|
| 1  | 206 ms | 9.4 ms  | 98  | 107 | 0.38 |
| 8  | 400 ms | 17.3 ms | 413 | 59  | 1.61 |
| 16 | 535 ms | 23.5 ms | 606 | 43  | 2.37 |

---

## 8. Why the pieces are needed (one-liners)

- **Forced causal** → draft leaves the non-causal MLA path (the cudagraph OOB
  source) and runs on fp8 asm; verify preserves accuracy.
- **aiter table key 80** → the fp8 block-size lookup for the 16×5 verify no longer
  `KeyError`s; defaulting `min_block_n` is safe (only bounds `num_kv_splits`).
- **`_mtp_decode_qlen=2*num_spec+1`** → persistent metadata buffers sized for the
  wide qlen=5 verify block (not the `else→1` fallthrough).
- **broadened persistent gate** → K3's 12-head rank still builds asm work-meta, so
  qlen=5 uses the **persistent** asm kernel (non-persistent caps at qo_len≤4 →
  fault at 5).
- **`ASM_PADDING=asm`** → pads 12→16 heads and folds qlen>1 onto asm
  (`use_gluon_decode` returns False for qlen>1 regardless).
- **KDA PR #27** → correct strided `state_indices` load (correctness, not a
  blocker).

---

## 9. Files touched (in-container; all have `.bak` backups)

| File | Change | §  |
|---|---|---|
| draft `config.json` | `dflash_config.causal = true` | 3 |
| `/root/aiter/aiter/mla.py` | `get_block_n_fp8` keys 80/96/112 + `.get(k,64)` | 4.2 |
| `.../mla/rocm_aiter_mla.py` | dspark `_mtp_decode_qlen = 2*num_spec+1` | 4.3 |
| `.../mla/rocm_aiter_mla.py` | broadened `use_persistent_metadata` gate | 4.4 |
| `.../kda/fused_recurrent.py` | PR #27 stride-aware state_indices | 4.5 |
| `/opt/aiter-local/aiter/configs/merged_bf16_tuned_gemm.csv` | decode-range FlyDSL dense rows → native torch | 4.6 |

Host-side scripts (copy into `/workspace`): `_serve_k3_bench_spec.sh` (§5),
`_bench_k3_dspark_fp8asm.sh` (§7), `_patch_flydsl_decode_to_torch.sh` (§4.6),
`_dspark_longctx_bench.sh` (long-ctx aiperf sweep).
