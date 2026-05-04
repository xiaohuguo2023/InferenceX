---
name: inferencex-bench-compare
description: >-
  Run a single-node sweep on AMD MI355x for an InferenceX-tracked model, pull
  the matching dashboard + B200/B300 reference numbers from the public IX-app
  release dump, and produce comparison + Pareto plots and a writeup. Use when
  the user asks to "do as we did for gptoss/kimik2.5", run a sweep and compare
  with InferenceX, compare MI355x vs B200/B300, refresh dashboard data, or
  produce pareto plots for a model recipe on MI355x.
---

# InferenceX bench + compare workflow

End-to-end recipe for: pick a model → run a widegraph-default sweep on MI355x →
compare with the IX dashboard (own MI355x + B200/B300) → produce plots + a
written analysis. Mirrors what was done for gptoss-fp4 and kimik2.5-fp4.

## Bundled templates (in this skill dir)

These are the working artifacts from the kimik2.5 + gptoss runs — copy them
into the repo and edit, don't write from scratch:

- **`sweep_kimik25_widegraph_default_mi355x.sh`** — MLA-model sweep template.
  Has all env-var filters (`TP_FILTER`, `CONC_FILTER`, `ISL_OSL_FILTER`,
  `OUT_BASE`, `MODEL`) and the firehose `tee + PIPESTATUS` driver-log layout.
  Best starting point for any DeepseekV3 / Kimi-style model.
- **`sweep_gptoss_widegraph_default_mi355x.sh`** — non-MLA (regular MHA) MoE
  template. Older layout (no env-var filters). Use as a structural reference
  when the model isn't MLA.
- **`plot_kimik25_compare.py`** — produces the 2x3 grid plots, interactivity
  paretos, e2el paretos, and CSV. Parameterize by editing `SOURCES`, `TPS`,
  `COLS`, `CONCS`, the sweep filename glob in `load_sweep()`, and the output
  filename prefix.
- **`example_writeup_kimik25.md`** — full worked example; mirror its section
  structure (Setup / Coverage / Image-version caveat / Findings TP=4 /
  Findings TP=8 / Pareto observations / Anomalies / Next steps).

```bash
# Standard copy-and-edit kickoff for a new model:
SKILL=$CLAUDE_PROJECT_DIR/.claude/skills/inferencex-bench-compare
cp "$SKILL/sweep_kimik25_widegraph_default_mi355x.sh"  sweep_<model>_widegraph_default_mi355x.sh
cp "$SKILL/plot_kimik25_compare.py"                    comparison_plots/plot_<model>_compare.py
cp "$SKILL/example_writeup_kimik25.md"                 sweep_<model>_comparison.md  # then edit
```

## Inputs the user usually gives you

- Model + precision + hw (e.g. "kimik2.5 fp4 mi355x"), or just a sweep script
  filename like `sweep_<model>_widegraph_default_mi355x.sh`.
- Optional: which container, which CONCs, whether to compare to B200/B300.

## Outputs you produce

- Sweep results in `/home/xiaohugu/work/sweep_<model>_output/sweep_<...>_<ts>/`
- `comparison_plots/<model>_mi355x_vs_b200_tp{4,8}.png` — 2x3 grids
- `comparison_plots/<model>_pareto_{1k1k,8k1k}_tp{4,8}.png` — interactivity pareto
- `comparison_plots/<model>_pareto_e2el_{1k1k,8k1k}_tp{4,8}.png` — e2el pareto
- `comparison_plots/<model>_data_table.csv` — per-combo numbers across all sources
- `sweep_<model>_comparison.md` — writeup at repo root

---

## 0. Survey what's already there

Don't re-derive. Read first:

```bash
ls comparison_plots/                    # prior model plots (gptoss, kimik2.5)
ls sweep_*_widegraph_default_mi355x.sh  # prior sweep scripts as templates
cat sweep_kimik25_comparison.md         # writeup template (most complete)
```

Two sweep scripts are the templates: `sweep_kimik25_widegraph_default_mi355x.sh`
(MLA model, 64 heads → TP=8 head-count gotcha) and
`sweep_gptoss_widegraph_default_mi355x.sh` (non-MLA). The kimik25 script has the
extra env-var filters (`TP_FILTER`, `CONC_FILTER`, `ISL_OSL_FILTER`, `OUT_BASE`)
and the firehose driver.log layout — copy from there.

## 1. Find the dashboard config

```bash
grep -n "<model>-<precision>-mi355x" .github/configs/amd-master.yaml
```

Note the `image:` field — that's the vLLM version the dashboard pinned. You
will probably NOT use the same image (see §3); flag the divergence in the
writeup.

## 2. Pull the IX dashboard dump (PUBLIC — no auth, no `gh` needed)

The InferenceX website at https://inferencex.com is just a frontend over a
weekly database dump published as a release asset on the public
`SemiAnalysisAI/InferenceX-app` GitHub repo. **One zip contains every
configuration the website renders — MI355x, MI300x, B200, B300, H200, etc.**
There is no separate per-hardware feed and no scraping needed. Authentication
is not required.

```bash
mkdir -p /tmp/inferencex_dump && cd /tmp/inferencex_dump
TAG=$(curl -s "https://api.github.com/repos/SemiAnalysisAI/InferenceX-app/releases?per_page=1" \
      | python3 -c "import json,sys;print(json.load(sys.stdin)[0]['tag_name'])")
DATE=${TAG##*/}   # e.g. 2026-04-27
curl -sL -o dump.zip "https://github.com/SemiAnalysisAI/InferenceX-app/releases/download/${TAG}/inferencex-dump-${DATE}.zip"
unzip -q -o dump.zip
ls inferencex-dump-${DATE}/
```

Releases drop **weekly**, tagged `db-dump/<YYYY-MM-DD>`. List recent ones with:

```bash
curl -s "https://api.github.com/repos/SemiAnalysisAI/InferenceX-app/releases?per_page=10" \
  | python3 -c "import json,sys;[print(r['tag_name'],r['name']) for r in json.load(sys.stdin)]"
```

Schema: `configs.json` has `id, hardware, framework, model, precision,
decode_tp, ...`. `benchmark_results.json` has rows with `config_id, isl, osl,
conc, date, image, metrics{tput_per_gpu, mean_ttft, mean_tpot, mean_e2el, ...}`.
Throughput in dump is **per-GPU**; total = `tput_per_gpu * decode_tp`. Times
are in **seconds**; ours are in **ms**.

## 3. Identify peer configs (own MI355x + B200/B300/H200)

One copy-paste filter for any model+precision; lists every cfg across vendors:

```python
import json
from collections import defaultdict

DUMP = "/tmp/inferencex_dump/inferencex-dump-2026-04-27"  # <-- date from §2
MODEL, PREC = "kimik2.5", "fp4"                           # <-- edit

cfgs = json.load(open(f"{DUMP}/configs.json"))
br   = json.load(open(f"{DUMP}/benchmark_results.json"))

target = [c for c in cfgs
          if c["model"].startswith(MODEL) and c["precision"] == PREC
          and not c["disagg"] and not c["is_multinode"]]
print(f"=== {MODEL} {PREC} single-node configs ===")
for c in sorted(target, key=lambda x: (x["hardware"], x["framework"], x["decode_tp"])):
    print(f"cfg {c['id']:>4} | {c['hardware']:<6} | {c['framework']:<6} | TP={c['decode_tp']}")

# For each cfg, audit ISL/OSL/CONC/date coverage BEFORE picking it as a series
print("\n=== Coverage per cfg ===")
ids = {c["id"] for c in target}
by = defaultdict(lambda: defaultdict(set))
for r in br:
    if r.get("config_id") in ids and not r.get("error"):
        by[r["config_id"]][(r["isl"], r["osl"])].add((r["conc"], r["date"][:10]))
for cid in sorted(by):
    print(f"\ncfg {cid}:")
    for (isl, osl), pts in sorted(by[cid].items()):
        concs = sorted({c for c, _ in pts})
        dates = sorted({d for _, d in pts})
        print(f"  ({isl},{osl}): conc={concs} dates={dates}")
```

For each (hw, TP) you may see multiple cfg IDs (older + newer reruns). Pick
the most recent with full CONC coverage (4–64); fall back to older for combos
the new one doesn't cover. **Always audit before picking** — some cfgs are
intentionally sparse and reran sparse (e.g., kimik2.5-fp4 B200 cfg 636 ran
3× across March-April and stayed CONC=4-only every time; no amount of waiting
will fill it in).

### Typical NV peer cfgs you'll want for a fair comparison

| Series | What to look for |
|---|---|
| IX MI355x vllm | `hardware=mi355x, framework=vllm` — your dashboard parity baseline |
| IX B200 vllm   | `hardware=b200,   framework=vllm` — direct vendor comparison |
| IX B200 trt    | `hardware=b200,   framework=trt`  — NV best-case (often only exists for popular models) |
| IX B300 vllm   | `hardware=b300,   framework=vllm` — newer NV |
| IX H200 vllm   | `hardware=h200,   framework=vllm` — older NV reference if relevant |

## 4. Write or adapt the sweep script

Copy `sweep_kimik25_widegraph_default_mi355x.sh` as a starting point. Required
edits per model:

- `MODEL` default path
- `RESULT_FILENAME` prefix (e.g. `kimik25_widegraph_default_mi355x_…`)
- AITER env vars per the AMD recipe for that model:
  - MLA models (DeepseekV3/Kimi backbones): `VLLM_ATTENTION_BACKEND=TRITON_MLA`
    (note: this name is ignored on vLLM 0.16; use `VLLM_ROCM_USE_AITER_MLA` to
    toggle the actual AITER MLA backend)
  - non-MLA: `VLLM_USE_AITER_UNIFIED_ATTENTION=1`,
    `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1`
  - MoE: `VLLM_ROCM_USE_AITER_MOE=1`; if FP4 weights: `VLLM_ROCM_USE_AITER_FUSED_MOE_A16W4=1`
  - RMSNORM toggling varies per recipe; check the AMD model README
- COMBOS array — match the dashboard's TP × ISL/OSL × CONC search-space (read
  it from `amd-master.yaml`).
- Per-model parsers: `--reasoning-parser`, `--tool-call-parser` (Kimi: `kimi_k2`).
- Keep the firehose layout from the kimik25 template:
  - `OUT_BASE="${OUT_BASE:-/workspace/sweep_<model>_widegraph_default_$(date +%Y%m%d-%H%M%S)}"`
  - `( run_one_combo … ) 2>&1 | tee "$OUT_BASE/<...>.stdout"` and `rc=${PIPESTATUS[0]}`
  - Filter helpers: `TP_FILTER`, `CONC_FILTER`, `ISL_OSL_FILTER` (all comma-separated)

## 5. Container setup

The default sweep container expectations:

```
mounts:  /home/xiaohugu → /home,  /data → /data,
         /home/xiaohugu/work/sweep_<model>_output → /workspace
devices: /dev/kfd, /dev/dri        groups: video        caps: SYS_PTRACE
network: host    ipc: host    shm-size: 16G
```

User's standard `drun` alias has most of this. To run detached, override
entrypoint with `sleep infinity`:

```bash
sudo docker run -d --name xguo-<model>-bench \
  --network=host --ipc=host --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size=16G --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $HOME:/home -v /data:/data \
  -v /home/xiaohugu/work/sweep_<model>_output:/workspace \
  --entrypoint sleep <image> infinity
```

### Choosing the image

- The dashboard `image:` from amd-master.yaml gives strict apples-to-apples but
  may be hard to obtain or older than what's on disk.
- `rocm/vllm-dev:nightly_main_20260224` ships **vLLM 0.16.0rc2.dev445** baked in.
  This is older than typical dashboard pins (often 0.18+). Document the drift.
- `xguo-comms4` is the same image, but with editable `/home/work/vllm`
  installed → reports as **0.19.1rc1.dev210**. Use this when you need newer
  vLLM. Note: `/data` is NOT mounted in xguo-comms4 — either move the model
  under `/home/xiaohugu/` (`mv` is free if same fs) or recreate the container
  with `-v /data:/data`.

### The model-mount trick

Models often live in `/data/amd/<MODEL>` which isn't mounted in xguo-comms4.
Same-filesystem `mv` is instant and uses no extra disk:

```bash
sudo mv /data/amd/<MODEL> /home/xiaohugu/<MODEL>
# now visible as /home/<MODEL> inside xguo-comms4
```

**Always check `df` first** — `/data` and `/home` may already be 95%+ full;
`mv` succeeds because it's a rename, but `cp` would fail.

### Downloading a missing model

`hf` CLI is in the vllm-dev image, not on host. Spin up a tiny container with
`/data:/data`:

```bash
sudo docker run -d --rm --name xguo-dl -v /data:/data <image> sleep infinity
sudo docker exec -d xguo-dl bash -c '
  export HF_HUB_ENABLE_HF_TRANSFER=1
  hf download <repo> --local-dir /data/amd/<MODEL> --max-workers 8 \
    > /data/amd/.<model>_download.log 2>&1
  echo DONE_RC=$? >> /data/amd/.<model>_download.log
'
```

Pre-flight: `curl -s "https://huggingface.co/api/models/<repo>/tree/main" | …`
to total bytes vs `df -h /data` free space.

## 6. Run the sweep

```bash
sudo docker exec -d xguo-<model>-bench bash -c "
  cd /home/work/InferenceX && \
  MODEL=/home/<MODEL> \
  bash sweep_<model>_widegraph_default_mi355x.sh > /workspace/driver.log 2>&1
"
```

Optional filters for partial reruns:

```bash
TP_FILTER=8 CONC_FILTER=4 ISL_OSL_FILTER=8192/1024 \
VLLM_ROCM_USE_AITER_MLA=0 \  # any per-run env toggle
bash sweep_<model>_widegraph_default_mi355x.sh
```

### Monitoring

```bash
tail -f /home/xiaohugu/work/sweep_<model>_output/driver.log  # firehose
tail -F /home/xiaohugu/work/sweep_<model>_output/server.log  # active vllm only
ls /home/xiaohugu/work/sweep_<model>_output/sweep_*/*.json | wc -l   # combos done
```

Per-combo wall: TP=4 ranges 5–10 min for 1k/1k and 8k/1k, 20–50 min for 1k/8k.
30-combo full sweep is ~3.5–5 h.

### Common failures

- **"Engine core initialization failed"** with `assert num_heads == 16 or 128`
  on Kimi/DeepSeek MLA at TP=8 in vLLM 0.16: rerun with
  `VLLM_ROCM_USE_AITER_MLA=0` to fall back to TRITON_MLA, OR move to vLLM 0.18+
  where the assertion is relaxed. Triton fallback is much slower at long ISL
  (see kimik25 writeup for the 2–3× TPOT regression).
- **"Server died before becoming healthy"** in 56 s with no useful
  traceback: search the `<combo>.stdout` (or `.server.log`) for
  `assert|RuntimeError|HIP|out of memory` before the `Server died` line.
- **Permission denied on `/workspace`/output dirs**: containers run as root,
  outputs end up root-owned. Use `sudo mkdir`/`sudo chown` to create writable
  subdirs first.
- **`pip install -e /home/work/vllm` fails with `CUDA_HOME is not set`**:
  don't try to install; either copy the existing `__editable__.vllm-*.pth`
  from xguo-comms4's site-packages, or set
  `PYTHONPATH=/home/work/vllm` (works for `python -m vllm.…`; may not for the
  `vllm` CLI).

## 7. Plot

The plot script `comparison_plots/plot_kimik25_compare.py` is the template.
Copy and adapt:

```bash
cp comparison_plots/plot_kimik25_compare.py comparison_plots/plot_<model>_compare.py
```

Edit constants near top:
- `SOURCES` cfg lists per series (own MI355x + B200 + B300; add TRT if it exists)
- `TPS`, `COLS` (= ISL/OSL grid), `CONCS`
- The sweep filename glob in `load_sweep()` (the prefix changes per model)
- Output filename prefix passed to `plot_tp` and `plot_pareto`

Then:

```bash
python3 comparison_plots/plot_<model>_compare.py \
  /home/xiaohugu/work/sweep_<model>_output/sweep_<...>_<ts>/
```

Produces: 2 `<model>_mi355x_vs_b200_tp{4,8}.png` + 4 interactivity paretos +
4 e2el paretos + 1 CSV.

### When to merge multiple sweep dirs

If you ran TP=4 and TP=8 in different containers (e.g., 0.16 for TP=4, 0.19 for
TP=8), merge into one dir before plotting:

```bash
sudo cp -p <tp8_dir>/*tp8* <tp4_dir>/
# Archive any prior TP=8 runs you're replacing:
sudo mkdir -p <tp4_dir>/v016_tp8_archive
sudo mv <tp4_dir>/*tp8.* <tp4_dir>/v016_tp8_archive/
```

Plot script reads the merged dir; record provenance per source in the writeup.

## 8. Writeup

Use `sweep_kimik25_comparison.md` as the template. Required sections:

1. **Setup** — model, hw, container, image, vLLM version, sweep dir, plot script
2. **Coverage** — TP × ISL/OSL × CONC table; mark missing combos
3. **Image-version caveat** — vLLM versions for ours vs IX-MI355x vs IX-B200/B300
4. **Headline plots** — links to all PNGs in `comparison_plots/`
5. **Findings TP=4** — sanity check vs IX MI355x (within ±10% expected); cross-vendor gap
6. **Findings TP=8** — same, plus any image/backend caveats
7. **Pareto observations** — what each pareto shows (curves dominating? sparse points?)
8. **Dashboard anomalies** — flag any obvious outliers in IX data (e.g. one ttft point 5× off-trend)
9. **What's missing / next steps** — coverage gaps in dashboard, follow-ups

### Standard caveats / always include

- IX `tput_per_gpu` is per-GPU, multiply by `decode_tp` for total throughput.
- IX times are seconds; ours are ms; conversion in `metric_from_*` helpers.
- Single-run measurements on our side; dashboard typically reruns across days.
- Keep the comparison "ours within 5–10% of IX MI355x" claim explicit if true —
  that's the headline that the recipe reproduces the dashboard.

## 9. Drilling into a perf gap (when ours diverges from dashboard)

Don't just shrug. Decompose:

1. Is it the **vLLM version**? Run a smoke test (single combo) on a different
   vLLM with same backends — see kimik25's "v0.16+Triton vs v0.19+Triton vs
   v0.19+AITER" 3-way decomposition.
2. Is it the **attention backend**? Toggle AITER MLA / unified-attention via env.
3. Is it the **MoE kernel path**? Check `VLLM_ROCM_USE_AITER_MOE`, A16W4 fusion.
4. Is it the **cudagraph capture** coverage? Check `cudagraph_capture_sizes` log line.

For each smoke test, run only the combo where the gap is biggest (often
8k/1k CONC=4) — ~5 min wall vs full sweep ~5 h.

## 10. Skeleton commands cheat-sheet

```bash
# Pull dump
mkdir -p /tmp/inferencex_dump && cd /tmp/inferencex_dump && \
  TAG=$(curl -s "https://api.github.com/repos/SemiAnalysisAI/InferenceX-app/releases?per_page=1" | \
        python3 -c "import json,sys;print(json.load(sys.stdin)[0]['tag_name'])") && \
  DATE=${TAG##*/} && \
  curl -sL -o dump.zip "https://github.com/SemiAnalysisAI/InferenceX-app/releases/download/${TAG}/inferencex-dump-${DATE}.zip" && \
  unzip -q -o dump.zip

# Move model into container-visible path (instant, same-fs rename)
sudo mv /data/amd/<MODEL> /home/xiaohugu/<MODEL>

# Launch sweep detached
sudo docker exec -d xguo-<model>-bench bash -c "
  cd /home/work/InferenceX && \
  MODEL=/home/<MODEL> \
  bash sweep_<model>_widegraph_default_mi355x.sh > /workspace/driver.log 2>&1
"

# Plot + writeup
python3 comparison_plots/plot_<model>_compare.py /home/xiaohugu/work/sweep_<model>_output/sweep_*/
${EDITOR:-vi} sweep_<model>_comparison.md
```
