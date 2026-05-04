# `sa_bench` — InferenceX-style sweep harness (PR1: MVP)

One script that runs the IX dashboard matrix for any `<model>-<prec>-<hw>-<framework>`
config in `.github/configs/amd-master.yaml`, reusing the per-model launcher in
`benchmarks/single_node/` as the recipe source-of-truth, and emitting an
IX-schema CSV the SA-style Pareto plot code can ingest directly.

## Why this exists

Per-model sweep scripts (`sweep_<model>_widegraph_default_mi355x.sh`) duplicate
~150 lines apiece and re-derive launch flags from each AMD recipe. That makes
adding a new model a day-long task. This harness:

1. Reads the matrix from `amd-master.yaml` (single source of truth for
   TP × ISL/OSL × CONC).
2. Runs the existing IX launcher per combo (single source of truth for AITER
   env + vllm CLI flags — same script the dashboard CI runs).
3. Captures per-combo JSON / server.log / gpu_metrics.csv into one out-dir.
4. Emits `sa_bench.csv` matching SA's `benchmark_results.json` columns
   (latency in seconds, throughput per-GPU) — drop-in for their Pareto plotter.
5. Writes `manifest.json` recording image, vllm version, env overlay, extra-env,
   git sha for provenance.

Adding a new model = 0 lines of new code if the launcher and amd-master.yaml
entry already exist (the IX team writes both for the dashboard).

## Quick start

Run inside a container with `/workspace` writable, the model present at the
path listed in `amd-master.yaml`, and this repo at `/home/work/InferenceX`.

```bash
# Full sweep for one config:
python3 benchmarks/harness/sa_bench.py \
    --config kimik2.5-fp4-mi355x-vllm \
    --out-dir /workspace/sweep_kimik25_$(date +%Y%m%d-%H%M%S)

# Or via shorter --model + --gpu form:
python3 benchmarks/harness/sa_bench.py \
    --model kimik2.5-fp4 --gpu mi355x \
    --out-dir /workspace/sweep_kimik25_$(date +%Y%m%d-%H%M%S)

# Subset / smoke test:
python3 benchmarks/harness/sa_bench.py \
    --config kimik2.5-fp4-mi355x-vllm \
    --tp 8 --conc 4 --isl-osl 8192/1024 \
    --extra-env VLLM_ROCM_USE_AITER_MLA=0 \
    --out-dir /workspace/smoke

# What would run, without launching anything:
python3 benchmarks/harness/sa_bench.py \
    --config gptoss-fp4-mi355x-vllm --dry-run
```

## CLI

| Flag | Description |
|---|---|
| `--config KEY` | Full amd-master.yaml key (e.g. `kimik2.5-fp4-mi355x-vllm`) |
| `--model NAME --gpu HW` | Shortcut: builds `<model>-<gpu>-<framework>` (default fw: vllm) |
| `--gpu {mi355x,mi300x,mi325x,b200,b300,h200}` | Hardware selector |
| `--framework FW` | When using `--model`; default `vllm` |
| `--tp 4,8` | Filter combos to these TPs |
| `--conc 4,8,16` | Filter combos to these concurrencies |
| `--isl-osl 8192/1024` | Filter to this ISL/OSL pair (repeatable) |
| `--env-overlay NAME` | `env_overlays/<NAME>.env`; default `widegraph_default`. Use `baseline` for none |
| `--extra-env K=V` | Per-run env override (after overlay). Repeatable |
| `--out-dir PATH` | Where artifacts and CSV land (required) |
| `--port N` | vLLM port (default 8891) |
| `--max-num-seqs N` | (default 256) |
| `--random-range-ratio F` | (default 0.8) |
| `--dry-run` | Print combos and exit |

## Outputs

In `--out-dir`:

```
<model>_<prec>_<hw>_isl<I>_osl<O>_tp<T>_conc<C>.json           # vllm-bench native (unchanged)
<model>_<prec>_<hw>_isl<I>_osl<O>_tp<T>_conc<C>.server.log     # vllm server output
<model>_<prec>_<hw>_isl<I>_osl<O>_tp<T>_conc<C>.gpu_metrics.csv
<model>_<prec>_<hw>_isl<I>_osl<O>_tp<T>_conc<C>.stdout         # combined launcher output
sa_bench.csv          # IX-schema CSV: per-GPU throughput, latency in seconds
manifest.json         # config + vllm version + env + git sha + per-combo rc/elapsed
```

## How env precedence works

For each combo the harness builds the environment as:

```
process env  +  env_overlays/<overlay>.env  +  --extra-env K=V…
       ↓
   bash benchmarks/single_node/<model>_<prec>_<hw>.sh
       ↓
   (launcher's `export VAR=...` lines win for the few they touch)
```

In practice: the launcher hard-sets a small set of recipe-critical vars
(`VLLM_ROCM_USE_AITER`, `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION`, model-specific
toggles). The overlay handles the larger set the launchers leave alone. To
override what a launcher hard-sets you'd need to fork the launcher into a
small overrides dir (not implemented in MVP).

## The `hf` shim

`benchmarks/harness/bin/hf` is a thin shim prepended to `PATH`. It no-ops
`hf download <local-dir>` (the per-combo download tax in every launcher) and
falls through to the real `hf` for everything else. Saves ~minutes per combo
on the benchmark machine where models live in `/data`.

## Compatibility with existing tooling

- The per-combo `.json` files are unchanged from what `vllm bench` produces,
  so the existing `comparison_plots/plot_*_compare.py` scripts read them as-is.
- `sa_bench.csv` is a flat per-row CSV using the same column names as SA's
  `benchmark_results.json` flat fields (one rename: `config_id` → `config_label`,
  since SA assigns the integer ID server-side and we don't have one).

## `--all --smoke` mode (PR2)

Iterate every config in amd-master.yaml matching `--gpu` and `--framework`,
run only the cheapest combo per config (smallest TP × smallest ISL+OSL ×
smallest CONC from the search-space), and emit a PASS/FAIL/DRIFT report
versus the IX dashboard:

```bash
python3 benchmarks/harness/sa_bench.py \
    --gpu mi355x --all --smoke \
    --ix-dump-dir /home/inferencex_dump/inferencex-dump-2026-04-27 \
    --out-dir /workspace/smoke_$(date +%Y%m%d-%H%M%S)

# Skip configs whose key matches a regex (e.g., known-broken or WIP):
sa_bench.py --gpu mi355x --all --smoke --exclude 'minimax|dsv4-fp8' --out-dir ...

# Resumable: re-run after a partial sweep, skip combos with JSON already on disk:
sa_bench.py --gpu mi355x --all --smoke --skip-existing --out-dir ...
```

### Verdicts

| Verdict | Meaning |
|---|---|
| **PASS**   | Combo ran, JSON produced, throughput within `--drift-pct` of the IX dashboard for the same combo |
| **DRIFT**  | Combo ran, JSON produced, throughput >`--drift-pct` off IX (default 20%) |
| **FAIL**   | Launcher non-zero exit OR no JSON produced |
| **NO_REF** | Combo ran fine, but no matching IX dashboard row exists to compare against |

### Outputs (root-of `--out-dir`)

```
<config-key-1>/
  ├── *.{json,server.log,gpu_metrics.csv,stdout}    (per combo)
  ├── sa_bench.csv                                  (this config's IX-schema rows)
  └── manifest.json
<config-key-2>/
  └── …
smoke_report.md       # human report: per-config table with verdicts and deltas
smoke_report.csv      # machine: same data, one row per smoke combo
```

### IX dump path inside containers

The dump auto-detect looks at `/tmp/inferencex_dump/inferencex-dump-*`. If your
container only mounts `/home/xiaohugu`, copy the dump under there and pass
`--ix-dump-dir` explicitly:

```bash
cp -r /tmp/inferencex_dump/inferencex-dump-2026-04-27 /home/xiaohugu/inferencex_dump/
sa_bench.py … --ix-dump-dir /home/inferencex_dump/inferencex-dump-2026-04-27
```

## Architectural rule: comparison-vs-NV is **always** through the IX dump

This is the load-bearing design choice for the harness. Internalize it:

- **Comparison** (our MI355x vs B200/B300/H200) is done by overlaying our
  `sa_bench.csv` (or per-combo JSONs) with rows pulled from the **IX dashboard
  dump**. No NV hardware required on our end. This works today via
  `find_ix_match` (DRIFT check) and `comparison_plots/plot_*_compare.py`
  (overlay plots).
- **Re-running NV** ourselves is *not* a comparison workflow. It's only useful
  for **profiling NV hardware on a vLLM patch we want to test** — i.e., when we
  don't trust the dashboard's NV numbers because we changed something the
  dashboard hasn't picked up yet.

So when someone asks "should we compare our run to B200?" the answer is "we
already do — via the IX dump." When someone asks "should we run on B200?" the
answer is "only if you have a profiling-on-NV use case AND access to NV
hardware."

## Roadmap

- **PR3** — `--plot` flag wires `comparison_plots/plot_*_compare.py` against
  the harness-produced `sa_bench.csv` so a single `sa_bench.py` invocation
  produces grid + Pareto PNGs end-to-end. Pure plumbing; no new analysis.

### Future / out of scope today

- **`--gpu b200` end-to-end** — only worth doing when there's a concrete
  profiling-on-NV need AND access to NV hardware. The mechanical bits work
  already (launchers exist, `--gpu b200` is accepted), but real validation
  would need: `nvidia-smi`-based cleanup vs the current `pkill` approach, and
  TRT-backend JSON field name mapping in `json_to_ix_row` (TRT emits
  different metric keys than vLLM). Defer until needed; do not preemptively
  build for this.
