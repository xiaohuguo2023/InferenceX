#!/usr/bin/env python3
"""sa_bench — InferenceX-style sweep harness.

Reuses benchmarks/single_node/<model>_<prec>_<gpu>[_<framework>].sh as the
recipe source-of-truth. Reads the matrix (TP × ISL/OSL × CONC) from
.github/configs/amd-master.yaml. Emits per-combo JSON (vllm-bench native)
plus an aggregated IX-schema CSV that the SA-style Pareto plot code can
consume directly.

Usage:
  sa_bench.py --gpu mi355x --model kimik2.5-fp4 --out-dir /workspace/sweep_$(date +%s)
  sa_bench.py --config kimik2.5-fp4-mi355x-vllm --tp 8 --conc 4 --isl-osl 8192/1024 \
              --extra-env VLLM_ROCM_USE_AITER_MLA=0
  sa_bench.py --config gptoss-fp4-mi355x-vllm --env-overlay widegraph_default

Conventions:
  - Run from inside a container with /workspace writable, /home mapped to the
    user's home, and the model present at the path listed in amd-master.yaml.
  - The harness prepends `bin/` (with the `hf` shim) to PATH to no-op
    `hf download` for already-local model paths.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml  # PyYAML; in the vllm-dev image already

HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parent.parent
SINGLE_NODE_DIR = REPO_ROOT / "benchmarks" / "single_node"
AMD_MASTER_YAML = REPO_ROOT / ".github" / "configs" / "amd-master.yaml"
OVERLAY_DIR = HARNESS_DIR / "env_overlays"
WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))


# ---------- amd-master.yaml lookup ----------

@dataclass
class IxConfig:
    key: str           # e.g. "kimik2.5-fp4-mi355x-vllm"
    model: str         # path or HF repo id, from yaml
    image: str
    runner: str        # mi355x / mi300x / b200 / ...
    framework: str     # vllm / sglang / atom / trt
    precision: str
    model_prefix: str  # short model name used in launcher filename (e.g. "kimik2.5")
    seq_len_configs: list[dict]
    raw: dict = field(repr=False)


def _seq_len_configs(raw: dict) -> list[dict]:
    # New schema: scenarios.fixed-seq-len. Old schema: top-level seq-len-configs.
    if "seq-len-configs" in raw:
        return raw["seq-len-configs"]
    return raw.get("scenarios", {}).get("fixed-seq-len", []) or []


def load_config(key: str) -> IxConfig:
    with open(AMD_MASTER_YAML) as f:
        all_cfgs = yaml.safe_load(f)
    if key not in all_cfgs:
        nearby = [k for k in all_cfgs if key in k or any(p in k for p in key.split("-"))][:8]
        raise SystemExit(f"config {key!r} not in {AMD_MASTER_YAML}; did you mean: {nearby}")
    raw = all_cfgs[key]
    return IxConfig(
        key=key,
        model=raw["model"],
        image=raw["image"],
        runner=raw["runner"],
        framework=raw["framework"],
        precision=raw["precision"],
        model_prefix=raw["model-prefix"],
        seq_len_configs=_seq_len_configs(raw),
        raw=raw,
    )


def derive_launcher_path(cfg: IxConfig) -> Path:
    """Map an IxConfig to its benchmarks/single_node/ launcher.

    Heuristics, matching observed file naming:
      - Default: <model_prefix>_<precision>_<runner>.sh                     (vllm)
      - With framework suffix: <model_prefix>_<precision>_<runner>_<framework>.sh
        for non-default frameworks (atom, sglang, trt).
    """
    candidates = []
    if cfg.framework == "vllm":
        candidates.append(SINGLE_NODE_DIR / f"{cfg.model_prefix}_{cfg.precision}_{cfg.runner}.sh")
        candidates.append(SINGLE_NODE_DIR / f"{cfg.model_prefix}_{cfg.precision}_{cfg.runner}_vllm.sh")
    candidates.append(
        SINGLE_NODE_DIR / f"{cfg.model_prefix}_{cfg.precision}_{cfg.runner}_{cfg.framework}.sh"
    )
    for p in candidates:
        if p.exists():
            return p
    raise SystemExit(f"no launcher found for {cfg.key}; tried: {[str(c) for c in candidates]}")


# ---------- combo expansion ----------

def expand_concs(conc_start: int, conc_end: int) -> list[int]:
    """Match dashboard expansion: powers-of-2 from start to end (inclusive)."""
    out = []
    c = conc_start
    while c <= conc_end:
        out.append(c)
        c *= 2
    if conc_start == conc_end and not out:
        out.append(conc_start)
    return out


def all_combos(cfg: IxConfig) -> list[tuple[int, int, int, int]]:
    """Expand seq-len-configs × search-space into (isl, osl, tp, conc) tuples."""
    combos = []
    for sl in cfg.seq_len_configs:
        isl, osl = sl["isl"], sl["osl"]
        for ss in sl["search-space"]:
            tp = ss["tp"]
            for conc in expand_concs(ss["conc-start"], ss["conc-end"]):
                combos.append((isl, osl, tp, conc))
    # de-dupe while preserving order
    seen, uniq = set(), []
    for c in combos:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def filter_combos(combos, *, tps=None, concs=None, isl_osls=None):
    out = []
    for isl, osl, tp, conc in combos:
        if tps and tp not in tps:
            continue
        if concs and conc not in concs:
            continue
        if isl_osls and (isl, osl) not in isl_osls:
            continue
        out.append((isl, osl, tp, conc))
    return out


# ---------- env overlay loading ----------

def load_overlay(name: str) -> dict[str, str]:
    if not name:
        return {}
    path = OVERLAY_DIR / f"{name}.env"
    if not path.exists():
        raise SystemExit(f"unknown overlay {name!r}; available: "
                         f"{[p.stem for p in OVERLAY_DIR.glob('*.env')]}")
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


def parse_extra_env(items: list[str]) -> dict[str, str]:
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        if not k or "=" not in it:
            raise SystemExit(f"--extra-env must be KEY=VALUE, got: {it!r}")
        out[k] = v
    return out


# ---------- per-combo runner ----------

def cleanup_vllm():
    for pat in ("vllm serve", "VLLM::", "gpu_monitor"):
        subprocess.run(["pkill", "-KILL", "-f", pat], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)


def detect_vllm_version() -> str:
    try:
        out = subprocess.check_output(
            ["python3", "-c", "import vllm; print(vllm.__version__)"],
            stderr=subprocess.DEVNULL, text=True, timeout=15)
        return out.strip()
    except Exception:
        return "unknown"


def run_one_combo(cfg: IxConfig, launcher: Path, out_dir: Path,
                  isl: int, osl: int, tp: int, conc: int,
                  base_env: dict[str, str], port: int,
                  random_range_ratio: float, max_num_seqs: int) -> dict:
    """Invoke the per-model launcher for one (isl, osl, tp, conc) combo.

    Returns a result dict; raises subprocess.CalledProcessError on launcher
    non-zero exit (combo failure — caller logs and continues).
    """
    result_filename = (
        f"{cfg.model_prefix}_{cfg.precision}_{cfg.runner}_isl{isl}_osl{osl}_tp{tp}_conc{conc}"
    )
    max_model_len = isl + osl + 256

    env = os.environ.copy()
    # Prepend our hf shim so `hf download <local-path>` no-ops.
    env["PATH"] = f"{HARNESS_DIR / 'bin'}:{env.get('PATH', '')}"
    # Required by the launcher's check_env_vars.
    env.update({
        "MODEL": cfg.model,
        "TP": str(tp),
        "CONC": str(conc),
        "ISL": str(isl),
        "OSL": str(osl),
        "MAX_MODEL_LEN": str(max_model_len),
        "RANDOM_RANGE_RATIO": str(random_range_ratio),
        "RESULT_FILENAME": result_filename,
        "PORT": str(port),
        "MAX_NUM_SEQS": str(max_num_seqs),
        # Defaults benchmark_lib.sh needs.
        "EVAL_ONLY": env.get("EVAL_ONLY", "false"),
        "RUN_EVAL": env.get("RUN_EVAL", "false"),
    })
    # Apply env overlay, then any user --extra-env on top.
    env.update(base_env)

    cleanup_vllm()
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    server_log = WORKSPACE / "server.log"
    gpu_metrics = WORKSPACE / "gpu_metrics.csv"
    json_out = WORKSPACE / f"{result_filename}.json"
    for f in (server_log, gpu_metrics, json_out):
        f.unlink(missing_ok=True)

    combo_stdout = out_dir / f"{result_filename}.stdout"
    print(f"\n=== [{isl}/{osl} tp={tp} conc={conc}] launcher: {launcher.name} ===", flush=True)

    start = time.time()
    rc = subprocess.run(
        ["bash", str(launcher)],
        env=env, stdin=subprocess.DEVNULL,
        stdout=open(combo_stdout, "w"), stderr=subprocess.STDOUT,
    ).returncode
    elapsed = int(time.time() - start)
    print(f"=== exit={rc} elapsed={elapsed}s ===", flush=True)

    # Move artifacts into out_dir whether or not the run succeeded.
    moved = []
    for src, dst_name in [
        (json_out,    f"{result_filename}.json"),
        (server_log,  f"{result_filename}.server.log"),
        (gpu_metrics, f"{result_filename}.gpu_metrics.csv"),
    ]:
        if src.exists():
            shutil.move(str(src), str(out_dir / dst_name))
            moved.append(dst_name)

    cleanup_vllm()
    return {
        "isl": isl, "osl": osl, "tp": tp, "conc": conc,
        "rc": rc, "elapsed_s": elapsed,
        "result_filename": result_filename,
        "json_present": f"{result_filename}.json" in moved,
    }


# ---------- IX-schema CSV emission ----------

# Columns mirror SA's benchmark_results.json metrics, flattened.
# Latency in seconds (to match dump units), throughput per-GPU.
IX_COLUMNS = [
    "config_label", "date", "isl", "osl", "conc", "tp",
    "image", "vllm_version", "elapsed_s",
    "tput_per_gpu", "input_tput_per_gpu", "output_tput_per_gpu",
    "mean_ttft", "p50_ttft", "p90_ttft", "p99_ttft",
    "mean_tpot", "p50_tpot", "p90_tpot", "p99_tpot",
    "mean_itl",  "p50_itl",  "p90_itl",  "p99_itl",
    "mean_e2el", "p50_e2el", "p90_e2el", "p99_e2el",
    "total_token_throughput", "completed", "duration_s",
]


def json_to_ix_row(j: dict, *, config_label: str, image: str, vllm_version: str,
                   isl: int, osl: int, tp: int, elapsed_s: int) -> dict:
    """Convert a vllm-bench JSON to one IX-schema CSV row.

    SA dump uses seconds for latency + per-GPU throughput; we convert from our
    native ms + total throughput. ISL/OSL come from the harness (the JSON
    doesn't store the CLI-arg targets, only post-hoc token counts).
    """
    def ms_to_s(k):
        v = j.get(k)
        return None if v is None else v / 1000.0

    total_tput = j.get("total_token_throughput") or 0.0
    out_tput = j.get("output_throughput") or 0.0
    in_tput = (total_tput - out_tput) if total_tput else 0.0
    return {
        "config_label": config_label,
        "date": j.get("date", ""),
        "isl": isl,
        "osl": osl,
        "conc": j.get("max_concurrency", ""),
        "tp": tp,
        "image": image,
        "vllm_version": vllm_version,
        "elapsed_s": elapsed_s,
        "tput_per_gpu":         total_tput / tp if tp else "",
        "input_tput_per_gpu":   in_tput / tp if tp else "",
        "output_tput_per_gpu":  out_tput / tp if tp else "",
        "mean_ttft": ms_to_s("mean_ttft_ms"),
        "p50_ttft":  ms_to_s("median_ttft_ms"),
        "p90_ttft":  ms_to_s("p90_ttft_ms"),
        "p99_ttft":  ms_to_s("p99_ttft_ms"),
        "mean_tpot": ms_to_s("mean_tpot_ms"),
        "p50_tpot":  ms_to_s("median_tpot_ms"),
        "p90_tpot":  ms_to_s("p90_tpot_ms"),
        "p99_tpot":  ms_to_s("p99_tpot_ms"),
        "mean_itl":  ms_to_s("mean_itl_ms"),
        "p50_itl":   ms_to_s("median_itl_ms"),
        "p90_itl":   ms_to_s("p90_itl_ms"),
        "p99_itl":   ms_to_s("p99_itl_ms"),
        "mean_e2el": ms_to_s("mean_e2el_ms"),
        "p50_e2el":  ms_to_s("median_e2el_ms"),
        "p90_e2el":  ms_to_s("p90_e2el_ms"),
        "p99_e2el":  ms_to_s("p99_e2el_ms"),
        "total_token_throughput": total_tput,
        "completed":  j.get("completed"),
        "duration_s": j.get("duration"),
    }


def write_csv(out_dir: Path, results: list[dict], cfg: IxConfig,
              vllm_version: str, env_overlay: str, extra_env: dict[str, str]):
    config_label = f"{cfg.key}--{env_overlay or 'baseline'}--vllm{vllm_version}"
    if extra_env:
        config_label += "--" + ",".join(f"{k}={v}" for k, v in sorted(extra_env.items()))

    rows = []
    for r in results:
        if not r["json_present"]:
            continue
        json_path = out_dir / f"{r['result_filename']}.json"
        try:
            j = json.loads(json_path.read_text())
        except Exception as e:
            print(f"WARN: failed to read {json_path}: {e}", file=sys.stderr)
            continue
        rows.append(json_to_ix_row(j, config_label=config_label, image=cfg.image,
                                   vllm_version=vllm_version,
                                   isl=r["isl"], osl=r["osl"], tp=r["tp"],
                                   elapsed_s=r["elapsed_s"]))

    csv_path = out_dir / "sa_bench.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=IX_COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"wrote {csv_path} ({len(rows)} rows)")
    return csv_path


def write_manifest(out_dir: Path, cfg: IxConfig, results: list[dict],
                   vllm_version: str, env_overlay: str, extra_env: dict[str, str],
                   launcher: Path, args: argparse.Namespace):
    manifest = {
        "config_key": cfg.key,
        "model": cfg.model,
        "image": cfg.image,
        "runner": cfg.runner,
        "framework": cfg.framework,
        "launcher": str(launcher.relative_to(REPO_ROOT)),
        "vllm_version": vllm_version,
        "env_overlay": env_overlay,
        "extra_env": extra_env,
        "cli_args": vars(args),
        "results": results,
        "n_combos": len(results),
        "n_passed": sum(1 for r in results if r["json_present"] and r["rc"] == 0),
        "git_sha": subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip() or None,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


# ---------- multi-config discovery (--all) ----------

def discover_configs(gpu: str, framework: str | None = None,
                     exclude_re: str | None = None) -> list[IxConfig]:
    """Return all amd-master.yaml configs matching --gpu and --framework.

    `exclude_re` is a Python regex applied to the config key; matching keys
    are dropped (useful to skip WIP / known-broken configs).
    """
    import re as _re
    with open(AMD_MASTER_YAML) as f:
        all_cfgs = yaml.safe_load(f)
    excl = _re.compile(exclude_re) if exclude_re else None
    out = []
    for key, raw in all_cfgs.items():
        if raw.get("runner") != gpu:
            continue
        if framework and raw.get("framework") != framework:
            continue
        if excl and excl.search(key):
            continue
        if raw.get("is_multinode") or raw.get("multinode"):
            continue
        sl = _seq_len_configs(raw)
        if not sl:
            # No fixed-seq-len scenario (e.g. agentic-coding only) — skip.
            continue
        out.append(IxConfig(
            key=key, model=raw["model"], image=raw["image"],
            runner=raw["runner"], framework=raw["framework"],
            precision=raw["precision"], model_prefix=raw["model-prefix"],
            seq_len_configs=sl, raw=raw,
        ))
    return sorted(out, key=lambda c: c.key)


def pick_smoke_combo(cfg: IxConfig) -> tuple[int, int, int, int]:
    """Pick the cheapest (isl, osl, tp, conc) combo for a config.

    Heuristic: smallest ISL+OSL, smallest TP within that, smallest CONC
    within that. Wall-time minimizer.
    """
    candidates = []
    for sl in cfg.seq_len_configs:
        isl, osl = sl["isl"], sl["osl"]
        for ss in sl["search-space"]:
            tp = ss["tp"]
            conc = ss["conc-start"]
            candidates.append((isl + osl, isl, osl, tp, conc))
    isl_osl_sum, isl, osl, tp, conc = min(candidates, key=lambda x: (x[0], x[3], x[4]))
    return (isl, osl, tp, conc)


# ---------- DRIFT detection vs IX dump ----------

def find_latest_ix_dump() -> Path | None:
    """Locate the most recent IX dump under /tmp/inferencex_dump/."""
    candidates = sorted(Path("/tmp/inferencex_dump").glob("inferencex-dump-*"))
    return candidates[-1] if candidates else None


def find_ix_match(dump_dir: Path, cfg: IxConfig,
                  isl: int, osl: int, tp: int, conc: int) -> dict | None:
    """Look up the IX dashboard row for the same (model, hw, fw, tp, isl, osl, conc).

    Returns the most recent row, or None if no match.
    """
    cfgs_path = dump_dir / "configs.json"
    br_path   = dump_dir / "benchmark_results.json"
    if not (cfgs_path.exists() and br_path.exists()):
        return None
    with open(cfgs_path) as f:
        ix_cfgs = json.load(f)
    matching_ids = {
        c["id"] for c in ix_cfgs
        if c["model"].startswith(cfg.model_prefix)
        and c["precision"] == cfg.precision
        and c["hardware"] == cfg.runner
        and c["framework"] == cfg.framework
        and c["decode_tp"] == tp
        and not c.get("disagg")
        and not c.get("is_multinode")
    }
    if not matching_ids:
        return None
    with open(br_path) as f:
        br = json.load(f)
    matches = [
        r for r in br
        if r.get("config_id") in matching_ids and not r.get("error")
        and r.get("isl") == isl and r.get("osl") == osl and r.get("conc") == conc
    ]
    if not matches:
        return None
    matches.sort(key=lambda r: r.get("date", ""), reverse=True)
    return matches[0]


def classify(combo_result: dict, json_data: dict | None,
             ix_match: dict | None, drift_pct: float = 20.0) -> dict:
    """Classify a smoke run as PASS / FAIL / DRIFT.

    PASS  — combo finished, JSON present, throughput within drift_pct of IX
    DRIFT — combo finished, JSON present, throughput >drift_pct off IX
    FAIL  — combo did not finish or no JSON
    NO_REF — passed but no IX row to compare against (still a useful smoke)
    """
    if not combo_result["json_present"] or combo_result["rc"] != 0:
        return {"verdict": "FAIL", "ours_tput": None, "ix_tput": None,
                "delta_pct": None, "reason": f"rc={combo_result['rc']}"}
    ours_tput = (json_data or {}).get("total_token_throughput")
    if ours_tput is None:
        return {"verdict": "FAIL", "ours_tput": None, "ix_tput": None,
                "delta_pct": None, "reason": "no total_token_throughput in JSON"}
    if ix_match is None:
        return {"verdict": "NO_REF", "ours_tput": ours_tput, "ix_tput": None,
                "delta_pct": None, "reason": "no matching IX dashboard row"}
    ix_tput = (ix_match.get("metrics", {}).get("tput_per_gpu") or 0.0) * combo_result["tp"]
    if not ix_tput:
        return {"verdict": "NO_REF", "ours_tput": ours_tput, "ix_tput": None,
                "delta_pct": None, "reason": "IX row has no tput_per_gpu"}
    delta = (ours_tput / ix_tput - 1) * 100
    verdict = "PASS" if abs(delta) <= drift_pct else "DRIFT"
    return {"verdict": verdict, "ours_tput": ours_tput, "ix_tput": ix_tput,
            "delta_pct": delta, "reason": ""}


def write_smoke_report(out_dir: Path, rows: list[dict], drift_pct: float,
                       ix_dump_dir: Path | None):
    """Write smoke_report.md (human) and smoke_report.csv (machine)."""
    md = out_dir / "smoke_report.md"
    cs = out_dir / "smoke_report.csv"

    counts = {"PASS": 0, "DRIFT": 0, "FAIL": 0, "NO_REF": 0}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1

    with open(md, "w") as f:
        f.write("# sa_bench --all --smoke report\n\n")
        f.write(f"- IX dump:   `{ix_dump_dir}`\n")
        f.write(f"- Drift threshold: ±{drift_pct:.0f}% on total_token_throughput\n")
        f.write(f"- Out dir:   `{out_dir}`\n")
        f.write(f"- Summary:   "
                f"PASS={counts['PASS']}  DRIFT={counts['DRIFT']}  "
                f"FAIL={counts['FAIL']}  NO_REF={counts['NO_REF']}  "
                f"(of {len(rows)} configs)\n\n")
        f.write("| config | tp | isl/osl | conc | verdict | ours tput | IX tput | Δ% | reason / elapsed |\n")
        f.write("|---|---:|---|---:|---|---:|---:|---:|---|\n")
        for r in rows:
            ours = f"{r['ours_tput']:.0f}" if r['ours_tput'] is not None else "—"
            ix   = f"{r['ix_tput']:.0f}"   if r['ix_tput']   is not None else "—"
            delt = f"{r['delta_pct']:+.1f}%" if r['delta_pct'] is not None else "—"
            extra = r["reason"] or f"{r['elapsed_s']}s"
            f.write(f"| `{r['config']}` | {r['tp']} | {r['isl']}/{r['osl']} | {r['conc']} | "
                    f"**{r['verdict']}** | {ours} | {ix} | {delt} | {extra} |\n")
    print(f"wrote {md}")

    with open(cs, "w", newline="") as f:
        cols = ["config", "tp", "isl", "osl", "conc", "verdict",
                "ours_tput", "ix_tput", "delta_pct", "elapsed_s", "rc", "reason"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"wrote {cs}")


# ---------- argparse + main ----------

def parse_isl_osl(s: str) -> tuple[int, int]:
    a, b = s.split("/")
    return (int(a), int(b))


def run_one_config(cfg: IxConfig, combos: list[tuple[int, int, int, int]],
                   out_dir: Path, base_env: dict[str, str], vllm_version: str,
                   args: argparse.Namespace) -> list[dict]:
    """Run all combos for a single config. Writes per-config csv + manifest."""
    launcher = derive_launcher_path(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- {cfg.key} ---")
    print(f"    image:    {cfg.image}")
    print(f"    launcher: {launcher.relative_to(REPO_ROOT)}")
    print(f"    combos:   {len(combos)} -> {out_dir}")

    results = []
    for i, (isl, osl, tp, conc) in enumerate(combos, 1):
        rf = f"{cfg.model_prefix}_{cfg.precision}_{cfg.runner}_isl{isl}_osl{osl}_tp{tp}_conc{conc}"
        if args.skip_existing and (out_dir / f"{rf}.json").exists():
            print(f"\n[{i}/{len(combos)}] SKIP — JSON exists at {rf}.json")
            results.append(dict(isl=isl, osl=osl, tp=tp, conc=conc, rc=0,
                                elapsed_s=0, result_filename=rf, json_present=True,
                                skipped=True))
            continue
        print(f"\n[{i}/{len(combos)}]", end=" ")
        try:
            r = run_one_combo(cfg, launcher, out_dir, isl, osl, tp, conc,
                              base_env, args.port, args.random_range_ratio,
                              args.max_num_seqs)
        except KeyboardInterrupt:
            cleanup_vllm()
            raise
        except Exception as e:
            print(f"WARN: combo crashed: {e}", file=sys.stderr)
            r = dict(isl=isl, osl=osl, tp=tp, conc=conc, rc=-1, elapsed_s=0,
                     result_filename="", json_present=False)
        results.append(r)

    extra_env = parse_extra_env(args.extra_env)
    write_manifest(out_dir, cfg, results, vllm_version,
                   args.env_overlay, extra_env, launcher, args)
    write_csv(out_dir, results, cfg, vllm_version, args.env_overlay, extra_env)

    n_pass = sum(1 for r in results if r["json_present"] and r["rc"] == 0)
    print(f"  -> {n_pass}/{len(results)} combos produced JSON")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sel = ap.add_argument_group("config selection")
    sel.add_argument("--config", help="Full amd-master.yaml key, e.g. kimik2.5-fp4-mi355x-vllm")
    sel.add_argument("--model", help="Short model spec like 'kimik2.5-fp4'; combine with --gpu/--framework")
    sel.add_argument("--gpu", choices=["mi355x", "mi300x", "mi325x", "b200", "b300", "h200"],
                     help="Hardware key when using --model or --all")
    sel.add_argument("--framework", default="vllm",
                     help="Framework when using --model or --all (default: vllm)")
    sel.add_argument("--all", action="store_true",
                     help="Iterate every config in amd-master.yaml matching --gpu and --framework")
    sel.add_argument("--exclude", default=None,
                     help="Regex; configs whose key matches are skipped (only with --all)")

    mode = ap.add_argument_group("run mode")
    mode.add_argument("--smoke", action="store_true",
                      help="Run only the cheapest combo per config (for --all sanity checks). "
                           "When used without --all, picks the smoke combo of the single config.")

    flt = ap.add_argument_group("combo filters (optional, ignored when --smoke)")
    flt.add_argument("--tp", help="Comma-separated TPs to keep (e.g. 4,8)")
    flt.add_argument("--conc", help="Comma-separated CONCs to keep (e.g. 4,8,16)")
    flt.add_argument("--isl-osl", action="append", default=[],
                     help="Slash-separated ISL/OSL to keep (e.g. 8192/1024). Repeatable.")

    rec = ap.add_argument_group("recipe overrides")
    rec.add_argument("--env-overlay", default="widegraph_default",
                     help=f"Overlay file in {OVERLAY_DIR.name}/ (no extension). "
                          f"Default: widegraph_default. Use 'baseline' for none.")
    rec.add_argument("--extra-env", action="append", default=[],
                     help="KEY=VALUE; repeatable. Applied AFTER overlay.")

    run = ap.add_argument_group("runtime")
    run.add_argument("--out-dir", required=True,
                     help="Where per-combo artifacts and CSV land. With --all, "
                          "each config gets a subdir <out-dir>/<config-key>/")
    run.add_argument("--port", type=int, default=8891)
    run.add_argument("--max-num-seqs", type=int, default=256)
    run.add_argument("--random-range-ratio", type=float, default=0.8)
    run.add_argument("--skip-existing", action="store_true",
                     help="Skip combos whose result JSON is already on disk (resumable)")
    run.add_argument("--dry-run", action="store_true", help="Print plan and exit")

    rep = ap.add_argument_group("smoke report")
    rep.add_argument("--ix-dump-dir", default=None,
                     help="Path to inferencex-dump-YYYY-MM-DD/ for DRIFT detection "
                          "and --plot. Auto-detected if omitted.")
    rep.add_argument("--drift-pct", type=float, default=20.0,
                     help="Throughput delta threshold for DRIFT verdict (default 20)")
    rep.add_argument("--plot", action="store_true",
                     help="After sweep, invoke comparison_plots/plot_<model_prefix>_compare.py "
                          "with the per-config sweep dir + IX_DUMP_DIR env. Writes grid + "
                          "Pareto PNGs to comparison_plots/. Skipped in --smoke (1 combo "
                          "doesn't make a curve).")

    args = ap.parse_args()

    # --- Validate selection mode ---
    if args.all:
        if not args.gpu:
            ap.error("--all requires --gpu")
        if args.config or args.model:
            ap.error("--all is exclusive with --config/--model")
    else:
        if not args.config and not args.model:
            ap.error("supply --config, (--model + --gpu), or --all + --gpu")
        if args.model:
            if not args.gpu:
                ap.error("--model requires --gpu")
            args.config = f"{args.model}-{args.gpu}-{args.framework}"

    # --- Build the config + combo list ---
    if args.all:
        configs = discover_configs(args.gpu, args.framework, args.exclude)
        if not configs:
            raise SystemExit(f"no configs match runner={args.gpu} framework={args.framework}")
    else:
        configs = [load_config(args.config)]

    tps     = [int(x) for x in args.tp.split(",")] if args.tp else None
    concs   = [int(x) for x in args.conc.split(",")] if args.conc else None
    isl_osls = [parse_isl_osl(s) for s in args.isl_osl] if args.isl_osl else None

    jobs: list[tuple[IxConfig, list[tuple[int, int, int, int]]]] = []
    for cfg in configs:
        if args.smoke:
            combos = [pick_smoke_combo(cfg)]
        else:
            combos = filter_combos(all_combos(cfg), tps=tps, concs=concs, isl_osls=isl_osls)
        if not combos:
            print(f"WARN: skip {cfg.key} — no combos after filter", file=sys.stderr)
            continue
        jobs.append((cfg, combos))
    if not jobs:
        raise SystemExit("no jobs to run")

    overlay = load_overlay(args.env_overlay if args.env_overlay != "baseline" else "")
    extra_env = parse_extra_env(args.extra_env)
    base_env = {**overlay, **extra_env}

    print(f"mode:      {'smoke' if args.smoke else 'full'}, "
          f"{'all' if args.all else 'single-config'}")
    print(f"gpu:       {args.gpu or '(from config)'}")
    print(f"framework: {args.framework}")
    print(f"overlay:   {args.env_overlay} ({len(overlay)} vars)")
    print(f"extra-env: {extra_env or '(none)'}")
    print(f"jobs:      {len(jobs)} configs, {sum(len(c) for _, c in jobs)} combos total")
    for cfg, combos in jobs:
        for isl, osl, tp, conc in combos:
            print(f"  {cfg.key:40s} isl={isl} osl={osl} tp={tp} conc={conc}")

    if args.dry_run:
        return

    root_out = Path(args.out_dir).resolve()
    root_out.mkdir(parents=True, exist_ok=True)
    vllm_version = detect_vllm_version()
    ix_dump_dir = Path(args.ix_dump_dir) if args.ix_dump_dir else find_latest_ix_dump()
    print(f"\nout_dir:   {root_out}")
    print(f"vllm:      {vllm_version}")
    print(f"ix dump:   {ix_dump_dir or '(none — DRIFT detection disabled)'}")

    signal.signal(signal.SIGINT, lambda *_: (cleanup_vllm(), sys.exit(130)))

    # --- Run all jobs, collect results ---
    all_runs: list[dict] = []
    for ji, (cfg, combos) in enumerate(jobs, 1):
        print(f"\n========== [{ji}/{len(jobs)}] {cfg.key} ==========")
        cfg_out = root_out / cfg.key if args.all else root_out
        results = run_one_config(cfg, combos, cfg_out, base_env, vllm_version, args)
        for r in results:
            all_runs.append({"cfg": cfg, "out_dir": cfg_out, **r})

    # --- Smoke-mode classification + report ---
    if args.smoke:
        report_rows = []
        for run in all_runs:
            cfg = run["cfg"]
            json_data = None
            if run["json_present"]:
                json_path = run["out_dir"] / f"{run['result_filename']}.json"
                try:
                    json_data = json.loads(json_path.read_text())
                except Exception:
                    pass
            ix_match = None
            if ix_dump_dir:
                ix_match = find_ix_match(ix_dump_dir, cfg, run["isl"], run["osl"],
                                         run["tp"], run["conc"])
            verdict = classify(run, json_data, ix_match, args.drift_pct)
            report_rows.append({
                "config": cfg.key, "tp": run["tp"], "isl": run["isl"], "osl": run["osl"],
                "conc": run["conc"], "rc": run["rc"], "elapsed_s": run["elapsed_s"],
                **verdict,
            })
        write_smoke_report(root_out, report_rows, args.drift_pct, ix_dump_dir)

    n_pass = sum(1 for r in all_runs if r["json_present"] and r["rc"] == 0)
    print(f"\nFinal: {n_pass}/{len(all_runs)} combos produced JSON. Out: {root_out}")

    # --- Plot pipeline (PR3) ---
    if args.plot:
        if args.smoke:
            print("\n--plot: skipped (smoke mode runs 1 combo per config — no curves to plot)")
        else:
            run_plot_pipeline(jobs, root_out, ix_dump_dir, args.all)


def run_plot_pipeline(jobs, root_out: Path, ix_dump_dir: Path | None, multi_config: bool):
    """Invoke comparison_plots/plot_<model_prefix>_compare.py for each config that has one."""
    plot_dir = REPO_ROOT / "comparison_plots"
    print("\n=== --plot ===")
    if ix_dump_dir is None:
        print("WARN: no IX dump found; plots will only show 'ours' series")

    for cfg, _ in jobs:
        # Plot script naming: strip dots from model_prefix (kimik2.5 -> kimik25,
        # minimaxm2.5 -> minimaxm25) to match the existing convention.
        slug = cfg.model_prefix.replace(".", "")
        plot_script = plot_dir / f"plot_{slug}_compare.py"
        sweep_dir = (root_out / cfg.key) if multi_config else root_out
        if not plot_script.exists():
            print(f"  [{cfg.key}] no plot script at {plot_script.relative_to(REPO_ROOT)} — skipping. "
                  f"To add: copy plot_kimik25_compare.py and edit constants for this model.")
            continue
        env = os.environ.copy()
        if ix_dump_dir:
            env["IX_DUMP_DIR"] = str(ix_dump_dir)
        print(f"  [{cfg.key}] running {plot_script.name} on {sweep_dir.relative_to(REPO_ROOT) if str(sweep_dir).startswith(str(REPO_ROOT)) else sweep_dir}")
        rc = subprocess.run(
            ["python3", str(plot_script), str(sweep_dir)],
            env=env, cwd=str(REPO_ROOT),
        ).returncode
        if rc != 0:
            print(f"  [{cfg.key}] plot script exited rc={rc}", file=sys.stderr)


if __name__ == "__main__":
    main()
