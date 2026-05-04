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
        seq_len_configs=raw["seq-len-configs"],
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


# ---------- argparse + main ----------

def parse_isl_osl(s: str) -> tuple[int, int]:
    a, b = s.split("/")
    return (int(a), int(b))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sel = ap.add_argument_group("config selection")
    sel.add_argument("--config", help="Full amd-master.yaml key, e.g. kimik2.5-fp4-mi355x-vllm")
    sel.add_argument("--model", help="Short model spec like 'kimik2.5-fp4'; combine with --gpu/--framework")
    sel.add_argument("--gpu", choices=["mi355x", "mi300x", "mi325x", "b200", "b300", "h200"],
                     help="Hardware key when using --model")
    sel.add_argument("--framework", default="vllm",
                     help="Framework when using --model (default: vllm)")

    flt = ap.add_argument_group("combo filters (optional)")
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
                     help="Where per-combo artifacts and CSV land")
    run.add_argument("--port", type=int, default=8891)
    run.add_argument("--max-num-seqs", type=int, default=256)
    run.add_argument("--random-range-ratio", type=float, default=0.8)
    run.add_argument("--dry-run", action="store_true", help="Print combos and exit")

    args = ap.parse_args()

    if not args.config and not args.model:
        ap.error("supply --config or (--model + --gpu)")
    if args.model:
        if not args.gpu:
            ap.error("--model requires --gpu")
        args.config = f"{args.model}-{args.gpu}-{args.framework}"

    cfg = load_config(args.config)
    launcher = derive_launcher_path(cfg)

    combos = all_combos(cfg)
    tps = [int(x) for x in args.tp.split(",")] if args.tp else None
    concs = [int(x) for x in args.conc.split(",")] if args.conc else None
    isl_osls = [parse_isl_osl(s) for s in args.isl_osl] if args.isl_osl else None
    combos = filter_combos(combos, tps=tps, concs=concs, isl_osls=isl_osls)
    if not combos:
        raise SystemExit("no combos after filtering")

    overlay = load_overlay(args.env_overlay if args.env_overlay != "baseline" else "")
    extra_env = parse_extra_env(args.extra_env)
    base_env = {**overlay, **extra_env}

    print(f"config:    {cfg.key}")
    print(f"model:     {cfg.model}")
    print(f"image:     {cfg.image}")
    print(f"launcher:  {launcher.relative_to(REPO_ROOT)}")
    print(f"overlay:   {args.env_overlay} ({len(overlay)} vars)")
    print(f"extra-env: {extra_env or '(none)'}")
    print(f"combos:    {len(combos)} after filter")
    for isl, osl, tp, conc in combos:
        print(f"  isl={isl} osl={osl} tp={tp} conc={conc}")

    if args.dry_run:
        return

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    vllm_version = detect_vllm_version()
    print(f"\nout_dir:   {out_dir}")
    print(f"vllm:      {vllm_version}\n")

    # Stop on Ctrl-C cleanly.
    signal.signal(signal.SIGINT, lambda *_: (cleanup_vllm(), sys.exit(130)))

    results = []
    for i, (isl, osl, tp, conc) in enumerate(combos, 1):
        print(f"\n[{i}/{len(combos)}]", end=" ")
        try:
            r = run_one_combo(cfg, launcher, out_dir, isl, osl, tp, conc,
                              base_env, args.port, args.random_range_ratio, args.max_num_seqs)
        except KeyboardInterrupt:
            cleanup_vllm()
            raise
        except Exception as e:
            print(f"WARN: combo crashed: {e}", file=sys.stderr)
            r = dict(isl=isl, osl=osl, tp=tp, conc=conc, rc=-1, elapsed_s=0,
                     result_filename="", json_present=False)
        results.append(r)

    write_manifest(out_dir, cfg, results, vllm_version,
                   args.env_overlay, extra_env, launcher, args)
    write_csv(out_dir, results, cfg, vllm_version, args.env_overlay, extra_env)

    n_pass = sum(1 for r in results if r["json_present"] and r["rc"] == 0)
    print(f"\nSummary: {n_pass}/{len(results)} combos produced JSON. Out: {out_dir}")


if __name__ == "__main__":
    main()
