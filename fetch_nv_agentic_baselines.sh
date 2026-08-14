#!/usr/bin/env bash
# Fetch Kimi-K3 agentic B300/B200 baseline rows for compare_agentic_sweep.py / plot_kimik3_pareto.py.
# Writes ${NV_JSON:-/tmp/k3_b300.json}.
#
# Option A — InferenceX dashboard API (default, no gh needed):
#   ./fetch_nv_agentic_baselines.sh dashboard
#
# Option B — manual dashboard export:
#   Export Kimi-K3 agentic rows from https://inferencex.com/ → save as /tmp/k3_b300.json
#
# Option C — GHA artifact (needs gh auth):
#   ./fetch_nv_agentic_baselines.sh <RUN_ID>
#
# Option D — weekly release (needs gh auth):
#   ./fetch_nv_agentic_baselines.sh release

set -euo pipefail

OUT="${NV_JSON:-/tmp/k3_b300.json}"
REPO="${INFERENCEX_REPO:-SemiAnalysisAI/InferenceX}"
DASHBOARD_API="${INFERENCEX_DASHBOARD_API:-https://inferencex.semianalysis.com/api/v1/benchmarks?model=Kimi-K3}"

if [ -f "$OUT" ] && [ "${1:-dashboard}" = "" ]; then
  echo "Using existing $OUT ($(wc -l < "$OUT") lines)"
  exit 0
fi

extract_agentic() {
  python3 - "$1" "$OUT" <<'PY'
import json, sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
raw = json.loads(src.read_text())
rows = raw if isinstance(raw, list) else raw.get("cells", raw.get("results", []))
if not isinstance(rows, list):
    # agg_bmk.json flat list
    rows = list(raw) if isinstance(raw, dict) and "hw" in next(iter(raw.values()), {}) else []

def norm_row(r):
    hw = (r.get("hardware") or r.get("hw") or "").lower()
    if "b300" in hw: hw = "b300"
    elif "b200" in hw: hw = "b200"
    else: return None
    model = (r.get("model") or r.get("infmax_model_prefix") or "").lower()
    if "kimik3" not in model and "kimi" not in model:
        return None
    m = r.get("metrics") or r
    conc = r.get("conc") or r.get("concurrency")
    if conc is None:
        return None
    tput = m.get("tput_per_gpu")
    tpot = m.get("mean_tpot")
    if tput is None or tpot is None:
        return None
    off = r.get("offload_mode") or m.get("offload_mode") or r.get("kv_offloading") or "none"
    return {
        "hardware": hw,
        "conc": int(conc),
        "metrics": {
            "tput_per_gpu": float(tput),
            "mean_tpot": float(tpot),
            "mean_ttft": float(m.get("mean_ttft", 0)),
            "offload_mode": off,
        },
    }

out_rows = [x for x in (norm_row(r) for r in rows) if x]
if not out_rows:
    sys.stderr.write(f"No Kimi-K3 agentic rows in {src}\n")
    sys.exit(1)
out.write_text(json.dumps(out_rows, indent=2))
print(f"Wrote {len(out_rows)} rows -> {out}")
PY
}

fetch_dashboard() {
  tmp=$(mktemp)
  echo "Fetching Kimi-K3 from $DASHBOARD_API ..."
  # --compressed: the dashboard serves gzip; without it the body is raw gzip
  # bytes and json.load chokes on 0x8b (UnicodeDecodeError).
  curl -fsSL --compressed "$DASHBOARD_API" -o "$tmp"
  python3 - "$tmp" "$OUT" <<'PY'
import json, sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
raw = json.loads(src.read_text())

def norm_row(r):
    # Keep ALL NV hardware and BOTH offload arms; the plotter filters per-series.
    hw = (r.get("hardware") or "").lower()
    if hw not in ("b300", "b200", "gb200", "h200"):
        return None
    if r.get("benchmark_type") != "agentic_traces":
        return None
    if r.get("conc") is None:
        return None
    off = r.get("offload_mode") or r.get("metrics", {}).get("offload_mode") or "none"
    m = r["metrics"]
    return {
        "hardware": hw,
        "conc": int(r["conc"]),
        # Kept top-level so the plotter can split MTP from non-MTP arms.
        "spec_method": r.get("spec_method") or "none",
        "offload_mode": off,
        "isl": r.get("isl"), "osl": r.get("osl"), "date": r.get("date"),
        "metrics": {
            "tput_per_gpu": float(m.get("tput_per_gpu", 0)),
            # mean_itl is the directly-measured streamed inter-token gap and is the
            # dashboard's own interactivity basis: CONFIRMED mean_intvty == 1000/mean_itl
            # across all NV rows (2026-08-14). mean_tpot is (e2e-TTFT)/(n-1), which
            # DEGENERATES for long-ISL agentic traces (prefill dominates e2e≈TTFT) and
            # on MTP rows inflates 1/tpot ~5-6x — do NOT use it for interactivity.
            "mean_itl": float(m.get("mean_itl", m.get("mean_tpot", 0))),
            "mean_tpot": float(m.get("mean_tpot", 0)),
            "mean_ttft": float(m.get("mean_ttft", 0)),
            # dashboard's own published interactivity (== 1000/mean_itl); kept for fidelity.
            "mean_intvty": float(m.get("mean_intvty", 0)),
            "mean_full_response_intvty": float(m.get("mean_full_response_intvty", 0)),
            "offload_mode": off,
        },
    }

rows = [x for x in (norm_row(r) for r in raw) if x]
if not rows:
    sys.stderr.write("No Kimi-K3 NV agentic rows in dashboard response\n")
    sys.exit(1)
rows.sort(key=lambda r: (r["hardware"], r["spec_method"], r["offload_mode"], r["conc"]))
out.write_text(json.dumps(rows, indent=2))
# Durable committed copy so a reboot (which wipes /tmp) doesn't strand the plot.
durable = Path(__file__).resolve().parent / "docs" / "kimik3_pareto" / "k3_nv_agentic_dashboard.json" \
    if "__file__" in globals() else Path("docs/kimik3_pareto/k3_nv_agentic_dashboard.json")
try:
    durable.parent.mkdir(parents=True, exist_ok=True)
    durable.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {len(rows)} rows -> {out}  and  {durable}")
except Exception as e:
    print(f"Wrote {len(rows)} rows -> {out}  (durable copy failed: {e})")
for x in rows:
    m = x["metrics"]
    itl_ms = m["mean_itl"] * 1000   # interactivity basis (NOT mean_tpot; see norm_row)
    iv = 1000/itl_ms if itl_ms else 0
    print(
        f"  {x['hardware']:>5} c{x['conc']:>3} {x['spec_method']:>4} off={m['offload_mode']:>3}: "
        f"tput/gpu={m['tput_per_gpu']:7.0f} ITL={itl_ms:6.1f}ms "
        f"interact={iv:6.1f} (dash={m['mean_intvty']:6.1f})"
    )
PY
  rm -f "$tmp"
}

case "${1:-dashboard}" in
  dashboard)
    fetch_dashboard
    ;;
  release)
    tmp=$(mktemp)
    gh release download --repo "$REPO" --pattern 'agg_bmk*.json' -D "$(dirname "$tmp")" 2>/dev/null || true
    f=$(find "$(dirname "$tmp")" -name 'agg_bmk*.json' 2>/dev/null | head -1)
    [ -n "$f" ] || { echo "No release artifact; try: $0 dashboard" >&2; exit 1; }
    extract_agentic "$f"
    ;;
  "")
    echo "Usage: $0 [dashboard|release|<RUN_ID>]" >&2
    exit 1
    ;;
  *)
    RUN_ID="$1"
    tmpdir=$(mktemp -d)
    gh run download "$RUN_ID" --repo "$REPO" -n results_bmk -D "$tmpdir" 2>/dev/null \
      || gh run download "$RUN_ID" --repo "$REPO" -D "$tmpdir"
    f=$(find "$tmpdir" -name 'agg_bmk.json' | head -1)
    [ -n "$f" ] || { echo "No agg_bmk.json in run $RUN_ID" >&2; exit 1; }
    extract_agentic "$f"
    rm -rf "$tmpdir"
    ;;
esac
