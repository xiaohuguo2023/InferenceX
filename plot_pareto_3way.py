"""Pareto plotting helpers for the Kimi-K3 agentic comparison.

Local replacement for the module that used to live in ~/work/sweep_gptoss_output,
which is no longer on this machine. Exposes the two names plot_kimik3_pareto.py
imports: `plot_combined` and `_pareto_frontier`.

Series format matches the callers: {label: {(_, _, ngpu, conc): {"total", "tpot"}}}.
`conc` may carry a +100000 offset to keep offload on/off points distinct, so it is
shown modulo that offset.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow the Agg backend selection)
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402

CONC_KEY_OFFSET = 100000

# InferenceX dashboard dark-theme tokens and chart constants (Aug 2026).
BACKGROUND = "#131416"
FOREGROUND = "#eaebec"
MUTED = "#27272a"
MUTED_FOREGROUND = "#b4b9bc"
BORDER_ALT = "#222426"
POINT_RADIUS_PX = 3.5
POINT_AREA = 3.14159 * POINT_RADIUS_PX**2
OFFLOAD_HALO_AREA = 3.14159 * (POINT_RADIUS_PX + 4) ** 2


def _pareto_frontier(points):
    """Points that maximize both interactivity (x) and throughput (y)."""
    front = []
    best_y = float("-inf")
    for x, y, c in sorted(points, key=lambda p: (-p[0], -p[1])):
        if y > best_y:
            front.append((x, y, c))
            best_y = y
    return front


def _series_points(data):
    pts = []
    for key, d in data.items():
        tpot = d.get("tpot", 0)
        if tpot <= 0:
            continue
        pts.append((1000.0 / tpot, d["total"] / key[2], key[3]))
    return pts


def _monotone_tangents(xs, ys):
    """Fritsch-Carlson tangents, matching d3 curveMonotoneX semantics closely."""
    if len(xs) < 2:
        return []
    dx = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    slopes = [(ys[i + 1] - ys[i]) / dx[i] for i in range(len(dx))]
    tangents = [slopes[0]]
    for i in range(1, len(xs) - 1):
        left, right = slopes[i - 1], slopes[i]
        tangents.append(0.0 if left * right <= 0 else 2 * left * right / (left + right))
    tangents.append(slopes[-1])
    return tangents


def _draw_monotone_frontier(ax, points, color, linestyle):
    """Draw the dashboard's curveMonotoneX-style 2px Pareto roofline."""
    ordered = sorted(points, key=lambda p: p[0])
    xs = [p[0] for p in ordered]
    ys = [p[1] for p in ordered]
    tangents = _monotone_tangents(xs, ys)
    vertices = [(xs[0], ys[0])]
    codes = [MplPath.MOVETO]
    for i in range(len(xs) - 1):
        width = xs[i + 1] - xs[i]
        vertices.extend(
            [
                (xs[i] + width / 3, ys[i] + tangents[i] * width / 3),
                (xs[i + 1] - width / 3, ys[i + 1] - tangents[i + 1] * width / 3),
                (xs[i + 1], ys[i + 1]),
            ]
        )
        codes.extend([MplPath.CURVE4] * 3)
    ax.add_patch(
        PathPatch(
            MplPath(vertices, codes),
            fill=False,
            color=color,
            linewidth=2,
            linestyle=linestyle,
            capstyle="round",
            joinstyle="round",
            zorder=2,
        )
    )


def _format_large_number(value):
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M".replace(".0M", "M")
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}k".replace(".0k", "k")
    return f"{value:g}"


def plot_combined(series, out_dir, title_prefix, series_styles=None, shapes=None):
    """Render a static counterpart of the InferenceX dashboard Pareto chart."""
    styles = series_styles or {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shape_label = ""
    if shapes:
        first = shapes[0]
        shape_label = str(first[-1]) if isinstance(first, (tuple, list)) else str(first)
    title = f"{title_prefix} — {shape_label}" if shape_label else str(title_prefix)

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DM Sans", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.facecolor": BACKGROUND,
            "figure.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "text.color": FOREGROUND,
            "axes.labelcolor": FOREGROUND,
            "axes.edgecolor": BORDER_ALT,
            "xtick.color": MUTED_FOREGROUND,
            "ytick.color": MUTED_FOREGROUND,
        }
    ):
        fig, ax = plt.subplots(figsize=(11.5, 7.25))
        fig.subplots_adjust(left=0.105, right=0.975, top=0.84, bottom=0.13)

        ax.set_axisbelow(True)
        ax.grid(True, color=BORDER_ALT, linewidth=0.8, alpha=0.75)
        for spine in ax.spines.values():
            spine.set_color(BORDER_ALT)
            spine.set_linewidth(0.8)

        for label, data in series.items():
            pts = _series_points(data)
            if not pts:
                continue
            st = styles.get(label, {})
            color = st.get("color", MUTED_FOREGROUND)
            marker = st.get("marker", "o")

            # The live dashboard identifies KV offload with a 1.5px dashed halo.
            offloaded = [p for p in pts if p[2] >= CONC_KEY_OFFSET]
            if offloaded:
                ax.scatter(
                    [p[0] for p in offloaded],
                    [p[1] for p in offloaded],
                    s=OFFLOAD_HALO_AREA,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.5,
                    linestyle=(0, (3, 2)),
                    zorder=2.5,
                )

            ax.scatter(
                [p[0] for p in pts],
                [p[1] for p in pts],
                label=label,
                s=POINT_AREA,
                color=color,
                marker=marker,
                linewidths=0,
                zorder=3,
            )
            for x, y, c in pts:
                ax.annotate(
                    f"C={c % CONC_KEY_OFFSET}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight=700,
                    color=FOREGROUND,
                    zorder=4,
                )
            front = _pareto_frontier(pts)
            if len(front) >= 2:
                _draw_monotone_frontier(
                    ax,
                    front,
                    color,
                    st.get("linestyle", "-"),
                )

        ax.set_xlabel("Interactivity (tok/s/user)", labelpad=9, fontweight=700)
        ax.set_ylabel("Token Throughput per Chip (tok/s/chip)", labelpad=10, fontweight=700)
        ax.xaxis.set_major_formatter(lambda x, _pos: _format_large_number(x))
        ax.yaxis.set_major_formatter(lambda y, _pos: _format_large_number(y))
        ax.tick_params(axis="both", which="major", length=0, pad=6, labelsize=9)

        fig.suptitle(
            "Inference Performance",
            x=0.105,
            y=0.955,
            ha="left",
            va="top",
            fontsize=18,
            fontweight=700,
            color=FOREGROUND,
        )
        ax.set_title(title, loc="left", pad=14, fontsize=10, color=MUTED_FOREGROUND)
        legend = ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=3,
            frameon=False,
            fontsize=9,
            handlelength=2.2,
            columnspacing=1.6,
        )
        for text in legend.get_texts():
            text.set_color(FOREGROUND)

        # Dashboard terminology and the source behind this static export.
        fig.text(
            0.975,
            0.025,
            "Source: InferenceX /api/v1/benchmarks?model=Kimi-K3 · "
            "Pareto-optimal concurrency points",
            ha="right",
            va="bottom",
            fontsize=7.5,
            color=MUTED_FOREGROUND,
        )

        combined = out_dir / "pareto_kimik3_mi355x_vs_b300_b200.png"
        fig.savefig(combined, dpi=180, bbox_inches="tight")
        named = out_dir / f"pareto_{title.replace('/', '-').replace(' ', '_')}.png"
        fig.savefig(named, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return combined
