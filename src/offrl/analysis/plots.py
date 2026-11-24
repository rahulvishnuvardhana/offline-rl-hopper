"""Report figures from the pre-registered analysis (report/phase7/prereg_analysis.json).

    python -m offrl.analysis.plots [--analysis <json>] [--out-dir report/figures]

Panel (a): final normalized score vs. the expert fraction p for BC and IQL (tau = 0.7): per-seed
scores as dots, the seed mean as a line, the 95% t-interval as a whisker and the 95% percentile
bootstrap as a thicker bar (preregistration Sections 2 and 3). Panel (b): the tau ablation at
p = 0.05 and p = 0.5. Grid levels are drawn evenly spaced (they are uneven in p). Colors are the
validated two-slot categorical pair; series are also told apart by marker and direct labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2, validated (light surface)
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
P_GRID = ["0", "0.05", "0.2", "0.5", "1"]
NEAR_EXPERT = 90.0


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "pdf.fonttype": 42,
        }
    )


def series(ax, x, cells, color, marker, label, offset, line=True):
    """Dots per seed, mean line, t whisker, bootstrap bar."""
    rng = np.random.default_rng(0)  # jitter only, for visibility
    xs = np.asarray(x, dtype=float) + offset
    means = [c["mean"] for c in cells]
    for xi, c in zip(xs, cells, strict=True):
        jit = rng.uniform(-0.05, 0.05, len(c["per_seed"]))
        ax.scatter(xi + jit, c["per_seed"], s=7, color=color, alpha=0.35, linewidths=0, zorder=2)
        ax.plot([xi, xi], c["t_95"], color=color, lw=0.8, zorder=3)
        ax.plot([xi, xi], c["bootstrap_95"], color=color, lw=2.6, solid_capstyle="butt", zorder=3)
    lw = 1.4 if line else 0
    ax.plot(xs, means, color=color, lw=lw, marker=marker, ms=4.5, mec="white", mew=0.8, label=label, zorder=4)
    return xs, means


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", type=Path, default=Path("report/phase7/prereg_analysis.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("report/figures"))
    args = ap.parse_args()
    cells = json.loads(args.analysis.read_text(encoding="utf-8"))["cells"]
    style()
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(5.5, 2.25), gridspec_kw={"width_ratios": [1.75, 1]}, sharey=True, layout="constrained"
    )

    x = np.arange(len(P_GRID))
    iql_label = r"IQL ($\tau$ = 0.7)"
    bcx, bcm = series(ax, x, [cells[f"bc p={p}"] for p in P_GRID], BLUE, "o", "BC", -0.09)
    iqx, iqm = series(ax, x, [cells[f"iql tau=0.7 p={p}"] for p in P_GRID], ORANGE, "s", iql_label, 0.09)
    ax.axhline(NEAR_EXPERT, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(1.5, NEAR_EXPERT + 1.5, "near-expert (90)", color=MUTED, fontsize=6.5, ha="center", va="bottom")
    ax.text(bcx[2], bcm[2] - 5, "BC", color=INK, fontsize=7.5, ha="center", va="top")
    ax.text(0.62, 60, iql_label, color=INK, fontsize=7.5, ha="left", va="center")  # beside the drop
    ax.set_xticks(x, P_GRID)
    ax.set_xlabel(r"expert fraction $p$ (grid levels, evenly spaced)")
    ax.set_ylabel("final normalized score")
    ax.set_ylim(-12, 115)  # the lowest t-whisker ends at -9.8
    ax.set_title("(a) BC vs. IQL across $p$", fontsize=8, color=INK, loc="left")
    ax.legend(loc="lower right", fontsize=7, handlelength=1.6)

    taus = [("0.5", "#f29a72"), ("0.7", ORANGE), ("0.9", "#a8441a")]  # one-hue ordinal ramp, validated
    ps = ["0.05", "0.5"]
    w = 0.22
    for i, (tau, col) in enumerate(taus):
        xs = np.arange(len(ps)) + (i - 1) * w
        cs = [cells[f"iql tau={tau} p={p}"] for p in ps]
        series(bx, xs, cs, col, "D", rf"$\tau$ = {tau}", 0.0, line=False)
    bx.set_xticks(np.arange(len(ps)), ps)
    bx.set_xlabel(r"expert fraction $p$")
    bx.set_title(r"(b) IQL $\tau$ ablation", fontsize=8, color=INK, loc="left")
    bx.axhline(NEAR_EXPERT, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    bx.legend(loc="upper left", fontsize=7, handlelength=1.2)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.out_dir / f"score_vs_p.{ext}", dpi=300)
    print(f"wrote {args.out_dir / 'score_vs_p.pdf'} and .png")


if __name__ == "__main__":
    main()
