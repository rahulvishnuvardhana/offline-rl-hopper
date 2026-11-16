"""Phase 2B figures. Each function takes computed numbers/arrays and returns a ``Figure``.

Uses ``matplotlib.figure.Figure`` directly (no pyplot, no global state, no GUI backend).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from offrl.analysis.eda import EXPERT, SIMPLE, SOURCES

# Categorical slots 1 and 2 of the validated palette; the same entity always gets the same color.
COLORS = {EXPERT: "#2a78d6", SIMPLE: "#eb6834"}
MUTED = "#9a9a94"
SIZE = (6.4, 4.0)
DPI = 120


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e3", linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig: Figure, path: str | Path) -> None:
    """PNG without a version stamp, so the bytes do not depend on the matplotlib version."""
    fig.savefig(path, dpi=DPI, metadata={"Software": None})


def hist_by_source(values: Mapping[str, np.ndarray], xlabel: str, title: str, bins: int = 50) -> Figure:
    fig = Figure(figsize=SIZE)
    ax = fig.add_subplot()
    edges = np.histogram_bin_edges(np.concatenate([values[s] for s in SOURCES]), bins=bins)
    for s in SOURCES:
        ax.hist(values[s], bins=edges, color=COLORS[s], alpha=0.6, label=f"{s} (n={len(values[s])})")
    ax.set(xlabel=xlabel, ylabel="episodes", title=title)
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


def returns_by_fall(
    returns: Mapping[str, np.ndarray], fell: Mapping[str, np.ndarray], bins: int = 50
) -> Figure:
    """One panel per dataset: no-fall episodes in the dataset color, fell episodes muted."""
    fig = Figure(figsize=(9.6, 4.0))
    axes = fig.subplots(1, 2, sharey=False)
    edges = np.histogram_bin_edges(np.concatenate([returns[s] for s in SOURCES]), bins=bins)
    for ax, s in zip(axes, SOURCES, strict=True):
        r, f = returns[s], fell[s]
        ax.hist(
            [r[~f], r[f]],
            bins=edges,
            stacked=True,
            color=[COLORS[s], MUTED],
            label=[f"no fall (n={(~f).sum()})", f"fell (n={f.sum()})"],
        )
        ax.set(xlabel="episode return", ylabel="episodes", title=f"{s}: fall rate {f.mean():.1%}")
        ax.legend(frameon=False)
        _style(ax)
    fig.tight_layout()
    return fig


def grouped_bars(
    values: Mapping[str, Sequence[float]], labels: Sequence[str], ylabel: str, title: str, log: bool = False
) -> Figure:
    fig = Figure(figsize=SIZE)
    ax = fig.add_subplot()
    x = np.arange(len(labels))
    width = 0.38
    for i, s in enumerate(SOURCES):
        ax.bar(x + (i - 0.5) * width, values[s], width * 0.95, color=COLORS[s], label=s)
    ax.set_xticks(x, labels)
    ax.set(ylabel=ylabel, title=title)
    if log:
        ax.set_yscale("log")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


def state_std(stats: Mapping[str, Mapping[str, Sequence[float]]]) -> Figure:
    labels = [f"obs[{i}]" for i in range(len(stats[EXPERT]["std"]))]
    fig = grouped_bars(
        {s: stats[s]["std"] for s in SOURCES},
        labels,
        "std (log scale)",
        "Per-dimension state std (descriptive only)",
        log=True,
    )
    fig.axes[0].tick_params(axis="x", labelrotation=45)
    return fig


def reward_terms(decomp: Mapping[str, Mapping[str, float]]) -> Figure:
    keys = ("mean_healthy", "mean_forward", "mean_ctrl", "mean_reward")
    labels = ["survival", "forward velocity", "- control cost", "recorded reward"]
    values = {s: [decomp[s][k] * (-1 if k == "mean_ctrl" else 1) for k in keys] for s in SOURCES}
    return grouped_bars(values, labels, "mean per step", "Reward decomposition per step")


def pca_coverage(proj: Mapping[str, np.ndarray], evr: Sequence[float], max_points: int = 5000) -> Figure:
    """Scatter of the first two PCs; rasterized and capped at ``max_points`` per source (file size).

    The cap is a seeded random subset: the rows arrive in dataset order, so the first rows
    would all come from the earliest episodes.
    """
    fig = Figure(figsize=(6.4, 5.2))
    ax = fig.add_subplot()
    rng = np.random.default_rng(0)
    for s in SOURCES:
        keep = rng.permutation(len(proj[s]))[:max_points]
        p = proj[s][keep]
        ax.scatter(p[:, 0], p[:, 1], s=3, alpha=0.35, color=COLORS[s], label=s, rasterized=True, linewidths=0)
    ax.set(
        xlabel=f"PC1 ({evr[0]:.0%} var)",
        ylabel=f"PC2 ({evr[1]:.0%} var)",
        title="State coverage (pooled standardized states)",
    )
    ax.legend(frameon=False, markerscale=4)
    _style(ax)
    fig.tight_layout()
    return fig


def knn_actions(d_mix: np.ndarray, d_same: np.ndarray, auc: float, bins: int = 50) -> Figure:
    fig = Figure(figsize=SIZE)
    ax = fig.add_subplot()
    edges = np.histogram_bin_edges(np.concatenate([d_mix, d_same]), bins=bins)
    ax.hist(
        d_same,
        bins=edges,
        color=COLORS[EXPERT],
        alpha=0.6,
        label=f"expert vs expert, median {np.median(d_same):.2f}",
    )
    ax.hist(
        d_mix,
        bins=edges,
        color=COLORS[SIMPLE],
        alpha=0.6,
        label=f"simple vs expert, median {np.median(d_mix):.2f}",
    )
    ax.set(
        xlabel="L2 distance between kNN mean actions",
        ylabel="test states",
        title=f"Action disagreement at shared states (source AUC {auc:.3f})",
    )
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


def mixture_reference(rows: Sequence[Mapping[str, Any]], simple_score: float) -> Figure:
    """Score of each mixture's own data vs p, with the seed range as a band.

    x uses evenly spaced positions labeled with p, since p = 0 rules out a log axis.
    """
    fig = Figure(figsize=SIZE)
    ax = fig.add_subplot()
    x = np.arange(len(rows))
    ax.fill_between(
        x,
        [r["score_min"] for r in rows],
        [r["score_max"] for r in rows],
        color=COLORS[EXPERT],
        alpha=0.2,
        linewidth=0,
    )
    ax.plot(
        x,
        [r["score_mean"] for r in rows],
        color=COLORS[EXPERT],
        linewidth=2,
        marker="o",
        markersize=5,
        label="mixture data (mean of 3 seeds)",
    )
    for y, text in ((100.0, "expert dataset"), (simple_score, "simple dataset"), (0.0, "random policy")):
        ax.axhline(y, color=MUTED, linewidth=1, linestyle="--")
        ax.annotate(
            text,
            (x[-1], y),
            xytext=(0, 3),
            textcoords="offset points",
            ha="right",
            fontsize=8,
            color="#555550",
        )
    ax.set_xticks(x, [f"{r['p']:g}" for r in rows])
    ax.set(
        xlabel="expert fraction p (by transitions)",
        ylabel="normalized score",
        title="Normalized score of each mixture's own episodes",
    )
    ax.legend(frameon=False, loc="upper left")
    _style(ax)
    fig.tight_layout()
    return fig
