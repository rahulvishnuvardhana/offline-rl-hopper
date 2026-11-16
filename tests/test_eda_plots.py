"""Every figure function runs on synthetic data and returns a Figure with the expected axes."""

import numpy as np
from matplotlib.figure import Figure

from offrl.analysis import eda_plots as plots
from offrl.analysis.eda import EXPERT, SIMPLE

RNG = np.random.default_rng(0)
BY_SOURCE = {EXPERT: RNG.normal(3000, 500, 50), SIMPLE: RNG.normal(1500, 900, 80)}


def _n_axes(fig):
    assert isinstance(fig, Figure)
    return len(fig.axes)


def test_histograms():
    assert _n_axes(plots.hist_by_source(BY_SOURCE, "x", "t")) == 1
    fell = {EXPERT: RNG.random(50) < 0.3, SIMPLE: RNG.random(80) < 0.6}
    assert _n_axes(plots.returns_by_fall(BY_SOURCE, fell)) == 2


def test_bars():
    stats = {s: {"std": list(RNG.random(11) + 0.1)} for s in (EXPERT, SIMPLE)}
    assert _n_axes(plots.state_std(stats)) == 1
    keys = ("mean_healthy", "mean_forward", "mean_ctrl", "mean_reward")
    decomp = {s: dict.fromkeys(keys, 1.0) for s in (EXPERT, SIMPLE)}
    fig = plots.reward_terms(decomp)
    assert _n_axes(fig) == 1 and len(fig.axes[0].patches) == 8  # 4 terms x 2 sources


def test_scatter_and_knn():
    proj = {EXPERT: RNG.normal(size=(100, 2)), SIMPLE: RNG.normal(size=(100, 2))}
    assert _n_axes(plots.pca_coverage(proj, [0.4, 0.2])) == 1
    assert _n_axes(plots.knn_actions(RNG.random(100), RNG.random(100), 0.9)) == 1


def test_mixture_reference_and_save(tmp_path):
    rows = [
        {"p": p, "score_mean": 40 + 60 * p, "score_min": 38 + 60 * p, "score_max": 42 + 60 * p}
        for p in (0.0, 0.05, 0.5, 1.0)
    ]
    fig = plots.mixture_reference(rows, simple_score=43.0)
    assert _n_axes(fig) == 1
    path = tmp_path / "f.png"
    plots.save(fig, path)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert b"matplotlib" not in path.read_bytes()  # no version stamp
