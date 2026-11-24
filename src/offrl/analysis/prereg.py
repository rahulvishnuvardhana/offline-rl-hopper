"""Pre-registered analysis of the Phase 7 main grid and the Phase 8 ablation (docs/preregistration.md,
approved 2026-11-21). Pure functions over per-seed final scores; ``scripts/prereg_analysis.py``
reads the runs. Section numbers below are the pre-registration's. Comparisons use unrounded
scores; rounding is for display only."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import spearmanr

SEEDS = (0, 1, 2, 3, 4)
P_GRID = (0.0, 0.05, 0.2, 0.5, 1.0)
MAIN_TAU = 0.7
ABL_TAUS = (0.5, 0.9)
ABL_PS = (0.05, 0.5)
LEVEL = 0.05  # Section 4
NEAR_EXPERT = 90.0  # Section 6
COLLAPSED = 10.0  # Section 9, rule 2
HIGH_FALL = 0.5  # Section 9, rule 2
NONMONO_DROP = 5.0  # Section 5
CLIMBING = 2.0  # Section 9
BOOT_N = 10_000  # Section 3
BOOT_SEED = 20261122  # Section 3
T_975_DF4 = 2.776  # Section 3
EPS = 1e-12  # sign-flip ties: a pattern mean equal to the observed one counts (absolute; fine at score scale)
INTERVAL_NOTE = (
    "5 seeds: the bootstrap has 126 distinct resamples, so its intervals are coarse and narrow; "
    "the t-interval assumes roughly normal per-seed values; both indicative only"
)

Cell = tuple[str, float | None, float]  # (algo, tau, p); BC carries tau = None


def bc(p: float) -> Cell:
    return ("bc", None, p)


def iql(tau: float, p: float) -> Cell:
    return ("iql", tau, p)


def cell_name(c: Cell) -> str:
    algo, tau, p = c
    return f"bc p={p:g}" if algo == "bc" else f"iql tau={tau:g} p={p:g}"


def final_score(score: Any) -> float:
    """Section 9 rule 2: a non-finite final score (written to final.json as a string) counts as 0."""
    if isinstance(score, str):
        return 0.0
    return float(score) if math.isfinite(score) else 0.0


def sign_flip_p(x: Sequence[float], alternative: str) -> float:
    """Section 4: exact one-sided paired sign-flip test; statistic the mean, null over all 2^n sign
    patterns. ``greater``: fraction of pattern means >= observed; ``less``: <= observed."""
    x = np.asarray(x, dtype=float)
    obs = x.mean()
    means = np.array([(x * s).mean() for s in itertools.product((1.0, -1.0), repeat=len(x))])
    if alternative == "greater":
        return float(np.mean(means >= obs - EPS))
    if alternative == "less":
        return float(np.mean(means <= obs + EPS))
    raise ValueError(alternative)


def boot_index(n: int = len(SEEDS)) -> np.ndarray:
    """One matrix of resampled seed indices, shared by every interval, so a contrast resamples
    the same seeds on both sides (Section 3, paired)."""
    return np.random.default_rng(BOOT_SEED).integers(0, n, size=(BOOT_N, n))


def boot_interval(stat: Callable[..., float], idx: np.ndarray, *xs: Sequence[float]) -> list[float]:
    """95% percentile bootstrap of ``stat`` over the shared resampled seeds. For the secondary
    statistics (Section 2) this is rliable's stratified bootstrap, which with one task is this one."""
    arrs = [np.asarray(x, dtype=float) for x in xs]
    boot = np.array([stat(*(a[i] for a in arrs)) for i in idx])
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def intervals(x: Sequence[float], idx: np.ndarray) -> dict[str, Any]:
    """Section 3: percentile bootstrap and t-interval, both 95%, for the mean of ``x``."""
    x = np.asarray(x, dtype=float)
    half = T_975_DF4 * x.std(ddof=1) / math.sqrt(len(x))
    return {
        "bootstrap_95": boot_interval(np.mean, idx, x),
        "t_95": [float(x.mean() - half), float(x.mean() + half)],
        "interval_note": INTERVAL_NOTE,
    }


def iqm(x: Sequence[float]) -> float:
    """Interquartile mean as rliable computes it (scipy trim_mean, 0.25): with 5 values, the mean
    of the middle 3."""
    x = np.sort(np.asarray(x, dtype=float))
    k = int(0.25 * len(x))
    return float(x[k : len(x) - k].mean())


def prob_improvement(a: Sequence[float], b: Sequence[float]) -> float:
    """P(A > B) + 0.5 P(A = B) over all run pairs (rliable's, one task)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.mean((a[:, None] > b[None, :]) + 0.5 * (a[:, None] == b[None, :])))


def describe(scores: Sequence[float], idx: np.ndarray) -> dict[str, Any]:
    x = np.asarray(scores, dtype=float)
    return {
        "per_seed": x.tolist(),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "sd": float(x.std(ddof=1)),
        "iqm": iqm(x),
        "iqm_bootstrap_95": boot_interval(iqm, idx, x),
        **intervals(x, idx),
    }


def compare(a: Sequence[float], b: Sequence[float], idx: np.ndarray) -> dict[str, Any]:
    """A named comparison: per-seed differences a - b, their description and both one-sided tests."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    out = describe(d, idx)
    out["p_greater"] = sign_flip_p(d, "greater")
    out["p_less"] = sign_flip_p(d, "less")
    out["prob_improvement"] = prob_improvement(a, b)
    out["prob_improvement_bootstrap_95"] = boot_interval(prob_improvement, idx, a, b)
    return out


def crossing(means: Mapping[float, float]) -> float | None:
    """Section 6: smallest grid p with mean >= 90 and every larger grid p >= 90 too."""
    cross = None
    for p in reversed(P_GRID):
        if means[p] >= NEAR_EXPERT:
            cross = p
        else:
            break
    return cross


def h1(score: Mapping[Cell, list[float]], idx: np.ndarray) -> dict[str, Any]:
    main = compare(score[bc(1.0)], score[bc(0.0)], idx)
    if main["p_greater"] <= LEVEL:
        verdict = "supported"
    elif main["p_less"] <= LEVEL:
        verdict = "not supported (reversed)"
    else:
        verdict = "inconclusive"
    means = {p: float(np.mean(score[bc(p)])) for p in P_GRID}
    rho = [float(spearmanr(P_GRID, [score[bc(p)][s] for p in P_GRID]).statistic) for s in range(len(SEEDS))]
    rho_nan = [SEEDS[s] for s, r in enumerate(rho) if not math.isfinite(r)]  # BC constant over p
    dips = [
        {"p": p, "mean": means[p], "per_seed": score[bc(p)]}
        for p in P_GRID[1:-1]
        if means[p] < means[0.0] and means[p] < means[1.0]
    ]
    adjacent = []
    for lo, hi in itertools.pairwise(P_GRID):
        c = compare(score[bc(lo)], score[bc(hi)], idx)  # positive = a drop from lo to hi
        drop = {"from": lo, "to": hi, "drop_mean": c["mean"], "p_drop": c["p_greater"]}
        adjacent.append({**drop, "per_seed": c["per_seed"]})
    nonmono = any(a["drop_mean"] > NONMONO_DROP and a["p_drop"] <= LEVEL for a in adjacent)
    return {
        "verdict": verdict,
        "test": "BC(p=1) - BC(p=0)",
        "comparison": main,
        "spearman_per_seed": rho,
        "spearman_mean": float(np.nanmean(rho)) if len(rho_nan) < len(rho) else None,
        "spearman_nan_seeds": rho_nan,
        "dips": dips,
        "adjacent_drops": adjacent,
        "shape": "non-monotone" if nonmono else "monotone (as far as 5 seeds show)",
    }


def h2(score: Mapping[Cell, list[float]], idx: np.ndarray) -> dict[str, Any]:
    bc_means = {p: float(np.mean(score[bc(p)])) for p in P_GRID}
    iql_means = {p: float(np.mean(score[iql(MAIN_TAU, p)])) for p in P_GRID}
    cb, ci = crossing(bc_means), crossing(iql_means)
    out: dict[str, Any] = {
        "bc_means": bc_means,
        "iql_means": iql_means,
        "bc_crossing_p": cb,
        "iql_crossing_p": ci,
        "comparison": None,
    }
    if ci is None:
        out["verdict"] = "not supported (IQL does not cross 90)"
        return out
    much_lower = cb is None or P_GRID.index(ci) <= P_GRID.index(cb) - 1
    out["much_lower"] = much_lower
    c = compare(score[iql(MAIN_TAU, ci)], score[bc(ci)], idx)
    out["comparison"] = c
    out["test"] = f"IQL(p={ci:g}) - BC(p={ci:g})"
    if not much_lower:
        out["verdict"] = "not supported (IQL's crossing p is not below BC's)"
    elif c["p_greater"] <= LEVEL:
        out["verdict"] = "supported"
    else:
        out["bc_leads"] = c["mean"] < 0  # Section 6: a BC lead at that p is reported as such
        out["verdict"] = "inconclusive" + (" (BC leads at IQL's crossing p)" if out["bc_leads"] else "")
    return out


def h3(score: Mapping[Cell, list[float]], idx: np.ndarray) -> dict[str, Any]:
    eff = {p: np.asarray(score[iql(0.9, p)]) - np.asarray(score[iql(0.5, p)]) for p in ABL_PS}
    x = eff[0.05] - eff[0.5]
    c = describe(x, idx)
    c["p_greater"] = sign_flip_p(x, "greater")
    c["p_less"] = sign_flip_p(x, "less")
    if c["p_greater"] <= LEVEL:
        # Section 7 wording rule; "the effect at p = 0.05" read as its seed mean
        positive = eff[0.05].mean() > 0
        verdict = "supported" if positive else "supported, worded: the tau effect is larger at p = 0.05"
    elif c["p_less"] <= LEVEL:
        verdict = "not supported (reversed)"
    else:
        verdict = "inconclusive"
    simple = {f"{p:g}": compare(score[iql(0.9, p)], score[iql(0.5, p)], idx) for p in ABL_PS}
    beside = {f"{p:g}": float(np.mean(score[iql(MAIN_TAU, p)])) for p in ABL_PS}
    return {
        "verdict": verdict,
        "test": "[tau0.9 - tau0.5](p=0.05) - [tau0.9 - tau0.5](p=0.5)",
        "contrast": c,
        "simple_effects_tau0.9_minus_tau0.5": simple,
        "tau0.7_means_beside": beside,
    }
