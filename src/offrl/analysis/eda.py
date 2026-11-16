"""Phase 2B exploratory data analysis (spec: docs/specs/phase-2b-eda.md).

Pure functions on ``Episode`` lists and arrays: no file I/O, no Minari, so every number
can be tested on synthetic episodes with known answers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from offrl.data.load import Episode

EXPERT, SIMPLE = "expert", "simple"
SOURCES = (EXPERT, SIMPLE)  # fixed order: the mixture RNG stream depends on it

VEL_DIMS = tuple(range(5, 11))  # obs[5:11] are velocities, clipped by the env
VEL_CLIP = 10.0
HEALTHY_REWARD = 1.0  # Hopper-v5 defaults
FORWARD_REWARD_WEIGHT = 1.0
CTRL_COST_WEIGHT = 1e-3

P_GRID = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0)
REWARD_SCALE_NUMERATOR = 1000.0  # IQL: r * 1000 / (max_return - min_return)


# ---------- A1 to A3b: episode statistics ----------


def describe(x: np.ndarray) -> dict[str, float] | None:
    """Summary statistics, or None for an empty group (JSON has no NaN)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return None
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "median": float(np.median(x)),
        "max": float(x.max()),
    }


def fell_mask(episodes: Sequence[Episode]) -> np.ndarray:
    """True where the episode ended by termination (the hopper fell), not the time limit."""
    return np.array([bool(ep.term[-1]) for ep in episodes])


def episode_stats(episodes: Sequence[Episode]) -> dict[str, Any]:
    lengths = np.array([ep.length for ep in episodes])
    returns = np.array([ep.ret for ep in episodes])
    fell = fell_mask(episodes)
    out: dict[str, Any] = {
        "n_episodes": len(episodes),
        "n_transitions": int(lengths.sum()),
        "fall_rate": float(fell.mean()),
        "length": describe(lengths),
        "return": describe(returns),
    }
    for name, mask in (("fell", fell), ("no_fall", ~fell)):
        out[name] = {
            "n_episodes": int(mask.sum()),
            "n_transitions": int(lengths[mask].sum()),
            "share_episodes": float(mask.mean()),
            "share_transitions": float(lengths[mask].sum() / lengths.sum()),
            "length": describe(lengths[mask]),
            "return": describe(returns[mask]),
        }
    return out


# ---------- A4 to A7: states, actions, rewards ----------


def state_matrix(episodes: Sequence[Episode]) -> np.ndarray:
    """States that have an action: ``obs[:-1]`` of every episode (the final obs is excluded)."""
    return np.concatenate([ep.obs[:-1] for ep in episodes])


def state_stats(episodes: Sequence[Episode]) -> dict[str, list[float]]:
    """Descriptive only. Training normalizers are fit per mixture (ADR-5), never on these."""
    s = state_matrix(episodes)
    return {"mean": s.mean(0).tolist(), "std": s.std(0).tolist()}


def velocity_clipping(episodes: Sequence[Episode]) -> dict[str, float]:
    s = state_matrix(episodes)
    return {
        f"obs[{d}]": float(np.isclose(np.abs(s[:, d]), VEL_CLIP, rtol=0.0, atol=1e-9).mean())
        for d in VEL_DIMS
    }


def action_stats(episodes: Sequence[Episode], bound: float = 1.0) -> dict[str, list[float]]:
    a = np.concatenate([ep.act for ep in episodes])
    return {
        "mean": a.mean(0).tolist(),
        "std": a.std(0).tolist(),
        "frac_at_bound": np.isclose(np.abs(a), bound, atol=1e-6).mean(0).tolist(),
    }


def reward_terms(episodes: Sequence[Episode]) -> dict[str, np.ndarray]:
    """Rebuild each reward from its parts. ``forward`` uses the next observation's x velocity.

    The env's forward term is a finite difference of x position over the step, so the
    residual is small but not zero; a wrong reward scale or a missing survival bonus makes
    it large. Hopper-v5 pays the survival bonus only while healthy, so the terminal (fall)
    step gets none. The control cost is too small (about 0.001 per step) for the residual
    to detect an error in it; that term is taken from the env definition, not verified.
    """
    act = np.concatenate([ep.act for ep in episodes])
    rew = np.concatenate([ep.rew for ep in episodes])
    forward = FORWARD_REWARD_WEIGHT * np.concatenate([ep.obs[1:, 5] for ep in episodes])
    healthy = HEALTHY_REWARD * ~np.concatenate([ep.term for ep in episodes])
    ctrl = CTRL_COST_WEIGHT * (act**2).sum(1)
    residual = rew - (healthy + forward - ctrl)
    return {"reward": rew, "healthy": healthy, "forward": forward, "ctrl": ctrl, "residual": residual}


def reward_decomposition(episodes: Sequence[Episode]) -> dict[str, float]:
    t = reward_terms(episodes)
    abs_res = np.abs(t["residual"])
    return {
        "mean_reward": float(t["reward"].mean()),
        "mean_healthy": float(t["healthy"].mean()),
        "mean_forward": float(t["forward"].mean()),
        "mean_ctrl": float(t["ctrl"].mean()),
        "residual_mean": float(t["residual"].mean()),
        "residual_abs_median": float(np.median(abs_res)),
        "residual_abs_p99": float(np.quantile(abs_res, 0.99)),
        "residual_abs_max": float(abs_res.max()),
    }


# ---------- A8, A9: subsampling, PCA, kNN, AUC ----------


def subsample_states(
    episodes: Sequence[Episode], n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``n`` states drawn uniformly over all transitions, with their actions and episode ids.

    Episode ids travel with the states so every train/test split is by episode: states of
    one episode (near-duplicates of each other) never sit on both sides.
    """
    states = state_matrix(episodes)
    if n > len(states):
        raise ValueError(f"asked for {n} states, only {len(states)} available")
    acts = np.concatenate([ep.act for ep in episodes])
    ids = np.repeat([ep.id for ep in episodes], [ep.length for ep in episodes])
    pick = np.sort(rng.choice(len(states), size=n, replace=False))
    return states[pick], acts[pick], ids[pick]


def split_by_episode(ids: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask: True for states whose episode falls in the first ``frac`` of a shuffled id list."""
    unique = rng.permutation(np.unique(ids))
    chosen = unique[: int(round(frac * len(unique)))]
    return np.isin(ids, chosen)


def pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(components (2, d), explained variance ratio (2,), projection (n, 2)).

    SVD signs are arbitrary, so each component is flipped to make its largest-abs loading
    positive; otherwise two correct runs could disagree.
    """
    xc = x - x.mean(0)
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    comps = vt[:2].copy()
    signs = np.sign(comps[np.arange(2), np.abs(comps).argmax(1)])
    comps *= signs[:, None]
    evr = s[:2] ** 2 / (s**2).sum()
    return comps, evr, xc @ comps.T


def knn_indices(query: np.ndarray, ref: np.ndarray, k: int, block: int = 512) -> np.ndarray:
    """Indices of the k nearest ``ref`` rows for each ``query`` row, nearest first.

    Distances are computed ``block`` query rows at a time (512 x 32k x 8 bytes is about
    130 MB), never as the full matrix.
    """
    ref_sq = (ref**2).sum(1)
    out = np.empty((len(query), k), dtype=np.int64)
    for start in range(0, len(query), block):
        q = query[start : start + block]
        d = (q**2).sum(1)[:, None] - 2.0 * q @ ref.T + ref_sq[None, :]
        part = np.argpartition(d, k - 1, axis=1)[:, :k]
        order = np.take_along_axis(d, part, 1).argsort(1, kind="stable")
        out[start : start + block] = np.take_along_axis(part, order, 1)
    return out


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based (Mann-Whitney) ROC AUC; tied scores get their average rank."""
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs both classes")
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(counts) - (counts - 1) / 2.0  # 1-based average rank of each tie group
    ranks = avg_rank[inverse]
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _knn_mean_action(query: np.ndarray, ref: np.ndarray, ref_act: np.ndarray, k: int) -> np.ndarray:
    return ref_act[knn_indices(query, ref, k)].mean(1)


def separability(
    expert: Sequence[Episode],
    simple: Sequence[Episode],
    n_per_source: int,
    seed: int,
    k: int = 10,
    train_frac: float = 0.8,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """A9: can the state alone reveal the source, and do the sources act differently there?

    Returns (numbers, arrays for the figure). Action disagreement uses equal-size pools on
    both sides (half of each source's train episodes), so ``d_mix`` and ``d_same`` are not
    biased by one pool being denser than the other.
    """
    rng = np.random.default_rng(seed)
    xe, ae, ge = subsample_states(expert, n_per_source, rng)
    xs, as_, gs = subsample_states(simple, n_per_source, rng)
    pooled = np.concatenate([xe, xs])
    mean, std = pooled.mean(0), pooled.std(0)
    ze, zs = (xe - mean) / std, (xs - mean) / std

    e_tr = split_by_episode(ge, train_frac, rng)
    s_tr = split_by_episode(gs, train_frac, rng)

    ref = np.concatenate([ze[e_tr], zs[s_tr]])
    ref_label = np.r_[np.ones(e_tr.sum()), np.zeros(s_tr.sum())]
    query = np.concatenate([ze[~e_tr], zs[~s_tr]])
    query_label = np.r_[np.ones((~e_tr).sum()), np.zeros((~s_tr).sum())]
    expert_frac = ref_label[knn_indices(query, ref, k)].mean(1)
    auc = roc_auc(expert_frac, query_label.astype(bool))

    e_half = split_by_episode(ge[e_tr], 0.5, rng)
    s_half = split_by_episode(gs[s_tr], 0.5, rng)
    ze_a, ae_a = ze[e_tr][e_half], ae[e_tr][e_half]
    ze_b, ae_b = ze[e_tr][~e_half], ae[e_tr][~e_half]
    zs_a, as_a = zs[s_tr][s_half], as_[s_tr][s_half]

    q_s, q_e = zs[~s_tr], ze[~e_tr]
    d_mix = np.linalg.norm(
        _knn_mean_action(q_s, ze_a, ae_a, k) - _knn_mean_action(q_s, zs_a, as_a, k), axis=1
    )
    d_same = np.linalg.norm(
        _knn_mean_action(q_e, ze_a, ae_a, k) - _knn_mean_action(q_e, ze_b, ae_b, k), axis=1
    )

    # How far the "shared" states really are: if simple states' expert neighbours are much
    # farther than their simple neighbours, d_mix partly measures state distance, not policy.
    def nn_dist(query, ref):
        return float(np.median(np.linalg.norm(query - ref[knn_indices(query, ref, 1)[:, 0]], axis=1)))

    numbers = {
        "auc": auc,
        "d_mix_median": float(np.median(d_mix)),
        "d_same_median": float(np.median(d_same)),
        "ratio": float(np.median(d_mix) / np.median(d_same)),
        "nn_dist_simple_to_expert_median": nn_dist(q_s, ze_a),
        "nn_dist_simple_to_simple_median": nn_dist(q_s, zs_a),
        "nn_dist_expert_to_expert_median": nn_dist(q_e, ze_b),
        "n_test_expert": int((~e_tr).sum()),
        "n_test_simple": int((~s_tr).sum()),
    }
    return numbers, {"d_mix": d_mix, "d_same": d_same}


def coverage_pca(
    expert: Sequence[Episode], simple: Sequence[Episode], n_per_source: int, seed: int
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """A8: PCA of pooled standardized states. Returns (numbers, projections by source)."""
    rng = np.random.default_rng(seed)
    xe, _, _ = subsample_states(expert, n_per_source, rng)
    xs, _, _ = subsample_states(simple, n_per_source, rng)
    pooled = np.concatenate([xe, xs])
    z = (pooled - pooled.mean(0)) / pooled.std(0)
    comps, evr, proj = pca_2d(z)
    numbers = {"explained_variance_ratio": evr.tolist(), "components": comps.tolist()}
    return numbers, {EXPERT: proj[:n_per_source], SIMPLE: proj[n_per_source:]}


# ---------- A10 to A12: score, mixtures ----------


def normalized_score(ret: float, r_random: float, r_expert: float) -> float:
    """ADR-6 (D4RL convention): 0 = random policy, 100 = expert dataset mean."""
    return 100.0 * (ret - r_random) / (r_expert - r_random)


def sample_mixture_episodes(
    lengths: Mapping[str, np.ndarray], p: float, n_transitions: int, seed: int
) -> dict[str, list[tuple[int, int]]]:
    """Phase 3 sampling rule: per source, shuffle episodes with the seed and take whole
    episodes until its budget is met; the last one is trimmed. Returns (index, steps taken).

    One generator draws the expert permutation first, then the simple one. Phase 3's
    ``mixture.py`` must use the same order to reproduce these mixtures.
    """
    rng = np.random.default_rng(seed)
    n_expert = int(round(p * n_transitions))
    budgets = {EXPERT: n_expert, SIMPLE: n_transitions - n_expert}
    out: dict[str, list[tuple[int, int]]] = {}
    for source in SOURCES:
        order = rng.permutation(len(lengths[source]))
        taken, remaining = [], budgets[source]
        for i in order:
            if remaining == 0:
                break
            n = min(int(lengths[source][i]), remaining)
            taken.append((int(i), n))
            remaining -= n
        if remaining:
            raise ValueError(f"{source}: budget {budgets[source]} exceeds the dataset")
        out[source] = taken
    return out


def mixture_preview(
    episodes: Mapping[str, Sequence[Episode]], p: float, n_transitions: int, seed: int
) -> dict[str, float]:
    """Mean episode return of the mixture (trimmed episodes at their trimmed return) and the
    IQL reward scale ``1000 / (max_ret - min_ret)`` it would get.

    The reward scale uses untrimmed episodes only: a trimmed tail can be a few steps long and
    would set the minimum, making the scale depend on where the budget cut fell.
    """
    lengths = {s: np.array([ep.length for ep in episodes[s]]) for s in SOURCES}
    picks = sample_mixture_episodes(lengths, p, n_transitions, seed)
    taken = [(episodes[s][i], n) for s in SOURCES for i, n in picks[s]]
    returns = np.array([float(ep.rew[:n].sum()) for ep, n in taken])
    whole = np.array([n == ep.length for ep, n in taken])
    return {
        "mean_return": float(returns.mean()),
        "n_episodes": len(returns),
        "n_trimmed": int((~whole).sum()),
        "reward_scale": REWARD_SCALE_NUMERATOR / float(returns[whole].max() - returns[whole].min()),
    }


def mixture_reference(
    episodes: Mapping[str, Sequence[Episode]],
    p_grid: Sequence[float],
    seeds: Sequence[int],
    n_transitions: int,
    r_random: float,
    r_expert: float,
) -> list[dict[str, Any]]:
    """A11 + A12 for every p: mean and range across seeds."""
    rows = []
    for p in p_grid:
        runs = [mixture_preview(episodes, p, n_transitions, s) for s in seeds]
        scores = [normalized_score(r["mean_return"], r_random, r_expert) for r in runs]
        scales = [r["reward_scale"] for r in runs]
        rows.append(
            {
                "p": float(p),
                "mean_return": float(np.mean([r["mean_return"] for r in runs])),
                "score_mean": float(np.mean(scores)),
                "score_min": float(np.min(scores)),
                "score_max": float(np.max(scores)),
                "reward_scale_mean": float(np.mean(scales)),
                "reward_scale_min": float(np.min(scales)),
                "reward_scale_max": float(np.max(scales)),
                "per_seed": runs,
            }
        )
    return rows


def sanity_bands(stats: dict[str, Any], r_random: float, r_expert: float) -> list[float]:
    """Phase 4 band for a dataset: [score of the fell mean, score of the no-fall mean]."""
    lo = stats["fell"]["return"]["mean"]
    hi = stats["no_fall"]["return"]["mean"]
    return [normalized_score(lo, r_random, r_expert), normalized_score(hi, r_random, r_expert)]


# ---------- C7: determinism ----------


def compare_numeric(a: Any, b: Any, rtol: float = 1e-9, path: str = "") -> list[str]:
    """Paths where two nested results differ: numbers by ``rtol``, everything else exactly."""
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if a.keys() != b.keys():
            return [f"{path}: keys differ"]
        return [m for key in a for m in compare_numeric(a[key], b[key], rtol, f"{path}.{key}")]
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        if len(a) != len(b):
            return [f"{path}: lengths differ"]
        return [
            m
            for i, (x, y) in enumerate(zip(a, b, strict=True))
            for m in compare_numeric(x, y, rtol, f"{path}[{i}]")
        ]
    numeric = (int, float, np.integer, np.floating)
    if isinstance(a, numeric) and isinstance(b, numeric) and not isinstance(a, bool):
        return [] if np.isclose(a, b, rtol=rtol, atol=0.0) else [f"{path}: {a} != {b}"]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]
