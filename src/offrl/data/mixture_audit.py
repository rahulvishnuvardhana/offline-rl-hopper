"""Build a mixture and check it against the original data (checks M1 to M7, spec Section 7).

Checks read only what is saved in the mixture arrays (rows, columns, stored normalizer and
reward scale) and re-derive every value from the float64 sources through ``SourceIndex``, a
code path independent of ``mixture.flatten``. A corrupted input yields a FAIL, never a crash.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from offrl.data import mixture as mx
from offrl.data.load import Episode
from offrl.data.mixture import SOURCES, Picks
from offrl.data.normalize import fit_state_normalizer, reward_scale
from offrl.data.validate import Check


def build_mixture(
    episodes: Mapping[str, Sequence[Episode]], p: float, seed: int, n_transitions: int, eps: float
) -> tuple[dict[str, np.ndarray], Picks]:
    """Arrays to save (raw rows, columns, normalizer, reward scale, eps) and the picks."""
    lengths = {s: np.array([ep.length for ep in episodes[s]]) for s in SOURCES}
    picks = mx.sample_mixture_episodes(lengths, p, n_transitions, seed)
    arrays = mx.flatten(episodes, picks)
    norm = fit_state_normalizer(episodes, picks, eps=eps)
    summ = mx.episode_summary(episodes, picks)
    arrays["state_mean"], arrays["state_std"] = norm.mean, norm.std
    arrays["state_std_eps"] = np.array(eps, dtype=np.float64)
    arrays["reward_scale"] = np.array(reward_scale(summ.returns, summ.whole), dtype=np.float64)
    return arrays, picks


class SourceIndex:
    """Flat float64 views of one dataset, addressed by (episode_id, t)."""

    def __init__(self, episodes: Sequence[Episode]) -> None:
        if [ep.id for ep in episodes] != list(range(len(episodes))):
            raise ValueError("episode ids must equal their index")
        self.lengths = np.array([ep.length for ep in episodes])
        self.offset = np.r_[0, np.cumsum(self.lengths)[:-1]]
        self.obs = np.concatenate([ep.obs[:-1] for ep in episodes])
        self.next_obs = np.concatenate([ep.obs[1:] for ep in episodes])
        self.act = np.concatenate([ep.act for ep in episodes])
        self.rew = np.concatenate([ep.rew for ep in episodes])
        self.term = np.concatenate([ep.term for ep in episodes])
        self.trunc = np.concatenate([ep.trunc for ep in episodes])

    def valid(self, episode_id: np.ndarray, t: np.ndarray) -> np.ndarray:
        ok = (episode_id >= 0) & (episode_id < len(self.lengths)) & (t >= 0)
        ok[ok] &= t[ok] < self.lengths[episode_id[ok]]
        return ok

    def rows(self, episode_id: np.ndarray, t: np.ndarray) -> np.ndarray:
        return self.offset[episode_id] + t


def pick_rows(src: SourceIndex, picks: list[tuple[int, int]]) -> np.ndarray:
    """Flat source rows of one source's picks, in pick order."""
    ids = np.repeat([i for i, _ in picks], [k for _, k in picks])
    t = np.concatenate([np.arange(k) for _, k in picks])
    return src.rows(ids, t)


def saco_cells(
    arrays: Mapping[str, np.ndarray], ref_mean: np.ndarray, ref_std: np.ndarray, width: float
) -> int:
    """Unique (s, a) cells on a grid of ``width`` reference std (analysis only)."""
    x = np.concatenate([arrays["obs"], arrays["act"]], axis=1).astype(np.float64)
    cells = np.ascontiguousarray(np.floor((x - ref_mean) / (ref_std + 1e-3) / width).astype(np.int32))
    return len(np.unique(cells.view(np.dtype((np.void, cells.dtype.itemsize * cells.shape[1])))))


def check_mixture(
    name: str,
    p: float,
    arrays: Mapping[str, np.ndarray],
    picks: Picks,
    index: Mapping[str, SourceIndex],
    episodes: Mapping[str, Sequence[Episode]],
    n_transitions: int,
    eps: float,
    phase2_ref: Mapping[str, Any] | None,
) -> list[Check]:
    checks = []
    n = len(arrays["obs"])
    n_exp = int((arrays["source"] == mx.SOURCE_CODE[mx.EXPERT]).sum())
    target = int(round(p * n_transitions))
    checks.append(
        Check(
            f"M1_size_and_p[{name}]",
            n == n_transitions and n_exp == target,
            f"{n} rows, {n_exp} expert (target {target}); guard",
        )
    )
    ok = (p > 0 or n_exp == 0) and (p < 1 or n_exp == n)
    checks.append(Check(f"M2_pure_endpoints[{name}]", ok, f"p={p:g}: {n_exp} expert rows"))

    # M3: every row equals the float64 source at (episode_id, t), cast to float32.
    known = np.isin(arrays["source"], list(mx.SOURCE_CODE.values()))
    bad = int((~known).sum())
    for source in SOURCES:
        m = arrays["source"] == mx.SOURCE_CODE[source]
        ids, t = arrays["episode_id"][m].astype(np.int64), arrays["t"][m].astype(np.int64)
        valid = index[source].valid(ids, t)
        bad += int((~valid).sum())
        src, r = index[source], index[source].rows(ids[valid], t[valid])
        for key, ref in (("obs", src.obs), ("act", src.act), ("rew", src.rew), ("next_obs", src.next_obs)):
            bad += int((arrays[key][m][valid] != ref[r].astype(np.float32)).sum())
        bad += int((arrays["done"][m][valid] != src.term[r]).sum())
    checks.append(Check(f"M3_rows_equal_source[{name}]", bad == 0, f"{bad} mismatching or invalid values"))

    # M4: done semantics, re-checked against the source flags and the picks.
    problems = []
    for source in SOURCES:
        m = arrays["source"] == mx.SOURCE_CODE[source]
        ids, t = arrays["episode_id"][m].astype(np.int64), arrays["t"][m].astype(np.int64)
        valid = index[source].valid(ids, t)
        r = index[source].rows(ids[valid], t[valid])
        if (arrays["done"][m][valid][index[source].trunc[r]] != 0).any():
            problems.append(f"{source}: a truncated step has done=1")
    end = 0
    for source in SOURCES:
        for i, k in picks[source]:
            end += k
            if k < episodes[source][i].length:  # trimmed pick: last row continues the episode
                if arrays["done"][end - 1] != 0:
                    problems.append(f"{source} ep {i}: trimmed last row has done=1")
                if end < n and np.array_equal(arrays["next_obs"][end - 1], arrays["obs"][end]):
                    problems.append(f"{source} ep {i}: trimmed next_obs equals the next row's obs")
    checks.append(Check(f"M4_done_semantics[{name}]", not problems, "; ".join(problems[:5]) or "ok"))

    # M5: picks re-derived from the saved columns; returns and scale vs. Phase 2 (when previewed).
    try:
        rederived = mx.picks_from_columns(arrays["source"], arrays["episode_id"], arrays["t"])
    except (ValueError, KeyError) as err:
        checks.append(Check(f"M5_picks_and_phase2[{name}]", False, f"columns unreadable: {err}"))
        rederived = None
    if rederived is not None:
        ok = rederived == picks
        detail = f"picks {'match' if ok else 'DIFFER'}"
        if phase2_ref is not None and ok:
            summ = mx.episode_summary(episodes, rederived)
            ok &= bool(np.isclose(summ.returns.mean(), phase2_ref["mean_return"], rtol=1e-12, atol=0))
            ok &= bool(
                np.isclose(
                    reward_scale(summ.returns, summ.whole), phase2_ref["reward_scale"], rtol=1e-12, atol=0
                )
            )
            detail += f"; mean return {summ.returns.mean():.6f} vs Phase 2 {phase2_ref['mean_return']:.6f}"
        elif phase2_ref is None:
            detail += "; no Phase 2 preview for this (p, seed)"
        checks.append(Check(f"M5_picks_and_phase2[{name}]", ok, detail))

    # M6: the STORED normalizer, eps and reward scale (what training reads) vs. a float64
    # recomputation from the sources via the index; plus the float32 file rows.
    if rederived is None:
        checks.append(Check(f"M6_stored_normalizer_and_scale[{name}]", False, "picks unavailable"))
        return checks
    ref = np.concatenate([index[s].obs[pick_rows(index[s], picks[s])] for s in SOURCES if picks[s]])
    summ = mx.episode_summary(episodes, picks)
    stored_mean, stored_std = arrays["state_mean"], arrays["state_std"]
    results = {
        "mean": np.allclose(ref.mean(0), stored_mean, rtol=1e-12, atol=0),
        "std": np.allclose(ref.std(0), stored_std, rtol=1e-12, atol=0),
        "eps": float(arrays["state_std_eps"]) == eps,
        "reward_scale": bool(
            np.isclose(
                float(arrays["reward_scale"]), reward_scale(summ.returns, summ.whole), rtol=1e-12, atol=0
            )
        ),
    }
    f32 = arrays["obs"].astype(np.float64)
    results["float32_rows"] = bool(
        (np.abs(f32.mean(0) - stored_mean) <= 1e-6 * stored_std).all()
        and (np.abs(f32.std(0) - stored_std) <= 1e-6 * stored_std).all()
    )
    failed = [k for k, v in results.items() if not v]
    checks.append(
        Check(
            f"M6_stored_normalizer_and_scale[{name}]",
            not failed,
            f"mismatch: {failed}" if failed else "mean, std, eps, reward_scale, float32 rows ok",
        )
    )
    return checks
