"""Phase 5 validation analysis (spec Section 8): pure functions, tested on synthetic data.

V0 recomputes the reward scale and the terminal count from the raw source episodes a manifest
picked, independently of the mixture builder. V3 bounds the logged values by the mixture's own
scaled reward range and compares V at the initial states with the discounted returns from them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from offrl.data.load import Episode

REWARD_SCALE_NUMERATOR = 1000.0
V1_MARGIN = 10.0  # IQL two-seed mean >= BC two-seed mean - 10 at p = 1 (D3)
V2_MARGIN = 10.0  # |ours - CORL| two-seed means at p = 0.2
V3_AFTER_STEP = 50_000
V3_V0_RATIO = (0.8, 1.5)
V3_DRIFT = 0.10
V3_DRIFT_STEPS = (200_000, 300_000)


def picked_returns(episodes: Mapping[str, Sequence[Episode]], picks: Mapping[str, Any]) -> np.ndarray:
    """Undiscounted returns of the picked episodes that were kept whole (not trimmed)."""
    out = [
        float(np.sum(episodes[src][i].rew, dtype=np.float64))
        for src, rows in picks.items()
        for i, n in rows
        if n == len(episodes[src][i].rew)
    ]
    return np.asarray(out)


def reward_scale_from_episodes(episodes: Mapping[str, Sequence[Episode]], picks: Mapping[str, Any]) -> float:
    r = picked_returns(episodes, picks)
    return REWARD_SCALE_NUMERATOR / float(r.max() - r.min())


def terminated_count(episodes: Mapping[str, Sequence[Episode]], picks: Mapping[str, Any]) -> int:
    """Picked episodes whose kept prefix ends in a true termination (a fall)."""
    return sum(bool(episodes[src][i].term[:n].any()) for src, rows in picks.items() for i, n in rows)


def value_bounds(scaled_rew: np.ndarray, gamma: float) -> tuple[float, float]:
    """``[min(r_min, 0), max(r_max, 0)] / (1 - gamma)``: no return can leave this interval."""
    return min(float(scaled_rew.min()), 0.0) / (1 - gamma), max(float(scaled_rew.max()), 0.0) / (1 - gamma)


def initial_states_and_returns(
    obs: np.ndarray, rew: np.ndarray, t: np.ndarray, scale: float, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rows with ``t = 0`` and each episode's discounted scaled return from there.

    Mixture rows run episode by episode with ``t`` ascending (``mixture.flatten``), so an episode
    starts wherever ``t == 0``. A trimmed episode's return is cut short (one or two per mixture).
    """
    starts = np.flatnonzero(t == 0)
    if len(starts) == 0 or starts[0] != 0:
        raise ValueError("mixture rows must start with t = 0")
    disc = rew.astype(np.float64) * scale * gamma ** t.astype(np.float64)
    return obs[starts], np.add.reduceat(disc, starts)


def bounded(rows: Sequence[Mapping[str, str]], keys: Sequence[str], lo: float, hi: float) -> tuple[bool, str]:
    """Every logged value after ``V3_AFTER_STEP`` finite and inside ``[lo, hi]``."""
    late = [r for r in rows if int(r["step"]) > V3_AFTER_STEP]
    if not late:
        return False, f"no metrics rows after step {V3_AFTER_STEP}"
    bad = [
        (int(r["step"]), k, float(r[k]))
        for r in late
        for k in keys
        if not (np.isfinite(float(r[k])) and lo <= float(r[k]) <= hi)
    ]
    return not bad, f"{len(late)} rows in [{lo:.1f}, {hi:.1f}]" + (f"; first bad {bad[0]}" if bad else "")


def q_drift(rows: Sequence[Mapping[str, str]]) -> tuple[bool, str]:
    """``q_mean`` at 300k within 10% of its value at 200k (the critic has settled)."""
    by_step = {int(r["step"]): float(r["q_mean"]) for r in rows}
    a, b = (by_step.get(s) for s in V3_DRIFT_STEPS)
    if a is None or b is None:
        return False, f"q_mean missing at {V3_DRIFT_STEPS}"
    rel = abs(b - a) / abs(a)
    return rel <= V3_DRIFT, f"q_mean {a:.2f} -> {b:.2f} ({rel:.1%}, tolerance {V3_DRIFT:.0%})"


def v0_ratio(v_s0: float, ret_s0: float) -> tuple[bool, str]:
    lo, hi = V3_V0_RATIO
    ratio = v_s0 / ret_s0
    detail = f"V(s0) {v_s0:.2f} / discounted return {ret_s0:.2f} = {ratio:.3f} in [{lo}, {hi}]"
    return lo <= ratio <= hi, detail


def mean_score(runs: Sequence[Mapping[str, Any]]) -> float:
    return float(np.mean([r["final"]["score"] for r in runs]))


def _clean_at(record: Mapping[str, Any], commit: str) -> bool:
    return record["git_commit"] == commit and record["git_dirty"] is False


def assemble_checks(
    runs: Sequence[Mapping[str, Any]],
    mixtures: Mapping[str, Mapping[str, Any]],
    commit: str,
    no_fall_check: Any,
) -> list[dict[str, Any]]:
    """V0 to V4 (spec Section 8) from per-run and per-mixture facts the script computed.

    ``runs``: ``algo``, ``p``, ``seed``, ``run``, ``mixture``, ``final``, ``provenance``,
    ``reward_scale``; IQL runs also ``v3`` (``bounded``, ``drift`` and, at p = 1, ``v_s0``,
    each ``[ok, detail]``) and ``no_fall_return``. ``no_fall_check(no_fall) -> (ok, detail)``.
    """
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    def cell(algo: str, p: float) -> list[Mapping[str, Any]]:
        return [r for r in runs if r["algo"] == algo and r["p"] == p]

    for r in runs:
        prov = r["provenance"]
        check(
            f"V0_provenance[{r['run']}]",
            _clean_at(prov, commit)
            and all(_clean_at(rec, commit) for rec in prov["resume_records"])
            and prov["content_sha256"] == prov["manifest_content_sha256"],
            f"commit {str(prov['git_commit'])[:7]}, dirty {prov['git_dirty']}, "
            f"resumes {len(prov['resume_records'])}, mixture {prov['content_sha256'][:12]}",
        )
    for name, m in mixtures.items():
        logged = [r["reward_scale"] for r in runs if r["mixture"] == name]
        scale_ok = bool(logged) and all(
            np.isclose(s, m["reward_scale_recomputed"], rtol=1e-12, atol=0) for s in logged
        )
        check(
            f"V0_data[{name}]",
            scale_ok and m["done_rows"] == m["terminated_episodes"],
            f"reward_scale {m['reward_scale_recomputed']:.6f} vs logged {sorted(set(logged))}; "
            f"done rows {m['done_rows']} vs terminated episodes {m['terminated_episodes']}",
        )

    iql1, bc1 = mean_score(cell("iql", 1.0)), mean_score(cell("bc", 1.0))
    check("V1_iql_vs_bc_p1", iql1 >= bc1 - V1_MARGIN, f"IQL {iql1:.1f} >= BC {bc1:.1f} - {V1_MARGIN:g} (D3)")
    ours, corl = mean_score(cell("iql", 0.2)), mean_score(cell("corl_iql", 0.2))
    check(
        "V2_ours_vs_corl_p0.2",
        abs(ours - corl) <= V2_MARGIN,
        f"|{ours:.1f} - {corl:.1f}| = {abs(ours - corl):.1f} <= {V2_MARGIN:g}",
    )
    for r in runs:
        if r["algo"] != "iql":
            continue
        for key in ("bounded", "drift", "v_s0"):
            if key in r["v3"]:
                ok, detail = r["v3"][key]
                check(f"V3_{key}[{r['run']}]", ok, detail)
        if r["p"] == 1.0:
            ok, detail = no_fall_check(r["no_fall_return"])
            check(f"V4_no_fall[{r['run']}]", ok, detail)
    return checks
