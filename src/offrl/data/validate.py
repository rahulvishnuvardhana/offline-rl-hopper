"""Integrity checks for offline RL episodes (ROADMAP Phase 2).

Each check returns a ``Check`` instead of raising, so one run reports every problem.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from offrl.data.load import Episode

# Documented on the Minari dataset pages (checked 2026-11-12).
EXPECTED = {
    "mujoco/hopper/expert-v0": {"total_steps": 999_164, "total_episodes": 1_086},
    "mujoco/hopper/simple-v0": {"total_steps": 999_206, "total_episodes": 1_952},
}
MAX_EPISODE_STEPS = 1_000

# Hopper-v5 health rule: healthy iff z > 0.7 and -0.2 < angle < 0.2 (obs[0] = z, obs[1] = angle).
HOPPER_MIN_Z = 0.7
HOPPER_MAX_ABS_ANGLE = 0.2


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str

    def as_dict(self) -> dict:
        return asdict(self)


def hopper_healthy(obs: np.ndarray) -> np.ndarray:
    """Vectorized health rule on observations (..., 11) -> bool (...)."""
    return (obs[..., 0] > HOPPER_MIN_Z) & (np.abs(obs[..., 1]) < HOPPER_MAX_ABS_ANGLE)


def check_counts(
    episodes: Sequence[Episode], documented: dict[str, int], metadata: dict[str, int] | None = None
) -> Check:
    """Three-way reconciliation: documented counts vs. dataset metadata vs. our own sum."""
    ours = {"total_steps": sum(ep.length for ep in episodes), "total_episodes": len(episodes)}
    sources = {"documented": documented, "ours": ours}
    if metadata is not None:
        sources["metadata"] = metadata
    agree = all(src == ours for src in sources.values())
    return Check("counts_reconcile", agree, "; ".join(f"{k}={v}" for k, v in sources.items()))


def check_unique_ids(episodes: Sequence[Episode]) -> Check:
    dupes = [i for i, n in Counter(ep.id for ep in episodes).items() if n > 1]
    return Check("unique_episode_ids", not dupes, f"duplicate ids: {dupes[:10]}" if dupes else "all unique")


def check_shapes(episodes: Sequence[Episode]) -> Check:
    bad = [
        ep.id
        for ep in episodes
        if not (
            len(ep.obs) == ep.length + 1
            and len(ep.rew) == len(ep.term) == len(ep.trunc) == ep.length
            and ep.length > 0
        )
    ]
    return Check("shapes_obs_is_n_plus_1", not bad, f"bad episodes: {bad[:10]}" if bad else "all consistent")


def check_finite(episodes: Sequence[Episode]) -> Check:
    bad = [
        ep.id
        for ep in episodes
        if not (np.isfinite(ep.obs).all() and np.isfinite(ep.act).all() and np.isfinite(ep.rew).all())
    ]
    return Check("no_nan_or_inf", not bad, f"episodes with NaN/inf: {bad[:10]}" if bad else "all finite")


def check_action_bounds(episodes: Sequence[Episode], low: float = -1.0, high: float = 1.0) -> Check:
    acts = np.concatenate([ep.act for ep in episodes])
    out = int(((acts < low) | (acts > high)).sum())
    saturated = float(np.isclose(np.abs(acts), high, atol=1e-6).mean())
    return Check(
        "actions_in_bounds",
        out == 0,
        f"{out} values outside [{low}, {high}]; {saturated:.2%} of values at the bound (informational)",
    )


def check_episode_ends(episodes: Sequence[Episode], max_steps: int = MAX_EPISODE_STEPS) -> Check:
    """Only the last step may be terminal or truncated; exactly one of them; truncation only at max_steps."""
    problems = []
    for ep in episodes:
        mid = ep.term[:-1].any() or ep.trunc[:-1].any()
        last_term, last_trunc = bool(ep.term[-1]), bool(ep.trunc[-1])
        if mid:
            problems.append((ep.id, "terminal/truncation flag before the last step"))
        elif last_term == last_trunc:
            problems.append((ep.id, f"last step term={last_term} trunc={last_trunc}"))
        elif last_trunc and ep.length != max_steps:
            problems.append((ep.id, f"truncated at {ep.length}, expected {max_steps}"))
        elif ep.length > max_steps:
            problems.append((ep.id, f"length {ep.length} > {max_steps}"))
    return Check(
        "episode_ends_well_formed", not problems, str(problems[:5]) if problems else "all well formed"
    )


def check_obs_bounds(episodes: Sequence[Episode], bound: float = 100.0) -> Check:
    worst = max(float(np.abs(ep.obs).max()) for ep in episodes)
    return Check("obs_within_health_bound", worst <= bound, f"max |obs| = {worst:.3f} (bound {bound})")


def check_termination_physics(episodes: Sequence[Episode]) -> Check:
    """Labels must agree with the health rule.

    Every observation after a non-terminal step must be healthy; the final observation of a
    terminated episode must be unhealthy. A mismatch means the done-mask would be wrong.
    """
    early_unhealthy, terminal_healthy = [], []
    for ep in episodes:
        next_obs = ep.obs[1:]
        not_last = ~ep.term
        if (~hopper_healthy(next_obs[not_last])).any():
            early_unhealthy.append(ep.id)
        if ep.term[-1] and hopper_healthy(ep.obs[-1]):
            terminal_healthy.append(ep.id)
    ok = not early_unhealthy and not terminal_healthy
    detail = (
        f"unhealthy before termination: {len(early_unhealthy)} episodes {early_unhealthy[:5]}; "
        f"healthy at termination: {len(terminal_healthy)} episodes {terminal_healthy[:5]}"
    )
    return Check("termination_matches_physics", ok, detail)


def check_no_duplicate_episodes(episodes: Sequence[Episode]) -> Check:
    digests = Counter(hashlib.sha256(ep.act.tobytes()).hexdigest() for ep in episodes)
    dupes = sum(n - 1 for n in digests.values() if n > 1)
    return Check("no_duplicate_episodes", dupes == 0, f"{dupes} duplicated episodes (by action hash)")


def run_all(
    episodes: Sequence[Episode], documented: dict[str, int], metadata: dict[str, int] | None = None
) -> list[Check]:
    return [
        check_counts(episodes, documented, metadata),
        check_unique_ids(episodes),
        check_shapes(episodes),
        check_finite(episodes),
        check_action_bounds(episodes),
        check_episode_ends(episodes),
        check_obs_bounds(episodes),
        check_termination_physics(episodes),
        check_no_duplicate_episodes(episodes),
    ]
