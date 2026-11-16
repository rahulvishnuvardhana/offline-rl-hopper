"""Leakage checks between the two datasets and between the data and evaluation.

Both datasets were collected with ``env.reset(seed=COLLECTION_SEED_OFFSET + i)`` for episode
i (verified on the data by ``check_collection_seeds``). MuJoCo is deterministic, so an
evaluation episode reset with one of those seeds starts from a training state, and a policy
that memorized that trajectory could replay it. Evaluation seeds must avoid the range.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from offrl.data.load import Episode
from offrl.data.validate import Check

COLLECTION_SEED_OFFSET = 123
# Inclusive ranges, from the episode counts: 1,086 expert and 1,952 simple episodes.
COLLECTION_SEEDS = {
    "mujoco/hopper/expert-v0": range(123, 123 + 1_086),
    "mujoco/hopper/simple-v0": range(123, 123 + 1_952),
}


def _digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def check_collection_seeds(
    episodes: Sequence[Episode], reset_obs: Callable[[int], np.ndarray], seeds: range
) -> Check:
    """Episode i's first observation equals ``reset_obs(seeds[i])`` exactly, for every i."""
    if len(seeds) != len(episodes):
        return Check("collection_seeds_verified", False, f"{len(seeds)} seeds for {len(episodes)} episodes")
    bad = [
        ep.id for ep, s in zip(episodes, seeds, strict=True) if not np.array_equal(ep.obs[0], reset_obs(s))
    ]
    detail = f"episode i starts at reset(seed={seeds.start}+i) for all {len(episodes)}"
    return Check("collection_seeds_verified", not bad, f"mismatched episodes: {bad[:10]}" if bad else detail)


def check_seeds_disjoint(name: str, seeds: Iterable[int], collection: Iterable[range]) -> Check:
    """Seeds used for rollouts (evaluation, random baseline) never reuse a collection seed."""
    used, ranges = set(seeds), list(collection)
    clash = sorted(s for s in used if any(s in r for r in ranges))
    detail = f"{len(clash)} overlapping seeds {clash[:10]}" if clash else f"{len(used)} seeds, no overlap"
    return Check(f"{name}_seeds_disjoint_from_collection", not clash, detail)


def check_no_cross_duplicates(a: Sequence[Episode], b: Sequence[Episode]) -> Check:
    """No episode appears in both datasets (by action hash or by observation hash)."""
    acts = {_digest(ep.act) for ep in a} & {_digest(ep.act) for ep in b}
    obs = {_digest(ep.obs) for ep in a} & {_digest(ep.obs) for ep in b}
    return Check(
        "no_episode_in_both_datasets",
        not acts and not obs,
        f"{len(acts)} shared by actions, {len(obs)} shared by observations",
    )


def check_shared_states_are_initial(a: Sequence[Episode], b: Sequence[Episode]) -> Check:
    """Exact state rows present in both datasets may only be initial states (same reset seeds)."""
    initial = {_digest(ep.obs[0]) for ep in a} | {_digest(ep.obs[0]) for ep in b}
    rows_a = {_digest(r) for ep in a for r in ep.obs[:-1]}
    rows_b = {_digest(r) for ep in b for r in ep.obs[:-1]}
    shared = rows_a & rows_b
    later = shared - initial
    return Check(
        "shared_states_only_initial",
        not later,
        f"{len(shared)} exact shared states, {len(later)} of them not initial states",
    )
