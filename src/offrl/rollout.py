"""Run a policy for seeded episodes. Used for the random baseline (Phase 2) and eval (Phase 4)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

# Seeds for every rollout we run. Both must stay disjoint from the datasets' collection seeds
# (offrl.data.leakage.COLLECTION_SEEDS); tests/test_leakage.py enforces it.
RANDOM_BASELINE_SEEDS = range(100)
# Evaluation seed offsets: base.yaml holds the development offset (Phases 4 to 6); the final
# grid (Phases 7, 8) overrides it with this one, so debugging never sees the final seeds.
FINAL_EVAL_SEED_OFFSET = 10_000
# Seeds reserved for the final grid. Development runs may not touch any of them.
FINAL_EVAL_RESERVED = range(FINAL_EVAL_SEED_OFFSET, FINAL_EVAL_SEED_OFFSET + 1_000)


def eval_seeds(eval_cfg: dict, n: int | None = None) -> range:
    """The evaluation seeds: ``n`` of them (default ``eval.episodes``) from ``eval.seed_offset``.
    The evaluator must use this, not its own arithmetic; larger ``n`` gives a superset."""
    n = eval_cfg["episodes"] if n is None else n
    return range(eval_cfg["seed_offset"], eval_cfg["seed_offset"] + n)


def run_policy(
    env, policy_fn: Callable[[np.ndarray], np.ndarray], seeds: Sequence[int]
) -> dict[str, np.ndarray]:
    """One episode per seed. Both ``env.reset`` and ``env.action_space`` are seeded, so a policy
    that samples from the action space is reproducible too.

    Returns arrays ``returns``, ``lengths`` and ``fell`` (ended by termination, not the time limit).
    """
    returns, lengths, fell = [], [], []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        env.action_space.seed(int(seed))
        total, steps, terminated, truncated = 0.0, 0, False, False
        while not (terminated or truncated):
            obs, reward, terminated, truncated, _ = env.step(policy_fn(obs))
            total += float(reward)
            steps += 1
        returns.append(total)
        lengths.append(steps)
        fell.append(bool(terminated))
    return {"returns": np.array(returns), "lengths": np.array(lengths), "fell": np.array(fell)}
