"""Run a policy for seeded episodes. Used for the random baseline (Phase 2) and eval (Phase 4)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

# Seeds for every rollout we run. Both must stay disjoint from the datasets' collection seeds
# (offrl.data.leakage.COLLECTION_SEEDS); tests/test_leakage.py enforces it.
RANDOM_BASELINE_SEEDS = range(100)


def eval_seeds(eval_cfg: dict) -> range:
    """The evaluation seeds. Phase 4's evaluator must use this, not its own arithmetic."""
    return range(eval_cfg["seed_offset"], eval_cfg["seed_offset"] + eval_cfg["episodes"])


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
