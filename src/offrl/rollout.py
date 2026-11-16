"""Run a policy for seeded episodes. Used for the random baseline (Phase 2) and eval (Phase 4)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


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
