"""Evaluation harness (spec Section 5): seeded episodes in the recovered Hopper-v5 env, scored by ADR-6.

Seeds come only from ``offrl.rollout.eval_seeds``; episodes only from ``offrl.rollout.run_policy``.
Observations are normalized with the buffer's own normalizer object, never a second copy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from offrl.algos.base import Trainer
from offrl.analysis.eda import normalized_score
from offrl.data.normalize import StateNormalizer
from offrl.rollout import eval_seeds, run_policy

PolicyFn = Callable[[np.ndarray], np.ndarray]


def make_env(dataset_id: str):
    """Hopper-v5 with the collection kwargs, recovered from the dataset."""
    import minari  # local import: CI tests pass their own env

    return minari.load_dataset(dataset_id).recover_environment()


def deterministic_policy(trainer: Trainer, normalizer: StateNormalizer, device: torch.device) -> PolicyFn:
    def act(obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(normalizer.transform(obs[None]), device=device)
        return trainer.act(x)[0].cpu().numpy().astype(np.float64)

    return act


def sampled_policy(
    policy: torch.nn.Module, normalizer: StateNormalizer, device: torch.device, generator: torch.Generator
) -> PolicyFn:
    """Descriptive diagnostic only: samples ``N(mean, std)`` with a dedicated CPU generator, so no
    other random stream moves. Not clipped: the env clips to its action box."""

    def act(obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(normalizer.transform(obs[None]), device=device)
        with torch.no_grad():
            dist = policy(x)
            noise = torch.randn(dist.mean.shape, generator=generator).to(device)
            return (dist.mean + dist.stddev * noise)[0].cpu().numpy().astype(np.float64)

    return act


def summarize(
    out: dict[str, np.ndarray], score_cfg: dict[str, float], mask: np.ndarray | None = None
) -> dict[str, float]:
    sel = slice(None) if mask is None else mask
    returns = out["returns"][sel]
    return {
        "n_episodes": len(returns),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "score_mean": normalized_score(float(returns.mean()), score_cfg["r_random"], score_cfg["r_expert"]),
        "fall_rate": float(out["fell"][sel].mean()),
        "length_mean": float(out["lengths"][sel].mean()),
    }


def evaluate(
    env, policy_fn: PolicyFn, cfg: dict[str, Any], n: int | None = None
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """``n`` episodes (default ``eval.episodes``) on ``eval_seeds``. Returns the summary, with the
    first ``eval.episodes`` seeds summarized again as ``sub_*`` (every evaluation then has a
    value on the same 10 seeds), and one row per episode."""
    seeds = eval_seeds(cfg["eval"], n)
    out = run_policy(env, policy_fn, seeds)
    summary = summarize(out, cfg["score"])
    n_sub = cfg["eval"]["episodes"]
    sub = summarize(out, cfg["score"], np.arange(len(seeds)) < n_sub)
    summary.update({f"sub_{k}": v for k, v in sub.items()})
    rows = [
        {"seed": int(s), "return": float(r), "length": int(ln), "fell": bool(f)}
        for s, r, ln, f in zip(seeds, out["returns"], out["lengths"], out["fell"], strict=True)
    ]
    return summary, rows
