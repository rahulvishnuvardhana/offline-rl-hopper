"""Shared fixtures. ``requires_data`` tests need the Minari datasets and skip without them (CI)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from offrl.config import load_checked
from offrl.data import mixture as mx
from offrl.data.load import dataset_dir

ROOT = Path(__file__).resolve().parents[1]
EXPERT_ID = "mujoco/hopper/expert-v0"


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_data: needs the Minari datasets (skipped in CI)")


def pytest_collection_modifyitems(config, items):
    if dataset_dir(EXPERT_ID).exists():
        return
    skip = pytest.mark.skip(reason="Minari datasets not downloaded")
    for item in items:
        if "requires_data" in item.keywords:
            item.add_marker(skip)


def synthetic_mixture(n: int = 2_000, seed: int = 0) -> dict[str, np.ndarray]:
    """A mixture-shaped dict: actions are a fixed function of obs plus small noise (learnable)."""
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(n, 11)).astype(np.float32)
    w = rng.normal(size=(11, 3)) / np.sqrt(11)
    act = np.clip(np.tanh(obs @ w) + 0.05 * rng.normal(size=(n, 3)), -1, 1).astype(np.float32)
    o64 = obs.astype(np.float64)
    return {
        "obs": obs,
        "act": act,
        "rew": rng.uniform(0, 4, n).astype(np.float32),
        "next_obs": rng.normal(size=(n, 11)).astype(np.float32),
        "done": (rng.random(n) < 0.01).astype(np.float32),
        "state_mean": o64.mean(0),
        "state_std": o64.std(0),
        "state_std_eps": np.array(1e-3),
        "reward_scale": np.array(0.25),
    }


@pytest.fixture
def make_mixture():
    return synthetic_mixture


@pytest.fixture
def small_run(tmp_path):
    """A fast config on a synthetic development mixture (seed 100) with its manifest.

    Returns ``make(*overrides, out=, algo=) -> cfg`` (``algo`` names ``configs/algo/<algo>.yaml``)
    and the manifest directory.
    """
    mix_dir, manifest_dir = tmp_path / "mixtures", tmp_path / "manifests"
    mix_dir.mkdir()
    manifest_dir.mkdir()

    def write_mixture(p: float, seed: int) -> None:
        arrays = synthetic_mixture(seed=seed)
        name = mx.mixture_name(p, seed)
        mx.save_mixture(mix_dir / f"{name}.npz", arrays)
        manifest = {"p_target": p, "seed": seed, "content_sha256": mx.content_sha256(arrays)}
        (manifest_dir / f"{name}.json").write_text(json.dumps(manifest), encoding="utf-8")

    def make(*extra: str, out: str = "runs", algo: str = "bc") -> dict:
        overrides = [
            "seed=100",
            "data.p_expert=1.0",
            f"data.mixture_dir={mix_dir.as_posix()}",
            f"train.out_dir={(tmp_path / out).as_posix()}",
            "train.steps=600",
            "train.batch_size=32",
            "train.log_every=50",
            "train.checkpoint_every=200",
            "train.device=cpu",
            "train.eval_every=100",
            "eval.episodes=2",
            "eval.final_episodes=3",
            "eval.final_score_last_n=2",
            "net.hidden=[32, 32]",
            *extra,
        ]
        cfg = load_checked(
            ROOT / "configs" / "base.yaml", ROOT / "configs" / "algo" / f"{algo}.yaml", overrides=overrides
        )
        write_mixture(cfg["data"]["p_expert"], cfg["seed"])
        return cfg

    return make, manifest_dir


@pytest.fixture
def f64():
    """float64 on the CPU with one thread, restored afterwards (Phase 5 exactness tests)."""
    dtype, threads = torch.get_default_dtype(), torch.get_num_threads()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    yield
    torch.set_default_dtype(dtype)
    torch.set_num_threads(threads)
