"""Phase 0 smoke test: versions, device, Hopper-v5 stepping, optional dataset check.

Usage:
    python scripts/smoke.py              # env + device only
    python scripts/smoke.py --datasets   # also download/load both Minari datasets
"""

from __future__ import annotations

import argparse
import os
import platform
import time

import gymnasium as gym
import minari
import mujoco
import numpy as np
import torch

DATASETS = {
    "mujoco/hopper/expert-v0": (999_164, 1_086),
    "mujoco/hopper/simple-v0": (999_206, 1_952),
}


def check_device(require_gpu: bool) -> None:
    print(
        f"python {platform.python_version()} | torch {torch.__version__} | "
        f"mujoco {mujoco.__version__} | gymnasium {gym.__version__} | minari {minari.__version__}"
    )
    if require_gpu and not torch.cuda.is_available():
        raise SystemExit(
            "FAIL: --require-gpu set but torch cannot use CUDA (check driver vs. torch CUDA build)"
        )
    if torch.cuda.is_available():
        print(f"device: cuda | {torch.cuda.get_device_name(0)}")
        x = torch.randn(4096, 4096, device="cuda")
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(10):
            x @ x
        torch.cuda.synchronize()
        print(f"10 matmuls 4096^2: {time.perf_counter() - t:.3f}s")
    else:
        print("device: cpu (no CUDA)")


def step_hopper(n_steps: int = 1_000) -> None:
    env = gym.make("Hopper-v5")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (11,), obs.shape
    resets, t = 0, time.perf_counter()
    for _ in range(n_steps):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, _ = env.reset()
            resets += 1
    env.close()
    dt = time.perf_counter() - t
    print(f"Hopper-v5: {n_steps} steps OK ({resets} resets, {n_steps / dt:,.0f} steps/s)")


def check_datasets() -> None:
    print(f"MINARI_DATASETS_PATH={os.environ.get('MINARI_DATASETS_PATH', '(default ~/.minari/datasets)')}")
    for dataset_id, (steps, episodes) in DATASETS.items():
        ds = minari.load_dataset(dataset_id, download=True)
        ok = ds.total_steps == steps and ds.total_episodes == episodes
        print(
            f"{dataset_id}: {ds.total_steps:,} steps, {ds.total_episodes:,} episodes "
            f"[{'OK' if ok else 'MISMATCH'}]"
        )
        assert ok, f"expected {steps} steps / {episodes} episodes"
        ep = next(ds.iterate_episodes())
        n = len(ep.actions)
        assert ep.observations.shape == (n + 1, 11), ep.observations.shape
        assert np.isfinite(ep.observations).all()
        env = ds.recover_environment()
        print(f"  recovered env: {env.spec.id}, max_episode_steps={env.spec.max_episode_steps}")
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", action="store_true", help="download and check Minari datasets")
    parser.add_argument("--require-gpu", action="store_true", help="fail if CUDA is unavailable")
    args = parser.parse_args()
    print(f"host: {platform.node()}")
    check_device(args.require_gpu)
    step_hopper()
    if args.datasets:
        check_datasets()
    print("SMOKE OK")


if __name__ == "__main__":
    main()
