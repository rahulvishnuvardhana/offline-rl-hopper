import numpy as np
import pytest
import torch

from offrl.data.buffer import Batch, ReplayBuffer
from offrl.data.normalize import StateNormalizer

N = 500


@pytest.fixture
def arrays():
    rng = np.random.default_rng(0)
    return {
        "obs": rng.normal(size=(N, 11)).astype(np.float32),
        "act": rng.uniform(-1, 1, size=(N, 3)).astype(np.float32),
        "rew": rng.uniform(0, 4, size=N).astype(np.float32),
        "next_obs": rng.normal(size=(N, 11)).astype(np.float32),
        "done": (rng.random(N) < 0.1).astype(np.float32),
    }


@pytest.fixture
def norm(arrays):
    obs = arrays["obs"].astype(np.float64)
    return StateNormalizer(obs.mean(0), obs.std(0))


def test_batch_contract(arrays, norm):
    batch = ReplayBuffer(arrays, norm, None, "cpu", seed=0).sample(64)
    assert isinstance(batch, Batch)
    shapes = {k: tuple(getattr(batch, k).shape) for k in ("obs", "act", "rew", "next_obs", "done")}
    assert shapes == {"obs": (64, 11), "act": (64, 3), "rew": (64, 1), "next_obs": (64, 11), "done": (64, 1)}
    assert all(getattr(batch, k).dtype == torch.float32 for k in shapes)
    assert set(batch.done.unique().tolist()) <= {0.0, 1.0}


def test_normalized_and_scaled_once(arrays, norm):
    raw = ReplayBuffer(arrays, norm, None, "cpu", seed=0)
    scaled = ReplayBuffer(arrays, norm, 0.25, "cpu", seed=0)
    np.testing.assert_allclose(raw.obs.numpy(), norm.transform(arrays["obs"]), rtol=1e-6)
    np.testing.assert_allclose(raw.rew.numpy()[:, 0], arrays["rew"], rtol=1e-6)  # BC: unscaled
    np.testing.assert_allclose(scaled.rew.numpy()[:, 0], arrays["rew"] * 0.25, rtol=1e-6)  # IQL
    np.testing.assert_allclose(raw.act.numpy(), arrays["act"])  # actions untouched


def test_every_array_comes_from_its_own_input(arrays, norm):
    """Traps a buffer that, e.g., normalizes obs into next_obs or drops done."""
    buf = ReplayBuffer(arrays, norm, None, "cpu", seed=0)
    np.testing.assert_allclose(buf.next_obs.numpy(), norm.transform(arrays["next_obs"]), rtol=1e-6)
    np.testing.assert_array_equal(buf.done.numpy()[:, 0], arrays["done"])


def test_from_mixture_uses_the_mixtures_own_statistics(arrays):
    stored = {
        **arrays,
        "state_mean": np.full(11, 2.0),
        "state_std": np.full(11, 4.0),
        "state_std_eps": np.array(1e-3),
        "reward_scale": np.array(0.5),
    }
    buf = ReplayBuffer.from_mixture(stored, scale_rewards=True, device="cpu", seed=0)
    np.testing.assert_allclose(buf.obs.numpy(), (arrays["obs"] - 2.0) / 4.001, rtol=1e-5)
    np.testing.assert_allclose(buf.rew.numpy()[:, 0], arrays["rew"] * 0.5, rtol=1e-6)
    bc = ReplayBuffer.from_mixture(stored, scale_rewards=False, device="cpu", seed=0)
    np.testing.assert_allclose(bc.rew.numpy()[:, 0], arrays["rew"], rtol=1e-6)


def test_same_seed_same_batches_other_seed_differs(arrays, norm):
    a = ReplayBuffer(arrays, norm, None, "cpu", seed=7)
    b = ReplayBuffer(arrays, norm, None, "cpu", seed=7)
    c = ReplayBuffer(arrays, norm, None, "cpu", seed=8)
    for _ in range(3):
        x, y, z = a.sample(32), b.sample(32), c.sample(32)
        assert torch.equal(x.obs, y.obs)
    assert not torch.equal(x.obs, z.obs)


def test_rows_stay_aligned(arrays, norm):
    """A sampled obs must come with its own act/rew/next_obs/done (one index for all five)."""
    buf = ReplayBuffer(arrays, norm, None, "cpu", seed=1)
    batch = buf.sample(50)
    idx = [int(torch.nonzero((buf.obs == o).all(1))[0]) for o in batch.obs]
    assert torch.equal(batch.act, buf.act[idx]) and torch.equal(batch.next_obs, buf.next_obs[idx])
    assert torch.equal(batch.rew, buf.rew[idx]) and torch.equal(batch.done, buf.done[idx])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CI and the laptop are CPU-only")
def test_cuda_matches_cpu_sequence(arrays, norm):
    cpu = ReplayBuffer(arrays, norm, None, "cpu", seed=3).sample(32)
    gpu = ReplayBuffer(arrays, norm, None, "cuda", seed=3).sample(32)
    assert torch.equal(cpu.obs, gpu.obs.cpu())
