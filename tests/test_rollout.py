import numpy as np

from offrl.rollout import run_policy


class FakeSpace:
    def __init__(self):
        self.rng = np.random.default_rng()

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def sample(self):
        return self.rng.uniform(-1, 1, size=3)


class FakeEnv:
    """Episode length = 3 + seed; even seeds fall (terminate), odd seeds hit the time limit."""

    def __init__(self):
        self.action_space = FakeSpace()

    def reset(self, seed):
        self.seed, self.t = seed, 0
        return np.zeros(11), {}

    def step(self, action):
        self.t += 1
        done = self.t == 3 + self.seed
        reward = float(np.sum(action))
        return np.zeros(11), reward, done and self.seed % 2 == 0, done and self.seed % 2 == 1, {}


def test_run_policy_counts_lengths_and_falls():
    env = FakeEnv()
    out = run_policy(env, lambda obs: np.ones(3), [0, 1, 2])
    np.testing.assert_array_equal(out["lengths"], [3, 4, 5])
    np.testing.assert_array_equal(out["returns"], [9.0, 12.0, 15.0])
    np.testing.assert_array_equal(out["fell"], [True, False, True])


def test_run_policy_seeds_action_space():
    env = FakeEnv()
    a = run_policy(env, lambda obs: env.action_space.sample(), [0, 1])
    b = run_policy(env, lambda obs: env.action_space.sample(), [0, 1])
    np.testing.assert_array_equal(a["returns"], b["returns"])
