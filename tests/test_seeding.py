import random

import numpy as np
import torch

from offrl.seeding import seed_everything


def _draw(seed: int):
    rng = seed_everything(seed)
    return random.random(), rng.normal(size=3), torch.randn(3)


def test_same_seed_same_draws():
    a, b = _draw(7), _draw(7)
    assert a[0] == b[0]
    np.testing.assert_array_equal(a[1], b[1])
    assert torch.equal(a[2], b[2])


def test_different_seed_different_draws():
    assert not torch.equal(_draw(0)[2], _draw(1)[2])
