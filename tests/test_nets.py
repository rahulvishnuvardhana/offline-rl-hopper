import torch

from offrl.nets import MLP, GaussianPolicy, TwinQ, ValueNet


def test_shapes():
    obs, act = torch.randn(7, 11), torch.rand(7, 3) * 2 - 1
    assert MLP(11, 5, (16, 16))(obs).shape == (7, 5)
    q1, q2 = TwinQ(11, 3, (16, 16))(obs, act)
    assert q1.shape == q2.shape == (7, 1)
    assert not torch.equal(q1, q2)  # independent heads
    assert ValueNet(11, (16, 16))(obs).shape == (7, 1)
    pi = GaussianPolicy(11, 3, (16, 16))
    assert pi.log_prob(obs, act).shape == (7,)


def test_act_is_deterministic_tanh_mean_in_bounds():
    pi = GaussianPolicy(11, 3, (16, 16))
    obs = 100 * torch.randn(64, 11)  # large inputs saturate tanh
    a = pi.act(obs)
    assert torch.equal(a, pi.act(obs))
    assert torch.equal(a, torch.tanh(pi.net(obs)).detach())
    assert a.abs().max() <= 1.0


def test_effective_log_std_is_clamped_raw_parameter_is_not():
    pi = GaussianPolicy(11, 3, (16, 16), log_std_min=-20.0, log_std_max=2.0)
    with torch.no_grad():
        pi.log_std.copy_(torch.tensor([5.0, -30.0, 0.5]))
    assert torch.equal(pi.effective_log_std(), torch.tensor([2.0, -20.0, 0.5]))
    assert pi.log_std[0] == 5.0  # clamp is applied in forward, as CORL
    std = pi(torch.zeros(1, 11)).stddev[0]
    torch.testing.assert_close(std, torch.tensor([2.0, -20.0, 0.5]).exp())


def test_bound_actions_have_finite_log_prob():
    pi = GaussianPolicy(11, 3, (16, 16))
    lp = pi.log_prob(torch.randn(4, 11), torch.tensor([[1.0, -1.0, 1.0]]).expand(4, 3))
    assert torch.isfinite(lp).all()
