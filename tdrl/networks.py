"""MLP actors and critics shared by every method."""
import torch
import torch.nn as nn

ACTS = {"relu": nn.ReLU, "tanh": nn.Tanh}


def mlp(inp, out, hidden=(256, 256), act="relu", norm=None, input_norm=False):
    """Linear stack. norm in {None, 'ln', 'bn'} is inserted after each hidden Linear.
    The final layer is a plain Linear and is always the last module (stabilisers rely on this)."""
    layers, d = [], inp
    if input_norm:  # CrossQ also normalises the input
        layers.append(nn.BatchNorm1d(inp, momentum=0.01))
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if norm == "ln":
            layers.append(nn.LayerNorm(h))
        elif norm == "bn":
            layers.append(nn.BatchNorm1d(h, momentum=0.01))
        layers.append(ACTS[act]())
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


def linear_layers(net):
    """All Linear layers in order; the last one is the output layer."""
    return [m for m in net.modules() if isinstance(m, nn.Linear)]


class Actor(nn.Module):
    """Deterministic policy mu(s) in [-1, 1]^d."""

    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.net = mlp(obs_dim, act_dim, hidden)

    def forward(self, s):
        return torch.tanh(self.net(s))


class GaussianActor(nn.Module):
    """Tanh-squashed Gaussian policy (SAC / CrossQ)."""

    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim, hidden)

    def dist(self, s):
        mu, log_std = self.net(s).chunk(2, -1)
        log_std = -5 + 3.5 * (torch.tanh(log_std) + 1)  # in [-5, 2], as in CleanRL
        return mu, log_std.exp()

    def sample(self, s):
        mu, std = self.dist(s)
        u = mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        logp = torch.distributions.Normal(mu, std).log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)
        return a, logp.sum(-1, keepdim=True)

    def forward(self, s):  # deterministic action for evaluation
        return torch.tanh(self.dist(s)[0])


class Critic(nn.Module):
    """Q(s, a); n independent heads stacked on dim 0 -> shape (n, B, 1)."""

    def __init__(self, obs_dim, act_dim, n=1, hidden=(256, 256), act="relu", norm=None, input_norm=False):
        super().__init__()
        self.qs = nn.ModuleList(mlp(obs_dim + act_dim, 1, hidden, act, norm, input_norm) for _ in range(n))

    def forward(self, s, a):
        x = torch.cat([s, a], -1)
        return torch.stack([q(x) for q in self.qs])
