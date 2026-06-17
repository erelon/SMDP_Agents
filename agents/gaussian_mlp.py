"""Gaussian feed-forward actor-critic used by the PPO-family agents.

Mirrors the MuJoCo policy used to produce this repo's results: an MLP for the
action mean ([obs->64->64->act], Tanh hidden + Tanh-squashed mean), a separate
MLP for the state value ([obs->64->64->1]), and a single state-independent
learnable ``log_std`` vector. torch + numpy only.
"""
import math

import torch
import torch.nn as nn

_LOG2PI = math.log(2.0 * math.pi)


def mlp(in_size, hidden, out_size, nonlinearity=nn.Tanh):
    """[in -> hidden... -> out] with a linear last layer."""
    sizes, layers = [in_size] + list(hidden), []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nonlinearity()]
    layers += [nn.Linear(sizes[-1], out_size)]
    return nn.Sequential(*layers)


class GaussianMLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(64, 64), init_log_std=0.0):
        super().__init__()
        self.mu = nn.Sequential(mlp(obs_dim, hidden, act_dim), nn.Tanh())
        self.v = mlp(obs_dim, hidden, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))

    def forward(self, obs):
        mu = self.mu(obs)
        value = self.v(obs).squeeze(-1)
        return mu, self.log_std.expand_as(mu), value


def gaussian_logp(action, mu, log_std):
    """Diagonal-Gaussian log-likelihood, summed over action dims."""
    z = (action - mu) / log_std.exp()
    return -(log_std + 0.5 * z ** 2).sum(-1) - 0.5 * action.shape[-1] * _LOG2PI


def gaussian_entropy(log_std):
    """Diagonal-Gaussian entropy, summed over action dims."""
    return (log_std + 0.5 * (_LOG2PI + 1.0)).sum(-1)
