"""Policy heads: the only part of the PPO-family agents that knows the action space.

An agent is assembled from three independent choices, each supplied by
inheritance:

    head        how actions are parameterised   (Gaussian / Categorical)
    rate        how the long-run rate rho is estimated (R-Learning, SMART, ...)
    core        the policy-optimisation algorithm (PPO, SMAPO)

A head owns the network and every distribution operation the core needs, so the
core never mentions a Gaussian or a softmax. Swapping continuous for discrete
control is then a change of base class and nothing else.

Both heads return ``(params, value)`` from ``forward_net``: ``params`` is
whatever that distribution needs (``(mu, log_std)`` or ``logits``) and is passed
straight back into the head's own ``logp``/``entropy``/``sample``.
"""
import torch
import torch.nn as nn

from .gaussian_mlp import (GaussianMLP, gaussian_entropy, gaussian_logp, mlp)


class GaussianHead:
    """Diagonal-Gaussian policy over a continuous action space."""

    #: Trailing shape of one stored action, for the rollout buffer.
    def action_shape(self, act_dim):
        return (act_dim,)

    def build_net(self, obs_dim, act_dim, hidden, init_log_std):
        return GaussianMLP(obs_dim, act_dim, hidden, init_log_std)

    def forward_net(self, obs):
        mu, log_std, value = self.net(obs)
        return (mu, log_std), value

    def sample(self, params):
        mu, log_std = params
        action = mu + log_std.exp() * torch.randn_like(mu)
        return action, gaussian_logp(action, mu, log_std)

    def mode(self, params):
        """The deterministic action used for evaluation: the distribution mean."""
        return params[0]

    def logp(self, params, action):
        return gaussian_logp(action, *params)

    def entropy(self, params):
        return gaussian_entropy(params[1])

    def store_action(self, action):
        return action

    def load_action(self, action):
        return action


class CategoricalMLP(nn.Module):
    """Actor-critic for a discrete action space: logits + state value."""

    def __init__(self, obs_dim, n_actions, hidden=(64, 64)):
        super().__init__()
        self.logits = mlp(obs_dim, hidden, n_actions)
        self.v = mlp(obs_dim, hidden, 1)

    def forward(self, obs):
        return self.logits(obs), self.v(obs).squeeze(-1)


class CategoricalHead:
    """Softmax policy over ``act_dim`` discrete actions.

    ``act_dim`` is the NUMBER OF ACTIONS, not a vector width. Actions are stored
    as scalars and cast back to ``long`` on use, because the rollout buffer keeps
    one float tensor per field; float32 represents every integer up to 2**24
    exactly, so an action index survives the round trip intact.
    """

    def action_shape(self, act_dim):
        return ()

    def build_net(self, obs_dim, act_dim, hidden, init_log_std):
        return CategoricalMLP(obs_dim, act_dim, hidden)

    def forward_net(self, obs):
        logits, value = self.net(obs)
        return logits, value

    @staticmethod
    def _dist(logits):
        return torch.distributions.Categorical(logits=logits)

    def sample(self, params):
        d = self._dist(params)
        action = d.sample()
        return action, d.log_prob(action)

    def mode(self, params):
        """Deterministic evaluation is the ARGMAX. Sampling here would report a
        worse policy than the one that was trained."""
        return params.argmax(-1)

    def logp(self, params, action):
        return self._dist(params).log_prob(action.long())

    def entropy(self, params):
        return self._dist(params).entropy()

    def store_action(self, action):
        return action.to(torch.float32)

    def load_action(self, action):
        return action.long()
