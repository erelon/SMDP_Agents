"""PPO and its average-reward (SMDP) variants — pure torch, agent-only.

The four agents share one PPO core (clipped surrogate + GAE) and differ ONLY in
how the long-run average reward ``eta`` is estimated, exactly mirroring the
tabular family (RLearning -> SMART / Harmonic via ``calc_new_rho``):

    PPO          discounted PPO, no average-reward correction (longrun=False)
    RsmartPPO    eta = EWMA(reward) / EWMA(time)        (Relaxed-SMART / APO)
    SmartPPO     eta = sum(reward) / sum(time)          (SMART)
    HarmonicPPO  eta = harmonic mean of reward/time     (Harmonic)

Defaults match the configuration used for this repo's results. ``time`` is the
per-step dwell (1.0 for an MDP; pass the macro-step duration for an SMDP).

You provide the env loop; the agents are env-agnostic (torch + numpy):

    agent = RsmartPPO(obs_dim, act_dim)
    buf = RolloutBuffer()
    obs = envs.reset()                                   # [B, obs_dim]
    for itr in range(n_itr):
        buf.clear()
        for t in range(agent.batch_T):
            action, value, logp = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = envs.step(action.numpy())
            buf.add(obs, action, reward, terminated, truncated, value, logp)
            obs = next_obs
        agent.update(buf, agent.value(obs))              # bootstrap from final obs
"""
import math

import numpy as np
import torch
from torch import optim

from .gaussian_mlp import GaussianMLP, gaussian_entropy, gaussian_logp


def _t(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


class RolloutBuffer:
    """Accumulates a time-major [T, B] on-policy batch, one ``add`` per step."""

    _FIELDS = ("obs", "act", "rew", "term", "trunc", "val", "logp", "time")

    def __init__(self):
        self.clear()

    def clear(self):
        self._d = {k: [] for k in self._FIELDS}

    def add(self, obs, action, reward, terminated, truncated, value, logp, time=1.0):
        vals = (obs, action, reward, terminated, truncated, value, logp, time)
        for k, v in zip(self._FIELDS, vals):
            self._d[k].append(torch.as_tensor(np.asarray(v)) if not torch.is_tensor(v)
                              else v.detach().cpu())

    def stacked(self, device):
        out = {}
        for k in self._FIELDS:
            seq = self._d[k]
            if k == "time" and all(t.ndim == 0 for t in seq):  # scalar dwell -> [T,B]
                seq = [t.expand_as(self._d["rew"][i]) for i, t in enumerate(seq)]
            out[k] = torch.stack([_t(t, device) for t in seq])
        return out


class PPO:
    """Discounted PPO. Base class for the average-reward variants below."""

    longrun = False  # subclasses flip this on to enable the eta correction

    def __init__(self, obs_dim, act_dim, hidden=(64, 64), init_log_std=0.0,
                 learning_rate=3e-4, lr_eta=0.1, rm_vbias_coeff=1.0,
                 value_loss_coeff=1.0, entropy_loss_coeff=0.01, clip_grad_norm=10.0,
                 discount=None, gae_lambda=0.95, epochs=10, minibatches=20,
                 ratio_clip=0.2, normalize_advantage=False, bootstrap_timelimit=True,
                 batch_T=200, device="cpu", seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        self.device = device
        self.net = GaussianMLP(obs_dim, act_dim, hidden, init_log_std).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate, foreach=True)
        # Average-reward variants default to no discounting; plain PPO to 0.99.
        self.discount = (1.0 if self.longrun else 0.99) if discount is None else discount
        self.gae_lambda = gae_lambda
        self.epochs, self.minibatches = epochs, minibatches
        self.ratio_clip = ratio_clip
        self.value_loss_coeff = value_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        self.clip_grad_norm = clip_grad_norm
        self.normalize_advantage = normalize_advantage
        self.bootstrap_timelimit = bootstrap_timelimit
        self.batch_T = batch_T
        # Off for plain PPO so the value-bias / eta terms vanish.
        self.lr_eta = lr_eta if self.longrun else 0.0
        self.rm_vbias_coeff = rm_vbias_coeff if self.longrun else 0.0
        self.eta = 0.0
        self.value_bias = None if self.longrun else 0.0

    # --- acting -------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs):
        """Sample an action. Returns (action, value, logp) as cpu tensors."""
        mu, log_std, value = self.net(_t(obs, self.device))
        action = mu + log_std.exp() * torch.randn_like(mu)
        logp = gaussian_logp(action, mu, log_std)
        return action.cpu(), value.cpu(), logp.cpu()

    @torch.no_grad()
    def eval_act(self, obs):
        """Deterministic (mean) action for evaluation."""
        return self.net(_t(obs, self.device))[0].cpu()

    @torch.no_grad()
    def value(self, obs):
        return self.net(_t(obs, self.device))[2].cpu()

    # --- average-reward estimator (the only thing the variants change) ------
    def update_eta(self, reward, value, time):
        """Refresh ``eta`` and ``value_bias`` from a fresh [T, B] batch."""
        if not self.longrun:
            return
        vm = value.mean().item()
        self.value_bias = vm if self.value_bias is None \
            else (1 - self.lr_eta) * self.value_bias + self.lr_eta * vm
        self.eta = self._estimate_eta(reward, time)

    def _estimate_eta(self, reward, time):
        # Relaxed-SMART / APO: ratio of EWMAs (== EWMA(reward) in the MDP case).
        rm, tm = reward.mean().item(), time.mean().item()
        if getattr(self, "_rho_r", None) is None:
            self._rho_r, self._rho_t = rm, tm
        else:
            self._rho_r = (1 - self.lr_eta) * self._rho_r + self.lr_eta * rm
            self._rho_t = (1 - self.lr_eta) * self._rho_t + self.lr_eta * tm
        return self._rho_r / max(self._rho_t, 1e-8)

    # --- update -------------------------------------------------------------
    def _gae(self, reward, value, nd, bootstrap_value, time):
        adv = torch.zeros_like(reward)
        nxt, gae = bootstrap_value, 0.0
        for t in reversed(range(reward.shape[0])):
            delta = reward[t] - self.eta * time[t] + self.discount * nxt * nd[t] - value[t]
            gae = delta + self.discount * self.gae_lambda * nd[t] * gae
            adv[t] = gae
            nxt = value[t]
        return adv, adv + value

    def update(self, buffer, bootstrap_value):
        """One PPO iteration over the collected [T, B] batch. Returns stats."""
        b = buffer.stacked(self.device)
        reward, value, time = b["rew"], b["val"], b["time"]
        nd = 1.0 - b["term"]  # terminations reset the return; truncations bootstrap

        self.update_eta(reward, value, time)
        adv, ret = self._gae(reward, value, nd, _t(bootstrap_value, self.device), time)
        ret = ret - self.rm_vbias_coeff * (self.value_bias or 0.0)

        valid = (1.0 - b["trunc"]) if self.bootstrap_timelimit else torch.ones_like(reward)
        if self.normalize_advantage:
            m = valid > 0
            adv = (adv - adv[m].mean()) / (adv[m].std() + 1e-6)

        # Flatten [T, B] -> [N] and run epochs x minibatches of SGD.
        obs = b["obs"].reshape(-1, b["obs"].shape[-1])
        act = b["act"].reshape(-1, b["act"].shape[-1])
        old_logp, adv, ret, valid = (x.reshape(-1) for x in (b["logp"], adv, ret, valid))
        n = obs.shape[0]
        mb = max(1, n // self.minibatches)
        stats = {"loss": [], "grad_norm": [], "entropy": []}
        for _ in range(self.epochs):
            for idx in torch.randperm(n, device=self.device).split(mb):
                mu, log_std, v = self.net(obs[idx])
                ratio = torch.exp(gaussian_logp(act[idx], mu, log_std) - old_logp[idx])
                clipped = torch.clamp(ratio, 1 - self.ratio_clip, 1 + self.ratio_clip)
                w = valid[idx]
                pi_loss = -_vmean(torch.min(ratio * adv[idx], clipped * adv[idx]), w)
                v_loss = self.value_loss_coeff * _vmean(0.5 * (v - ret[idx]) ** 2, w)
                entropy = _vmean(gaussian_entropy(log_std), w)
                loss = pi_loss + v_loss - self.entropy_loss_coeff * entropy

                self.optimizer.zero_grad()
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_grad_norm)
                self.optimizer.step()
                stats["loss"].append(loss.item())
                stats["grad_norm"].append(gn.item())
                stats["entropy"].append(entropy.item())
        return {"eta": self.eta, "value_bias": self.value_bias or 0.0,
                **{k: float(np.mean(v)) for k, v in stats.items()}}


def _vmean(x, valid):
    """Mean of x over valid (>0) entries; falls back to plain mean if none."""
    s = valid.sum()
    return (x * valid).sum() / s if s > 0 else x.mean()


class RsmartPPO(PPO):
    """APO / Relaxed-SMART: eta = EWMA(reward) / EWMA(time)."""
    longrun = True


class SmartPPO(PPO):
    """SMART: eta = sum(reward) / sum(time) (cumulative running average)."""
    longrun = True

    def _estimate_eta(self, reward, time):
        self._tot_r = getattr(self, "_tot_r", 0.0) + reward.sum().item()
        self._tot_t = getattr(self, "_tot_t", 0.0) + time.sum().item()
        return self._tot_r / max(self._tot_t, 1e-8)


class HarmonicPPO(PPO):
    """Harmonic-mean estimator over the positive/negative reward streams.

    ``weighted=True`` uses w=reward (WeightedHarmonic); False uses w=1 (Harmonic).
    """
    longrun = True

    def __init__(self, *args, weighted=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.weighted = weighted
        self._pr = self._nr = self._pw1 = self._nw1 = self._pw2 = self._nw2 = self._zw = 0.0

    def _estimate_eta(self, reward, time):
        a = self.lr_eta
        for r, t in zip(reward.reshape(-1).tolist(), time.reshape(-1).tolist()):
            pos, neg, zero = float(r > 0), float(r < 0), float(r == 0)
            w = r if self.weighted else 1.0
            rate = 0.0 if zero else t / r
            self._pr = (1 - a) * self._pr + a * rate * pos * w
            self._pw1 = (1 - a) * self._pw1 + a * pos * w
            self._pw2 = (1 - a) * self._pw2 + a * pos
            self._nr = (1 - a) * self._nr + a * rate * neg * w
            self._nw1 = (1 - a) * self._nw1 + a * neg * w
            self._nw2 = (1 - a) * self._nw2 + a * neg
            self._zw = (1 - a) * self._zw + a * zero
        h_pos = 0.0 if self._pr == 0 else self._pw1 / self._pr
        h_neg = 0.0 if self._nr == 0 else self._nw1 / self._nr
        denom = self._pw2 + self._nw2 + self._zw
        return 0.0 if denom == 0 else (h_pos * self._pw2 + h_neg * self._nw2) / denom
