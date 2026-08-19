"""PPO and its average-reward (SMDP) variants — pure torch, agent-only.

One PPO core (clipped surrogate + GAE) plus an average-reward correction whose
long-run rate ``rho`` is estimated by REUSING the tabular agents' ``calc_new_rho``
via multiple inheritance — so the rho logic lives in exactly one place and is
shared by the tabular and deep agents alike:

    PPO          discounted PPO, no correction        (longrun=False)
    RsmartPPO    (PPO, RelaxedSMART)  rho = EWMA(reward)/EWMA(time)   [APO]
    SmartPPO     (PPO, SMART)         rho = Σreward/Σtime
    HarmonicPPO  (PPO, Harmonic)      rho = harmonic mean of reward/time

Adding a variant is one line — inherit the deep core and a tabular rho agent,
and pick how the batch feeds the rho updater (``rho_reduce``):

    class WeightedHarmonicPPO(PPO, WeightedHarmonic):
        longrun = True
        rho_reduce = "none"          # per-transition (needs each reward's sign)

``rho`` is updated each batch via the inherited ``calc_new_rho``: Rsmart/Smart
aggregate the batch (mean / sum -> one O(1) call), Harmonic iterates per
transition. ``time`` is the per-step dwell (1.0 = MDP, macro-step duration =
SMDP). Defaults match this project's configuration.

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
import numpy as np
import torch
from torch import optim

from .base import Agent
from .gaussian_mlp import GaussianMLP, gaussian_entropy, gaussian_logp
from .harmonic_r import Harmonic
from .smart_r import SMART
from .relaxed_smart import RelaxedSMART


def _t(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


def _vmean(x, valid):
    """Mean of x over valid (>0) entries; plain mean if none are valid."""
    s = valid.sum()
    return (x * valid).sum() / s if s > 0 else x.mean()


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
            self._d[k].append(v.detach().cpu() if torch.is_tensor(v)
                              else torch.as_tensor(np.asarray(v)))

    def stacked(self, device):
        out = {}
        for k in self._FIELDS:
            seq = self._d[k]
            if k == "time" and all(t.ndim == 0 for t in seq):  # scalar dwell -> [T,B]
                seq = [t.expand_as(self._d["rew"][i]) for i, t in enumerate(seq)]
            out[k] = torch.stack([_t(t, device) for t in seq])
        return out


class PPO(Agent):
    """Discounted PPO + the deep-RL machinery. Base for the average-reward
    variants, which add the ``rho`` correction by also inheriting a tabular
    rho agent (see module docstring)."""

    longrun = False  # variants flip this on to enable the rho correction

    def __init__(self, obs_dim, act_dim, hidden=(64, 64), init_log_std=0.0,
                 learning_rate=3e-4, rho_lr=0.1, rm_vbias_coeff=1.0,
                 value_loss_coeff=1.0, entropy_loss_coeff=0.01, clip_grad_norm=10.0,
                 discount=None, gae_lambda=0.95, epochs=10, minibatches=20,
                 ratio_clip=0.2, normalize_advantage=False, bootstrap_timelimit=True,
                 batch_T=200, device="cpu", seed=None):
        # Initialise the inherited (tabular) rho machinery — calc_new_rho, the
        # per-variant accumulators and self.rho. For plain PPO this just runs
        # Agent.__init__. action_space is unused by the deep agents (placeholder
        # satisfies Agent's check); rho_lr maps to the tabular rho_learning_rate.
        super().__init__(name=type(self).__name__, action_space=[0],
                         seed=42 if seed is None else seed,
                         learning_rate=learning_rate, rho_learning_rate=rho_lr)
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
        self.rho_lr = rho_lr
        self.rho_learning_rate = rho_lr  # used by the inherited calc_new_rho
        self.rho = 0.0
        # Off for plain PPO so the value-bias / rho terms vanish.
        self.rm_vbias_coeff = rm_vbias_coeff if self.longrun else 0.0
        self.value_bias = None if self.longrun else 0.0

    # --- acting -------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs):
        """Sample an action. Returns (action, value, logp) as cpu tensors."""
        mu, log_std, value = self.net(_t(obs, self.device))
        action = mu + log_std.exp() * torch.randn_like(mu)
        return action.cpu(), value.cpu(), gaussian_logp(action, mu, log_std).cpu()

    @torch.no_grad()
    def eval_act(self, obs):
        """Deterministic (mean) action for evaluation."""
        return self.net(_t(obs, self.device))[0].cpu()

    @torch.no_grad()
    def value(self, obs):
        return self.net(_t(obs, self.device))[2].cpu()

    # --- average-reward rate (delegated to the inherited tabular calc_new_rho)
    #
    # ``rho_reduce`` controls how the [T, B] batch feeds the tabular updater:
    #   "mean" -> one call with the batch means  (per-batch EWMA, e.g. Rsmart/APO)
    #   "sum"  -> one call with the batch sums    (cumulative, e.g. SMART)
    #   "none" -> one call per transition         (needed by Harmonic's pos/neg
    #                                              split; matches the source)
    # Aggregating ("mean"/"sum") keeps the update O(1) per batch instead of
    # O(T*B) Python calls.
    rho_reduce = "mean"

    def update_rho(self, reward, value, time):
        """Refresh ``rho`` and ``value_bias`` from a fresh [T, B] batch."""
        if not self.longrun:
            return
        vm = value.mean().item()
        self.value_bias = vm if self.value_bias is None \
            else (1 - self.rho_lr) * self.value_bias + self.rho_lr * vm
        r, t = reward.reshape(-1), time.reshape(-1)
        if self.rho_reduce == "none":
            for ri, ti in zip(r.tolist(), t.tolist()):
                self.calc_new_rho(ri, ti, None, None)
        elif self.rho_reduce == "sum":
            self.calc_new_rho(r.sum().item(), t.sum().item(), None, None)
        else:  # "mean"
            self.calc_new_rho(r.mean().item(), t.mean().item(), None, None)

    # --- update -------------------------------------------------------------
    def _gae(self, reward, value, nd, bootstrap_value, time):
        adv = torch.zeros_like(reward)
        nxt, gae = bootstrap_value, 0.0
        for t in reversed(range(reward.shape[0])):
            delta = reward[t] - self.rho * time[t] + self.discount * nxt * nd[t] - value[t]
            gae = delta + self.discount * self.gae_lambda * nd[t] * gae
            adv[t] = gae
            nxt = value[t]
        return adv, adv + value

    def update(self, buffer, bootstrap_value):
        """One PPO iteration over the collected [T, B] batch. Returns stats."""
        b = buffer.stacked(self.device)
        reward, value, time = b["rew"], b["val"], b["time"]
        nd = 1.0 - b["term"]  # terminations reset the return; truncations bootstrap

        self.update_rho(reward, value, time)
        adv, ret = self._gae(reward, value, nd, _t(bootstrap_value, self.device), time)
        ret = ret - self.rm_vbias_coeff * (self.value_bias or 0.0)

        valid = (1.0 - b["trunc"]) if self.bootstrap_timelimit else torch.ones_like(reward)
        if self.normalize_advantage:
            m = valid > 0
            adv = (adv - adv[m].mean()) / (adv[m].std() + 1e-6)

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
        return {"rho": self.rho, "value_bias": self.value_bias or 0.0,
                **{k: float(np.mean(v)) for k, v in stats.items()}}


class RsmartPPO(PPO, RelaxedSMART):
    """APO / Relaxed-SMART: rho = EWMA(reward) / EWMA(time), per batch."""
    longrun = True
    rho_reduce = "mean"


class SmartPPO(PPO, SMART):
    """SMART: rho = Σreward / Σtime (cumulative running average)."""
    longrun = True
    rho_reduce = "sum"


class HarmonicPPO(PPO, Harmonic):
    """Harmonic-mean rho over the positive/negative reward streams."""
    longrun = True
    rho_reduce = "none"  # per-transition: the pos/neg split needs each reward
