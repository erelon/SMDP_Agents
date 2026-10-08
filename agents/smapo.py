"""SMAPO — average-reward policy optimisation for semi-Markov decision processes.

SMAPO optimises the long-run reward RATE rather than a discounted return. It is
PPO's clipped surrogate with three changes, and nothing else:

1. **No discounting.** ``gamma = 1``. The TD residual carries the average-reward
   correction ``r - rho*tau``, where ``tau`` is the decision's holding time and
   ``rho`` is an estimate of the long-run rate.

2. **A-centering.** The batch advantage is centred on its own mean before the
   optimisation epochs. Any estimate of ``rho`` carries tracking error, and that
   error enters every advantage in the batch as a SHARED offset. Zero-mean noise
   averages out over an epoch of minibatches; a shared offset does not — it
   accumulates linearly in the update while the noise grows as its square root.
   Subtracting the batch mean removes it without touching the ranking of actions,
   which is all the policy gradient uses.

3. **Calibrated entropy pressure.** The entropy bonus ``c_H`` is specified as a
   RATIO to the advantage scale rather than as an absolute coefficient (see
   ``_update_entropy_coeff``).

``rho`` is not SMAPO's business: it is supplied by inheriting one of the rate
estimators, so the algorithm and the estimator vary independently::

    APO                   tau is ignored (tau == 1); rho = EWMA(reward)
    RsmartSMAPO           rho = EWMA(reward) / EWMA(tau)
    SmartSMAPO            rho = sum(reward) / sum(tau)
    SmoothedSmartSMAPO    rho smoothed in ELAPSED TIME, not per transition

The action space is supplied the same way, by a policy head, so each variant has
a discrete counterpart that differs only in its base classes::

    class DiscreteRsmartSMAPO(CategoricalHead, RsmartSMAPO):
        pass

You provide the environment loop; the agents are env-agnostic (torch + numpy)::

    agent = RsmartSMAPO(obs_dim, act_dim)
    buf = RolloutBuffer()
    ...
    buf.add(o, a, r, terminated, truncated, v, logp, time=tau)
    stats = agent.update(buf, bootstrap_value)

``time`` is the per-decision holding time: 1.0 for an MDP, the macro-step
duration for an SMDP. Passing it is what makes the agent average-REWARD-RATE
rather than average-reward-per-decision; leaving it at its default silently
turns every rate estimator into its MDP special case.
"""
import math

import torch

from .policy_heads import CategoricalHead
from .ppo import PPO
from .relaxed_smart import RelaxedSMART
from .smart_r import SMART, SmoothedSMART


class SMAPO(PPO):
    """Average-reward PPO with A-centering and calibrated entropy pressure.

    Subclass it together with a rate estimator; see the module docstring.
    """

    longrun = True

    def __init__(self, obs_dim, act_dim, entropy_pressure=0.02,
                 entropy_warmup_iters=20, pressure_track=True,
                 pressure_track_steps=2_000_000, pressure_boost=True,
                 boost_efold_steps=400_000, boost_frac=0.30, boost_max=8.0,
                 **kwargs):
        """
        entropy_pressure
            The target ratio ``p = c_H / std(A)``. This is the hyperparameter to
            search; it is environment-specific and we make no claim of a
            universal value.
        entropy_warmup_iters
            Iterations used to measure the advantage scale before the bonus is
            switched on. ``c_H`` is whatever ``entropy_loss_coeff`` was (0 by
            default) until the measurement completes.
        pressure_track, pressure_track_steps
            Re-derive ``c_H`` from a slow estimate of the advantage scale, so the
            delivered ratio stays at ``p`` as the scale moves. The time constant
            is in ENVIRONMENT STEPS, not iterations: how many iterations fall in
            a given span of simulated time depends on the dwell regime, so an
            iteration-based rate would mean different things in different
            environments.
        pressure_boost, boost_efold_steps, boost_frac, boost_max
            Raise ``c_H`` above the tracked value while the advantage scale sits
            below ``boost_frac`` of its own running peak, which is the signal
            that exploration is starving. The trigger is RELATIVE to the run's
            own history because absolute set-points did not transfer across
            environments. The boost is suppressed while policy entropy is already
            rising, floored at 1 (it only ever adds) and capped at ``boost_max``.
        """
        kwargs.setdefault("entropy_loss_coeff", 0.0)
        super().__init__(obs_dim, act_dim, **kwargs)
        self.entropy_pressure = float(entropy_pressure)
        self.entropy_warmup_iters = int(entropy_warmup_iters)
        self.pressure_track = bool(pressure_track)
        self.pressure_track_steps = float(pressure_track_steps)
        self.pressure_boost = bool(pressure_boost)
        self.boost_efold_steps = float(boost_efold_steps)
        self.boost_frac = float(boost_frac)
        self.boost_max = float(boost_max)

        self.adv_std = float("nan")          # advantage scale of the last batch
        self.realised_pressure = float("nan")  # c_H / std(A), the delivered ratio
        self.boost = 1.0
        self._warmup = []
        self._adv_std_frozen = None
        self._scale_ema = None
        self._scale_peak = 0.0
        self._log_boost = 0.0
        self._entropy_ema = None
        self._entropy_slope = 0.0

    # --- holding time -------------------------------------------------------
    def dwell(self, time):
        """The holding time the algorithm attributes to each decision.

        Overridden by :class:`APO`, which ignores duration entirely.
        """
        return time

    def rate_residual(self, reward, time):
        return reward - self.rho * self.dwell(time)

    def update_rho(self, reward, value, time):
        return super().update_rho(reward, value, self.dwell(time))

    # --- advantage conditioning --------------------------------------------
    def shape_advantage(self, adv, valid, batch):
        """Measure the advantage scale, set ``c_H`` from it, then A-centre."""
        m = valid > 0
        a = adv[m]
        if a.numel() < 2:
            return adv
        self.adv_std = float(a.std().item())
        self._update_entropy_coeff(batch)
        return adv - a.mean()

    # --- calibrated entropy pressure ---------------------------------------
    def _update_entropy_coeff(self, batch):
        """Set ``c_H`` so that ``c_H / std(A)`` tracks the configured ratio ``p``.

        The entropy bonus and the policy-gradient term act on the same quantity —
        the width of the policy — but in different units. The objective's force
        on ``log sigma`` scales with the curvature of the advantage landscape,
        which carries the task's reward units and moves during training; the
        bonus contributes a constant ``c_H``. Under a shared gradient-norm clip
        only the RATIO of the two survives into the update direction, so the
        ratio is what we specify. A fixed coefficient is a fixed force against a
        moving one: it means something different on every task, and something
        different at the end of a run than at its start.
        """
        scale = self.adv_std
        if not scale == scale or scale <= 0.0:       # NaN or degenerate
            return

        if self._adv_std_frozen is None:
            self._warmup.append(scale)
            if len(self._warmup) >= self.entropy_warmup_iters:
                w = sorted(self._warmup)
                n = len(w)
                self._adv_std_frozen = (w[n // 2] if n % 2
                                        else 0.5 * (w[n // 2 - 1] + w[n // 2]))
                self._scale_ema = self._adv_std_frozen
                self.entropy_loss_coeff = self.entropy_pressure * self._adv_std_frozen
            return

        steps = self._batch_steps(batch)
        if self.pressure_track:
            g = self._rate(steps, self.pressure_track_steps)
            self._scale_ema = (1.0 - g) * self._scale_ema + g * scale
        floor = max(self._scale_ema, self._adv_std_frozen)
        self.entropy_loss_coeff = self.entropy_pressure * floor
        if self.pressure_boost:
            self.entropy_loss_coeff *= self._boost_step(batch, steps)
        self.realised_pressure = self.entropy_loss_coeff / scale

    def _boost_step(self, batch, steps):
        """Multiplicative boost over the tracked floor while exploration starves."""
        self._scale_peak = max(self._scale_peak, self._scale_ema)
        starving = (self._scale_peak > 0.0
                    and self._scale_ema < self.boost_frac * self._scale_peak)

        # Is the width already recovering? Measured on the behaviour policy's
        # entropy and smoothed twice (level, then slope) so one noisy batch
        # cannot flip the decision.
        with torch.no_grad():
            obs = batch["obs"].reshape(-1, batch["obs"].shape[-1])
            params, _ = self.forward_net(obs)
            entropy = float(self.entropy(params).mean().item())
        g = self._rate(steps, self.pressure_track_steps)
        if self._entropy_ema is None:
            self._entropy_ema, slope = entropy, 0.0
        else:
            slope = entropy - self._entropy_ema
            self._entropy_ema = (1.0 - g) * self._entropy_ema + g * entropy
        self._entropy_slope = (1.0 - g) * self._entropy_slope + g * slope

        beta = self._rate(steps, self.boost_efold_steps)
        step = beta if (starving and self._entropy_slope <= 0.0) else -beta
        self._log_boost = min(max(self._log_boost + step, 0.0),
                              math.log(self.boost_max))
        self.boost = math.exp(self._log_boost)
        return self.boost

    @staticmethod
    def _rate(steps, horizon):
        """Per-iteration rate for a time constant given in environment steps."""
        if horizon <= 0.0:
            return 0.0
        return min(1.0, float(steps) / float(horizon))

    def _batch_steps(self, batch):
        """Environment steps this iteration consumed: the summed holding time."""
        time = batch.get("time")
        if time is None:
            return float(batch["rew"].numel())
        return float(self.dwell(time).sum().item())

    def update(self, buffer, bootstrap_value):
        stats = super().update(buffer, bootstrap_value)
        stats.update(adv_std=self.adv_std, entropy_coeff=self.entropy_loss_coeff,
                     realised_pressure=self.realised_pressure, boost=self.boost)
        return stats


# --------------------------------------------------------------- variants ---
class APO(SMAPO, RelaxedSMART):
    """Average-reward PPO that ignores duration: every decision counts as one.

    The MDP reading of average reward. On an SMDP it optimises reward per
    DECISION rather than reward per unit time, so it is the control that isolates
    what time-awareness buys.
    """

    rho_reduce = "mean"

    def dwell(self, time):
        return torch.ones_like(time)


class RsmartSMAPO(SMAPO, RelaxedSMART):
    """``rho = EWMA(reward) / EWMA(tau)`` — a ratio of two smoothed quantities."""

    rho_reduce = "mean"


class SmartSMAPO(SMAPO, SMART):
    """``rho = sum(reward) / sum(tau)`` over the whole run; it cannot forget."""

    rho_reduce = "sum"


class SmoothedSmartSMAPO(SMAPO, SmoothedSMART):
    """``rho`` smoothed in ELAPSED TIME: it forgets per unit of simulated time
    rather than per transition, so its memory does not change when the dwell does.

    ``rho_reduce="none"`` because the estimator decays by ``exp(-lambda*tau)``
    per transition; aggregating the batch would only be exact if the rate were
    constant across it, which a rollout does not guarantee.
    """

    rho_reduce = "none"


# Discrete-action counterparts: the head is the only difference.
class DiscreteAPO(CategoricalHead, APO):
    pass


class DiscreteRsmartSMAPO(CategoricalHead, RsmartSMAPO):
    pass


class DiscreteSmartSMAPO(CategoricalHead, SmartSMAPO):
    pass


class DiscreteSmoothedSmartSMAPO(CategoricalHead, SmoothedSmartSMAPO):
    pass
