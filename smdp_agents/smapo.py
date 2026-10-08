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
    HarmonicSMAPO         rho from a harmonic mean of the reward/time streams
                          (and the weighted, cumulative and |rho|-scaled
                           members of that family -- see below)
    SmoothedSmartSMAPO    rho smoothed in ELAPSED TIME, not per transition

The action space is a constructor flag inherited from :class:`~smdp_agents.ppo.PPO`:
``discrete=True`` swaps the Gaussian actor for a categorical one. Every variant
below has a ``Discrete*`` convenience class that sets it.

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

from .experemental_harmonic_r import (
    ExperimentalCumulativeWeightedHarmonic, ExperimentalWeightedHarmonic,
    abs_rho_scaled_advantage)
from .harmonic_r import (CumulativeHarmonic, CumulativeWeightedHarmonic,
                         Harmonic, WeightedHarmonic)
from .ppo import PPO
from .relaxed_smart import RelaxedSMART
from .smart_r import SMART, SmoothedSMART


class SMAPO(PPO):
    """Average-reward PPO with A-centering and calibrated entropy pressure.

    Subclass it together with a rate estimator; see the module docstring.
    """

    longrun = True

    def __init__(self, obs_dim, act_dim, a_centering=True,
                 calibrated_pressure=True, entropy_pressure=0.02,
                 entropy_warmup_iters=20, pressure_track=True,
                 pressure_track_steps=2_000_000, pressure_boost=True,
                 boost_efold_steps=400_000, boost_frac=0.30, boost_max=8.0,
                 **kwargs):
        """
        a_centering, calibrated_pressure
            The two changes SMAPO makes beyond undiscounted PPO, both on by
            default. Turn one off to measure what it contributes; with both off
            this is average-reward PPO with a fixed entropy coefficient.

            Note that ``entropy_pressure=0`` is NOT how to disable the bonus: the
            mechanism would still run and overwrite ``entropy_loss_coeff`` with 0
            every iteration, silently discarding a fixed coefficient you had set.
            Use ``calibrated_pressure=False``, which leaves the coefficient alone.
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
        if calibrated_pressure and float(entropy_pressure) <= 0.0:
            raise ValueError(
                "entropy_pressure must be > 0 when calibrated_pressure is on; "
                "pass calibrated_pressure=False to use a fixed "
                "entropy_loss_coeff instead.")
        kwargs.setdefault("entropy_loss_coeff", 0.0)
        super().__init__(obs_dim, act_dim, **kwargs)
        self.a_centering = bool(a_centering)
        self.calibrated_pressure = bool(calibrated_pressure)
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
        if self.calibrated_pressure:
            self._update_entropy_coeff(batch)
        if not self.a_centering:
            return super().shape_advantage(adv, valid, batch)
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
        entropy = self.policy_entropy(
            batch["obs"].reshape(-1, batch["obs"].shape[-1]))
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


class HarmonicSMAPO(SMAPO, Harmonic):
    """``rho`` from a harmonic mean over the positive and negative reward streams.

    ``rho_reduce="none"`` because the sign-stratified split needs each reward
    individually; aggregating the batch would lose it.
    """

    rho_reduce = "none"


class SmoothedSmartSMAPO(SMAPO, SmoothedSMART):
    """``rho`` smoothed in ELAPSED TIME: it forgets per unit of simulated time
    rather than per transition, so its memory does not change when the dwell does.

    ``rho_reduce="none"`` because the estimator decays by ``exp(-lambda*tau)``
    per transition; aggregating the batch would only be exact if the rate were
    constant across it, which a rollout does not guarantee.
    """

    rho_reduce = "none"


# --- the rest of the harmonic family ----------------------------------------
# Every one of these is the same algorithm with a different rho; they are listed
# individually rather than generated so that `from smdp_agents import X` works and
# docstring can say what each one's rho is. All take `rho_reduce = "none"`: the
# harmonic estimators stratify by the sign of the reward, so they need each
# transition rather than a batch aggregate.
#
# A discrete-action counterpart of any of these is one class line, as above:
#     WeightedHarmonicSMAPO(obs_dim, n_options, discrete=True)

class WeightedHarmonicSMAPO(SMAPO, WeightedHarmonic):
    """Harmonic mean weighted by the reward itself."""

    rho_reduce = "none"


class CumulativeHarmonicSMAPO(SMAPO, CumulativeHarmonic):
    """Harmonic mean over the whole run, unweighted; it does not forget."""

    rho_reduce = "none"


class CumulativeWeightedHarmonicSMAPO(SMAPO, CumulativeWeightedHarmonic):
    """Harmonic mean over the whole run, weighted by reward.

    On a domain whose rewards are all positive the weight makes this degenerate
    to ``sum(r) / sum(tau)``, which is exactly :class:`SmartSMAPO`'s rate.
    """

    rho_reduce = "none"


class _AbsRhoScaledSMAPO(SMAPO):
    """Shared override for the ``|rho|``-scaled variants.

    The scaling MUST be spelled out here rather than inherited from
    ``AbsRhoScaledTarget``. That mixin provides ``set_target``, which is the
    tabular entry point; the deep agents never call it, and ``SMAPO`` precedes
    the mixin in the MRO, so inheriting it alone would compile, run, and
    silently apply no scaling at all.
    """

    def rate_residual(self, reward, time):
        return abs_rho_scaled_advantage(reward, self.dwell(time), self.rho)


class ExperimentalWeightedHarmonicSMAPO(_AbsRhoScaledSMAPO,
                                        ExperimentalWeightedHarmonic):
    """Weighted harmonic rho, with the residual divided by ``|rho|``."""

    rho_reduce = "none"


class ExperimentalCumulativeWeightedHarmonicSMAPO(
        _AbsRhoScaledSMAPO, ExperimentalCumulativeWeightedHarmonic):
    """Cumulative weighted harmonic rho, with the residual divided by ``|rho|``."""

    rho_reduce = "none"


# --- discrete-action convenience classes ------------------------------------
# `discrete=True` is a PPO constructor flag, so these only pre-set it; `act_dim`
# is then the NUMBER OF ACTIONS rather than a vector width. Any other variant
# above can be used discretely by passing the flag directly.

class _Discrete:
    def __init__(self, obs_dim, n_actions, **kwargs):
        kwargs.setdefault("discrete", True)
        super().__init__(obs_dim, n_actions, **kwargs)


class DiscreteAPO(_Discrete, APO):
    pass


class DiscreteRsmartSMAPO(_Discrete, RsmartSMAPO):
    pass


class DiscreteSmartSMAPO(_Discrete, SmartSMAPO):
    pass


class DiscreteHarmonicSMAPO(_Discrete, HarmonicSMAPO):
    pass


class DiscreteSmoothedSmartSMAPO(_Discrete, SmoothedSmartSMAPO):
    pass
