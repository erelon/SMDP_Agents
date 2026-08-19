"""Reusable weighted averaging primitives and reward-rate estimators.

This module is intentionally independent from the agent hierarchy.  Agent
integration belongs to a later phase; the classes here only own averaging
state and return the latest estimate from each update.

Building blocks average an arbitrary quantity and expose it as ``value``.
Reward-rate estimators derive from ``RewardRateEstimator``, consume
``(reward, duration, weight)`` transitions, and expose the estimate as ``rho``.
"""

import math


def _require_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _require_nonnegative_duration(duration: float) -> float:
    duration = _require_finite("duration", duration)
    if duration < 0:
        raise ValueError("duration must not be negative")
    return duration


def _require_duration(duration: float) -> float:
    duration = _require_nonnegative_duration(duration)
    if duration == 0:
        raise ValueError("duration must be greater than zero")
    return duration


def _require_beta(beta: float) -> float:
    beta = _require_finite("beta", beta)
    if not 0 < beta <= 1:
        raise ValueError("beta must be in the interval (0, 1]")
    return beta


def _time_decay(beta: float) -> float:
    """Decay rate whose retention over a unit duration is ``1 - beta``.

    ``log1p`` because ``1 - beta`` cancels for small beta; ``beta == 1`` means
    no memory.
    """
    beta = _require_beta(beta)
    return math.inf if beta == 1 else -math.log1p(-beta)


# Building blocks:  EMA, normalized EMA, and time-decayed EMA

class ExponentialMovingAverage:
    """Zero-initialized EMA with a call-specific multiplicative weight."""

    def __init__(self, beta: float):
        self.beta = _require_beta(beta)
        self.reset()

    def reset(self) -> None:
        self.value = 0.0

    def update(self, value: float, weight: float) -> float:
        value = _require_finite("value", value)
        weight = _require_finite("weight", weight)
        self.value = (1 - self.beta) * self.value + self.beta * value * weight
        return self.value


class CumulativeAverage:
    """Running mean of ``value * weight``, with the EMA's call signature.

    The whole-history counterpart of :class:`ExponentialMovingAverage`: a gain of
    ``1 / n`` instead of a fixed ``beta``, so an estimator written against this
    interface can be built over either and forget or not accordingly.  There is no
    zero-initialization bias to remove, so no normalized variant is needed.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.value = 0.0
        self.count = 0

    def update(self, value: float, weight: float) -> float:
        value = _require_finite("value", value)
        weight = _require_finite("weight", weight)
        self.count += 1
        # Incremental rather than sum/count: the running form does not lose the
        # small terms once the totals grow large.
        self.value += (value * weight - self.value) / self.count
        return self.value


class NormalizedEMA:
    """EMA divided by the EMA of its weights, removing the zero-init bias."""

    def __init__(self, beta: float):
        self.beta = _require_beta(beta)
        self.unnormalized = ExponentialMovingAverage(self.beta)
        self.normalizer = ExponentialMovingAverage(self.beta)
        self.reset()

    def reset(self) -> None:
        self.value = 0.0
        self.unnormalized.reset()
        self.normalizer.reset()

    def update(self, value: float, weight: float) -> float:
        value = _require_finite("value", value)
        weight = _require_finite("weight", weight)
        unnormalized = self.unnormalized.update(value, weight)
        normalizer = self.normalizer.update(weight, 1.0)
        self.value = unnormalized / normalizer
        return self.value


class TimeDecayedEMA:
    """Zero-initialized EMA that decays per unit of elapsed time.

    Retains ``exp(-lambda * duration)`` instead of a fixed ``1 - beta`` per
    step, so the average depends on elapsed time rather than on how that time
    was chopped into transitions.  ``level`` is already a rate.
    """

    def __init__(self, beta: float):
        self.beta = _require_beta(beta)
        self.lambda_ = _time_decay(self.beta)
        self.reset()

    def reset(self) -> None:
        self.value = 0.0

    def update(self, level: float, duration: float) -> float:
        level = _require_finite("level", level)
        duration = _require_duration(duration)
        retention = math.exp(-self.lambda_ * duration)
        gain = -math.expm1(-self.lambda_ * duration)
        self.value = retention * self.value + gain * level
        return self.value


# The estimator contract

class RewardRateEstimator:
    """Base class for reward-rate estimators.

    Subclasses implement ``_update``, which receives a validated transition and
    returns the new ``rho``, and extend ``reset`` when they hold further state.
    A weight of 1.0 is the unweighted estimator, so callers state a weight only
    to deviate from it -- as ``WeightedHarmonic`` does by passing the reward.
    """

    #: Estimators that divide by the duration need it strictly positive.  Those
    #: that only ever multiply by it accept an instantaneous transition and set
    #: this to True; a negative duration stays an error either way.
    accepts_zero_duration = False

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.rho = 0.0

    def update(self, reward: float, duration: float, weight: float = 1.0) -> float:
        reward = _require_finite("reward", reward)
        duration = (_require_nonnegative_duration(duration)
                    if self.accepts_zero_duration else _require_duration(duration))
        weight = _require_finite("weight", weight)
        self.rho = self._update(reward, duration, weight)
        return self.rho

    def _update(self, reward: float, duration: float, weight: float) -> float:
        raise NotImplementedError


class ExponentialRateEstimator(RewardRateEstimator):
    """Estimator parameterized by a smoothing rate ``beta``.

    Subclasses must construct their building blocks before delegating here,
    because ``__init__`` resets the whole estimator.
    """

    def __init__(self, beta: float):
        self.beta = _require_beta(beta)
        super().__init__()


# Exponential reward-rate estimators

class _TimeDecayedRate(ExponentialRateEstimator):
    """Estimator backed by a time-decayed average of the transition rate."""

    def __init__(self, beta: float):
        self.unnormalized = TimeDecayedEMA(beta)
        super().__init__(beta)

    def reset(self) -> None:
        super().reset()
        self.unnormalized.reset()

    @property
    def lambda_(self) -> float:
        return self.unnormalized.lambda_

    def _smoothed_rate(self, reward: float, duration: float, weight: float) -> float:
        return self.unnormalized.update(reward * weight / duration, duration)


class ExponentialMovingTimeRate(_TimeDecayedRate):
    """Unnormalized exponential moving time-rate estimator."""

    def _update(self, reward: float, duration: float, weight: float) -> float:
        return self._smoothed_rate(reward, duration, weight)


class NormalizedExponentialMovingTimeRate(_TimeDecayedRate):
    """Time-rate estimator divided by the decayed weight, removing the zero-init bias.

    Same construction as ``NormalizedEMA``, in elapsed time: the driving weight
    is the constant 1.0, so the normalizer converges to
    ``1 - prod(exp(-lambda * duration_i))``.
    """

    def __init__(self, beta: float):
        self.normalizer = TimeDecayedEMA(beta)
        super().__init__(beta)

    def reset(self) -> None:
        super().reset()
        self.normalizer.reset()

    def _update(self, reward: float, duration: float, weight: float) -> float:
        unnormalized = self._smoothed_rate(reward, duration, weight)
        return unnormalized / self.normalizer.update(1.0, duration)


class ExponentialMovingRatioRate(ExponentialRateEstimator):
    """Ratio of a smoothed reward to a smoothed duration."""

    def __init__(self, beta: float):
        self.reward_ema = NormalizedEMA(beta)
        self.duration_ema = NormalizedEMA(beta)
        super().__init__(beta)

    def reset(self) -> None:
        super().reset()
        self.reward_ema.reset()
        self.duration_ema.reset()

    @property
    def mean_reward(self) -> float:
        return self.reward_ema.value

    @property
    def mean_duration(self) -> float:
        return self.duration_ema.value

    def _update(self, reward: float, duration: float, weight: float) -> float:
        reward_ema = self.reward_ema.update(reward, weight)
        duration_ema = self.duration_ema.update(duration, 1.0)
        # _require_duration keeps duration_ema != 0
        return reward_ema / duration_ema


# Cumulative reward-rate estimators

class CumulativeTimeRate(RewardRateEstimator):
    """Cumulative weighted reward divided by unweighted elapsed duration."""

    def reset(self) -> None:
        super().reset()
        self.total_reward = 0.0
        self.total_duration = 0.0

    def _update(self, reward: float, duration: float, weight: float) -> float:
        self.total_reward += reward * weight
        self.total_duration += duration
        return self.total_reward / self.total_duration


class CumulativeStepRate(RewardRateEstimator):
    """Mean of the weighted per-transition reward rate."""

    def reset(self) -> None:
        super().reset()
        self.total_rates = 0.0
        self.total_steps = 0.0

    def _update(self, reward: float, duration: float, weight: float) -> float:
        self.total_rates += (reward / duration) * weight
        self.total_steps += 1
        return self.total_rates / self.total_steps


# Harmonic reward-rate estimators

class _SignedHarmonicBranch:
    """One sign branch of the signed harmonic average.

    ``indicator`` is 1.0 when the reward carries this branch's sign, keeping the
    other sign out of the averages while still ageing this branch every step.

    ``make_average`` builds the three averaging blocks, so the same branch serves
    the exponential and the cumulative estimators.  Whatever it returns, the
    branch's harmonic mean is a ratio of two averages taken the same way, so the
    normalizer -- ``1 - (1 - beta)**n`` for an EMA, ``1 / n`` for a running mean
    -- cancels and never has to be tracked.
    """

    def __init__(self, make_average):
        self.reciprocal = make_average()
        self.weighted_occurrence = make_average()
        self.occurrence = make_average()

    def reset(self) -> None:
        self.reciprocal.reset()
        self.weighted_occurrence.reset()
        self.occurrence.reset()

    def update(self, indicator: float, reciprocal_rate: float,
               weight: float) -> tuple[float, float]:
        """Return this branch's harmonic rate and its occurrence share."""
        reciprocal = self.reciprocal.update(reciprocal_rate * indicator, weight)
        weighted = self.weighted_occurrence.update(indicator, weight)
        occurrence = self.occurrence.update(indicator, 1.0)

        harmonic = 0.0 if reciprocal == 0 else weighted / reciprocal
        return harmonic, occurrence


class _SignedHarmonicMixing:
    """The signed harmonic estimator, over whatever averaging blocks it is given.

    Each sign is averaged in its own branch -- a harmonic mean across a sign
    change is meaningless -- and the branch rates are mixed by how often each
    sign occurs.  Zero rewards have no rate, so they enter only as occurrences.

    The duration enters as ``duration / reward``, never as a divisor, so an
    instantaneous transition is well defined: it contributes nothing to the
    branch's reciprocal average and only ages it.

    Mixed in *before* an estimator base so that ``reset`` and ``_update`` resolve
    here first and cooperate upward; a subclass calls ``_build_branches`` before
    delegating to that base, which resets everything.
    """

    accepts_zero_duration = True

    def _build_branches(self, make_average) -> None:
        self.positive = _SignedHarmonicBranch(make_average)
        self.negative = _SignedHarmonicBranch(make_average)
        self.zero_occurrence = make_average()

    def reset(self) -> None:
        super().reset()
        self.positive.reset()
        self.negative.reset()
        self.zero_occurrence.reset()

    def _update(self, reward: float, duration: float, weight: float) -> float:
        positive = float(reward > 0)
        negative = float(reward < 0)
        zero = float(reward == 0)
        reciprocal_rate = 0.0 if zero else duration / reward

        positive_rate, positive_occurrence = \
            self.positive.update(positive, reciprocal_rate, weight)
        negative_rate, negative_occurrence = \
            self.negative.update(negative, reciprocal_rate, weight)
        zero_occurrence = self.zero_occurrence.update(zero, 1.0)

        return (
            positive_rate * positive_occurrence
            + negative_rate * negative_occurrence
        ) / (positive_occurrence + negative_occurrence + zero_occurrence)


class WeightedHarmonicRate(_SignedHarmonicMixing, ExponentialRateEstimator):
    """Signed harmonic *moving*-average reward-rate estimator.

    The forgetting one: every average behind it is an EMA at ``beta``, so the
    estimate tracks a drifting rate rather than averaging the whole history.
    """

    def __init__(self, beta: float):
        self._build_branches(lambda: ExponentialMovingAverage(beta))
        super().__init__(beta)


class CumulativeWeightedHarmonicRate(_SignedHarmonicMixing, RewardRateEstimator):
    """Signed harmonic average over the whole history, with no forgetting.

    The same estimator as :class:`WeightedHarmonicRate` with running means in
    place of EMAs -- what ``CumulativeTimeRate`` is to
    ``ExponentialMovingRatioRate``.  Since each branch is a ratio of two averages
    taken the same way, the ``1 / n`` cancels and the branch reduces to a ratio of
    plain sums; the sign indicators partition every step, so the occurrence shares
    sum to exactly 1 and the mixing denominator is a no-op.

    Two special cases are worth knowing, both of which hold only while no reward
    is zero:

    * with **unit weight** and all rewards positive it is the harmonic mean of
      the per-transition rates, ``n / sum(tau_i / r_i)`` -- the third member of
      the family beside ``CumulativeTimeRate``'s ``sum(r) / sum(tau)`` and
      ``CumulativeStepRate``'s ``mean(r_i / tau_i)``;
    * with the **reward as the weight** and all rewards positive the ``r`` in
      ``(tau / r) * r`` cancels, the branch becomes ``sum(r) / sum(tau)``, and it
      coincides exactly with :class:`CumulativeTimeRate`.  A negative reward puts
      the two branches back to work and the two part company again.
    """

    def __init__(self):
        self._build_branches(CumulativeAverage)
        super().__init__()


class NormHMA(WeightedHarmonicRate):
    """Alias for :class:`WeightedHarmonicRate`; normalizing it is a no-op.

    This used to be a copy built from ``NormalizedEMA`` instead of
    ``ExponentialMovingAverage``, to debias the zero initialization.  There is
    no bias to remove: the estimator only ever consumes ratios of equally
    biased EMAs, so the normalizer ``N_n = 1 - (1 - beta)**n`` cancels in each
    branch's harmonic mean, and the sign indicators partition every step, so the
    raw occurrences already sum to ``N_n``.  ``NormHMATests`` confirms it
    numerically.
    """
