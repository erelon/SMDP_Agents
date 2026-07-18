"""Reusable weighted averaging primitives and reward-rate estimators.

This module is intentionally independent from the agent hierarchy.  Agent
integration belongs to a later phase; the classes here only own averaging
state and return the latest estimate from each update.
"""

import math


def _require_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _require_duration(duration: float) -> float:
    duration = _require_finite("duration", duration)
    if duration <= 0:
        raise ValueError("duration must be greater than zero")
    return duration


class ExponentialMovingAverage:
    """Zero-initialized EMA with a call-specific multiplicative weight."""

    def __init__(self, beta: float):
        beta = _require_finite("beta", beta)
        if not 0 < beta <= 1:
            raise ValueError("beta must be in the interval (0, 1]")
        self.beta = beta
        self.value = 0.0

    def reset(self) -> None:
        self.value = 0.0

    def update(self, value: float, weight: float) -> float:
        value = _require_finite("value", value)
        weight = _require_finite("weight", weight)
        self.value = (1 - self.beta) * self.value + self.beta * value * weight
        return self.value


class CumulativeTimeRate:
    """Cumulative weighted reward divided by unweighted elapsed duration."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.total_reward = 0.0
        self.total_duration = 0.0
        self.value = 0.0

    @property
    def rho(self) -> float:
        return self.value	# TODO: Seems wasteful. If all algorithms use rho, then use rho. Otherwise use value.

    def update(self, reward: float, duration: float, weight: float) -> float:
        reward = _require_finite("reward", reward)
        duration = _require_duration(duration)
        weight = _require_finite("weight", weight)
        self.total_reward += reward * weight
        self.total_duration += duration
        self.value = self.total_reward / self.total_duration
        return self.value


class RatioEmaRate:
    """Weighted reward EMA divided by an unweighted duration EMA."""

    def __init__(self, beta: float, duration_beta: float = None):
        if duration_beta is None:
            duration_beta = beta
        self.reward_ema = ExponentialMovingAverage(beta)
        self.duration_ema = ExponentialMovingAverage(duration_beta)
        self.value = 0.0

    def reset(self) -> None:
        self.reward_ema.reset()
        self.duration_ema.reset()
        self.value = 0.0

    @property
    def rho(self) -> float:
        return self.value

    def update(self, reward: float, duration: float, weight: float) -> float:
        duration = _require_duration(duration)
        weighted_reward = self.reward_ema.update(reward, weight)
        average_duration = self.duration_ema.update(duration, 1.0)
        self.value = weighted_reward / average_duration
        return self.value


class WeightedHarmonicRate:
    """General signed harmonic moving-average reward-rate estimator."""

    def __init__(self, beta: float):
        self.positive_reciprocal = ExponentialMovingAverage(beta)
        self.negative_reciprocal = ExponentialMovingAverage(beta)
        self.positive_weight = ExponentialMovingAverage(beta)
        self.negative_weight = ExponentialMovingAverage(beta)
        self.positive_occurrence = ExponentialMovingAverage(beta)
        self.negative_occurrence = ExponentialMovingAverage(beta)
        self.zero_occurrence = ExponentialMovingAverage(beta)
        self.value = 0.0

    def reset(self) -> None:
        self.positive_reciprocal.reset()
        self.negative_reciprocal.reset()
        self.positive_weight.reset()
        self.negative_weight.reset()
        self.positive_occurrence.reset()
        self.negative_occurrence.reset()
        self.zero_occurrence.reset()
        self.value = 0.0

    @property
    def rho(self) -> float:
        return self.value

    def update(self, reward: float, duration: float, weight: float) -> float:
        reward = _require_finite("reward", reward)
        duration = _require_duration(duration)
        weight = _require_finite("weight", weight)

        positive = float(reward > 0)
        negative = float(reward < 0)
        zero = float(reward == 0)
        reciprocal_rate = 0.0 if zero else duration / reward

        positive_reciprocal = self.positive_reciprocal.update(
            reciprocal_rate * positive, weight
        )
        positive_weight = self.positive_weight.update(positive, weight)
        positive_occurrence = self.positive_occurrence.update(positive, 1.0)

        negative_reciprocal = self.negative_reciprocal.update(
            reciprocal_rate * negative, weight
        )
        negative_weight = self.negative_weight.update(negative, weight)
        negative_occurrence = self.negative_occurrence.update(negative, 1.0)
        zero_occurrence = self.zero_occurrence.update(zero, 1.0)

        positive_harmonic = (
            0.0 if positive_reciprocal == 0
            else positive_weight / positive_reciprocal
        )
        negative_harmonic = (
            0.0 if negative_reciprocal == 0
            else negative_weight / negative_reciprocal
        )
        occurrence_total = (
            positive_occurrence + negative_occurrence + zero_occurrence
        )
        self.value = (
            positive_harmonic * positive_occurrence
            + negative_harmonic * negative_occurrence
        ) / occurrence_total
        return self.value



