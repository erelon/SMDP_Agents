"""Reusable weighted averaging primitives and reward-rate estimators.

This module is intentionally independent from the agent hierarchy.  Agent
integration belongs to a later phase; the classes here only own averaging
state and return the latest estimate from each update.
"""

import importlib.util
import math
import pathlib

try:
    from .value_checks import require_duration, require_finite
except ImportError:  # Support direct loading by file path.
    _VALUE_CHECKS_PATH = pathlib.Path(__file__).with_name("value_checks.py")
    _VALUE_CHECKS_SPEC = importlib.util.spec_from_file_location(
        "_average_rates_value_checks", _VALUE_CHECKS_PATH
    )
    if _VALUE_CHECKS_SPEC is None or _VALUE_CHECKS_SPEC.loader is None:
        raise ImportError(f"cannot load value checks from {_VALUE_CHECKS_PATH}")
    _value_checks = importlib.util.module_from_spec(_VALUE_CHECKS_SPEC)
    _VALUE_CHECKS_SPEC.loader.exec_module(_value_checks)
    require_duration = _value_checks.require_duration
    require_finite = _value_checks.require_finite

# Building blocks:  EMA and normalized EMA

class ExponentialMovingAverage:
    """Zero-initialized EMA with a call-specific multiplicative weight."""

    def __init__(self, beta: float):
        beta = require_finite("beta", beta)
        if not 0 < beta <= 1:
            raise ValueError("beta must be in the interval (0, 1]")
        self.beta = beta
        self.value = 0.0

    def reset(self) -> None:
        self.value = 0.0

    def update(self, value: float, weight: float) -> float:
        value = require_finite("value", value)
        weight = require_finite("weight", weight)
        self.value = (1 - self.beta) * self.value + self.beta * value * weight
        return self.value

class NormalizedEMA:
    """NORMALIZED EMA with a call-specific multiplicative weight.  This eliminates the bias of 0-initialization"""

    def __init__(self, beta: float):
        beta = require_finite("beta", beta)
        if not 0 < beta <= 1:
            raise ValueError("beta must be in the interval (0, 1]")
        self.beta = beta
        self.unnorm = ExponentialMovingAverage(self.beta)
        self.weight = ExponentialMovingAverage(self.beta)
        self.reset()

    def reset(self) -> None:
        self.value = 0.0
        self.unnorm.reset()
        self.weight.reset()

    def update(self, value: float, weight: float) -> float:
        value = require_finite("value", value)
        weight = require_finite("weight", weight)
        self.unnorm.update(value,weight)
        self.weight.update(weight,1.0)
        self.value = self.unnorm.value / self.weight.value
        return self.value

class ExponentialMovingTimeRate:
    """Unnormalized exponential moving time-rate estimator."""

    def __init__(self, beta: float):
        beta = require_finite("beta", beta)
        if not 0 < beta <= 1:
            raise ValueError("beta must be in the interval (0, 1]")

        self.beta = beta
        self.lambda_ = -math.log(1-beta)
        self.reset()

    def reset(self) -> None:
        self.value = 0.0

    @property
    def rho(self) -> float:
        return self.value

    def update(
        self,
        reward: float,
        time: float,
        weight: float = 1.0,
    ) -> float:
        reward = require_finite("reward", reward)
        time = require_duration(time)
        weight = require_finite("weight", weight)

        gain = -math.expm1(-self.lambda_ * time)

        self.value += (
            gain / time
        ) * (
            reward * weight - self.value * time
        )

        return self.value


class NormalizedExponentialMovingTimeRate:
    """Normalized exponential moving time-rate estimator.

    Normalization removes the bias caused by zero initialization.
    """

    def __init__(self, beta: float):
        beta = require_finite("beta", beta)
        if not 0 < beta <= 1:
            raise ValueError("beta must be in the interval (0, 1]")

        self.beta = beta
        self.lambda_ = -math.log(1-beta)
        self.reset()

    def reset(self) -> None:
        self.unnormalized_value = 0.0
        self.normalizer = 0.0
        self.value = 0.0

    @property
    def rho(self) -> float:
        return self.value

    def update(
        self,
        reward: float,
        time: float,
        weight: float = 1.0,
    ) -> float:
        reward = require_finite("reward", reward)
        time = require_duration(time)
        weight = require_finite("weight", weight)

        retention = math.exp(-self.lambda_ * time)
        gain = -math.expm1(-self.lambda_ * time)

        self.unnormalized_value = (
            retention * self.unnormalized_value
            + gain * (reward * weight / time)
        )

        self.normalizer = (
            retention * self.normalizer
            + gain
        )

        self.value = (
            self.unnormalized_value / self.normalizer
        )

        return self.value

class ExponentialMovingRatioRate:
    def __init__(self, beta: float):
        beta = require_finite("beta", beta)
        if not 0 < beta <= 1:
            raise ValueError("beta must be in the interval (0, 1]")
        self.beta = beta
        self.reward_ema = NormalizedEMA(beta)
        self.duration_ema = NormalizedEMA(beta)
        self.reset()

    def reset(self) -> None:
        self.reward_ema.reset()
        self.duration_ema.reset()
        self.value = 0.0

    @property
    def rho(self) -> float:
        return self.value

    def update(self,reward: float, time: float, weight: float = 1.0) -> float:
        reward = require_finite("reward", reward)
        time = require_duration(time)
        weight = require_finite("weight", weight)

        reward_ema = self.reward_ema.update(reward, weight)
        duration_ema = self.duration_ema.update(time, 1.0)
        try:
            self.value = reward_ema / duration_ema
        except: 
            if duration_ema == 0:
                raise ZeroDivisionError("Relaxed SMART requires nonzero elapsed time") from None
            raise

        return self.value


# Cumulative reward-rate estimators

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
        reward = require_finite("reward", reward)
        duration = require_duration(duration)
        weight = require_finite("weight", weight)
        self.total_reward += reward * weight
        self.total_duration += duration
        self.value = self.total_reward / self.total_duration
        return self.value


class CumulativeStepRate:
    def __init__(self):
        self.reset()
        
    def reset(self) -> None:
        self.total_rates = 0.0
        self.total_steps = 0.0
        self.value = 0.0

    def update(self, reward: float, duration: float, weight: float) -> float:
        reward = require_finite("reward", reward)
        duration = require_duration(duration)
        weight = require_finite("weight", weight)
        self.total_rates += (reward/duration) * weight
        self.total_steps += 1
        self.value = self.total_rates/ self.total_steps
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
        reward = require_finite("reward", reward)
        duration = require_duration(duration)
        weight = require_finite("weight", weight)

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




class NormHMA:
    """General signed harmonic moving-average reward-rate estimator."""

    def __init__(self, beta: float):
        self.positive_reciprocal = NormalizedEMA(beta)
        self.negative_reciprocal = NormalizedEMA(beta)
        self.positive_weight = NormalizedEMA(beta)
        self.negative_weight = NormalizedEMA(beta)
        self.positive_occurrence = NormalizedEMA(beta)
        self.negative_occurrence = NormalizedEMA(beta)
        self.zero_occurrence = NormalizedEMA(beta)
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
        reward = require_finite("reward", reward)
        duration = require_duration(duration)
        weight = require_finite("weight", weight)

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
        self.value = ( positive_harmonic + negative_harmonic+zero_occurrence )
        #     positive_harmonic * positive_occurrence
        #     + negative_harmonic * negative_occurrence
        # ) / occurrence_total
        return self.value
