"""Power-mean estimators over positive local reward rates."""

import math

from .power_means import CumulativePowerMean, NormalizedExponentialPowerMean


def _require_finite(name: str, value: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _local_rate(reward: float, duration: float) -> float:
    reward = _require_finite("reward", reward)
    duration = _require_finite("duration", duration)
    if duration <= 0:
        raise ValueError("duration must be greater than zero")
    return reward / duration


class CumulativePowerMeanRate:
    """Cumulative weighted power mean of positive reward/duration rates."""

    def __init__(self, p: float):
        self.mean = CumulativePowerMean(p)
        self.p = self.mean.p
        self.reset()

    def reset(self) -> None:
        self.mean.reset()
        self.value = 0.0

    @property
    def rho(self) -> float:
        return self.value

    def update(
        self,
        reward: float,
        duration: float,
        weight: float = 1.0,
    ) -> float:
        self.value = self.mean.update(_local_rate(reward, duration), weight)
        return self.value


class NormalizedExponentialPowerMeanRate:
    """Normalized exponentially smoothed weighted mean of positive rates."""

    def __init__(self, p: float, beta: float):
        self.mean = NormalizedExponentialPowerMean(p, beta)
        self.p = self.mean.p
        self.beta = self.mean.beta
        self.reset()

    def reset(self) -> None:
        self.mean.reset()
        self.value = 0.0

    @property
    def rho(self) -> float:
        return self.value

    def update(
        self,
        reward: float,
        duration: float,
        weight: float = 1.0,
    ) -> float:
        self.value = self.mean.update(_local_rate(reward, duration), weight)
        return self.value
