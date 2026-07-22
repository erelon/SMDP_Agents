"""Power-mean estimators over positive local reward rates."""

from .power_means import CumulativePowerMean, NormalizedExponentialPowerMean
from .value_checks import local_rate


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
        self.value = self.mean.update(local_rate(reward, duration), weight)
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
        self.value = self.mean.update(local_rate(reward, duration), weight)
        return self.value
