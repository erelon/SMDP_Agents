"""Exponentially smoothed power-mean rate R-learning agent."""

from .power_rates import NormalizedExponentialPowerMeanRate
from .r_learning import ContinuousRLearning


def _observation_weight(mode: str, reward: float, duration: float) -> float:
    if mode == "unit":
        return 1.0
    if mode == "reward":
        return reward
    if mode == "duration":
        return duration
    raise ValueError("rate_weight must be 'unit', 'reward', or 'duration'")


class RiskSmoothedR(ContinuousRLearning):
    """R-learning with a normalized smoothed power mean of local rates.

    The CRRA risk parameter is ``theta = 1 - p``. Thus ``p > 1`` is
    risk-seeking, ``p = 1`` is risk-neutral, ``0 < p < 1`` is risk-averse,
    ``p = 0`` is logarithmic, and ``p < 0`` increasingly emphasizes low
    rates. Rewards, durations, and selected observation weights must produce
    strictly positive local rates and weights.

    Unit weighting is the default because at ``p=-1`` it reproduces the
    positive-domain Harmonic estimator. Select ``rate_weight="duration"`` at
    ``p=1`` to reproduce RelaxedSMART.
    """

    def __init__(
        self,
        name: str,
        action_space=None,
        learning_rate=0.1,
        exploration_rate=0.1,
        with_rho_trick=True,
        rho_learning_rate=0.3,
        p=-1.0,
        rate_weight="unit",
        **kwargs,
    ):
        super().__init__(
            name,
            action_space,
            learning_rate,
            exploration_rate,
            with_rho_trick,
            rho_learning_rate,
            **kwargs,
        )
        if rate_weight not in ("unit", "reward", "duration"):
            raise ValueError("rate_weight must be 'unit', 'reward', or 'duration'")
        self.p = float(p)
        self.rate_weight = rate_weight
        self.rate = NormalizedExponentialPowerMeanRate(
            self.p, rho_learning_rate
        )
        self.reset()

    def reset(self):
        super().reset()
        self.rate.reset()

    def calc_new_rho(self, reward, time, td_target, td_error):
        try:
            weight = _observation_weight(self.rate_weight, reward, time)
            self.rho = self.rate.update(reward, time, weight)
        except ValueError:
            if time == 0:
                raise ZeroDivisionError(
                    "RiskSmoothedR requires nonzero elapsed time"
                ) from None
            raise
