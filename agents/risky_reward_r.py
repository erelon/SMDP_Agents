"""CRRA utility-transformed, average-reward SMDP R-learning."""

from .r_learning import ContinuousRLearning
from .risky_reward_tools import (
    CumulativeCRRAUtilityRate,
    require_finite,
    utility_differential_reward,
)


class RiskyRewardR(ContinuousRLearning):
    """Risk-sensitive R-learning using CRRA utility of positive local rates.

    ``theta > 0`` is risk-averse, ``theta = 0`` is risk-neutral, and
    ``theta < 0`` is risk-seeking. The corresponding power-mean parameter is
    ``p = 1 - theta``.

    The TD reward is ``duration * (u(local_rate) - u(rho))``. The baseline is
    the duration-weighted average CRRA utility and ``rho`` is its inverse-
    utility certainty equivalent. At ``theta=0`` the TD target is exactly
    ``reward - rho * duration + next_q``, the ordinary continuous R-learning
    target. The initial utility baseline is zero, whose certainty equivalent
    is the positive reference rate ``rho=1``.

    Direct CRRA utility requires positive rates. Zero/negative rewards, rate
    shifting, cost conversion, and sign-partitioned extensions are outside
    this implementation.
    """

    def __init__(
        self,
        name: str,
        action_space=None,
        learning_rate=0.1,
        exploration_rate=0.1,
        with_rho_trick=True,
        rho_learning_rate=0.3,
        theta=0.0,
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
        self.theta = require_finite("theta", theta)
        self.p = 1.0 - self.theta
        self.utility_baseline = CumulativeCRRAUtilityRate(self.theta)
        self.reset()

    def reset(self):
        super().reset()
        self.utility_baseline.reset()
        self.rho = self.utility_baseline.rho

    @property
    def utility_rate(self):
        return self.utility_baseline.utility_rate

    @property
    def total_time(self):
        return self.utility_baseline.total_duration

    def set_target(self, reward, time, next_q):
        differential_reward = utility_differential_reward(
            reward, time, self.rho, self.theta
        )
        return differential_reward + next_q

    def calc_new_rho(self, reward, time, td_target, td_error):
        try:
            self.rho = self.utility_baseline.update(reward, time)
        except ValueError:
            if time == 0:
                raise ZeroDivisionError(
                    "RiskyRewardR requires nonzero elapsed time"
                ) from None
            raise
