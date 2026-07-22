"""Exponentially smoothed Risk-Sensitive SMDP R Learning."""

import math

from .power_rates import NormalizedExponentialPowerMeanRate
from .r_learning import ContinuousRLearning
from .risk_ce import crra
from .value_checks import require_finite

class RiskSmoothedR(ContinuousRLearning):
    """R-learning with a normalized smoothed power mean of local rates.

    The CRRA risk parameter is ``theta = 1 - p``. Thus ``p > 1`` is
    risk-seeking, ``p = 1`` is risk-neutral, ``0 < p < 1`` is risk-averse,
    ``p = 0`` is logarithmic, and ``p < 0`` increasingly emphasizes low
    rates. Rewards, durations, and selected observation weights must produce
    strictly positive local rates and weights.

    With theta = 0.0 and weight_parameter = -1, we get time-weighted step rate, which is the same as RelaxedSmart
    With theta = 2.0 and weight_parameter =  1, we get reward-weighted harmonic step rate, which is the same as RelaxedSmart
    """

    def __init__(
        self,
        name: str,
        action_space=None,
        learning_rate=0.1,
        exploration_rate=0.1,
        with_rho_trick=True,
        rho_learning_rate=0.3,
        theta=1.0,
        weight_parameter=0.0,
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
        self.p = 1.0-theta
        self.rate = NormalizedExponentialPowerMeanRate(self.p, rho_learning_rate)
        self.weight_parameter = weight_parameter
        self.reset()

    def reset(self):
        super().reset()
        self.rate.reset()

    def weight(self, r: float, t:float):
        """ translate weight parameter to a balance between ratio and time.
            parameter =  0 is weight=1.0.
            parameter =  1 is weight = reward.
            parameter = -1 is weight = time.
        """

        # Special cases handled for efficiency
        if self.weight_parameter == 0:
            return 1.0
        if self.weight_parameter == 1:
            return r
        if self.weight_parameter == -1:
            return t
        
        # ((r*t)**((self.weight_parameter**2)/2.0) * ((r/t)**(self.weight_parameter/2)        
    
        # for efficiency:  These are constants given the weight_parameter
        reward_power = self.weight_parameter * (self.weight_parameter + 1) / 2
        time_power = self.weight_parameter * (self.weight_parameter - 1) / 2
        return r**reward_power * t**time_power

    def set_target(self, reward, time, next_q):
        # continuous r-learning:  (reward - self.rho * time) + next_q

        step_rate = reward/time
        weight = self.weight(reward,time)
        target = weight*(crra(step_rate, self.theta) - crra(self.rho, self.theta)) + next_q

        # sanity check to be moved to a test
        if self.theta == 0 and weight==time:
            a_target = reward - self.rho * time + next_q
            if not math.isclose(a_target, target):
                raise ValueError("sanity target does not match.")
        # end of sanity check

        return target


    def calc_new_rho(self, reward, time, td_target, td_error):

        """ Cumulative version:
            rho is exactly weighted power mean with power p, weight = self.weight(r_i,t_i)
        """
        try:
            weight = self.weight(reward, time)
            self.rho = self.rate.update(reward, time, weight)
        except ValueError:
            if time == 0:
                raise ZeroDivisionError(
                    "RiskSmoothedR requires nonzero elapsed time"
                ) from None
            raise
