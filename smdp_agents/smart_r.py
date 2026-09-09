from .average_rates import (CumulativeTimeRate,
                            NormalizedExponentialMovingTimeRate)
from .r_learning import ContinuousRLearning


class SMART(ContinuousRLearning):
    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1, with_rho_trick=True,
                 rho_learning_rate=0.3, **kwargs):
        super().__init__(name, action_space, learning_rate, exploration_rate, with_rho_trick, rho_learning_rate,
                         **kwargs)
        self.rate = CumulativeTimeRate()

    def reset(self):
        super().reset()
        self.rate.reset()

    def calc_new_rho(self, reward, time, td_target, td_error):
        try:
            self.rho = self.rate.update(reward, time, 1.0)
        except ValueError:
            if time == 0:
                raise ZeroDivisionError("SMART requires nonzero elapsed time") from None
            raise

    @property
    def total_time(self):
        return self.rate.total_duration

    @property
    def total_reward(self):
        return self.rate.total_reward


class SmoothedSMART(ContinuousRLearning):
    """SMART with the cumulative rate replaced by one smoothed in *elapsed time*.

    SMART's ``rho`` is the whole-history ``sum(r) / sum(tau)``, which cannot
    forget; ``RelaxedSMART`` makes it forgetful by smoothing the reward and the
    duration separately and dividing, which forgets **per transition**.  This
    agent forgets **per unit of elapsed time** instead::

        rho <- exp(-lambda * tau) * rho + (1 - exp(-lambda * tau)) * (r / tau)

    the exact integral of the continuous-time filter ``d(rho)/dt =
    lambda * (q(t) - rho)`` over the transition, where ``q(t) = r / tau`` is the
    realised rate while the action runs.  ``lambda`` is derived from
    ``rho_learning_rate`` as ``-log(1 - beta)``, so a transition of unit duration
    has exactly the gain ``beta`` a plain EMA would give it, and
    :class:`RelaxedSMART` at the same ``rho_learning_rate`` is the comparable
    estimator.

    Two things follow, and they are the reason to prefer this over the ratio of
    two smoothed quantities when the holding times vary:

    * **No ratio of two noisy filters.**  ``E[EMA(r) / EMA(tau)]`` is not
      ``E[r] / E[tau]``; the gap grows with the variance of the holding times and
      with the dependence between reward and duration.  There is only one filter
      here, so there is no ratio bias to acquire.
    * **Segmentation invariance.**  Because ``exp(-lambda * tau_1) *
      exp(-lambda * tau_2) == exp(-lambda * (tau_1 + tau_2))``, splitting one
      transition into two that cover the same time at the same rate leaves the
      estimate unchanged.  A per-transition EMA does not have this property: chop
      a run of transitions finer and its estimate moves, because its memory is
      measured in events rather than in seconds.

    Deliberately not a ``SMART`` subclass, for the same reason
    :class:`RelaxedSMART` is not: the smoothed rate replaces the cumulative one
    outright, so SMART's ``total_reward`` and ``total_time`` would describe an
    accumulator this agent never feeds.
    """

    def __init__(self, name: str, action_space=None, learning_rate=0.1,
                 exploration_rate=0.1, with_rho_trick=True,
                 rho_learning_rate=0.3, **kwargs):
        super().__init__(name, action_space, learning_rate, exploration_rate,
                         with_rho_trick, rho_learning_rate, **kwargs)
        self.time_rate = NormalizedExponentialMovingTimeRate(rho_learning_rate)

    @property
    def lambda_(self):
        """The forgetting rate per unit time, ``-log(1 - rho_learning_rate)``."""
        return self.time_rate.lambda_

    def reset(self):
        super().reset()
        self.time_rate.reset()

    def calc_new_rho(self, reward, time, td_target, td_error):
        try:
            self.rho = self.time_rate.update(reward, time)
        except ValueError as ex:
            if time == 0:
                raise ZeroDivisionError(
                    "Smoothed SMART requires nonzero elapsed time") from None
            raise ex
