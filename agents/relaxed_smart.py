from .average_rates import ExponentialMovingRatioRate
from .r_learning import ContinuousRLearning


class RelaxedSMART(ContinuousRLearning):
    """Continuous reinforcement-learning agent using the Relaxed SMART rate.

    Deliberately not a ``SMART`` subclass.  The exponentially smoothed rate
    replaces the cumulative one outright, so SMART's ``total_reward`` and
    ``total_time`` would describe an accumulator this agent never feeds; a
    caller reaching for them should get an ``AttributeError``, not a zero.
    """

    def __init__(self, name: str, action_space=None, learning_rate=0.1,
                 exploration_rate=0.1, with_rho_trick=True,
                 rho_learning_rate=0.3, **kwargs):
        super().__init__(
            name, action_space, learning_rate, exploration_rate,
            with_rho_trick, rho_learning_rate, **kwargs
        )
        self.ratio_rate = ExponentialMovingRatioRate(rho_learning_rate)

    @property
    def rho_reward(self):
        return self.ratio_rate.mean_reward

    @property
    def rho_time(self):
        return self.ratio_rate.mean_duration

    def reset(self):
        super().reset()
        self.ratio_rate.reset()

    def calc_new_rho(self, reward: float, time: float, td_target, td_error):
        try:
            self.rho = self.ratio_rate.update(reward, time)
        except ValueError:
            if time == 0:
                raise ZeroDivisionError("Relaxed SMART requires nonzero elapsed time") from None
            raise
