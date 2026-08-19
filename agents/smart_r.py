from .average_rates import CumulativeTimeRate
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
