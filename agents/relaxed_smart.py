from .average_rates import ExponentialMovingAverage
from .r_learning import ContinuousRLearning
from .smart_r import SMART


class RelaxedSMART(SMART):
    """Continuous reinforcement-learning agent using the Relaxed SMART rate."""

    def __init__(self, name: str, action_space=None, learning_rate=0.1,
                 exploration_rate=0.1, with_rho_trick=True,
                 rho_learning_rate=0.3, **kwargs):
        super().__init__(
            name, action_space, learning_rate, exploration_rate,
            with_rho_trick, rho_learning_rate, **kwargs
        )
        self.reward_ema = ExponentialMovingAverage(rho_learning_rate)
        self.duration_ema = ExponentialMovingAverage(rho_learning_rate)

    @property
    def rho_reward(self):
        return self.reward_ema.value

    @property
    def rho_time(self):
        return self.duration_ema.value

    def reset(self):
        super().reset()
        self.reward_ema.reset()
        self.duration_ema.reset()

    def calc_new_rho(self, reward: float, time: float, td_target, td_error):
        # Retain SMART's counters, then replace its cumulative estimate with
        # the Relaxed SMART exponentially smoothed estimate.
        # super().calc_new_rho(reward, time, td_target, td_error)
        reward_ema = self.reward_ema.update(reward, 1.0)
        duration_ema = self.duration_ema.update(time, 1.0)
        try:
            self.rho = reward_ema / duration_ema
        except: 
            if duration_ema == 0:
                raise ZeroDivisionError("Relaxed SMART requires nonzero elapsed time") from None
            raise

