from .r_learning import ContinuousRLearning
from .average_rates import WeightedHarmonicRate


class WeightedHarmonic(ContinuousRLearning):
    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1, with_rho_trick=True,
                 rho_learning_rate=0.3, **kwargs):
        super().__init__(name, action_space, learning_rate, exploration_rate, with_rho_trick, rho_learning_rate,
                         **kwargs)
        self.hma = WeightedHarmonicRate(rho_learning_rate)
        self.reset()

    def reset(self):
        super().reset()
        self.hma.reset()

    def calc_new_rho(self, reward, time, td_target, td_error):
        self.rho = self.hma.update(reward, time, reward)  # Weighted HMA with weight = reward


class Harmonic(WeightedHarmonic):
    def calc_new_rho(self, reward, time, td_target, td_error):
        self.rho = self.hma.update(reward, time)  # Unweighted HMA: the default weight of 1.0
