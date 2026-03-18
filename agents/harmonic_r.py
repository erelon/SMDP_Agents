from .r_learning import ContinuousRLearning


class WeightedHarmonic(ContinuousRLearning):
    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1, with_rho_trick=True,
                 rho_learning_rate=0.3, **kwargs):
        super().__init__(name, action_space, learning_rate, exploration_rate, with_rho_trick, rho_learning_rate,
                         **kwargs)
        self.reset()

    def reset(self):
        super().reset()
        self.pos_reciprocal_rho = 0.0
        self.neg_reciprocal_rho = 0.0
        self.neg_w1 = 0
        self.neg_w2 = 0
        self.pos_w1 = 0
        self.pos_w2 = 0
        self.zero_w = 0

    def HMA_rho(self, reward, time, weight=1.0):
        pos = 1 if reward > 0 else 0
        neg = 1 if reward < 0 else 0
        zero = 1 if reward == 0 else 0

        reciprocal_rate = 0 if zero == 1 else time / reward
        self.pos_reciprocal_rho = ((1 - self.rho_learning_rate) * self.pos_reciprocal_rho +
                                   self.rho_learning_rate * reciprocal_rate * pos * weight)
        self.pos_w1 = (1 - self.rho_learning_rate) * self.pos_w1 + self.rho_learning_rate * pos * weight
        self.pos_w2 = (1 - self.rho_learning_rate) * self.pos_w2 + self.rho_learning_rate * pos * 1

        self.neg_reciprocal_rho = ((1 - self.rho_learning_rate) * self.neg_reciprocal_rho +
                                   self.rho_learning_rate * reciprocal_rate * neg * weight)
        self.neg_w1 = (1 - self.rho_learning_rate) * self.neg_w1 + self.rho_learning_rate * neg * weight
        self.neg_w2 = (1 - self.rho_learning_rate) * self.neg_w2 + self.rho_learning_rate * neg * 1

        H_pos = 0 if self.pos_reciprocal_rho == 0 else self.pos_w1 / self.pos_reciprocal_rho

        H_neg = 0 if self.neg_reciprocal_rho == 0 else self.neg_w1 / self.neg_reciprocal_rho
        self.zero_w = (1 - self.rho_learning_rate) * self.zero_w + self.rho_learning_rate * zero

        self.rho = (H_pos * self.pos_w2 + H_neg * self.neg_w2) / (self.pos_w2 + self.neg_w2 + self.zero_w)

    def calc_new_rho(self, reward, time, td_target, td_error):
        self.HMA_rho(reward, time, reward)  # Weighted HMA with weight = reward


class Harmonic(WeightedHarmonic):
    def calc_new_rho(self, reward, time, td_target, td_error):
        self.HMA_rho(reward, time, 1.0)  # Weighted HMA with weight = 1.0