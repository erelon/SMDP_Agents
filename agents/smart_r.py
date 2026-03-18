from .r_learning import ContinuousRLearning


class SMART(ContinuousRLearning):
    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1, with_rho_trick=True,
                 rho_learning_rate=0.3, **kwargs):
        super().__init__(name, action_space, learning_rate, exploration_rate, with_rho_trick, rho_learning_rate,
                         **kwargs)
        self.total_time = 0
        self.total_reward = 0
        self.step_count = 0
        self.total_totals = 0
        self.beta = rho_learning_rate

    def reset(self):
        super().reset()
        self.total_time = 0
        self.total_reward = 0
        self.step_count = 0
        self.total_totals = 0

    def calc_new_rho(self, reward, time, td_target, td_error):
        self.step_count += 1
        self.total_time += time
        self.total_reward += reward
        # SMART
        self.rho = self.total_reward / self.total_time


class RelaxedSMART(SMART):
    """
    Continuous Reinforcement Learning Agent based on Gostabi 2004: Relaxed SMART.
    This agent is designed for environments with continuous rewards.
    """

    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1, with_rho_trick=True,
                 rho_learning_rate=0.3, **kwargs):
        super().__init__(
            name, action_space, learning_rate, exploration_rate, with_rho_trick, rho_learning_rate, **kwargs
        )
        self.rho_time = 0
        self.rho_reward = 0

    def reset(self):
        super().reset()
        self.rho_time = 0
        self.rho_reward = 0

    def calc_new_rho(self, reward, time, td_target, td_error):
        super().calc_new_rho(reward, time, td_target, td_error)  # Really, only needed to update the step count

        # Now override whatever super() did for self.rho
        b1 = self.rho_learning_rate
        b2 = self.rho_learning_rate

        self.rho_time = (1 - b1) * self.rho_time + b1 * time
        self.rho_reward = (1 - b2) * self.rho_reward + b2 * reward

        self.rho = self.rho_reward / self.rho_time
