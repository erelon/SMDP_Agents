from .r_learning import ContinuousRLearning


class SMART(ContinuousRLearning):
    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1, with_rho_trick=True,
                 rho_learning_rate=0.3, **kwargs):
        super().__init__(name, action_space, learning_rate, exploration_rate, with_rho_trick, rho_learning_rate,
                         **kwargs)
        self.total_time = 0
        self.total_reward = 0
        # self.step_count = 0
        # self.total_totals = 0
        self.beta = rho_learning_rate

    def reset(self):
        super().reset()
        self.total_time = 0
        self.total_reward = 0
        # self.step_count = 0
        # self.total_totals = 0

    def calc_new_rho(self, reward, time, td_target, td_error):
        self.total_time += time
        self.total_reward += reward
        # SMART
        self.rho = self.total_reward / self.total_time
