from .q_learning import ContinuousQLearning

class ContinuousRLearning(ContinuousQLearning):
    def __init__(self, name: str, action_space=None, learning_rate=0.2, exploration_rate=0.1, with_rho_trick=True,
                 rho_learning_rate=0.03, **kwargs):
        super().__init__(name, action_space, learning_rate, exploration_rate=exploration_rate, **kwargs)
        self.with_rho_trick = with_rho_trick
        self.rho_learning_rate = rho_learning_rate

    def calc_new_rho(self, reward, time, td_target, td_error):
        self.rho += self.rho_learning_rate * td_error

    def set_target(self, reward, time, next_q):
        return (reward - self.rho * time) + next_q

    def update_table(self, state, action, reward, time, td_target, td_error, onpolicy):
        super().update_table(state, action, reward, time, td_target, td_error, onpolicy)
        if not self.with_rho_trick or (self.with_rho_trick and onpolicy):
            self.calc_new_rho(reward, time, td_target, td_error)


class RLearning(ContinuousRLearning):
    def learn(self, state, action, reward, next_state, time):
        super().learn(state, action, reward, next_state, 1.0)
