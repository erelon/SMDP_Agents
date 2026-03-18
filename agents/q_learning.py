from .base import Agent, MAX_REWARDS


class ContinuousQLearning(Agent):
    def __init__(self, name: str, action_space=None, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1,
                 _lambda=0.01, **kwargs):
        super().__init__(name, action_space, **kwargs)
        self._lambda = _lambda
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = {}
        self.policy_changed = False
        self.rho = 0
        self.q_table_past_a = []
        self.q_table_past_b = []

    def reset(self):
        self.q_table = {}
        self.rho = 0

    def initialize_table(self, state):
        if state not in self.q_table:
            available_actions = self.get_available_actions(state)
            self.q_table[state] = {action: MAX_REWARDS for action in available_actions}

    def eval(self, state):  # returns best action in state
        return max(self.q_table[state], key=self.q_table[state].get)

    def act(self, state):
        self.initialize_table(state)
        if self.rng.random() < self.exploration_rate:
            available_actions = list(self.q_table[state].keys())
            return self.rng.choice(available_actions)
        return self.eval(state)

    def set_target(self, reward, time, next_q):
        df = self.discount_factor ** time
        return reward + df * next_q


    def update_table(self, state, action, reward, time, td_target, td_error, onpolicy):
        self.q_table[state][action] += self.learning_rate * td_error
        self.q_table_past_a.append(self.q_table["s1"][0])
        self.q_table_past_b.append(self.q_table["s1"][1])

    def learn(self, state, action, reward, next_state, time):
        self.initialize_table(next_state)
        best_next_action = self.eval(next_state)
        # Sometimes, it matters to the table updates -- see r-learning on-policy updates
        best_old_action = self.eval(state)
        td_target = self.set_target(reward, time, self.q_table[next_state][best_next_action])
        td_error = td_target - self.q_table[state][action]

        self._check_convergence(state, action, self.learning_rate * td_error)
        self.update_table(state, action, reward, time, td_target, td_error, (action == best_old_action))


class QLearning(ContinuousQLearning):
    def learn(self, state, action, reward, next_state, time):
        super().learn(state, action, reward, next_state, 1.0)
