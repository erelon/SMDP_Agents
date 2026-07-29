from .base import Agent


class ContinuesMAB(Agent):
    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1, **kwargs):
        super().__init__(name, action_space, **kwargs)
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.total_time = {}
        self.total_reward = {}

    def reset(self):
        super().reset()
        self.q_table = {}
        self.total_time = {}
        self.total_reward = {}

    def act(self, state):
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.action_space}
        if self.rng.random() < self.exploration_rate:
            return self.rng.choice(self.action_space)
        return max(self.q_table[state], key=self.q_table[state].get)

    def eval(self, state):
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.action_space}
        return max(self.q_table[state], key=self.q_table[state].get)

    def learn(self, state, action, reward, next_state, time):
        super().learn(state, action, reward, next_state, time)
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.action_space}
        if state not in self.total_time:
            self.total_time[state] = {action: 0 for action in self.action_space}
            self.total_reward[state] = {action: 0 for action in self.action_space}
        self.total_time[state][action] += time
        self.total_reward[state][action] += reward
        if self.total_time[state][action] == 0:
            return
        self._check_convergence(state, action, self.total_reward[state][action] / self.total_time[state][action], True)
        self.q_table[state][action] = self.total_reward[state][action] / self.total_time[state][action]


class MAB(Agent):
    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1, **kwargs):
        super().__init__(name, action_space, **kwargs)
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.total_steps = {}
        self.total_reward = {}

    def reset(self):
        super().reset()
        self.q_table = {}
        self.total_steps = {}
        self.total_reward = {}

    def act(self, state):
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.action_space}
        if self.rng.random() < self.exploration_rate:
            return self.rng.choice(self.action_space)
        return max(self.q_table[state], key=self.q_table[state].get)

    def eval(self, state):
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.action_space}
        return max(self.q_table[state], key=self.q_table[state].get)

    def learn(self, state, action, reward, next_state, time):
        super().learn(state, action, reward, next_state, time)
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.action_space}
        if state not in self.total_steps:
            self.total_steps[state] = {action: 0 for action in self.action_space}
            self.total_reward[state] = {action: 0 for action in self.action_space}
        self.total_steps[state][action] += 1
        self.total_reward[state][action] += reward
        if self.total_steps[state][action] == 0:
            return
        self._check_convergence(state, action, self.total_reward[state][action] / self.total_steps[state][action], True)
        self.q_table[state][action] = self.total_reward[state][action] / self.total_steps[state][action]
