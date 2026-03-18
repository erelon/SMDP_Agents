import math
from .base import Agent

class ContinuesMAB(Agent):
    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1, **kwargs):
        super().__init__(name, action_space, **kwargs)
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.total_time = {}
        self.total_reward = {}

    def reset(self):
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

class UCB(Agent):
    def __init__(self, name: str, action_space=None, exploration_constant=1.0, **kwargs):
        super().__init__(name, action_space, **kwargs)
        self.exploration_constant = exploration_constant
        self.total_steps = {}
        self.total_reward = {}

    def reset(self):
        self.q_table = {}
        self.total_steps = {}
        self.total_reward = {}

    def act(self, state):
        if state not in self.q_table:
            self.q_table[state] = {action: 0.000000000001 for action in self.action_space}
        if state not in self.total_steps:
            self.total_steps[state] = {action: 1 for action in self.action_space}
            self.total_reward[state] = {action: 0.000000000001 for action in self.action_space}
        ucb_values = {action: (self.q_table[state][action] + self.exploration_constant * math.sqrt(2 * (math.log(sum(self.total_steps[state].values())) / (self.total_steps[state][action])))) for action in self.action_space}
        return max(ucb_values, key=ucb_values.get)

    def eval(self, state):
        return max(self.q_table[state], key=self.q_table[state].get)

    def learn(self, state, action, reward, next_state, time):
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.action_space}
        if state not in self.total_steps:
            self.total_steps[state] = {action: 0 for action in self.action_space}
            self.total_reward[state] = {action: 0 for action in self.action_space}
        self.total_steps[state][action] += 1
        self.total_reward[state][action] += reward
        self._check_convergence(state, action, self.total_reward[state][action] / self.total_steps[state][action], True)
        self.q_table[state][action] = self.total_reward[state][action] / self.total_steps[state][action]

class ContinuosUCB(Agent):
    def __init__(self, name: str, action_space=None, exploration_constant=1.0, **kwargs):
        super().__init__(name, action_space, **kwargs)
        self.exploration_constant = exploration_constant
        self.reset()

    def reset(self):
        self.q_table = {}
        self.total_time = {}
        self.total_reward = {}
        self.total_count = {}

    def act(self, state):
        if state not in self.q_table:
            self.q_table[state] = {action: 0.000001 for action in self.action_space}
        if state not in self.total_time:
            self.total_time[state] = {action: 0.000000000001 for action in self.action_space}
            self.total_count[state] = {action: 1 for action in self.action_space}
            self.total_reward[state] = {action: 0 for action in self.action_space}
        ucb_values = {action: (self.q_table[state][action] + self.exploration_constant * math.sqrt(2 * (math.log(sum(self.total_count[state].values())) / (self.total_count[state][action])))) for action in self.action_space}
        return max(ucb_values, key=ucb_values.get)

    def eval(self, state):
        return max(self.q_table[state], key=self.q_table[state].get)

    def learn(self, state, action, reward, next_state, time):
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.action_space}
        if state not in self.total_time:
            self.total_time[state] = {action: 0 for action in self.action_space}
            self.total_reward[state] = {action: 0 for action in self.action_space}
            self.total_count[state] = {action: 1 for action in self.action_space}
        self.total_time[state][action] += time
        self.total_reward[state][action] += reward
        self._check_convergence(state, action, self.total_reward[state][action] / self.total_time[state][action], True)
        self.q_table[state][action] = (self.total_reward[state][action] / self.total_time[state][action])

