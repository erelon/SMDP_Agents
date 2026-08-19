from .base import Agent


class RandomAgent(Agent):
    def __init__(self, name: str, action_space=None):
        super().__init__(name, action_space)

    def reset(self):
        super().reset()

    def act(self, state):
        return self.rng.choice(self.action_space)

    def eval(self, state):
        return self.rng.choice(self.action_space)

    def learn(self, state, action, reward, next_state, time):
        super().learn(state, action, reward, next_state, time)
