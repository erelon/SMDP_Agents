from .base import Agent


class RandomAgent(Agent):
    def __init__(self, name: str, action_space=None, **kwargs):
        super().__init__(name, action_space, **kwargs)

    def reset(self):
        super().reset()

    def act(self, state):
        return self.rng.choice(self.get_available_actions(state))

    def eval(self, state):
        return self.rng.choice(self.get_available_actions(state))

    def learn(self, state, action, reward, next_state, time):
        super().learn(state, action, reward, next_state, time)
