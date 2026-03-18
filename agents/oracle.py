from .base import Agent

class Oracle(Agent):
    def __init__(self, name: str, action_space=None, env_secret=None):
        super().__init__(name, action_space)
        if env_secret is None:
            raise ValueError("OracleAgent requires an environment secret to provide optimal actions.")
        self.env_secret = env_secret

    def reset(self):
        pass

    def act(self, state):
        return self.env_secret(state)

    def eval(self, state):
        return self.act(state)

    def learn(self, state, action, reward, next_state, time):
        pass

