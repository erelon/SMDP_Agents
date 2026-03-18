import copy
from random import Random
from typing import Any, Dict, List, Optional

MAX_REWARDS = 0 

class Agent:
    def __init__(self, name: str, action_space: Optional[List[int]] = None, seed: int = 42, env=None, **kwargs):
        self.name = name
        self.q_table: Dict[Any, Dict[int, float]] = {}
        if action_space is None:
            raise ValueError("Action space must be provided for the agent.")
        self.action_space = action_space
        self.env = env  # Store environment reference to query available actions
        self.seed = seed
        self.rng = Random(self.seed)
        self.rng.seed(self.seed)
        self.policy_changed = False
        self.last_policy_changed_at = 0  # track last episode of policy change

    def get_available_actions(self, state):
        """Get available actions for a state. If env is set, use it; otherwise use full action_space."""
        if self.env is not None and hasattr(self.env, 'get_available_actions'):
            return self.env.get_available_actions(state)
        return self.action_space

    def __repr__(self):
        return f"Agent(name={self.name})"

    def set_seed(self, seed):
        self.seed = seed

    def reset(self):
        self.rng = Random(self.seed)
        self.rng.seed(self.seed)
        self.q_table = {}
        self.policy_changed = False
        self.last_policy_changed_at = 0

    def act(self, state):
        raise NotImplementedError

    def eval(self, state):
        raise NotImplementedError

    def learn(self, state, action, reward, next_state, time):
        raise NotImplementedError

    def _check_convergence(self, state, action, update_value, assign=False):
        if len(self.q_table) == 0 or state not in self.q_table:
            return False
        best_current_action = max(self.q_table[state], key=self.q_table[state].get)
        table = copy.copy(self.q_table[state])
        if not assign:
            table[action] += update_value
        else:
            table[action] = update_value
        best_action_after_update = max(table, key=table.get)
        self.policy_changed = best_current_action != best_action_after_update
        if self.policy_changed:
            # caller should set last_policy_changed_at externally (episode index)
            pass

    def get_policy_changed(self):
        return self.policy_changed
