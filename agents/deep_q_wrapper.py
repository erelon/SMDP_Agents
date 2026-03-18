import copy
import collections
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .q_learning import ContinuousQLearning


class DeepQWrapper(ContinuousQLearning):
    """
    A wrapper that converts any ContinuousQLearning-derived agent into a Deep Q-Network agent.

    The Q-table is replaced by a PyTorch nn.Module. All variant-specific hooks
    (set_target, calc_new_rho, etc.) from the wrapped agent's class chain are preserved.

    Args:
        agent (ContinuousQLearning): An instantiated agent whose class hierarchy provides
            the TD-target and rho-update logic (e.g. RLearning, SMART, Harmonic, ...).
        network (nn.Module): PyTorch model with input shape matching the encoded state
            and output shape (len(action_space),) — one Q-value per action.
        replay_buffer_size (int | None): If set, enables experience replay with a deque
            of this capacity. A gradient step is only taken when the buffer holds at
            least `batch_size` transitions.
        batch_size (int): Number of transitions sampled per gradient step when replay
            is enabled. Ignored when replay_buffer_size is None. Default: 32.
        target_update_freq (int | None): If set, a frozen copy of `network` is used to
            compute next-state Q-values, and its weights are synced every
            `target_update_freq` learn() calls.
    """

    def __init__(
            self,
            agent: ContinuousQLearning,
            network: nn.Module,
            replay_buffer_size: Optional[int] = None,
            batch_size: int = 32,
            target_update_freq: Optional[int] = None,
    ):
        # Copy all attributes from the wrapped agent so that subclass hooks
        # (set_target, calc_new_rho, update_table overrides) work transparently.
        # We deliberately do NOT call super().__init__ here because the wrapped
        # agent is already fully constructed; we just re-use its state.
        self.__class__ = type(
            f"Deep{type(agent).__name__}",
            (DeepQWrapper, type(agent)),
            {},
        )
        self.__dict__.update(copy.deepcopy(agent.__dict__))

        # ── Neural network ────────────────────────────────────────────────────
        self.network = network
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        # ── Target network (optional) ─────────────────────────────────────────
        self.target_update_freq = target_update_freq
        self.target_network: Optional[nn.Module] = None
        if target_update_freq is not None:
            self.target_network = copy.deepcopy(network)
            for p in self.target_network.parameters():
                p.requires_grad_(False)

        # ── Replay buffer (optional) ──────────────────────────────────────────
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.replay_buffer: Optional[collections.deque] = None
        if replay_buffer_size is not None:
            self.replay_buffer = collections.deque(maxlen=replay_buffer_size)

        # Internal counters
        self._learn_step = 0

    # ── State encoding ────────────────────────────────────────────────────────

    @staticmethod
    def _to_tensor(state) -> torch.Tensor:
        """Convert a Python scalar/list or numpy array to a float32 1-D tensor."""
        if isinstance(state, torch.Tensor):
            t = state.float()
        elif isinstance(state, np.ndarray):
            t = torch.from_numpy(state).float()
        elif isinstance(state, (list, tuple)):
            t = torch.tensor(state, dtype=torch.float32)
        else:
            # scalar
            t = torch.tensor([state], dtype=torch.float32)
        return t.flatten()

    # ── Q-value helpers ───────────────────────────────────────────────────────

    def _q_values(self, state, network: nn.Module) -> torch.Tensor:
        """Return a 1-D tensor of Q-values (one per action) from `network`."""
        s = self._to_tensor(state).unsqueeze(0)  # (1, state_dim)
        with torch.no_grad():
            q = network(s).squeeze(0)  # (n_actions,)
        return q

    def _q_values_grad(self, state) -> torch.Tensor:
        """Return Q-values from the main network *with* gradients (for training)."""
        s = self._to_tensor(state).unsqueeze(0)
        return self.network(s).squeeze(0)  # (n_actions,)

    def _action_index(self, action) -> int:
        return self.action_space.index(action)

    # ── ContinuousQLearning API overrides ─────────────────────────────────────

    def initialize_table(self, state):
        """No-op: the network replaces the Q-table."""
        pass

    def eval(self, state):
        """Return the greedy action according to the network."""
        q = self._q_values(state, self.network)
        idx = int(q.argmax().item())
        return self.action_space[idx]

    def act(self, state):
        """ε-greedy action selection using the network for exploitation."""
        if self.rng.random() < self.exploration_rate:
            return self.rng.choice(self.action_space)
        return self.eval(state)

    # ── Core learn step ───────────────────────────────────────────────────────

    def _gradient_step(self, states, actions, td_targets):
        """
        Perform one gradient descent step.

        Args:
            states:     list of raw states (will be encoded individually).
            actions:    list of actions taken.
            td_targets: list of scalar TD targets.
        """
        self.optimizer.zero_grad()

        loss = torch.tensor(0.0, requires_grad=True)
        for s, a, tgt in zip(states, actions, td_targets):
            q_vals = self._q_values_grad(s)  # (n_actions,)
            q_sa = q_vals[self._action_index(a)]  # scalar with grad
            target_t = torch.tensor(float(tgt), dtype=torch.float32)
            loss = loss + self.loss_fn(q_sa.unsqueeze(0), target_t.unsqueeze(0))

        loss = loss / len(states)
        loss.backward()
        self.optimizer.step()

    def _compute_td_target(self, state, action, reward, next_state, time):
        """
        Compute the TD target using the wrapped agent's set_target logic and
        the appropriate network (target or main) for next-state Q-values.
        """
        net = self.target_network if self.target_network is not None else self.network
        next_q_vals = self._q_values(next_state, net)
        best_next_q = float(next_q_vals.max().item())
        return self.set_target(reward, time, best_next_q)

    def learn(self, state, action, reward, next_state, time):
        # ── Compute TD quantities using current Q-network ──────────────────
        td_target = self._compute_td_target(state, action, reward, next_state, time)
        q_vals_now = self._q_values(state, self.network)
        q_sa_now = float(q_vals_now[self._action_index(action)].item())
        td_error = td_target - q_sa_now

        # ── Subclass side-effects (rho updates, convergence check, etc.) ───
        self._check_convergence(state, action, self.learning_rate * td_error)

        # Call calc_new_rho if the wrapped class defines it (R-Learning variants)
        if hasattr(self, 'calc_new_rho'):
            best_next_action = self.action_space[int(self._q_values(next_state, self.network).argmax())]
            best_old_action = self.eval(state)
            onpolicy = (action == best_old_action)
            self.calc_new_rho(reward, time, td_target, td_error)

        # ── Gradient update ────────────────────────────────────────────────
        if self.replay_buffer is not None:
            self.replay_buffer.append((state, action, reward, next_state, time))
            if len(self.replay_buffer) >= self.batch_size:
                batch = random.sample(self.replay_buffer, self.batch_size)
                b_states, b_actions, b_rewards, b_next_states, b_times = zip(*batch)
                b_targets = [
                    self._compute_td_target(s, a, r, ns, t)
                    for s, a, r, ns, t in zip(b_states, b_actions, b_rewards, b_next_states, b_times)
                ]
                self._gradient_step(list(b_states), list(b_actions), b_targets)
        else:
            self._gradient_step([state], [action], [td_target])

        # ── Target network sync ────────────────────────────────────────────
        self._learn_step += 1
        if (
                self.target_network is not None
                and self._learn_step % self.target_update_freq == 0
        ):
            self.target_network.load_state_dict(copy.deepcopy(self.network.state_dict()))

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self):
        super().reset()
        # Re-initialise optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        # Clear replay buffer
        if self.replay_buffer is not None:
            self.replay_buffer.clear()
        # Re-sync target network
        if self.target_network is not None:
            self.target_network.load_state_dict(copy.deepcopy(self.network.state_dict()))
        self._learn_step = 0
