"""Two two-step paths: take the small reward first, or the big one first.

::

    state0 --A (r=1)--> state1a --A (r=100)--> state2a  [terminal]   total 101
    state0 --B (r=50)-> state1b --B (r=50)---> state2b  [terminal]   total 100

Path A is worth more in total but pays almost nothing up front; path B front-loads
its reward. Every transition takes ``tau = 1``, so this is a plain MDP — the only
thing it probes is how far ahead an agent looks. A myopic or heavily discounted
agent takes B; anything valuing the episode's total takes A. With two decisions
per episode and a fully deterministic layout, an agent's choice can be read
straight off its Q-table, which is what makes it the fixture for
``examples/hvsa_demo.py``.

Once past ``state0`` the path is forced — ``get_available_actions`` offers exactly
one action — so the second decision is bookkeeping, not choice. The terminal
states are absorbing rather than action-less, for the reason noted on
:data:`TRANSITIONS`.

Moved from ``two_path_env.py`` at the repository root; the observation is now the
state index rather than a bare int-as-obs, and ``info`` carries the holding time
required by the contract.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from gymnasium import spaces

from .base import SMDPEnv

# --- State / action indices --------------------------------------------------
STATE0 = 0
STATE1A = 1
STATE1B = 2
STATE2A = 3
STATE2B = 4
ACTION_A = 0
ACTION_B = 1

# (state, action) -> (next_state, reward, terminated)
#
# The two terminal states carry a zero-reward self-loop. Nothing ever takes it —
# ``step`` has already reported ``terminated`` by then and a runner resets — but
# the agents in ``agents/`` bootstrap unconditionally from ``eval(next_state)``,
# so a terminal state needs *some* entry in the Q-table for the final update to
# be well defined. A single absorbing action worth 0 makes that bootstrap
# exactly the terminal value it should be.
TRANSITIONS = {
    (STATE0, ACTION_A): (STATE1A, 1, False),
    (STATE0, ACTION_B): (STATE1B, 50, False),
    (STATE1A, ACTION_A): (STATE2A, 100, True),
    (STATE1B, ACTION_B): (STATE2B, 50, True),
    (STATE2A, ACTION_A): (STATE2A, 0, True),
    (STATE2B, ACTION_B): (STATE2B, 0, True),
}
VALID_ACTIONS = {
    STATE0: [ACTION_A, ACTION_B],
    STATE1A: [ACTION_A],
    STATE1B: [ACTION_B],
    STATE2A: [ACTION_A],
    STATE2B: [ACTION_B],
}
STATE_NAMES = {0: "state0", 1: "state1a", 2: "state1b",
               3: "state2a (terminal)", 4: "state2b (terminal)"}
ACTION_NAMES = {ACTION_A: "A", ACTION_B: "B"}


class TwoPathEnv(SMDPEnv):
    """The two-path chain above, with unit holding times."""

    metadata = {"render_modes": ["human", "ansi"]}
    is_mdp = True

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.observation_space = spaces.Discrete(5)
        self.action_space = spaces.Discrete(2)
        self.render_mode = render_mode
        self._state = STATE0
        self.time = 0.0

    # ------------------------------------------------------------------ hooks
    def get_available_actions(self, state: Optional[int] = None) -> List[int]:
        """The single legal continuation, or both actions at the start."""
        return list(VALID_ACTIONS[self._state if state is None else int(state)])

    def secret(self):
        """Take the delayed jackpot: A wherever there is a choice."""
        return lambda state: (ACTION_B if int(state) in (STATE1B, STATE2B)
                              else ACTION_A)

    # ---------------------------------------------------------------- gym api
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._state = STATE0
        self.time = 0.0
        info = self._info(0.0, state_name=STATE_NAMES[self._state])
        if self.render_mode == "human":
            print(f"Reset  ->  {STATE_NAMES[self._state]}")
        return self._state, info

    def step(self, action: int):
        action = int(action)
        valid = VALID_ACTIONS[self._state]
        if action not in valid:
            raise ValueError(
                f"Action {ACTION_NAMES.get(action, action)!r} invalid in "
                f"{STATE_NAMES[self._state]!r}. Valid: {[ACTION_NAMES[a] for a in valid]}"
            )
        next_state, reward, terminated = TRANSITIONS[(self._state, action)]
        self._state = next_state
        self.time += 1.0
        info = self._info(1.0, state_name=STATE_NAMES[self._state],
                          action_name=ACTION_NAMES[action])
        if self.render_mode == "human":
            print(f"  action={ACTION_NAMES[action]}  reward={reward:>4}  "
                  f"->  {STATE_NAMES[self._state]}"
                  + ("  [DONE]" if terminated else ""))
        return next_state, float(reward), terminated, False, info

    # ----------------------------------------------------------------- render
    def render(self):
        text = f"Current state: {STATE_NAMES[self._state]}"
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def close(self):
        pass


if __name__ == "__main__":
    env = TwoPathEnv(render_mode="human")
    print("=== Path A  (total reward = 101) ===")
    env.reset()
    env.step(ACTION_A)
    env.step(ACTION_A)
    print("\n=== Path B  (total reward = 100) ===")
    env.reset()
    env.step(ACTION_B)
    env.step(ACTION_B)
    env.close()
