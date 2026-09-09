"""Whack-a-Mole: the same world under a unit-time and a distance-time control.

The *underlying* process is always semi-Markov: the agent occupies a cell of an
``rows x cols`` grid, every action consumes an amount of time ``tau``, and while
that time elapses the moles evolve on their own. What differs between the two
environments is the control interface, and therefore whether the decision
process the agent faces is an MDP or an SMDP:

``WhackAMoleMDP``
    King moves only — step to one of the 8 neighbours or stay — and every action
    costs exactly ``tau = 1``. Unit holding times collapse the SMDP formulas onto
    their MDP form, so this is the control experiment.
``WhackAMoleSMDP``
    The agent may jump to *any* cell, paying the travel time
    ``tau = time_per_distance * D[i, j]`` from a Euclidean distance matrix (so a
    diagonal costs ~1.414 and a jump across the board costs several units).
    Whacking costs ``whack_time``.

Because the two share their dynamics and rewards, the pair isolates one
variable: whether holding times vary. That is what makes it the headline
comparison — an agent that estimates reward *per unit time* should gain nothing
on the MDP and win on the SMDP.

Mole dynamics (``mole_dynamics``)
    ``"until_whacked"`` — a down hole pops up at rate ``mole_up_prob`` per unit
    time and, once up, stays up until whacked. ``"up_down"`` — up holes also drop
    back down at rate ``mole_down_prob``, so the signal is sparser and the task
    considerably harder. Over a holding time ``tau`` a per-unit-time probability
    ``p`` compounds as ``1 - (1 - p) ** tau``; at ``tau == 1`` that is just ``p``.

Reward modes (``reward_mode``)
    ``"whack"`` — +1 per successful whack. ``"whack_downed"`` — on a successful
    whack, the number of currently-down holes, so a clean board pays more and the
    rewards compound. ``"step_downed"`` — the number of down holes every step;
    **MDP only**, since a per-step reward is ill-defined when steps have
    different lengths (the SMDP analogue would be a reward *rate* integrated over
    ``tau``, which is a different environment).

Ported from ``whackAmole/whack_a_mole_smdp.py``. The one behavioural change:
``get_available_actions`` now honours the environment's own action mask, so
agents no longer keep table entries for the redundant off-grid king moves or for
the illegal self-move. Numbers are therefore not directly comparable with that
repo's reports.
"""

from __future__ import annotations

from typing import Any, Dict, Hashable, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .base import SMDPEnv

# King-move offsets; index 4 == (0, 0) == stay in place.
_KING_OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),  (0, 0),  (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]

MOLE_DYNAMICS = ("until_whacked", "up_down")
REWARD_MODES = ("whack", "whack_downed", "step_downed")


class _WhackAMoleBase(SMDPEnv):
    """Shared SMDP core: state, mole dynamics, rewards, rendering.

    Subclasses define the action space and implement ``_decode_action`` (mapping
    a discrete action to ``(action_type, tau, target)``) and ``_action_mask``.
    """

    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        rows: int = 3,
        cols: int = 3,
        mole_dynamics: str = "until_whacked",
        reward_mode: str = "whack",
        mole_up_prob: float = 0.2,
        mole_down_prob: float = 0.1,
        whack_time: float = 1.0,
        init_evolve_time: float = 1.0,
        max_steps: Optional[int] = 200,
        max_time: Optional[float] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        if rows < 1 or cols < 1:
            raise ValueError("rows and cols must be >= 1")
        if mole_dynamics not in MOLE_DYNAMICS:
            raise ValueError(f"mole_dynamics must be one of {MOLE_DYNAMICS}")
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}")
        for name, p in (("mole_up_prob", mole_up_prob),
                        ("mole_down_prob", mole_down_prob)):
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

        self.rows = int(rows)
        self.cols = int(cols)
        self.n = self.rows * self.cols
        self.mole_dynamics = mole_dynamics
        self.reward_mode = reward_mode
        self.mole_up_prob = float(mole_up_prob)
        self.mole_down_prob = float(mole_down_prob)
        self.whack_time = float(whack_time)
        self.init_evolve_time = float(init_evolve_time)
        self.max_steps = None if max_steps is None else int(max_steps)
        self.max_time = None if max_time is None else float(max_time)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Cell coordinates and Euclidean distance matrix (diagonals ~ sqrt(2)).
        rr, cc = np.divmod(np.arange(self.n), self.cols)
        self._coords = np.stack([rr, cc], axis=1).astype(np.float64)
        diff = self._coords[:, None, :] - self._coords[None, :, :]
        self.distance_matrix = np.sqrt((diff ** 2).sum(-1))

        self.observation_space = spaces.Dict({
            "moles": spaces.MultiBinary(self.n),
            "agent": spaces.Discrete(self.n),
        })
        # action_space is set by the subclass.

        self.moles = np.zeros(self.n, dtype=bool)
        self.agent_pos = 0
        self.time = 0.0
        self.n_steps = 0

    # ------------------------------------------------------------------ hooks
    def state_of(self, obs: Dict[str, Any]) -> Hashable:
        """``(moles, agent_position)`` — hashable and Markov."""
        return tuple(int(x) for x in obs["moles"]), int(obs["agent"])

    def get_available_actions(self, state: Optional[Hashable] = None) -> List[int]:
        """The legal actions, from the mask for ``state``'s agent position."""
        pos = self.agent_pos if state is None else int(state[1])
        mask = self._mask_at(pos)
        return [a for a, ok in enumerate(mask) if ok]

    # ---------------------------------------------------------------- dynamics
    def _evolve(self, tau: float) -> None:
        """Advance the mole field by a holding time ``tau``."""
        if tau <= 0.0:
            return
        r = self.np_random.random(self.n)
        p_up = 1.0 - (1.0 - self.mole_up_prob) ** tau
        if self.mole_dynamics == "until_whacked":
            self.moles[(~self.moles) & (r < p_up)] = True
        else:  # up_down
            p_down = 1.0 - (1.0 - self.mole_down_prob) ** tau
            up = self.moles.copy()
            self.moles[(~up) & (r < p_up)] = True
            self.moles[up & (r < p_down)] = False

    def _downed(self) -> int:
        return self.n - int(self.moles.sum())

    def _reward(self, action_type: str, whack_success: bool) -> float:
        if self.reward_mode == "whack":
            return 1.0 if whack_success else 0.0
        if self.reward_mode == "whack_downed":
            return float(self._downed()) if whack_success else 0.0
        # step_downed (MDP-gated in the subclass)
        return float(self._downed())

    # ------------------------------------------------------------ to override
    def _decode_action(self, action: int):
        raise NotImplementedError

    def _mask_at(self, pos: int) -> np.ndarray:
        """The action mask that applies when the agent stands on ``pos``."""
        raise NotImplementedError

    def _action_mask(self) -> np.ndarray:
        return self._mask_at(self.agent_pos)

    # ----------------------------------------------------------------- gym api
    def _get_obs(self):
        return {"moles": self.moles.astype(np.int8), "agent": int(self.agent_pos)}

    def _get_info(self, action_type, tau, whack_success):
        return self._info(
            tau,
            n_steps=int(self.n_steps),
            action_type=action_type,          # "move" | "whack" | "wait" | "reset"
            whack_success=bool(whack_success),
            moles_up=int(self.moles.sum()),
            moles_down=self._downed(),
            agent_position=int(self.agent_pos),
            action_mask=self._action_mask(),
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.moles = np.zeros(self.n, dtype=bool)
        self.time = 0.0
        self.n_steps = 0
        start = 0
        if options and "agent_start" in options:
            start = int(options["agent_start"])
            if not 0 <= start < self.n:
                raise ValueError("agent_start out of range")
        self.agent_pos = start
        self._evolve(self.init_evolve_time)  # let the world run before arrival
        obs, info = self._get_obs(), self._get_info("reset", 0.0, False)
        if self.render_mode == "human":
            self.render()
        return obs, info

    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action} for {self.action_space}")

        action_type, tau, target = self._decode_action(action)

        whack_success = False
        if action_type == "whack":
            if self.moles[self.agent_pos]:
                self.moles[self.agent_pos] = False
                whack_success = True
        elif action_type == "move":
            self.agent_pos = int(target)
        # "wait": position and moles unchanged until the evolution below.

        reward = self._reward(action_type, whack_success)  # measured pre-evolve
        self._evolve(tau)
        self.time += tau
        self.n_steps += 1

        terminated = False  # continuing task: no terminal state
        truncated = (
            (self.max_steps is not None and self.n_steps >= self.max_steps)
            or (self.max_time is not None and self.time >= self.max_time)
        )
        obs = self._get_obs()
        info = self._get_info(action_type, tau, whack_success)
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, bool(truncated), info

    # ------------------------------------------------------------------ render
    def render(self):
        if self.render_mode in ("ansi", "human"):
            s = self._render_text()
            if self.render_mode == "human":
                print(s)
                return None
            return s
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        return None

    def _render_text(self) -> str:
        lines = [f"t={self.time:.2f} steps={self.n_steps} "
                 f"up={int(self.moles.sum())} down={self._downed()}"]
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                i = r * self.cols + c
                ch = "M" if self.moles[i] else "."
                row.append(f"[{ch}]" if i == self.agent_pos else f" {ch} ")
            lines.append("".join(row))
        return "\n".join(lines)

    def _render_rgb(self) -> np.ndarray:
        cell, pad = 32, 2
        img = np.full((self.rows * cell, self.cols * cell, 3), 30, np.uint8)
        for i in range(self.n):
            r, c = divmod(i, self.cols)
            y0, x0 = r * cell + pad, c * cell + pad
            y1, x1 = (r + 1) * cell - pad, (c + 1) * cell - pad
            img[y0:y1, x0:x1] = (40, 190, 40) if self.moles[i] else (70, 55, 45)
            if i == self.agent_pos:  # white border marks the agent
                img[y0:y0 + 3, x0:x1] = img[y1 - 3:y1, x0:x1] = 255
                img[y0:y1, x0:x0 + 3] = img[y0:y1, x1 - 3:x1] = 255
        return img

    def close(self):
        pass


class WhackAMoleMDP(_WhackAMoleBase):
    """Unit-time king-move control -> a plain MDP."""

    is_mdp = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 0..8 : king moves (incl. stay); 9 : whack.
        self.action_space = spaces.Discrete(10)

    def _decode_action(self, action):
        if action == 9:
            return "whack", 1.0, None
        dr, dc = _KING_OFFSETS[action]
        r, c = divmod(self.agent_pos, self.cols)
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.rows and 0 <= nc < self.cols:
            return "move", 1.0, nr * self.cols + nc
        return "move", 1.0, self.agent_pos  # off-grid -> stay, still costs 1

    def _mask_at(self, pos: int) -> np.ndarray:
        mask = np.ones(10, dtype=np.int8)
        r, c = divmod(int(pos), self.cols)
        for a, (dr, dc) in enumerate(_KING_OFFSETS):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                mask[a] = 0  # redundant off-grid move (stay is action 4)
        return mask


class WhackAMoleSMDP(_WhackAMoleBase):
    """Move-anywhere control with distance-based holding times -> an SMDP."""

    def __init__(self, time_per_distance: float = 1.0,
                 noop_time: Optional[float] = None,
                 distance_matrix: Optional[np.ndarray] = None, **kwargs):
        super().__init__(**kwargs)
        if self.reward_mode == "step_downed":
            raise ValueError(
                "reward_mode='step_downed' is MDP-only; per-step reward is "
                "undefined under variable holding times (use 'whack' or "
                "'whack_downed', or an SMDP reward-rate variant)."
            )
        self.time_per_distance = float(time_per_distance)
        self.noop_time = float(self.whack_time if noop_time is None else noop_time)
        if distance_matrix is not None:
            D = np.asarray(distance_matrix, dtype=np.float64)
            if D.shape != (self.n, self.n):
                raise ValueError(f"distance_matrix must be {(self.n, self.n)}")
            self.distance_matrix = D
        # 0..n-1 : move to that cell; n : whack.
        self.action_space = spaces.Discrete(self.n + 1)

    def _decode_action(self, action):
        if action == self.n:
            return "whack", self.whack_time, None
        if action == self.agent_pos:
            return "wait", self.noop_time, None  # self-move is illegal -> wait
        tau = self.time_per_distance * self.distance_matrix[self.agent_pos, action]
        return "move", float(tau), action

    def _mask_at(self, pos: int) -> np.ndarray:
        mask = np.ones(self.n + 1, dtype=np.int8)
        mask[int(pos)] = 0  # cannot move to the cell you are on
        return mask


def heuristic_policy(env: _WhackAMoleBase, state: Hashable) -> int:
    """A decent hand-written policy: whack if on a mole, else head for the nearest.

    Not optimal — it ignores the moles it will pass on the way and, under
    ``up_down``, that its target may drop before it arrives — so it is offered as
    a strong baseline rather than through ``secret()``.
    """
    moles, pos = state
    up = [i for i, m in enumerate(moles) if m]
    whack = env.action_space.n - 1
    if not up or moles[pos]:
        return whack
    if isinstance(env, WhackAMoleSMDP):
        return min(up, key=lambda i: env.distance_matrix[pos, i])  # jump straight there
    # MDP: one king-move toward the nearest mole (by Euclidean distance).
    target = min(up, key=lambda i: env.distance_matrix[pos, i])
    r, c = divmod(pos, env.cols)
    tr, tc = divmod(target, env.cols)
    dr = (tr > r) - (tr < r) + 1
    dc = (tc > c) - (tc < c) + 1
    return dr * 3 + dc


def register_gym_ids() -> None:
    """Expose the two environments as ``gym.make`` ids, idempotently.

    Note that ``gym.make`` returns a *wrapped* environment, so the contract
    hooks live at ``env.unwrapped.state_of``. The registry in ``envs/__init__``
    hands back unwrapped instances instead.
    """
    for env_id, cls in (("WhackAMoleMDP-v0", WhackAMoleMDP),
                        ("WhackAMoleSMDP-v0", WhackAMoleSMDP)):
        if env_id not in gym.registry:
            gym.register(id=env_id, entry_point=f"{__name__}:{cls.__name__}")
