"""A configurable finite SMDP: states, actions, and weighted transitions.

Most of the classic examples in the average-reward literature are small labelled
graphs — "from ``s1``, action *a* goes to ``s2`` with probability ½, reward 20,
holding time 1". ``SMDPConfig`` is that graph written down directly, and
``TabularSMDPEnv`` turns one into an environment obeying the contract in
``base.py``. ``envs/configs.py`` is the catalogue of graphs.

A transition's reward, holding time, probability and target may each be either a
constant or a zero-argument callable, so the same machinery expresses both a
stationary textbook example and a drifting one built from
``envs/distributions.py``.

Observations are the state's *index* (a proper ``Discrete`` space); the
human-readable label is what ``state_of`` returns and what the agents key their
Q-tables on, so a printed Q-table reads ``{'s1': {...}, 's2': {...}}``.

Ported from ``PythonProject3/smdp_env.py``, with the transition sampling fixed:
probabilities are validated at construction, normalised at sample time, drawn
from the Gymnasium RNG so ``reset(seed=...)`` controls them, and the sampled
probability — rather than a leaked loop variable — is what lands in ``info``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from gymnasium import spaces

from .base import SMDPEnv

State = Hashable
Action = int

#: A transition field may be a fixed number/state or something callable.
Maybe = Any


def _value(spec: Maybe) -> Any:
    """Resolve a transition field: call it if callable, else use it as-is."""
    return spec() if callable(spec) else spec


class Transition:
    """One weighted outcome of taking an action in a state.

    Each of ``next_state``, ``prob``, ``reward`` and ``duration`` is either a
    constant or a zero-argument callable resolved afresh on every visit.
    """

    __slots__ = ("_next_state", "_prob", "_reward", "_duration")

    def __init__(self, next_state: Maybe, prob: Maybe = 1.0,
                 reward: Maybe = 0.0, duration: Maybe = 1.0):
        self._next_state = next_state
        self._prob = prob
        self._reward = reward
        self._duration = duration

    def next_state(self) -> State:
        return _value(self._next_state)

    def prob(self) -> float:
        return float(_value(self._prob))

    def reward(self) -> float:
        return float(_value(self._reward))

    def duration(self) -> float:
        return float(_value(self._duration))

    # --- introspection used for validation and for the "is this an MDP?" check
    @property
    def specs(self) -> Tuple[Maybe, Maybe, Maybe, Maybe]:
        return self._next_state, self._prob, self._reward, self._duration

    def constant(self, name: str) -> Optional[Any]:
        """The field's value if it is a constant, else ``None``."""
        spec = getattr(self, f"_{name}")
        return None if callable(spec) else spec

    def processes(self) -> Iterable[Any]:
        """The callable fields, i.e. the stateful reward/duration processes."""
        return (spec for spec in self.specs if callable(spec))

    def __repr__(self) -> str:
        return (f"Transition(next_state={self._next_state!r}, prob={self._prob!r}, "
                f"reward={self._reward!r}, duration={self._duration!r})")


@dataclass
class SMDPConfig:
    """A finite SMDP as a labelled graph.

    ``transitions`` maps ``(state, action)`` to the list of possible outcomes;
    a ``(state, action)`` pair with no entry is an action illegal in that state.
    ``terminal_states`` is usually empty: these are continuing tasks, and the
    runner bounds the trajectory rather than the environment.
    """

    states: List[State]
    actions: List[Action]
    transitions: Dict[Tuple[State, Action], List[Transition]]
    start_state: State
    terminal_states: List[State] = field(default_factory=list)
    #: Free-text note explaining what the config demonstrates; surfaced by the
    #: registry and the report.
    note: str = ""

    def __post_init__(self) -> None:
        self.states = list(dict.fromkeys(self.states))
        self.actions = sorted(set(self.actions))
        self._validate()

    def _validate(self) -> None:
        known = set(self.states)
        if self.start_state not in known:
            raise ValueError(
                f"start_state {self.start_state!r} is not among the states {self.states!r}"
            )
        for bad in set(self.terminal_states) - known:
            raise ValueError(f"terminal state {bad!r} is not among the states")

        for (state, action), outcomes in self.transitions.items():
            where = f"transition ({state!r}, {action!r})"
            if state not in known:
                raise ValueError(f"{where} starts from an unlisted state")
            if action not in self.actions:
                raise ValueError(
                    f"{where} uses an action outside {self.actions!r}"
                )
            if not outcomes:
                raise ValueError(f"{where} has no outcomes")
            for t in outcomes:
                target = t.constant("next_state")
                if target is not None and target not in known:
                    raise ValueError(f"{where} leads to the unlisted state {target!r}")
                duration = t.constant("duration")
                if duration is not None and (not math.isfinite(duration) or duration < 0):
                    raise ValueError(
                        f"{where} has duration {duration!r}; holding times must be "
                        f"finite and non-negative"
                    )
            probs = [t.constant("prob") for t in outcomes]
            if all(p is not None for p in probs):
                total = float(sum(probs))
                if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError(
                        f"{where} probabilities sum to {total}, not 1: {probs!r}"
                    )
                if any(p < 0 for p in probs):
                    raise ValueError(f"{where} has a negative probability: {probs!r}")

        unreachable = known - {s for (s, _) in self.transitions}
        unreachable -= set(self.terminal_states)
        if unreachable:
            raise ValueError(
                f"states {sorted(map(repr, unreachable))} have no outgoing "
                f"transitions and are not listed as terminal"
            )

    # ---------------------------------------------------------------- helpers
    def available_actions(self, state: State) -> List[Action]:
        return [a for a in self.actions if (state, a) in self.transitions]

    def processes(self) -> List[Any]:
        """Every stateful reward/duration process, de-duplicated, in a stable order."""
        seen: Dict[int, Any] = {}
        for outcomes in self.transitions.values():
            for t in outcomes:
                for proc in t.processes():
                    seen.setdefault(id(proc), proc)
        return list(seen.values())

    @property
    def is_mdp(self) -> bool:
        """True when every holding time is a constant 1 — a plain MDP."""
        return all(
            t.constant("duration") == 1.0
            for outcomes in self.transitions.values()
            for t in outcomes
        )


class TabularSMDPEnv(SMDPEnv):
    """A finite SMDP built from an :class:`SMDPConfig`."""

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, config: SMDPConfig, name: str = "",
                 max_steps: Optional[int] = None, max_time: Optional[float] = None,
                 reseed_processes: bool = True, render_mode: Optional[str] = None):
        super().__init__()
        self.config = config
        self.name = name or "TabularSMDP"
        self.max_steps = None if max_steps is None else int(max_steps)
        self.max_time = None if max_time is None else float(max_time)
        self.reseed_processes = bool(reseed_processes)
        self.render_mode = render_mode
        self.is_mdp = config.is_mdp

        self.state_labels: List[State] = list(config.states)
        self._index = {label: i for i, label in enumerate(self.state_labels)}
        self.terminal_states = set(config.terminal_states)

        self.observation_space = spaces.Discrete(len(self.state_labels))
        self.action_space = spaces.Discrete(max(config.actions) + 1)

        self.state: State = config.start_state
        self.time = 0.0
        self.total_reward = 0.0
        self.n_steps = 0

    # ------------------------------------------------------------------ hooks
    def state_of(self, obs: Any) -> State:
        """The state label; accepts either an index or a label."""
        if isinstance(obs, (int, np.integer)) and not isinstance(obs, bool):
            index = int(obs)
            if 0 <= index < len(self.state_labels):
                return self.state_labels[index]
        return obs

    def get_available_actions(self, state: Optional[State] = None) -> List[Action]:
        label = self.state if state is None else self.state_of(state)
        return self.config.available_actions(label)

    # ---------------------------------------------------------------- gym api
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.state = self.config.start_state
        self.time = 0.0
        self.total_reward = 0.0
        self.n_steps = 0

        for i, proc in enumerate(self.config.processes()):
            if hasattr(proc, "reset"):
                proc.reset()
            if seed is not None and self.reseed_processes and hasattr(proc, "reseed"):
                # Distinct, seed-derived streams so a multi-seed study varies the
                # reward/duration noise and not only the transition draws.
                proc.reseed(int(seed) * 7919 + i)

        if self.render_mode == "human":
            self.render()
        return self._obs(), self._info(0.0, state=self.state, prob=1.0)

    def step(self, action: Action):
        action = int(action)
        key = (self.state, action)
        outcomes = self.config.transitions.get(key)
        if outcomes is None:
            raise ValueError(
                f"{self.name}: action {action!r} is not available in state "
                f"{self.state!r}; available: {self.get_available_actions()}"
            )

        chosen, prob = self._sample(outcomes)
        reward = chosen.reward()
        duration = chosen.duration()
        if not math.isfinite(duration) or duration < 0.0:
            raise ValueError(
                f"{self.name}: transition from {self.state!r} under action "
                f"{action!r} produced the holding time {duration!r}"
            )
        next_state = chosen.next_state()
        if next_state not in self._index:
            raise ValueError(
                f"{self.name}: transition from {self.state!r} under action "
                f"{action!r} led to the unknown state {next_state!r}"
            )

        self.state = next_state
        self.time += duration
        self.total_reward += reward
        self.n_steps += 1

        terminated = self.state in self.terminal_states
        truncated = (
            (self.max_steps is not None and self.n_steps >= self.max_steps)
            or (self.max_time is not None and self.time >= self.max_time)
        )
        info = self._info(duration, state=self.state, prob=prob,
                          n_steps=self.n_steps, total_reward=self.total_reward)
        if self.render_mode == "human":
            self.render()
        return self._obs(), reward, terminated, bool(truncated), info

    # ------------------------------------------------------------------ inner
    def _obs(self) -> int:
        return self._index[self.state]

    def _sample(self, outcomes: Sequence[Transition]) -> Tuple[Transition, float]:
        """Draw one outcome, normalising the weights so they cannot silently miss."""
        if len(outcomes) == 1:
            return outcomes[0], outcomes[0].prob()
        probs = [t.prob() for t in outcomes]
        if any(p < 0 for p in probs):
            raise ValueError(f"{self.name}: negative transition probability in {probs!r}")
        total = float(sum(probs))
        if total <= 0.0:
            raise ValueError(
                f"{self.name}: transition probabilities from {self.state!r} sum to "
                f"{total}; nothing can be sampled"
            )
        threshold = self.np_random.random() * total
        cumulative = 0.0
        for t, p in zip(outcomes, probs):
            cumulative += p
            if threshold < cumulative:
                return t, p / total
        return outcomes[-1], probs[-1] / total

    # ----------------------------------------------------------------- render
    def render(self):
        text = (f"{self.name}: state={self.state!r} t={self.time:.3f} "
                f"steps={self.n_steps} total_reward={self.total_reward:.3f}")
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def __repr__(self) -> str:
        return (f"TabularSMDPEnv(name={self.name!r}, "
                f"states={len(self.state_labels)}, actions={self.config.actions!r})")
