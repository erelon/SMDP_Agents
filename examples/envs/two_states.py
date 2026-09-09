"""Small SMDPs where the reward is earned *at a rate* over a random duration.

Every environment here shares one shape. Each decision draws a holding time
``tau ~ U(interval)`` — a wide range, 1 to 500 by default — and pays

    reward = clip(N(tau * rate(state, action), noise), interval[0], tau)

so the reward scales with the time it took to earn and the quantity an action
actually controls is the *rate*. That coupling is the point: reward and holding
time are strongly correlated, which is precisely the regime where the candidate
rho estimators come apart, and where an agent comparing raw rewards instead of
rates cannot tell a good action from a slow one.

The clip is inherited from the original: the ceiling ``tau`` caps any rate at 1,
and the floor is the smallest possible holding time, so no reward is ever below
1.

The variants differ in what ``rate`` depends on:

``StatelessRates``
    One state, two rates, very heavy noise. A pure bandit: the only difficulty is
    telling 0.6 from 0.55 through a standard deviation of 100.
``EvenTwoStates``
    Two states, and the reward does **not** scale with ``tau`` — it is 2 or 1
    flat. So the rate is dominated by the holding time draw and the reward barely
    matters, the mirror image of the rest of the family.
``UnevenTwoStates``
    Two states with a genuine trade-off: state 0 is the lucrative one, and the
    action that pays best *inside* state 1 is also the one least likely to escape
    it. Taking the immediate loss is optimal.
``CyclingUnevenTwoStates`` / ``LatentCyclingUnevenTwoStates``
    The same, but which action is best flips every ``cycle`` decisions — and the
    cycle phase is **not** observable, so from the agent's side the world is
    simply non-stationary.
``ShiftingUnevenTwoStates`` / ``SlopeShiftingUnevenTwoStates``
    All rates rescaled by a common factor that moves over time, randomly or along
    a chosen trajectory (``exp``, ``linear``, ``toward_target``, ``logistic``,
    ``sinusoidal``). A common factor leaves the *ordering* of the actions alone,
    so the optimal policy never changes; what changes is the scale a rho
    estimator has to track.

Ported from ``TimeBasedAgentsComparer/envs/``. The pull-model
``get_reward(agent, action) -> (T, r)`` becomes a Gymnasium ``step``, and
``secret`` now reads the state it is passed rather than the environment's
already-updated ``self.state`` — in the shifting variants those were different
states, so the original returned the optimal action for the wrong one.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

from gymnasium import spaces

from .base import SMDPEnv

#: Default holding-time range, from the original ``interval_min_len`` /
#: ``interval_max_len``. Deliberately wide: a factor of 500 between the shortest
#: and longest action.
DEFAULT_INTERVAL = (1.0, 500.0)


class RateEnv(SMDPEnv):
    """Base class: a random holding time, and a reward earned at a rate over it."""

    #: Number of observable states.
    n_states = 1
    #: Whether ``reward_mean`` is multiplied by the holding time.
    scales_with_tau = True

    def __init__(self, name: Optional[str] = None,
                 interval: Tuple[float, float] = DEFAULT_INTERVAL,
                 noise: float = 5.0, max_steps: Optional[int] = None,
                 max_time: Optional[float] = None,
                 render_mode: Optional[str] = None):
        super().__init__()
        lo, hi = float(interval[0]), float(interval[1])
        if lo <= 0 or hi < lo:
            raise ValueError(f"interval must satisfy 0 < min <= max, got {interval!r}")
        if noise < 0:
            raise ValueError("noise must be >= 0")
        self.name = name or type(self).__name__
        self.interval = (lo, hi)
        self.noise = float(noise)
        self.max_steps = None if max_steps is None else int(max_steps)
        self.max_time = None if max_time is None else float(max_time)
        self.render_mode = render_mode

        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(2)

        self.state = 0
        self.time = 0.0
        self.n_steps = 0

    # ------------------------------------------------------------ to override
    def rate(self, state: int, action: int) -> float:
        """The reward per unit time this state-action earns, before noise."""
        raise NotImplementedError

    def next_state(self, state: int, action: int) -> int:
        """Where the process goes next. Default: stay put."""
        return state

    def advance_clock(self) -> None:
        """Advance any latent process (a cycle phase, a drifting scale)."""

    # ---------------------------------------------------------------- gym api
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.state = 0
        self.time = 0.0
        self.n_steps = 0
        self.reset_latent()
        if self.render_mode == "human":
            self.render()
        return self.state, self._info(0.0, rate=self.rate(self.state, 0))

    def reset_latent(self) -> None:
        """Re-initialise any latent process. Called from ``reset``."""

    def step(self, action: int):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action} for {self.action_space}")

        lo, hi = self.interval
        tau = float(self.np_random.uniform(lo, hi))
        rate = self.rate(self.state, action)
        mean = rate * tau if self.scales_with_tau else rate
        reward = float(self.np_random.normal(mean, self.noise)) if self.noise else mean
        # The original clips to [smallest holding time, this holding time], which
        # caps any achievable rate at 1.
        reward = max(lo, min(tau, reward))

        previous = self.state
        self.state = int(self.next_state(previous, action))
        self.advance_clock()
        self.time += tau
        self.n_steps += 1

        terminated = False  # continuing task
        truncated = (
            (self.max_steps is not None and self.n_steps >= self.max_steps)
            or (self.max_time is not None and self.time >= self.max_time)
        )
        info = self._info(tau, rate=rate, mean_reward=mean, previous_state=previous,
                          n_steps=self.n_steps)
        if self.render_mode == "human":
            self.render()
        return self.state, reward, terminated, bool(truncated), info

    # ----------------------------------------------------------------- render
    def render(self):
        text = (f"{self.name}: state={self.state} t={self.time:.1f} "
                f"steps={self.n_steps}")
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(interval={self.interval}, "
                f"noise={self.noise})")


class StatelessRates(RateEnv):
    """One state, two nearly equal rates, and a standard deviation of 100.

    ``a=0`` earns 0.6 per unit time and ``a=1`` earns 0.55, a 9% gap buried in
    noise twenty times its size. There is no state and no transition structure,
    so this isolates one question: how many samples does an estimator need to
    resolve two similar rates when the durations vary by a factor of 500?
    """

    n_states = 1
    RATES = (0.6, 0.55)

    def __init__(self, name: Optional[str] = None, noise: float = 100.0, **kwargs):
        super().__init__(name, noise=noise, **kwargs)

    def rate(self, state: int, action: int) -> float:
        return self.RATES[action]

    def secret(self) -> Callable[[int], int]:
        return lambda state: 0


class EvenTwoStates(RateEnv):
    """Two states, a flat reward, and a duration that swamps it.

    The matching action pays 2 and the other pays 1 — flat, *not* scaled by the
    holding time — while ``tau`` ranges over 1 to 500. So the realised rate is
    almost entirely the luck of the duration draw, and an agent that compares
    rewards without dividing by time will still get this one right while
    learning nothing transferable. The state flips uniformly at random and is
    unaffected by the action, so there is no long-run trade-off: greedy is
    optimal.
    """

    n_states = 2
    scales_with_tau = False

    def __init__(self, name: Optional[str] = None, noise: float = 0.0, **kwargs):
        # The original is deterministic here: the reward is exactly 2 or 1.
        super().__init__(name, noise=noise, **kwargs)

    def rate(self, state: int, action: int) -> float:
        # Named "rate" by inheritance; with scales_with_tau False it is the flat
        # reward. 1 + the floor when the action matches the state, else the floor.
        floor = self.interval[0]
        return floor + 1.0 if action == state else floor

    def next_state(self, state: int, action: int) -> int:
        return int(self.np_random.integers(0, 2))

    def secret(self) -> Callable[[int], int]:
        return lambda state: int(state)


class UnevenTwoStates(RateEnv):
    """Two states where the best action in the bad state is the one that leaves it.

    Rates, at the default ``maxp=0.8``::

                     a=0                  a=1
        state 0      maxp / 1.0 = 0.80    maxp / 2.0 = 0.40
        state 1      maxp / 1.5 = 0.53    maxp / 3.0 = 0.27

    and the transitions::

        state 0            -> state 1 with p=0.2, else stay
        state 1, a=0       -> state 0 with p=0.2, else stay
        state 1, a=1       -> state 0 with p=0.8, else stay

    State 0 is the lucrative one, and inside state 1 the *worse-paying* action is
    the one that escapes: ``a=1`` earns half what ``a=0`` does there but leaves
    four times as often. Taking that loss is correct. Always playing ``a=0``
    spends half its time in state 1 and reaches a rate of 0.667; playing ``a=1``
    there spends only a fifth of its time in state 1 and reaches 0.693. The gap
    is 4%, which makes this a discrimination test rather than a giveaway — and an
    agent that maximises immediate reward gets it wrong.
    """

    n_states = 2

    def __init__(self, name: Optional[str] = None, maxp: float = 0.8, **kwargs):
        super().__init__(name, **kwargs)
        if maxp <= 0:
            raise ValueError("maxp must be > 0")
        self.maxp = float(maxp)

    #: Divisors of ``maxp`` per ``(state, action)``. The shifting variants in the
    #: original used a different table for state 1; they override this.
    DIVISORS = ((1.0, 2.0), (1.5, 3.0))

    def rate(self, state: int, action: int) -> float:
        return self.maxp / self.DIVISORS[state][action] * self.scale()

    def scale(self) -> float:
        """A common multiplier on every rate. Overridden by the shifting variants."""
        return 1.0

    def next_state(self, state: int, action: int) -> int:
        draw = self.np_random.random()
        if state == 0:
            return 1 if draw < 0.2 else 0
        return 0 if draw < (0.2 if action == 0 else 0.8) else 1

    def secret(self) -> Callable[[int], int]:
        return lambda state: 0 if int(state) == 0 else 1


class CyclingUnevenTwoStates(UnevenTwoStates):
    """``UnevenTwoStates`` where state 0's better action flips every ``cycle`` steps.

    In the first phase the rates are as in :class:`UnevenTwoStates`; in the second
    the two actions in state 0 swap, so what was worth 0.8 is worth 0.089 and
    vice versa. State 1 is untouched.

    The phase is **latent** — the observation is still just the state — so an
    agent cannot condition on it and the environment simply looks non-stationary.
    With the default ``cycle=5`` the flips come faster than a whole-history
    average can follow.
    """

    def __init__(self, name: Optional[str] = None, cycle: int = 5, **kwargs):
        super().__init__(name, **kwargs)
        if cycle < 1:
            raise ValueError("cycle must be >= 1")
        self.cycle = int(cycle)
        self.clock = 0
        self.phase = 0

    def reset_latent(self) -> None:
        self.clock = 0
        self.phase = 0

    def advance_clock(self) -> None:
        self.clock += 1
        self.phase = (self.clock // self.cycle) % 2

    #: ``DIVISORS`` per phase; phase 1 swaps state 0's two actions.
    PHASE_DIVISORS = (((1.0, 9.0), (3.0, 1.5)),
                      ((9.0, 1.0), (3.0, 1.5)))

    def rate(self, state: int, action: int) -> float:
        divisor = self.PHASE_DIVISORS[self.phase][state][action]
        return self.maxp / divisor * self.scale()

    def secret(self) -> Callable[[int], int]:
        def secret(state):
            if self.phase == 0:
                return 0 if int(state) == 0 else 1
            return 1
        return secret


class LatentCyclingUnevenTwoStates(CyclingUnevenTwoStates):
    """A latent cycle that ignores the state entirely: ``a=0`` then ``a=1``.

    Phase 0 favours ``a=0`` (0.8 against 0.4) and phase 1 favours ``a=1``
    (0.53 against 0.27), in *both* states, on a slower ``cycle=50``. So the
    observable state carries no information about which action to take and the
    whole signal is in the unobservable phase — the hardest case in the family
    for anything that assumes stationarity.
    """

    def __init__(self, name: Optional[str] = None, cycle: int = 50, **kwargs):
        super().__init__(name, cycle=cycle, **kwargs)

    def rate(self, state: int, action: int) -> float:
        divisor = (1.0, 2.0)[action] if self.phase == 0 else (3.0, 1.5)[action]
        return self.maxp / divisor * self.scale()

    def secret(self) -> Callable[[int], int]:
        return lambda state: 0 if self.phase == 0 else 1


class ShiftingUnevenTwoStates(UnevenTwoStates):
    """``UnevenTwoStates`` with every rate rescaled by a factor that jumps at random.

    Every ``shift_steps`` decisions the common multiplier is redrawn from
    ``U(shift_min, shift_max)``. Because it is *common* to all four rates the
    optimal policy never changes — only the scale of the rewards does, by up to a
    factor of ten. What that tests is whether an agent's rho tracks a moving
    scale, and whether its action preferences survive when it does not.

    Note the state-1 rates are the transpose of :class:`UnevenTwoStates`'s: here
    ``a=1`` both pays more (0.53 against 0.27) and escapes faster, so the
    trade-off that environment poses is absent and the greedy action is optimal.
    That difference is inherited from the original ``shifting.py``, which encoded
    state 1 the other way round from ``two_states_uneven.py``.
    """

    #: State 1 transposed relative to :class:`UnevenTwoStates`, as in the original.
    DIVISORS = ((1.0, 2.0), (3.0, 1.5))

    def __init__(self, name: Optional[str] = None, shift_steps: int = 50,
                 shift_min: float = 0.1, shift_max: float = 1.0, **kwargs):
        super().__init__(name, **kwargs)
        if shift_steps < 1:
            raise ValueError("shift_steps must be >= 1")
        if not 0 < shift_min <= shift_max:
            raise ValueError("must satisfy 0 < shift_min <= shift_max")
        self.shift_steps = int(shift_steps)
        self.shift_min = float(shift_min)
        self.shift_max = float(shift_max)
        self.clock = 0
        self.shift_constant = 1.0

    def reset_latent(self) -> None:
        self.clock = 0
        self.shift_constant = 1.0

    def scale(self) -> float:
        return self.shift_constant

    def advance_clock(self) -> None:
        self.clock += 1
        if self.clock % self.shift_steps == 0:
            self.shift_constant = float(
                self.np_random.uniform(self.shift_min, self.shift_max))


SCALE_MODES = ("exp", "linear", "toward_target", "logistic", "sinusoidal")


class SlopeShiftingUnevenTwoStates(UnevenTwoStates):
    """``UnevenTwoStates`` with a common multiplier on a chosen deterministic path.

    ``scale_mode`` picks the trajectory the multiplier follows, starting from 1:

    * ``"exp"`` — ``scale *= slope``; decay for ``0 < slope < 1``, growth above 1.
    * ``"linear"`` — ``scale += slope``; use a negative slope to decay.
    * ``"toward_target"`` — ``scale += (target - scale) * slope``, a fractional
      approach with ``slope`` in ``(0, 1]``.
    * ``"logistic"`` — ``scale += slope * scale * (1 - scale / target)``.
    * ``"sinusoidal"`` — ``scale = base + amplitude * sin(2*pi*clock/period)``.

    The update fires every ``shift_steps`` decisions, or every decision with
    ``apply_every_step``, and is clamped to ``[min_scale, max_scale]`` when those
    are given. Deterministic, so unlike :class:`ShiftingUnevenTwoStates` the whole
    trajectory is known in advance and a run is exactly reproducible even if the
    agent's own choices change. State 1's rates are transposed relative to
    :class:`UnevenTwoStates`, as in :class:`ShiftingUnevenTwoStates`.
    """

    #: State 1 transposed relative to :class:`UnevenTwoStates`, as in the original.
    DIVISORS = ((1.0, 2.0), (3.0, 1.5))

    def __init__(self, name: Optional[str] = None, shift_steps: int = 50,
                 slope: float = 0.99, apply_every_step: bool = False,
                 scale_mode: str = "exp", min_scale: Optional[float] = None,
                 max_scale: Optional[float] = None, target: Optional[float] = None,
                 period: Optional[float] = None, amplitude: float = 0.0,
                 base: float = 1.0, **kwargs):
        super().__init__(name, **kwargs)
        if scale_mode not in SCALE_MODES:
            raise ValueError(f"scale_mode must be one of {SCALE_MODES}")
        if shift_steps < 1:
            raise ValueError("shift_steps must be >= 1")
        if scale_mode in ("toward_target", "logistic") and target is None:
            raise ValueError(f"target is required when scale_mode={scale_mode!r}")
        if scale_mode == "logistic" and target <= 0:
            raise ValueError("target must be > 0 for the logistic mode")
        if scale_mode == "sinusoidal":
            if period is None or period <= 0:
                raise ValueError("period must be > 0 for the sinusoidal mode")
            if amplitude == 0.0:
                raise ValueError("amplitude must be non-zero for the sinusoidal mode")

        self.shift_steps = int(shift_steps)
        self.slope = float(slope)
        self.apply_every_step = bool(apply_every_step)
        self.scale_mode = scale_mode
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.target = target
        self.period = period
        self.amplitude = float(amplitude)
        self.base = float(base)
        self.clock = 0
        self.shift_constant = 1.0

    def reset_latent(self) -> None:
        self.clock = 0
        self.shift_constant = 1.0

    def scale(self) -> float:
        return self.shift_constant

    def _update_shift(self) -> None:
        if self.scale_mode == "exp":
            self.shift_constant *= self.slope
        elif self.scale_mode == "linear":
            self.shift_constant += self.slope
        elif self.scale_mode == "toward_target":
            self.shift_constant += (self.target - self.shift_constant) * self.slope
        elif self.scale_mode == "logistic":
            self.shift_constant += (self.slope * self.shift_constant
                                    * (1.0 - self.shift_constant / self.target))
        else:  # sinusoidal
            self.shift_constant = self.base + self.amplitude * math.sin(
                2.0 * math.pi * self.clock / self.period)
        if self.min_scale is not None:
            self.shift_constant = max(self.min_scale, self.shift_constant)
        if self.max_scale is not None:
            self.shift_constant = min(self.max_scale, self.shift_constant)

    def advance_clock(self) -> None:
        self.clock += 1
        if self.apply_every_step or self.clock % self.shift_steps == 0:
            self._update_shift()
