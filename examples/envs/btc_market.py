"""Take a long or short position on hourly Bitcoin bars, for a variable duration.

One decision per bar: go long or go short. The reward is
``position * return * tau`` and the holding time ``tau`` is derived from the size
of the return, so the *rate* an action earns is ``position * return`` — the
holding time scales the payoff without changing the rate. That is the interesting
shape here: the market decides how long each of your positions is held, and it
decides it from the same quantity that decides whether you were right.

The observable state is the sign of each of the previous ``state_size`` returns,
so ``3 ** state_size`` states (a return can be exactly zero on a quiet bar). It
carries no magnitude and no lookahead, which is deliberate: the point is not to
trade Bitcoin well but to have a real, messy, non-stationary reward stream with
genuinely coupled holding times. The rate ceiling is available through
``secret()``, which peeks at the bar about to be paid — the Oracle baseline it
feeds is what a perfect one-bar forecaster would earn, and nothing without
lookahead should come near it.

Holding times (``dependency``)
-----------------------------
The return is first mapped into ``[0, 1]``, then to a holding time in
``1 - time_window``: a mapped value of 0 gives the longest hold and 1 the
shortest. ``"linear"``, ``"quadratic"`` and ``"exponential"`` differ in the curve
between them, and ``"random"`` ignores the return and draws uniformly, which
decouples reward from duration and is the control condition.

Note the mapping is piecewise across the return distribution's quartiles: the
lower tail, the interquartile range and the upper tail are each stretched over
the whole ``[0, 1]`` interval separately. So it is a sawtooth in the return, not
monotone — a bar just below the first quartile gets the *shortest* hold and one
just above it the longest. That is inherited from the original and is deliberate
there ("distance from segment start"): it models three regimes rather than one
smooth relationship.

Ported from ``BtcSwarm/env.py``, one hardcoded configuration of it. Two
deviations, both flagged by keyword arguments: ``percentage=True`` uses relative
returns rather than the original's absolute dollar differences, which over this
window would otherwise drift the reward scale by four orders of magnitude and
conflate the market signal with a scale drift; and ``dependency="linear"``
replaces the original config's ``"random"``, since a holding time coupled to the
market is the reason this environment is here.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Callable, Hashable, List, Optional, Sequence, Tuple

import numpy as np
from gymnasium import spaces

from .base import SMDPEnv

#: The committed hourly slice; see ``examples/data/make_btc_slice.py``.
DEFAULT_DATA = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "btc_1h.csv.gz")

DEPENDENCIES = ("linear", "quadratic", "exponential", "random")


@lru_cache(maxsize=8)
def load_series(path: str, percentage: bool) -> np.ndarray:
    """Per-bar returns, cached so repeated environment builds parse the file once."""
    import pandas as pd

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no price data at {path}. Build it with "
            f"`python examples/data/make_btc_slice.py`, or pass data_path="
        )
    bars = pd.read_csv(path)
    missing = {"open", "close"} - set(bars.columns)
    if missing:
        raise ValueError(f"{path} is missing the columns {sorted(missing)}")
    difference = bars["close"].to_numpy(float) - bars["open"].to_numpy(float)
    if percentage:
        difference = difference / bars["open"].to_numpy(float)
    return difference


def _mapper(returns: np.ndarray) -> Callable[[float], float]:
    """Map a return into ``[0, 1]``, per quartile segment. See the module docstring."""
    low, high = float(returns.min()), float(returns.max())
    q25, q75 = (float(q) for q in np.quantile(returns, [0.25, 0.75]))
    if q25 == q75:  # a degenerate slice: fall back to the full range
        q25, q75 = low, high

    def mapped(value: float) -> float:
        if value < q25:
            return (value - low) / (q25 - low) if q25 != low else 0.0
        if value > q75:
            return (value - q75) / (high - q75) if high != q75 else 0.0
        return (value - q25) / (q75 - q25) if q75 != q25 else 0.0

    return mapped


_CURVES = {
    "linear": lambda t: t,
    "quadratic": lambda t: t ** 2,
    "exponential": lambda t: (math.exp(t) - 1.0) / (math.e - 1.0),
}


class BtcMarketEnv(SMDPEnv):
    """Long or short on hourly BTC bars, with return-dependent holding times."""

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, name: str = "btc_market", data_path: str = DEFAULT_DATA,
                 state_size: int = 3, positions: Sequence[float] = (-1.0, 1.0),
                 time_window: Tuple[float, float] = (0.1, 0.75),
                 dependency: str = "linear", percentage: bool = True,
                 random_start: bool = True, max_steps: Optional[int] = None,
                 max_time: Optional[float] = None,
                 render_mode: Optional[str] = None):
        super().__init__()
        if state_size < 1:
            raise ValueError("state_size must be >= 1")
        if dependency not in DEPENDENCIES:
            raise ValueError(f"dependency must be one of {DEPENDENCIES}")
        low, high = float(time_window[0]), float(time_window[1])
        if not 0.0 <= low <= high < 1.0:
            raise ValueError("time_window must satisfy 0 <= low <= high < 1")
        if len(positions) < 2:
            raise ValueError("at least two positions are needed for a decision")

        self.name = name
        self.state_size = int(state_size)
        self.positions = [float(p) for p in positions]
        self.time_window = (low, high)
        self.dependency = dependency
        self.percentage = bool(percentage)
        self.random_start = bool(random_start)
        self.max_steps = None if max_steps is None else int(max_steps)
        self.max_time = None if max_time is None else float(max_time)
        self.render_mode = render_mode

        self.returns = load_series(data_path, self.percentage)
        if len(self.returns) <= self.state_size + 1:
            raise ValueError(f"{data_path} has too few bars for state_size="
                             f"{self.state_size}")
        self.signs = np.sign(self.returns).astype(np.int8)
        self._codes = self._encode_states()
        self._fixed_taus = self._precompute_taus()

        self.observation_space = spaces.Discrete(3 ** self.state_size)
        self.action_space = spaces.Discrete(len(self.positions))

        self.t = self.state_size
        self.time = 0.0
        self.n_steps = 0
        self.total_reward = 0.0

    # ---------------------------------------------------------------- indexing
    def _encode_states(self) -> np.ndarray:
        """Base-3 code of the previous ``state_size`` signs, per decision index."""
        n = len(self.returns)
        codes = np.zeros(n, dtype=np.int64)
        for offset in range(self.state_size):
            shifted = np.roll(self.signs, offset + 1) + 1  # {-1,0,1} -> {0,1,2}
            codes = codes * 3 + shifted
        codes[: self.state_size] = -1  # not enough history yet
        return codes

    def _precompute_taus(self) -> Optional[np.ndarray]:
        """Per-bar holding times, or ``None`` when they are drawn at run time."""
        if self.dependency == "random":
            return None
        mapped = _mapper(self.returns)
        curve = _CURVES[self.dependency]
        low, high = self.time_window
        return np.array([1.0 - (low + (high - low) * curve(mapped(float(r))))
                         for r in self.returns])

    def _tau(self, index: int) -> float:
        if self._fixed_taus is not None:
            return float(self._fixed_taus[index])
        low, high = self.time_window
        return 1.0 - float(self.np_random.uniform(low, high))

    # ------------------------------------------------------------------ hooks
    def state_of(self, obs) -> Hashable:
        """The sign tuple the code stands for, oldest first."""
        code = int(obs)
        digits: List[int] = []
        for _ in range(self.state_size):
            digits.append(code % 3 - 1)
            code //= 3
        return tuple(digits)

    def secret(self) -> Callable[[Hashable], int]:
        """The best position for the bar about to be paid — privileged lookahead.

        The observable state cannot support this: it holds only past signs. What
        it gives the Oracle is the rate ceiling of a perfect one-bar forecaster,
        which is the right thing to measure everything else against.
        """
        def best(state) -> int:
            ret = float(self.returns[min(self.t, len(self.returns) - 1)])
            return max(range(len(self.positions)),
                       key=lambda a: self.positions[a] * ret)
        return best

    # ---------------------------------------------------------------- gym api
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        last = len(self.returns) - 2
        if self.random_start:
            # A fresh window per pass, so different seeds see different history.
            self.t = int(self.np_random.integers(self.state_size, last))
        else:
            self.t = self.state_size
        if options and "start" in options:
            self.t = max(self.state_size, min(int(options["start"]), last))
        self.time = 0.0
        self.n_steps = 0
        self.total_reward = 0.0
        if self.render_mode == "human":
            self.render()
        return self._obs(), self._info(0.0, bar=self.t)

    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action} for {self.action_space}")

        ret = float(self.returns[self.t])
        tau = self._tau(self.t)
        position = self.positions[action]
        reward = position * ret * tau

        self.t += 1
        self.time += tau
        self.total_reward += reward
        self.n_steps += 1

        # Running off the end of the series is not a terminal state of the market,
        # it is the end of the sample: report it as truncation so a runner restarts.
        exhausted = self.t >= len(self.returns) - 1
        truncated = bool(
            exhausted
            or (self.max_steps is not None and self.n_steps >= self.max_steps)
            or (self.max_time is not None and self.time >= self.max_time)
        )
        info = self._info(tau, bar=self.t, market_return=ret, position=position,
                          n_steps=self.n_steps, exhausted=exhausted)
        if self.render_mode == "human":
            self.render()
        return self._obs(), reward, False, truncated, info

    def _obs(self) -> int:
        return int(self._codes[min(self.t, len(self._codes) - 1)])

    # ----------------------------------------------------------------- render
    def render(self):
        text = (f"{self.name}: bar={self.t} t={self.time:.2f} "
                f"steps={self.n_steps} total_reward={self.total_reward:+.4f}")
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def __repr__(self) -> str:
        return (f"BtcMarketEnv(bars={len(self.returns)}, "
                f"state_size={self.state_size}, dependency={self.dependency!r}, "
                f"percentage={self.percentage})")
