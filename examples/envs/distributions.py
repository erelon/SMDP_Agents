"""Callable reward and duration processes for the configurable SMDPs.

A ``Distribution`` produces one number per call and may be non-stationary: it
counts its own calls and, every ``interval`` calls, *drifts* — moving its mean,
its level, or the phase of an oscillation. ``envs/tabular.py`` accepts one of
these anywhere a constant reward or duration is expected, which is how the
drifting environments in ``envs/configs.py`` are built.

Two wrappers give the numbers their meaning. ``Reward`` passes the sample
through unchanged; ``Duration`` clamps it up to ``Duration.MIN_DURATION``, since
a holding time may be zero-ish but never negative.

Hooks
-----
``a.register_hook(b)`` makes every sample of ``a`` also advance ``b``'s clock,
discarding ``b``'s value. Two distributions hooked to each other therefore drift
in lockstep no matter which action the agent picks — the difference between an
action whose reward drifts *when chosen* (local drift) and a world that moves on
regardless (global drift). Mutual hooks do not recurse: the hooked call is made
with ``call_hook=False``.

Ported from ``PythonProject3/dist_factory.py``, trimmed to the processes the
shipped configs use, plus ``reseed`` so a runner can vary the noise stream per
experiment seed.
"""

from __future__ import annotations

import math
from random import Random
from typing import Dict, Literal, Optional, Sequence, Type, Union

DriftOp = Literal["add", "mul"]


def _apply_drift(value: float, drift: float, op: DriftOp) -> float:
    if op == "add":
        return value + drift
    if op == "mul":
        return value * drift
    raise ValueError(f"Unknown drift op: {op!r}. Expected 'add' or 'mul'.")


class Distribution:
    """Something that yields a number per call, optionally drifting as it goes.

    ``interval`` sets how many calls pass between drifts and ``max_drifts`` caps
    how many ever happen — a compounding process left uncapped will run a
    time-budgeted experiment off the end of its clock in a few dozen steps.
    """

    def __init__(self, *, interval: int = 1, max_drifts: Optional[int] = None):
        if interval < 1:
            raise ValueError("interval must be >= 1")
        if max_drifts is not None and max_drifts < 0:
            raise ValueError("max_drifts must be >= 0")
        self.interval = interval
        self.max_drifts = max_drifts
        self.step = 0
        self.drifts = 0
        self.hook: Optional[Distribution] = None

    def register_hook(self, hook: Union["Distribution", "Reward", "Duration"]) -> None:
        """Advance ``hook`` on every sample of this distribution."""
        self.hook = hook.dist if isinstance(hook, (Reward, Duration)) else hook

    def reset(self) -> None:
        """Return to the initial state, keeping the seed."""
        self.step = 0
        self.drifts = 0

    def reseed(self, seed: int) -> None:
        """Re-seed the noise stream. A no-op for deterministic processes."""

    def __call__(self, call_hook: bool = True) -> float:
        self.step += 1
        if self.step % self.interval == 0 and (
            self.max_drifts is None or self.drifts < self.max_drifts
        ):
            self._drift()
            self.drifts += 1
        if self.hook is not None and call_hook:
            self.hook(call_hook=False)
        return self._sample()

    # --- subclass API ---
    def _sample(self) -> float:
        raise NotImplementedError

    def _drift(self) -> None:
        return

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Reward:
    """Semantic wrapper: this distribution's samples are rewards."""

    def __init__(self, dist: Distribution):
        self.dist = dist

    def __call__(self) -> float:
        return self.dist()

    def reset(self) -> None:
        self.dist.reset()

    def reseed(self, seed: int) -> None:
        self.dist.reseed(seed)

    def register_hook(self, hook) -> None:
        self.dist.register_hook(hook)

    def __repr__(self) -> str:
        return f"Reward({self.dist!r})"


class Duration:
    """Semantic wrapper: this distribution's samples are holding times."""

    #: Holding times are clamped up to this, so a duration is never negative.
    MIN_DURATION = 0.001

    def __init__(self, dist: Distribution):
        self.dist = dist

    def __call__(self) -> float:
        value = self.dist()
        return value if value >= self.MIN_DURATION else self.MIN_DURATION

    def reset(self) -> None:
        self.dist.reset()

    def reseed(self, seed: int) -> None:
        self.dist.reseed(seed)

    def register_hook(self, hook) -> None:
        self.dist.register_hook(hook)

    def __repr__(self) -> str:
        return f"Duration({self.dist!r})"


# ------------------------------------------------------------- distributions
class ConstantDist(Distribution):
    """Always ``value``."""

    def __init__(self, value: float = 1.0):
        super().__init__(interval=1)
        self.value = float(value)

    def _sample(self) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"ConstantDist({self.value})"


class LinearDriftDist(Distribution):
    """``start``, then drifted by ``step`` every ``interval`` calls."""

    def __init__(self, start: float = 0.0, step: float = 1.0, *,
                 interval: int = 1, max_drifts: Optional[int] = None,
                 drift_op: DriftOp = "add"):
        super().__init__(interval=interval, max_drifts=max_drifts)
        self.start = float(start)
        self.step_value = float(step)
        self.drift_op = drift_op
        self.current = self.start

    def reset(self) -> None:
        super().reset()
        self.current = self.start

    def _drift(self) -> None:
        self.current = _apply_drift(self.current, self.step_value, self.drift_op)

    def _sample(self) -> float:
        return self.current

    def __repr__(self) -> str:
        return (f"LinearDriftDist(start={self.start}, step={self.step_value}, "
                f"interval={self.interval}, drift_op={self.drift_op!r})")


class NormalDriftDist(Distribution):
    """Gaussian samples whose mean drifts by ``mean_drift`` every ``interval``."""

    def __init__(self, mean: float = 0.0, stddev: float = 1.0, *,
                 mean_drift: float = 0.0, drift_op: DriftOp = "add",
                 seed: int = 42, interval: int = 1,
                 max_drifts: Optional[int] = None):
        super().__init__(interval=interval, max_drifts=max_drifts)
        if stddev < 0:
            raise ValueError("stddev must be >= 0")
        self.seed = seed
        self.start_mean = float(mean)
        self.stddev = float(stddev)
        self.mean_drift = float(mean_drift)
        self.drift_op = drift_op
        self.rng = Random(seed)
        self.mean = self.start_mean

    def reset(self) -> None:
        super().reset()
        self.rng = Random(self.seed)
        self.mean = self.start_mean

    def reseed(self, seed: int) -> None:
        self.seed = int(seed)
        self.rng = Random(self.seed)

    def _drift(self) -> None:
        self.mean = _apply_drift(self.mean, self.mean_drift, self.drift_op)

    def _sample(self) -> float:
        return self.rng.gauss(self.mean, self.stddev)

    def __repr__(self) -> str:
        return (f"NormalDriftDist(mean={self.start_mean}, stddev={self.stddev}, "
                f"mean_drift={self.mean_drift}, interval={self.interval})")


class _TrigLogDist(Distribution):
    """Shared core of the sine/cosine oscillations under exponential scaling.

    ``(amplitude * trig(step * frequency + phase) + offset) * log_base ** (start_exp + step * log_scale)``

    The oscillation carries the shape and the power term the scale, so a small
    ``log_scale`` gives a slow exponential ramp with a wave riding on it — the
    process the ``sincoslog`` environment uses for both its reward and its
    holding time.
    """

    _trig = staticmethod(math.sin)

    def __init__(self, amplitude: float = 1.0, frequency: float = 1.0,
                 phase: float = 0.0, offset: float = 0.0, *,
                 log_base: float = 10.0, start_exp: float = 0.0,
                 log_scale: float = 0.001, interval: int = 1):
        super().__init__(interval=interval)
        if log_base <= 0 or log_base == 1:
            raise ValueError("log_base must be > 0 and != 1")
        self.amplitude = float(amplitude)
        self.frequency = float(frequency)
        self.phase = float(phase)
        self.offset = float(offset)
        self.log_base = float(log_base)
        self.start_exp = float(start_exp)
        self.log_scale = float(log_scale)

    def _sample(self) -> float:
        wave = self.amplitude * self._trig(self.step * self.frequency + self.phase)
        multiplier = self.log_base ** (self.start_exp + self.step * self.log_scale)
        return (wave + self.offset) * multiplier

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(amplitude={self.amplitude}, "
                f"frequency={self.frequency}, offset={self.offset}, "
                f"log_base={self.log_base}, log_scale={self.log_scale})")


class SinLogDist(_TrigLogDist):
    """``sin`` oscillation times an exponential ramp."""

    _trig = staticmethod(math.sin)


class CosLogDist(_TrigLogDist):
    """``cos`` oscillation times an exponential ramp."""

    _trig = staticmethod(math.cos)


class SinDist(_TrigLogDist):
    """``amplitude * sin(step * frequency + phase) + offset``, no scaling."""

    _trig = staticmethod(math.sin)

    def __init__(self, amplitude: float = 1.0, frequency: float = 1.0,
                 phase: float = 0.0, offset: float = 0.0, *, interval: int = 1):
        super().__init__(amplitude, frequency, phase, offset,
                         log_scale=0.0, start_exp=0.0, interval=interval)


class CosDist(SinDist):
    """``amplitude * cos(step * frequency + phase) + offset``, no scaling."""

    _trig = staticmethod(math.cos)


class ChoiceDist(Distribution):
    """One of ``values``, drawn with optional ``weights``."""

    def __init__(self, values: Sequence[float], weights: Optional[Sequence[float]] = None,
                 *, seed: int = 42, interval: int = 1):
        super().__init__(interval=interval)
        if not values:
            raise ValueError("values must be non-empty")
        if weights is not None and len(weights) != len(values):
            raise ValueError("weights must be the same length as values")
        self.values = [float(v) for v in values]
        self.weights = None if weights is None else [float(w) for w in weights]
        if self.weights is not None and sum(self.weights) <= 0:
            raise ValueError("weights must sum to something positive")
        self.seed = seed
        self.rng = Random(seed)

    def reset(self) -> None:
        super().reset()
        self.rng = Random(self.seed)

    def reseed(self, seed: int) -> None:
        self.seed = int(seed)
        self.rng = Random(self.seed)

    def _sample(self) -> float:
        if self.weights is None:
            return self.values[self.rng.randrange(len(self.values))]
        return self.rng.choices(self.values, weights=self.weights, k=1)[0]

    def __repr__(self) -> str:
        return f"ChoiceDist({self.values!r}, weights={self.weights!r})"


class UniformDist(Distribution):
    """Uniform samples on ``[low, high]``, both ends optionally drifting."""

    def __init__(self, low: float = 0.0, high: float = 1.0, *,
                 low_drift: float = 0.0, high_drift: float = 0.0,
                 drift_op: DriftOp = "add", seed: int = 42, interval: int = 1,
                 max_drifts: Optional[int] = None):
        super().__init__(interval=interval, max_drifts=max_drifts)
        if high < low:
            raise ValueError("high must be >= low")
        self.seed = seed
        self.start_low, self.start_high = float(low), float(high)
        self.low, self.high = self.start_low, self.start_high
        self.low_drift = float(low_drift)
        self.high_drift = float(high_drift)
        self.drift_op = drift_op
        self.rng = Random(seed)

    def reset(self) -> None:
        super().reset()
        self.rng = Random(self.seed)
        self.low, self.high = self.start_low, self.start_high

    def reseed(self, seed: int) -> None:
        self.seed = int(seed)
        self.rng = Random(self.seed)

    def _drift(self) -> None:
        self.low = _apply_drift(self.low, self.low_drift, self.drift_op)
        self.high = _apply_drift(self.high, self.high_drift, self.drift_op)

    def _sample(self) -> float:
        low, high = (self.low, self.high) if self.low <= self.high else (self.high, self.low)
        return self.rng.uniform(low, high)

    def __repr__(self) -> str:
        return (f"UniformDist(low={self.start_low}, high={self.start_high}, "
                f"low_drift={self.low_drift}, high_drift={self.high_drift})")


# ------------------------------------------------------------------- factory
_REGISTRY: Dict[str, Type[Distribution]] = {
    "constant": ConstantDist,
    "linear": LinearDriftDist,
    "normal": NormalDriftDist,
    "gauss": NormalDriftDist,
    "uniform": UniformDist,
    "choice": ChoiceDist,
    "sin": SinDist,
    "cos": CosDist,
    "sin_log": SinLogDist,
    "cos_log": CosLogDist,
}


def make_dist(kind: str, **kwargs) -> Distribution:
    """Build a distribution by name; see ``_REGISTRY`` for the roster."""
    key = kind.strip().lower().replace("-", "_")
    if key not in _REGISTRY:
        raise KeyError(
            f"Unknown distribution {kind!r}. Known: {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[key](**kwargs)


def make_reward(kind: str, **kwargs) -> Reward:
    """A reward process, e.g. ``make_reward("sin_log", log_scale=1e-3)``."""
    return Reward(make_dist(kind, **kwargs))


def make_duration(kind: str, **kwargs) -> Duration:
    """A holding-time process, clamped to stay non-negative."""
    return Duration(make_dist(kind, **kwargs))
