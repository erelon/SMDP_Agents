"""Cumulative and normalized exponentially smoothed power means.

The standard real-valued power mean is used, so observations must be finite
and strictly positive for every power. Restricting the complete family to one
domain keeps the behavior consistent for negative, zero, fractional, and
positive powers and avoids implicit complex-valued results.
"""

import importlib.util
import math
import pathlib


_AVERAGE_RATES_PATH = pathlib.Path(__file__).resolve().parent / "average_rates.py"
_AVERAGE_RATES_SPEC = importlib.util.spec_from_file_location(
    "_power_means_average_rates", _AVERAGE_RATES_PATH
)
if _AVERAGE_RATES_SPEC is None or _AVERAGE_RATES_SPEC.loader is None:
    raise ImportError(f"cannot load average-rate helpers from {_AVERAGE_RATES_PATH}")
_average_rates = importlib.util.module_from_spec(_AVERAGE_RATES_SPEC)
_AVERAGE_RATES_SPEC.loader.exec_module(_average_rates)

NormalizedEMA = _average_rates.NormalizedEMA


def _require_finite(name: str, value: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _require_observation(value: float) -> float:
    value = _require_finite("value", value)
    if value <= 0:
        raise ValueError("power means require values greater than zero")
    return value


def _transform(value: float, p: float) -> float:
    try:
        transformed = math.log(value) if p == 0 else math.pow(value, p)
    except (OverflowError, ValueError) as error:
        raise ValueError(
            "transformed value is not finite in floating-point arithmetic"
        ) from error
    if not math.isfinite(transformed):
        raise ValueError(
            "transformed value is not finite in floating-point arithmetic"
        )
    return transformed


def _inverse_transform(transformed_mean: float, p: float) -> float:
    try:
        value = (
            math.exp(transformed_mean)
            if p == 0
            else math.pow(transformed_mean, 1.0 / p)
        )
    except (OverflowError, ValueError, ZeroDivisionError) as error:
        raise ValueError(
            "power mean is not finite in floating-point arithmetic"
        ) from error
    if not math.isfinite(value):
        raise ValueError("power mean is not finite in floating-point arithmetic")
    return value


class CumulativePowerMean:
    """Power mean over all strictly positive observations seen so far."""

    def __init__(self, p: float):
        self.p = _require_finite("p", p)
        self.reset()

    def reset(self) -> None:
        self.transformed_total = 0.0
        self.count = 0
        self.value = 0.0

    def update(self, value: float) -> float:
        value = _require_observation(value)
        self.transformed_total += _transform(value, self.p)
        self.count += 1
        transformed_mean = self.transformed_total / self.count
        self.value = _inverse_transform(transformed_mean, self.p)
        return self.value


class NormalizedExponentialPowerMean:
    """Normalized exponentially smoothed mean of positive observations."""

    def __init__(self, p: float, beta: float):
        self.p = _require_finite("p", p)
        self.beta = _require_finite("beta", beta)
        self.transformed_mean = NormalizedEMA(self.beta)
        self.reset()

    def reset(self) -> None:
        self.transformed_mean.reset()
        self.value = 0.0

    def update(self, value: float) -> float:
        value = _require_observation(value)
        transformed = _transform(value, self.p)
        normalized_mean = self.transformed_mean.update(transformed, 1.0)
        self.value = _inverse_transform(normalized_mean, self.p)
        return self.value
