"""Reusable validation helpers for numeric values and reward rates."""

import math


def require_finite(name: str, value: float) -> float:
    """Return ``value`` as a float, or reject non-numeric/non-finite values."""

    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def require_positive(name: str, value: float) -> float:
    """Return a finite float that is strictly greater than zero."""

    value = require_finite(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def require_duration(duration: float) -> float:
    """Return a finite, strictly positive duration."""

    return require_positive("duration", duration)


def require_power_mean_observation(value: float) -> float:
    """Validate an observation in the common real-valued power-mean domain."""

    value = require_finite("value", value)
    if value <= 0:
        raise ValueError("power means require values greater than zero")
    return value


def require_power_mean_weight(weight: float) -> float:
    """Validate a weight used by a power mean."""

    weight = require_finite("weight", weight)
    if weight <= 0:
        raise ValueError("power means require weights greater than zero")
    return weight


def local_rate(reward: float, duration: float) -> float:
    """Validate a transition and return its strictly positive local rate."""

    reward = require_finite("reward", reward)
    duration = require_duration(duration)
    return require_positive("reward / duration", reward / duration)
