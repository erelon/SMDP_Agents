"""CRRA utility helpers for positive-rate risk-sensitive R-learning."""

import math


def require_finite(name: str, value: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def require_positive(name: str, value: float) -> float:
    value = require_finite(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def power_from_theta(theta: float) -> float:
    """Return the power-mean parameter p = 1 - theta."""

    return 1.0 - require_finite("theta", theta)


def crra_utility(rate: float, theta: float) -> float:
    """Return CRRA utility of a finite, strictly positive rate."""

    rate = require_positive("rate", rate)
    theta = require_finite("theta", theta)
    if theta == 0:
        return rate - 1.0
    if theta == 1:
        return math.log(rate)

    p = 1.0 - theta
    try:
        value = math.expm1(p * math.log(rate)) / p
    except OverflowError as error:
        raise ValueError("CRRA utility is not finite") from error
    if not math.isfinite(value):
        raise ValueError("CRRA utility is not finite")
    return value


def inverse_crra_utility(utility: float, theta: float) -> float:
    """Map a valid CRRA utility value to its certainty-equivalent rate."""

    utility = require_finite("utility", utility)
    theta = require_finite("theta", theta)
    if theta == 0:
        return require_positive("certainty-equivalent rate", utility + 1.0)
    if theta == 1:
        try:
            rate = math.exp(utility)
        except OverflowError as error:
            raise ValueError("certainty-equivalent rate is not finite") from error
    else:
        p = 1.0 - theta
        base = 1.0 + p * utility
        if base <= 0:
            raise ValueError("utility is outside the inverse CRRA domain")
        try:
            rate = math.exp(math.log1p(p * utility) / p)
        except (OverflowError, ValueError) as error:
            raise ValueError("certainty-equivalent rate is not finite") from error
    return require_positive("certainty-equivalent rate", rate)


def local_rate(reward: float, duration: float) -> float:
    """Validate a transition and return its strictly positive local rate."""

    reward = require_finite("reward", reward)
    duration = require_positive("duration", duration)
    return require_positive("reward / duration", reward / duration)


def utility_differential_reward(
    reward: float,
    duration: float,
    rho: float,
    theta: float,
) -> float:
    """Return duration * (u(reward/duration) - u(rho))."""

    rate = local_rate(reward, duration)
    rho = require_positive("rho", rho)
    theta = require_finite("theta", theta)
    if theta == 0:
        return reward - rho * duration
    return duration * (crra_utility(rate, theta) - crra_utility(rho, theta))


class CumulativeCRRAUtilityRate:
    """Duration-weighted CRRA utility baseline and certainty equivalent."""

    def __init__(self, theta: float):
        self.theta = require_finite("theta", theta)
        self.p = power_from_theta(self.theta)
        self.reset()

    def reset(self) -> None:
        self.total_utility = 0.0
        self.total_duration = 0.0
        self.utility_rate = 0.0
        self.value = inverse_crra_utility(self.utility_rate, self.theta)

    @property
    def rho(self) -> float:
        return self.value

    def update(self, reward: float, duration: float) -> float:
        rate = local_rate(reward, duration)
        duration = float(duration)
        utility_total = self.total_utility + duration * crra_utility(
            rate, self.theta
        )
        total_duration = self.total_duration + duration
        if not math.isfinite(utility_total) or not math.isfinite(total_duration):
            raise ValueError("cumulative utility totals must be finite")

        self.total_utility = utility_total
        self.total_duration = total_duration
        self.utility_rate = self.total_utility / self.total_duration
        self.value = inverse_crra_utility(self.utility_rate, self.theta)
        return self.value


def run_self_checks() -> None:
    """Run dependency-free checks for the Phase 4 mathematical contracts."""

    for theta in (-1.0, 0.0, 1.0, 2.0):
        for rate in (0.25, 1.0, 3.5, 10.0):
            recovered = inverse_crra_utility(crra_utility(rate, theta), theta)
            if not math.isclose(recovered, rate, rel_tol=1e-12, abs_tol=1e-12):
                raise AssertionError((theta, rate, recovered))

    reward, duration, rho = 7.0, 2.0, 1.25
    differential = utility_differential_reward(
        reward, duration, rho, theta=0.0
    )
    if differential != reward - rho * duration:
        raise AssertionError("theta=0 does not reduce to ordinary R-learning")

    scale = 10.0
    for theta in (-1.0, 0.0, 1.0, 2.0):
        original = utility_differential_reward(7.0, 2.0, 1.25, theta)
        scaled = utility_differential_reward(
            scale * 7.0, 2.0, scale * 1.25, theta
        )
        p = power_from_theta(theta)
        expected = original if p == 0 else scale**p * original
        if not math.isclose(scaled, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError((theta, scaled, expected))

    transitions = ((2.0, 1.0), (12.0, 3.0))
    for theta in (-1.0, 0.0, 1.0, 2.0):
        baseline = CumulativeCRRAUtilityRate(theta)
        for transition in transitions:
            baseline.update(*transition)
        p = power_from_theta(theta)
        if p == 0:
            expected = math.exp((math.log(2.0) + 3.0 * math.log(4.0)) / 4.0)
        else:
            expected = ((2.0**p + 3.0 * 4.0**p) / 4.0) ** (1.0 / p)
        if not math.isclose(
            baseline.rho, expected, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise AssertionError((theta, baseline.rho, expected))


if __name__ == "__main__":
    run_self_checks()
