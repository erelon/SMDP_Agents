"""Where the reward-rate estimators disagree, and what controls the size of the gap.

Not an environment — a direct probe of the estimators in ``agents/average_rates.py``
on synthetic ``(reward, duration)`` pairs, with no agent and no policy in the way.

The question is when "average reward per unit time" is ambiguous. Two readings
compete:

* the **ratio of expectations**, ``E[R] / E[T]``, which ``CumulativeTimeRate``
  (behind SMART) converges to, and
* the **mean of the per-transition rates**, ``E[R / T]``, which
  ``CumulativeStepRate`` converges to.

They are related by an exact identity. Since ``E[R] = E[T · (R/T)]``,

    E[R] / E[T]  =  E[R/T]  +  Cov(T, R/T) / E[T]

so the gap between them *is* the covariance between how long a transition takes
and how fast it pays, normalised by the mean duration. When durations carry no
information about rates the two readings coincide and the choice of estimator does
not matter; the more they covary, the more it does.

This script sweeps that coupling and reports where each estimator lands. Following
the original, the sweep is parameterised by a target ``Cov(R, R/T)``: rewards are
generated and then scaled, which moves both covariances together, so the sweep is
in effect a sweep over reward magnitude at a fixed duration distribution. The
governing quantity ``Cov(T, R/T) / E[T]`` is reported alongside it.

    python -m examples.rate_covariance
    python -m examples.rate_covariance --size 20000 --beta 0.05
    python -m examples.rate_covariance --no-plot

Ported from ``PythonProject4``, with its private re-implementations of the three
operators replaced by the library's own estimators — which makes this a
consistency check on ``agents/average_rates.py`` as well as a demonstration.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from agents.average_rates import (CumulativeStepRate, CumulativeTimeRate,
                                  ExponentialMovingRatioRate,
                                  WeightedHarmonicRate)

DEFAULT_PLOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results", "plots", "rate_covariance.png")


def estimators(beta: float) -> Dict[str, Tuple[object, str]]:
    """The estimator under test, and how each one weights a sample."""
    return {
        # name: (estimator, weighting)   "one" -> weight 1, "reward" -> weight r
        "CumulativeTimeRate": (CumulativeTimeRate(), "one"),
        "CumulativeStepRate": (CumulativeStepRate(), "one"),
        "ExponentialMovingRatioRate": (ExponentialMovingRatioRate(beta), "one"),
        "WeightedHarmonicRate(w=1)": (WeightedHarmonicRate(beta), "one"),
        "WeightedHarmonicRate(w=r)": (WeightedHarmonicRate(beta), "reward"),
    }


def generate_pair(target_cov: float, size: int = 5000,
                  seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Rewards and durations with ``Cov(R, R/T) ≈ target_cov``.

    Durations are uniform on ``[1, 5]`` and rewards Gaussian around 5. Scaling the
    rewards by ``k`` scales both ``R`` and ``R/T`` and therefore scales
    ``Cov(R, R/T)`` by ``k**2``, which is the knob; the sign is set by flipping the
    rewards when it does not already match.

    At ``target_cov == 0`` the construction is different in kind: rewards are made
    exactly proportional to durations, so every per-transition rate is the same
    constant and every estimator must agree.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=5.0, scale=2.0, size=size)
    durations = rng.uniform(low=1.0, high=5.0, size=size)  # strictly positive

    if target_cov == 0.0:
        # R = c * T exactly, so R/T is constant and its covariance with anything is 0.
        return float(rng.uniform(1.0, 5.0)) * durations, durations

    current = float(np.cov(base, base / durations)[0, 1])
    if current == 0.0:
        raise ValueError(f"degenerate sample at seed {seed}: zero covariance")
    if np.sign(target_cov) != np.sign(current):
        base = -base
        current = float(np.cov(base, base / durations)[0, 1])
    return base * float(np.sqrt(abs(target_cov) / abs(current))), durations


def run_estimators(rewards: Sequence[float], durations: Sequence[float],
                   beta: float) -> Dict[str, float]:
    """Feed the whole sample through each estimator; return the final rho of each."""
    bank = estimators(beta)
    rho: Dict[str, float] = {}
    for name, (estimator, weighting) in bank.items():
        estimator.reset()
        value = 0.0
        for reward, duration in zip(rewards, durations):
            weight = float(reward) if weighting == "reward" else 1.0
            value = estimator.update(float(reward), float(duration), weight)
        rho[name] = value
    return rho


def measure(rewards: np.ndarray, durations: np.ndarray) -> Dict[str, float]:
    """The two criteria and the covariance term relating them, computed directly."""
    rates = rewards / durations
    ratio_of_expectations = float(rewards.mean() / durations.mean())
    mean_of_rates = float(rates.mean())
    # ddof=0: the identity is exact only when the covariance is normalised the same
    # way as the means it is being compared against.
    coupling = float(np.cov(durations, rates, ddof=0)[0, 1] / durations.mean())
    return {"E[R]/E[T]": ratio_of_expectations,
            "E[R/T]": mean_of_rates,
            "Cov(T,R/T)/E[T]": coupling,
            "identity_residual": ratio_of_expectations - mean_of_rates - coupling}


def sweep(covariances: Sequence[float], size: int, seed: int,
          beta: float) -> List[Dict[str, float]]:
    rows = []
    for target in covariances:
        rewards, durations = generate_pair(target, size=size, seed=seed)
        row = {"target_cov": float(target)}
        row.update(measure(rewards, durations))
        row.update(run_estimators(rewards, durations, beta))
        rows.append(row)
    return rows


def table(rows: Sequence[Dict[str, float]], every: int = 1) -> str:
    keys = ["target_cov", "Cov(T,R/T)/E[T]", "E[R]/E[T]", "E[R/T]",
            *estimators(0.1)]
    widths = {key: max(len(key), 12) for key in keys}
    lines = ["  ".join(key.rjust(widths[key]) for key in keys)]
    lines.append("  ".join("-" * widths[key] for key in keys))
    for row in rows[::every]:
        lines.append("  ".join(f"{row[key]:>{widths[key]}.4f}" for key in keys))
    return "\n".join(lines)


def plot(rows: Sequence[Dict[str, float]], path: str) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    names = list(estimators(0.1))
    covs = [row["target_cov"] for row in rows]
    baseline = "CumulativeTimeRate"

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.5))
    for name in names:
        left.plot(covs, [row[name] for row in rows], label=name, linewidth=1.4)
    left.plot(covs, [row["E[R/T]"] for row in rows], "k--", linewidth=1.0,
              label="E[R/T] (exact)")
    left.set_xlabel("target Cov(R, R/T)")
    left.set_ylabel("estimated rho")
    left.set_title("Where each estimator lands")
    left.legend(fontsize=7)
    left.grid(True, alpha=0.3)

    # Relative rather than absolute differences: scaling the reward scales every
    # estimate together, so absolute gaps grow trivially with the sweep.
    for name in names:
        if name == baseline:
            continue
        relative = [(row[name] - row[baseline]) / abs(row[baseline])
                    if row[baseline] else 0.0 for row in rows]
        right.plot(covs, relative, label=f"{name} vs {baseline}", linewidth=1.4)
    right.plot(covs, [row["Cov(T,R/T)/E[T]"] / row["E[R]/E[T]"]
                      if row["E[R]/E[T]"] else 0.0 for row in rows],
               "k--", linewidth=1.0, label="Cov(T,R/T)/E[T], normalised")
    right.set_xlabel("target Cov(R, R/T)")
    right.set_ylabel("relative difference in rho")
    right.set_title(f"Disagreement with {baseline}")
    right.legend(fontsize=7)
    right.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def agreement_check(size: int, beta: float, seeds: int = 20) -> Tuple[bool, float]:
    """At ``Cov = 0`` every per-transition rate is identical, so all estimators agree.

    Returns whether they did and the worst relative spread seen.
    """
    worst = 0.0
    for seed in range(1, seeds + 1):
        rewards, durations = generate_pair(0.0, size=size, seed=seed)
        rho = run_estimators(rewards, durations, beta)
        values = list(rho.values())
        spread = (max(values) - min(values)) / abs(values[0]) if values[0] else 0.0
        worst = max(worst, spread)
    return worst < 1e-9, worst


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=5000,
                        help="samples per covariance setting")
    parser.add_argument("--points", type=int, default=41,
                        help="covariance settings across the sweep")
    parser.add_argument("--max-cov", type=float, default=1.0,
                        help="largest target Cov(R, R/T)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="smoothing rate for the exponential estimators")
    parser.add_argument("--every", type=int, default=5,
                        help="print every Nth row of the sweep")
    parser.add_argument("--plot", default=DEFAULT_PLOT)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    covariances = np.linspace(0.0, args.max_cov, args.points)
    rows = sweep(covariances, size=args.size, seed=args.seed, beta=args.beta)

    print(f"{args.size} samples per point, beta={args.beta}, seed={args.seed}\n")
    print(table(rows, every=args.every))

    residual = max(abs(row["identity_residual"]) for row in rows)
    print(f"\nE[R]/E[T] = E[R/T] + Cov(T,R/T)/E[T] holds to {residual:.2e} "
          f"across the sweep.")

    agreed, worst = agreement_check(args.size, args.beta)
    print(f"At Cov = 0 (reward exactly proportional to duration) every estimator "
          f"{'agrees' if agreed else 'DISAGREES'}; worst relative spread "
          f"{worst:.2e}.")

    last = rows[-1]
    print(f"\nAt the far end of the sweep the two criteria are "
          f"{last['E[R]/E[T]']:.4f} (ratio of expectations) against "
          f"{last['E[R/T]']:.4f} (mean of rates) — a gap of "
          f"{last['Cov(T,R/T)/E[T]']:+.4f}.")

    if not args.no_plot:
        written = plot(rows, args.plot)
        print(f"plot -> {written}" if written
              else "matplotlib is not installed; skipped the plot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
