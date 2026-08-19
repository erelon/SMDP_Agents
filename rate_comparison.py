#!/usr/bin/env python3
"""Write a CSV comparison of reward-rate averages for a reward/duration log."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
from pathlib import Path

# Load the dependency-free module without executing agents/__init__.py, whose
# optional deep-RL imports require packages that this utility does not need.
_AVERAGE_RATES_PATH = Path(__file__).resolve().parent / "agents" / "average_rates.py"
_SPEC = importlib.util.spec_from_file_location("average_rates", _AVERAGE_RATES_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import machinery guard
    raise ImportError(f"cannot load {_AVERAGE_RATES_PATH}")
_average_rates = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_average_rates)

CumulativeStepRate = _average_rates.CumulativeStepRate
CumulativeTimeRate = _average_rates.CumulativeTimeRate
ExponentialMovingRatioRate = _average_rates.ExponentialMovingRatioRate
NormalizedExponentialMovingTimeRate = _average_rates.NormalizedExponentialMovingTimeRate
WeightedHarmonicRate = _average_rates.WeightedHarmonicRate

DEFAULT_BETA = 0.3


def read_transitions(path: Path):
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) != 2:
                raise ValueError(
                    f"{path}:{line_number}: expected reward and duration"
                )
            try:
                reward, duration = map(float, fields)
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: reward and duration must be numbers"
                ) from error
            if not math.isfinite(reward) or not math.isfinite(duration):
                raise ValueError(f"{path}:{line_number}: values must be finite")
            if duration <= 0:
                raise ValueError(f"{path}:{line_number}: duration must be positive")
            yield reward, duration


def write_comparison(input_path: Path, output_path: Path, beta: float) -> None:
    cumulative = CumulativeTimeRate()
    exponential_time = NormalizedExponentialMovingTimeRate(beta)
    step_mean = CumulativeStepRate()
    weighted_harmonic = WeightedHarmonicRate(beta)
    harmonic = WeightedHarmonicRate(beta)
    ratio_rate = ExponentialMovingRatioRate(beta)

    headings = [
        "row #",
        "reward",
        "duration",
        "rate",
        "cumulative time rate",
        "exponential moving cumulative time rate",
        "mean rate per step",
        "weighted harmonic rate",
        "harmonic rate",
        "ratio rate",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(headings)
        for row_number, (reward, duration) in enumerate(
                read_transitions(input_path), start=1
        ):
            writer.writerow(
                [
                    row_number,
                    reward,
                    duration,
                    reward / duration,
                    cumulative.update(reward, duration),
                    exponential_time.update(reward, duration),
                    step_mean.update(reward, duration),
                    weighted_harmonic.update(reward, duration, reward),
                    harmonic.update(reward, duration),
                    ratio_rate.update(reward, duration),
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="whitespace-delimited reward/duration file")
    parser.add_argument(
        "--beta",
        type=float,
        default=DEFAULT_BETA,
        help=f"EMA update rate (default: {DEFAULT_BETA})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.beta <= 1:
        raise SystemExit("error: --beta must be in the interval (0, 1]")
    output_path = Path(f"rate-comparison-{args.file.name}.csv")
    try:
        write_comparison(args.file, output_path, args.beta)
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print(output_path)


if __name__ == "__main__":
    main()
