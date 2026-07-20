import importlib.util
import pathlib
import unittest


AVERAGE_RATES_PATH = pathlib.Path(__file__).resolve().parents[1] / "agents" / "average_rates.py"
AVERAGE_RATES_SPEC = importlib.util.spec_from_file_location(
    "average_rates", AVERAGE_RATES_PATH
)
average_rates = importlib.util.module_from_spec(AVERAGE_RATES_SPEC)
AVERAGE_RATES_SPEC.loader.exec_module(average_rates)

CumulativeStepRate = average_rates.CumulativeStepRate
CumulativeTimeRate = average_rates.CumulativeTimeRate
ExponentialMovingRatioRate = average_rates.ExponentialMovingRatioRate
NormalizedExponentialMovingTimeRate = (
    average_rates.NormalizedExponentialMovingTimeRate
)
WeightedHarmonicRate = average_rates.WeightedHarmonicRate


def print_rate_comparison_table(sequence, beta):
    class RewardWeightedHarmonicRate(WeightedHarmonicRate):
        def update(self, reward, duration, weight):
            return super().update(reward, duration, reward)

    class ReciprocalHarmonicRate(WeightedHarmonicRate):
        def update(self, reward, duration, weight):
            harmonic_rate = super().update(reward, duration, reward)
            self.value = 1.0 / harmonic_rate
            return self.value

    avgs = [
        ("TimeRate", CumulativeTimeRate()),
        ("StepRate", CumulativeStepRate()),
        ("Mov.RatioRate", ExponentialMovingRatioRate(beta)),
        ("Mov.TimeRate", NormalizedExponentialMovingTimeRate(beta)),
        ("Reward-w-H", RewardWeightedHarmonicRate(beta)),
        ("Harmonic", WeightedHarmonicRate(beta)),
        ("1/H", ReciprocalHarmonicRate(beta)),
    ]

    headers = ["step", "reward", "duration", "rate", *[name for name, _ in avgs]]
    rows = []
    for step, (reward, duration) in enumerate(sequence, start=1):
        row = [
            str(step),
            f"{reward:.3f}",
            f"{duration:.3f}",
            f"{reward / duration:.3f}",
        ]
        for _, average in avgs:
            row.append(f"{average.update(reward, duration, 1.0):.3f}")
        rows.append(row)

    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    print()
    print(" | ".join(header.rjust(width) for header, width in zip(headers, widths)))
    print("-|-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.rjust(width) for value, width in zip(row, widths)))

    return headers, rows


def test_file(filename, beta):
    path = pathlib.Path(filename)
    if not path.is_absolute():
        path = pathlib.Path(__file__).resolve().parent / path

    sequence = []
    with path.open(encoding="utf-8") as data_file:
        for line_number, line in enumerate(data_file, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 2:
                raise ValueError(
                    f"{path}:{line_number}: expected reward and duration"
                )
            reward, duration = map(float, fields)
            sequence.append((reward, duration))

    return print_rate_comparison_table(sequence, beta)


# This is a helper whose name was requested as test_file, not a standalone pytest test.
test_file.__test__ = False


class SMDPCriteriaTests(unittest.TestCase):
    def test_slow_then_fast(self):
        sequence = [
            (10.0, 10.0),  # rate = 1
            (1.0, 0.1),  # rate = 10
            (1.0, 0.1),
            (1.0, 0.1),
            (1.0, 0.1),
            (1.0, 0.1),
            (1.0, 0.1),
            (1.0, 0.1),
            (1.0, 0.1),
            (1.0, 0.1),
            (1.0, 0.1),
            # *[(1.0, 0.1)]*1000
        ]
        headers, rows = print_rate_comparison_table(sequence, beta=0.01)

        self.assertEqual(len(headers), len(rows[0]))

    def test_dependent_rewards(self):
        sequence = [
            (10.0, 1.0),  # rate = 10
            (1.0, 0.1),  # rate = 10
            (2.0, 0.2),
            (1000.0, 100.0),
            (1.0, 0.1),
            (1.0, 0.1),
            (1.0, 0.1),
            (1.0, 0.1),
            (1.0, 0.1),
            (1.0, 0.1),
            # *[(1.0, 0.1)] * 100,
        ]
        headers, rows = print_rate_comparison_table(sequence, beta=0.3)

        self.assertEqual(len(headers), len(rows[0]))

    def test_sincos(self):
        headers, rows = test_file("test-data/sincoslog.data", beta=0.001)

        self.assertGreater(len(rows), 0)
        self.assertTrue(all(len(row) == len(headers) for row in rows))

    def test_short_seq(self):
        sequence = [
            (2.0, 1.0),  # rate = 10
            (1.0, 3.0),  # rate = 10
         ] * 100
        headers, rows = print_rate_comparison_table(sequence, beta=0.01)

        self.assertEqual(len(headers), len(rows[0]))

if __name__ == "__main__":
    unittest.main()
