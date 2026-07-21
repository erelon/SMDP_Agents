import importlib.util
import math
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


power_means = load_module("power_means", ROOT / "agents" / "power_means.py")
average_rates = load_module(
    "average_rates_for_power_means", ROOT / "agents" / "average_rates.py"
)

CumulativePowerMean = power_means.CumulativePowerMean
NormalizedExponentialPowerMean = power_means.NormalizedExponentialPowerMean
NormalizedEMA = average_rates.NormalizedEMA

SEQUENCE = [0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
POWERS = (-1.0, 0.0, 1.0, 2.0)


def direct_power_mean(values, p):
    if p == 0:
        return math.exp(sum(math.log(value) for value in values) / len(values))
    return (sum(value**p for value in values) / len(values)) ** (1.0 / p)


class CumulativePowerMeanTests(unittest.TestCase):
    def test_matches_direct_power_means_after_every_observation(self):
        for p in POWERS:
            with self.subTest(p=p):
                mean = CumulativePowerMean(p)
                values = []
                for value in SEQUENCE:
                    values.append(value)
                    self.assertAlmostEqual(
                        mean.update(value), direct_power_mean(values, p)
                    )

    def test_single_item_is_identity_for_each_power(self):
        for p in POWERS:
            with self.subTest(p=p):
                self.assertAlmostEqual(CumulativePowerMean(p).update(3.5), 3.5)

    def test_reset_replays_the_same_sequence(self):
        mean = CumulativePowerMean(-1)
        first = [mean.update(value) for value in SEQUENCE]
        mean.reset()
        second = [mean.update(value) for value in SEQUENCE]

        self.assertEqual(first, second)
        self.assertEqual(mean.count, len(SEQUENCE))


class NormalizedExponentialPowerMeanTests(unittest.TestCase):
    def test_matches_independent_normalized_smoothed_calculation(self):
        for beta in (0.0001, 0.999):
            for p in POWERS:
                with self.subTest(beta=beta, p=p):
                    mean = NormalizedExponentialPowerMean(p, beta)
                    transformed_ema = 0.0
                    normalizer = 0.0
                    for value in SEQUENCE:
                        transformed = math.log(value) if p == 0 else value**p
                        transformed_ema = (
                            (1.0 - beta) * transformed_ema + beta * transformed
                        )
                        normalizer = (1.0 - beta) * normalizer + beta
                        normalized = transformed_ema / normalizer
                        expected = (
                            math.exp(normalized)
                            if p == 0
                            else normalized ** (1.0 / p)
                        )
                        self.assertAlmostEqual(mean.update(value), expected)

    def test_p_one_matches_existing_normalized_ema_item_by_item(self):
        for beta in (0.0001, 0.25, 0.999):
            with self.subTest(beta=beta):
                power_mean = NormalizedExponentialPowerMean(1, beta)
                normalized_ema = NormalizedEMA(beta)
                for value in SEQUENCE:
                    self.assertEqual(
                        power_mean.update(value),
                        normalized_ema.update(value, 1.0),
                    )

    def test_single_item_is_identity_for_each_power(self):
        for p in POWERS:
            with self.subTest(p=p):
                mean = NormalizedExponentialPowerMean(p, beta=0.35)
                self.assertAlmostEqual(mean.update(3.5), 3.5)

    def test_reset_replays_the_same_sequence(self):
        mean = NormalizedExponentialPowerMean(2, beta=0.2)
        first = [mean.update(value) for value in SEQUENCE]
        mean.reset()
        second = [mean.update(value) for value in SEQUENCE]

        self.assertEqual(first, second)


class PowerMeanValidationTests(unittest.TestCase):
    def test_rejects_invalid_power(self):
        for p in (float("nan"), float("inf"), float("-inf"), "invalid"):
            for estimator_type in (
                CumulativePowerMean,
                NormalizedExponentialPowerMean,
            ):
                with self.subTest(p=p, estimator=estimator_type.__name__):
                    with self.assertRaises(ValueError):
                        if estimator_type is CumulativePowerMean:
                            estimator_type(p)
                        else:
                            estimator_type(p, beta=0.3)

    def test_rejects_invalid_beta(self):
        for beta in (0, -0.1, 1.1, float("nan"), float("inf"), "invalid"):
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                NormalizedExponentialPowerMean(1, beta)

    def test_rejects_zero_negative_and_nonfinite_observations(self):
        invalid_values = (0, -1, float("nan"), float("inf"), float("-inf"))
        for estimator_type in (CumulativePowerMean, NormalizedExponentialPowerMean):
            for p in POWERS:
                for value in invalid_values:
                    with self.subTest(
                        estimator=estimator_type.__name__, p=p, value=value
                    ):
                        if estimator_type is CumulativePowerMean:
                            estimator = estimator_type(p)
                        else:
                            estimator = estimator_type(p, beta=0.3)
                        with self.assertRaises(ValueError):
                            estimator.update(value)


if __name__ == "__main__":
    unittest.main()
