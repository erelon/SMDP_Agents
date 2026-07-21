import importlib.util
import math
import pathlib
import sys
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
AGENTS_DIR = ROOT / "agents"
PACKAGE_NAME = "_power_rate_agents"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(AGENTS_DIR)]
package.__package__ = PACKAGE_NAME
sys.modules[PACKAGE_NAME] = package

average_rates = load_module(
    f"{PACKAGE_NAME}.average_rates", AGENTS_DIR / "average_rates.py"
)
load_module(f"{PACKAGE_NAME}.power_means", AGENTS_DIR / "power_means.py")
power_rates = load_module(
    f"{PACKAGE_NAME}.power_rates", AGENTS_DIR / "power_rates.py"
)

CumulativePowerMeanRate = power_rates.CumulativePowerMeanRate
NormalizedExponentialPowerMeanRate = (
    power_rates.NormalizedExponentialPowerMeanRate
)
CumulativeStepRate = average_rates.CumulativeStepRate
CumulativeTimeRate = average_rates.CumulativeTimeRate
ExponentialMovingRatioRate = average_rates.ExponentialMovingRatioRate
WeightedHarmonicRate = average_rates.WeightedHarmonicRate

SEQUENCE = [
    (1.0, 2.0),
    (3.0, 1.0),
    (2.0, 4.0),
    (5.0, 3.0),
    (0.5, 2.0),
    (4.0, 1.0),
    (7.0, 5.0),
    (1.5, 2.5),
    (8.0, 4.0),
    (2.5, 1.5),
    (6.0, 3.0),
]
WEIGHTS = [0.5, 1.0, 2.0, 0.25, 3.0, 1.5, 4.0, 0.75, 2.5, 1.25, 5.0]
POWERS = (-1.0, 0.0, 1.0, 2.0)


def inverse_transform(moment, p):
    return math.exp(moment) if p == 0 else moment ** (1.0 / p)


class CumulativePowerMeanRateTests(unittest.TestCase):
    def test_unit_weights_match_direct_event_power_rates_item_by_item(self):
        for p in POWERS:
            with self.subTest(p=p):
                estimator = CumulativePowerMeanRate(p)
                transformed_total = 0.0
                for step, (reward, duration) in enumerate(SEQUENCE, start=1):
                    rate = reward / duration
                    transformed_total += math.log(rate) if p == 0 else rate**p
                    expected = inverse_transform(transformed_total / step, p)
                    self.assertAlmostEqual(
                        estimator.update(reward, duration), expected
                    )

    def test_nonunit_weights_match_direct_weighted_rates_item_by_item(self):
        for p in POWERS:
            with self.subTest(p=p):
                estimator = CumulativePowerMeanRate(p)
                transformed_total = 0.0
                total_weight = 0.0
                for (reward, duration), weight in zip(SEQUENCE, WEIGHTS):
                    rate = reward / duration
                    transformed = math.log(rate) if p == 0 else rate**p
                    transformed_total += weight * transformed
                    total_weight += weight
                    expected = inverse_transform(
                        transformed_total / total_weight, p
                    )
                    self.assertAlmostEqual(
                        estimator.update(reward, duration, weight), expected
                    )

    def test_p_one_with_unit_weight_matches_cumulative_step_rate(self):
        power_rate = CumulativePowerMeanRate(1)
        step_rate = CumulativeStepRate()
        for reward, duration in SEQUENCE:
            self.assertAlmostEqual(
                power_rate.update(reward, duration),
                step_rate.update(reward, duration, 1.0),
            )

    def test_p_one_with_time_weight_matches_cumulative_time_rate(self):
        power_rate = CumulativePowerMeanRate(1)
        time_rate = CumulativeTimeRate()
        for reward, duration in SEQUENCE:
            self.assertAlmostEqual(
                power_rate.update(reward, duration, duration),
                time_rate.update(reward, duration, 1.0),
            )

    def test_reward_weighted_p_minus_one_equals_time_weighted_p_one(self):
        harmonic = CumulativePowerMeanRate(-1)
        arithmetic = CumulativePowerMeanRate(1)
        for reward, duration in SEQUENCE:
            self.assertAlmostEqual(
                harmonic.update(reward, duration, reward),
                arithmetic.update(reward, duration, duration),
            )

    def test_reset_replays_sequence_and_clears_value(self):
        estimator = CumulativePowerMeanRate(2)
        first = [estimator.update(*transition) for transition in SEQUENCE]
        estimator.reset()
        self.assertEqual((estimator.value, estimator.rho), (0.0, 0.0))
        second = [estimator.update(*transition) for transition in SEQUENCE]
        self.assertEqual(first, second)


class NormalizedExponentialPowerMeanRateTests(unittest.TestCase):
    def test_unit_weights_match_independent_smoothed_rates_item_by_item(self):
        for beta in (0.0001, 0.999):
            for p in POWERS:
                with self.subTest(beta=beta, p=p):
                    estimator = NormalizedExponentialPowerMeanRate(p, beta)
                    transformed_ema = 0.0
                    normalizer = 0.0
                    for reward, duration in SEQUENCE:
                        rate = reward / duration
                        transformed = math.log(rate) if p == 0 else rate**p
                        transformed_ema = (
                            (1.0 - beta) * transformed_ema + beta * transformed
                        )
                        normalizer = (1.0 - beta) * normalizer + beta
                        expected = inverse_transform(
                            transformed_ema / normalizer, p
                        )
                        self.assertAlmostEqual(
                            estimator.update(reward, duration), expected
                        )

    def test_time_weighted_p_one_matches_moving_ratio_rate(self):
        for beta in (0.0001, 0.3, 0.999):
            with self.subTest(beta=beta):
                power_rate = NormalizedExponentialPowerMeanRate(1, beta)
                ratio_rate = ExponentialMovingRatioRate(beta)
                for reward, duration in SEQUENCE:
                    self.assertAlmostEqual(
                        power_rate.update(reward, duration, duration),
                        ratio_rate.update(reward, duration, 1.0),
                    )

    def test_positive_unit_weight_p_minus_one_matches_weighted_harmonic(self):
        for beta in (0.0001, 0.3, 0.999):
            with self.subTest(beta=beta):
                power_rate = NormalizedExponentialPowerMeanRate(-1, beta)
                harmonic_rate = WeightedHarmonicRate(beta)
                for reward, duration in SEQUENCE:
                    self.assertAlmostEqual(
                        power_rate.update(reward, duration),
                        harmonic_rate.update(reward, duration, 1.0),
                    )

    def test_positive_reward_weighted_p_minus_one_matches_weighted_harmonic(self):
        for beta in (0.0001, 0.3, 0.999):
            with self.subTest(beta=beta):
                power_rate = NormalizedExponentialPowerMeanRate(-1, beta)
                harmonic_rate = WeightedHarmonicRate(beta)
                for reward, duration in SEQUENCE:
                    self.assertAlmostEqual(
                        power_rate.update(reward, duration, reward),
                        harmonic_rate.update(reward, duration, reward),
                    )

    def test_reward_weighted_p_minus_one_equals_time_weighted_p_one(self):
        for beta in (0.0001, 0.3, 0.999):
            with self.subTest(beta=beta):
                harmonic = NormalizedExponentialPowerMeanRate(-1, beta)
                arithmetic = NormalizedExponentialPowerMeanRate(1, beta)
                for reward, duration in SEQUENCE:
                    self.assertAlmostEqual(
                        harmonic.update(reward, duration, reward),
                        arithmetic.update(reward, duration, duration),
                    )

    def test_reset_replays_sequence_and_clears_value(self):
        estimator = NormalizedExponentialPowerMeanRate(0, beta=0.2)
        first = [estimator.update(*transition) for transition in SEQUENCE]
        estimator.reset()
        self.assertEqual((estimator.value, estimator.rho), (0.0, 0.0))
        second = [estimator.update(*transition) for transition in SEQUENCE]
        self.assertEqual(first, second)


class PowerMeanRateValidationTests(unittest.TestCase):
    def test_single_observation_returns_its_local_rate(self):
        for estimator in (
            CumulativePowerMeanRate(2),
            NormalizedExponentialPowerMeanRate(-1, beta=0.3),
        ):
            with self.subTest(estimator=type(estimator).__name__):
                self.assertAlmostEqual(estimator.update(6.0, 4.0, 2.5), 1.5)

    def test_rejects_invalid_durations(self):
        for duration in (0, -1, float("nan"), float("inf"), "invalid"):
            with self.subTest(duration=duration):
                estimator = CumulativePowerMeanRate(1)
                with self.assertRaises(ValueError):
                    estimator.update(1.0, duration)

    def test_rejects_nonpositive_and_nonfinite_rewards(self):
        for reward in (0, -1, float("nan"), float("inf"), "invalid"):
            with self.subTest(reward=reward):
                estimator = NormalizedExponentialPowerMeanRate(1, beta=0.3)
                with self.assertRaises(ValueError):
                    estimator.update(reward, 1.0)

    def test_rejects_nonpositive_and_nonfinite_weights(self):
        for weight in (0, -1, float("nan"), float("inf"), "invalid"):
            with self.subTest(weight=weight):
                estimator = CumulativePowerMeanRate(1)
                with self.assertRaises(ValueError):
                    estimator.update(1.0, 1.0, weight)


if __name__ == "__main__":
    unittest.main()
