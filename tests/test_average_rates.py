import importlib.util
import pathlib
import unittest


MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "agents" / "average_rates.py"
SPEC = importlib.util.spec_from_file_location("average_rates", MODULE_PATH)
average_rates = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(average_rates)

ExponentialMovingAverage = average_rates.ExponentialMovingAverage
CumulativeTimeRate = average_rates.CumulativeTimeRate
WeightedHarmonicRate = average_rates.WeightedHarmonicRate
NormHMA = average_rates.NormHMA


class ExponentialMovingAverageTests(unittest.TestCase):
    def test_updates_with_a_different_weight_on_each_call(self):
        average = ExponentialMovingAverage(0.25)
        self.assertAlmostEqual(average.update(8.0, 0.5), 1.0)
        self.assertAlmostEqual(average.update(-4.0, 2.0), -1.25)
        self.assertAlmostEqual(average.update(10.0, 0.0), -0.9375)

    def test_unit_weight_matches_standard_zero_initialized_ema(self):
        average = ExponentialMovingAverage(0.5)
        self.assertEqual(average.update(4.0, 1.0), 2.0)
        self.assertEqual(average.update(2.0, 1.0), 2.0)

    def test_reset_reproduces_sequence(self):
        average = ExponentialMovingAverage(0.2)
        first = [average.update(value, weight) for value, weight in ((2, 1), (3, -1))]
        average.reset()
        second = [average.update(value, weight) for value, weight in ((2, 1), (3, -1))]
        self.assertEqual(first, second)

    def test_rejects_invalid_beta_and_nonfinite_inputs(self):
        for beta in (0, -0.1, 1.1, float("nan")):
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                ExponentialMovingAverage(beta)
        average = ExponentialMovingAverage(0.5)
        with self.assertRaises(ValueError):
            average.update(float("inf"), 1)
        with self.assertRaises(ValueError):
            average.update(1, float("nan"))


class CumulativeTimeRateTests(unittest.TestCase):
    def test_accumulates_weighted_reward_over_unweighted_duration(self):
        rate = CumulativeTimeRate()
        self.assertEqual(rate.update(4, 2, 0.5), 1.0)
        self.assertEqual(rate.update(-1, 1, 2.0), 0.0)
        self.assertEqual(rate.total_reward, 0.0)
        self.assertEqual(rate.total_duration, 3.0)

    def test_reset_and_invalid_duration(self):
        rate = CumulativeTimeRate()
        rate.update(3, 1, 1)
        rate.reset()
        self.assertEqual((rate.value, rate.rho, rate.total_reward, rate.total_duration),
                         (0.0, 0.0, 0.0, 0.0))
        for duration in (0, -1, float("inf")):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                rate.update(1, duration, 1)


""" class RatioEmaRateTests(unittest.TestCase):
    def test_matches_composed_reward_and_duration_emas(self):
        rate = RatioEmaRate(0.25, duration_beta=0.5)
        reward_ema = ExponentialMovingAverage(0.25)
        duration_ema = ExponentialMovingAverage(0.5)
        sequence = ((4, 2, 0.5), (-1, 3, 2.0), (0, 1, -3.0))
        for reward, duration, weight in sequence:
            expected = (
                reward_ema.update(reward, weight)
                / duration_ema.update(duration, 1.0)
            )
            self.assertAlmostEqual(rate.update(reward, duration, weight), expected)

    def test_reset_clears_composed_emas(self):
        rate = RatioEmaRate(0.5)
        rate.update(2, 1, 1)
        rate.reset()
        self.assertEqual(rate.value, 0.0)
        self.assertEqual(rate.reward_ema.value, 0.0)
        self.assertEqual(rate.duration_ema.value, 0.0)
"""


def legacy_harmonic(sequence, beta):
    pos_recip = neg_recip = 0.0
    pos_w1 = pos_w2 = neg_w1 = neg_w2 = zero_w = 0.0
    results = []
    for reward, duration, weight in sequence:
        pos, neg, zero = reward > 0, reward < 0, reward == 0
        reciprocal = 0 if zero else duration / reward
        pos_recip = (1 - beta) * pos_recip + beta * reciprocal * pos * weight
        pos_w1 = (1 - beta) * pos_w1 + beta * pos * weight
        pos_w2 = (1 - beta) * pos_w2 + beta * pos
        neg_recip = (1 - beta) * neg_recip + beta * reciprocal * neg * weight
        neg_w1 = (1 - beta) * neg_w1 + beta * neg * weight
        neg_w2 = (1 - beta) * neg_w2 + beta * neg
        zero_w = (1 - beta) * zero_w + beta * zero
        h_pos = 0 if pos_recip == 0 else pos_w1 / pos_recip
        h_neg = 0 if neg_recip == 0 else neg_w1 / neg_recip
        results.append(
            (h_pos * pos_w2 + h_neg * neg_w2) / (pos_w2 + neg_w2 + zero_w)
        )
    return results


class WeightedHarmonicRateTests(unittest.TestCase):
    def test_matches_legacy_formula_with_signed_rewards_and_changing_weights(self):
        sequence = ((4, 2, 4), (-2, 1, -2), (0, 3, 0), (1, 4, 0.5))
        expected = legacy_harmonic(sequence, beta=0.3)
        rate = WeightedHarmonicRate(0.3)
        for transition, wanted in zip(sequence, expected):
            self.assertAlmostEqual(rate.update(*transition), wanted)

    def test_all_zero_rewards_return_zero(self):
        rate = WeightedHarmonicRate(0.4)
        self.assertEqual(rate.update(0, 5, 3), 0.0)
        self.assertEqual(rate.update(0, 2, -1), 0.0)

    def test_reset_reproduces_sequence(self):
        sequence = ((2, 1, 2), (-1, 2, -1))
        rate = WeightedHarmonicRate(0.2)
        first = [rate.update(*transition) for transition in sequence]
        rate.reset()
        second = [rate.update(*transition) for transition in sequence]
        self.assertEqual(first, second)

    def test_is_weighted_harmonic_specialized_to_unit_weight(self):
        sequence = ((4, 1), (-2, 1), (0, 3), (2, 4))
        harmonic = WeightedHarmonicRate(0.3)
        weighted = WeightedHarmonicRate(0.3)
        for reward, duration in sequence:
            h = harmonic.update(reward, duration, reward)
            w = weighted.update(reward, duration, 1.0)
        self.assertNotAlmostEqual(h,w)

    def test_harmonic_is_harmonic(self):
        sequence = ((1,2), (2,1))
        harmonic = WeightedHarmonicRate(0.00000000001)
        for reward, duration in sequence:
            r  = harmonic.update(reward, duration, 1.0)
        self.assertAlmostEqual(r,0.8)


class NormHMATests(unittest.TestCase):
    def assert_matches_weighted_harmonic_rate(self, sequence, beta=0.05):
        weighted_harmonic = WeightedHarmonicRate(beta)
        norm_hma = NormHMA(beta)

        for step, (reward, duration) in enumerate(sequence, start=1):
            weighted_value = weighted_harmonic.update(reward, duration, 1.0)
            norm_value = norm_hma.update(reward, duration, 1.0)

            self.assertAlmostEqual(
                norm_value,
                weighted_value,
                msg=(
                    f"NormHMA differs from WeightedHarmonicRate at step {step}: "
                    f"reward={reward}, duration={duration}, "
                    f"norm_hma={norm_value}, weighted_harmonic={weighted_value}"
                ),
            )

    def test_matches_weighted_harmonic_rate_for_positive_and_zero_rewards(self):
        sequence = [
            (1.0, 2.0), (3.0, 1.0), (0.0, 4.0), (2.0, 3.0),
            (5.0, 2.0), (0.0, 1.0), (4.0, 5.0), (1.5, 2.5),
            (8.0, 4.0), (2.5, 1.5), (6.0, 3.0),
        ]
        self.assert_matches_weighted_harmonic_rate(sequence)

    def test_matches_weighted_harmonic_rate_for_negative_rewards(self):
        sequence = [
            (-1.0, 2.0), (-3.0, 1.0), (-2.0, 4.0), (-5.0, 3.0),
            (-0.5, 2.0), (-4.0, 1.0), (-7.0, 5.0), (-1.5, 2.5),
            (-8.0, 4.0), (-2.5, 1.5), (-6.0, 3.0),
        ]
        self.assert_matches_weighted_harmonic_rate(sequence)

    def test_matches_weighted_harmonic_rate_for_mixed_sign_rewards(self):
        sequence = [
            (2.0, 1.0), (-1.0, 2.0), (4.0, 3.0), (-3.0, 1.0),
            (0.0, 2.0), (1.5, 4.0), (-2.5, 2.5), (6.0, 3.0),
            (-4.0, 1.5), (3.0, 2.0), (-0.5, 1.0),
        ]
        self.assert_matches_weighted_harmonic_rate(sequence)

    def test_matches_weighted_harmonic_rate_for_strictly_positive_rewards(self):
        negative_sequence = [
            (-1.0, 2.0), (-3.0, 1.0), (-2.0, 4.0), (-5.0, 3.0),
            (-0.5, 2.0), (-4.0, 1.0), (-7.0, 5.0), (-1.5, 2.5),
            (-8.0, 4.0), (-2.5, 1.5), (-6.0, 3.0),
        ]
        sequence = [(abs(reward), duration) for reward, duration in negative_sequence]
        self.assert_matches_weighted_harmonic_rate(sequence)


if __name__ == "__main__":
    unittest.main()
