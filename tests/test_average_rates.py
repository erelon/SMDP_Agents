import math
import unittest

from agents.average_rates import (
    CumulativeAverage,
    CumulativeTimeRate,
    CumulativeWeightedHarmonicRate,
    ExponentialMovingAverage,
    ExponentialMovingRatioRate,
    NormalizedExponentialMovingTimeRate,
    NormHMA,
    TimeDecayedEMA,
    WeightedHarmonicRate,
)

#: One long slow transition followed by ten short fast ones -- the worked example
#: that motivated the time-domain smoother, and the shape of the
#: ``high_time_variance`` environment. Half the elapsed time is the first
#: transition; one eleventh of the transitions is.
BURST_SEQUENCE = [(10.0, 10.0)] + [(1.0, 0.1)] * 10


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
        self.assertEqual((rate.rho, rate.total_reward, rate.total_duration),
                         (0.0, 0.0, 0.0))
        for duration in (0, -1, float("inf")):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                rate.update(1, duration, 1)


class TimeDecayedEMATests(unittest.TestCase):
    """Forgetting per unit of elapsed time, not per update."""

    def test_beta_is_the_gain_of_a_unit_duration_update(self):
        # lambda = -log(1 - beta), so a tau = 1 transition gets exactly beta and a
        # time-decayed EMA is indistinguishable from a plain one on a unit clock.
        for beta in (0.1, 0.3, 0.5):
            with self.subTest(beta=beta):
                decayed = TimeDecayedEMA(beta)
                plain = ExponentialMovingAverage(beta)
                for level in (4.0, -2.0, 7.5):
                    self.assertAlmostEqual(decayed.update(level, 1.0),
                                           plain.update(level, 1.0))
                self.assertAlmostEqual(decayed.lambda_, -math.log(1 - beta))

    def test_a_longer_transition_forgets_more(self):
        # The gain climbs with the duration: 0.5 at tau = 1, 0.75 at tau = 2,
        # 0.067 at tau = 0.1 -- the worked example's own numbers.
        for duration, gain in ((1.0, 0.5), (2.0, 0.75), (0.1, 1 - 0.5 ** 0.1)):
            with self.subTest(duration=duration):
                average = TimeDecayedEMA(0.5)
                self.assertAlmostEqual(average.update(1.0, duration), gain)

    def test_beta_of_one_keeps_only_the_latest_level(self):
        average = TimeDecayedEMA(1.0)
        self.assertEqual(average.lambda_, math.inf)
        average.update(3.0, 5.0)
        self.assertEqual(average.update(-2.0, 0.001), -2.0)

    def test_rejects_an_instantaneous_transition(self):
        # It divides the reward by the duration to get a rate, so tau = 0 has no
        # meaning here -- unlike the harmonic estimators, which accept it.
        with self.assertRaises(ValueError):
            TimeDecayedEMA(0.5).update(1.0, 0.0)


class ExponentialTimeRateTests(unittest.TestCase):
    """The time-domain smoother against the per-transition ratio of EMAs."""

    @staticmethod
    def feed(estimator, sequence):
        value = 0.0
        for reward, duration in sequence:
            value = estimator.update(reward, duration)
        return value

    def test_the_burst_separates_the_two_smoothers(self):
        # Both see the same eleven transitions. The ratio of EMAs has performed
        # ten forgetting operations since the slow step and has all but discarded
        # it; the time-domain filter has seen only one time unit pass and has not.
        # The pathwise time average of the whole sequence is 20/11 = 1.818.
        ratio = self.feed(ExponentialMovingRatioRate(0.5), BURST_SEQUENCE)
        timed = self.feed(NormalizedExponentialMovingTimeRate(0.5), BURST_SEQUENCE)
        self.assertAlmostEqual(ratio, 9.5806, places=4)
        self.assertAlmostEqual(timed, 5.5022, places=4)
        self.assertAlmostEqual(self.feed(CumulativeTimeRate(), BURST_SEQUENCE),
                               20 / 11, places=4)
        self.assertGreater(ratio / timed, 1.7)

    def test_both_smoothers_converge_on_the_time_average_as_beta_vanishes(self):
        # The disagreement is a fixed-gain effect, not an asymptotic bias: the
        # ratio of two *sample averages* is exactly the pathwise time average.
        cycles = BURST_SEQUENCE * 200
        for estimator in (ExponentialMovingRatioRate(1e-4),
                          NormalizedExponentialMovingTimeRate(1e-4)):
            with self.subTest(estimator=type(estimator).__name__):
                self.assertAlmostEqual(self.feed(estimator, cycles), 20 / 11,
                                       places=2)

    def test_the_time_rate_does_not_care_how_a_transition_is_split(self):
        # exp(-lambda*t1) * exp(-lambda*t2) == exp(-lambda*(t1+t2)), so cutting
        # every transition into n pieces covering the same time at the same rate
        # leaves the estimate alone.
        whole = self.feed(NormalizedExponentialMovingTimeRate(0.5), BURST_SEQUENCE)
        for pieces in (2, 5, 20):
            with self.subTest(pieces=pieces):
                split = [(r / pieces, d / pieces) for r, d in BURST_SEQUENCE
                         for _ in range(pieces)]
                self.assertAlmostEqual(
                    self.feed(NormalizedExponentialMovingTimeRate(0.5), split),
                    whole, places=9)

    def test_the_ratio_of_emas_does_care(self):
        # The same split walks it from 9.58 to the burst's own rate of 10: its
        # memory is measured in transitions, so finer transitions are a shorter
        # memory in seconds.
        previous = self.feed(ExponentialMovingRatioRate(0.5), BURST_SEQUENCE)
        for pieces in (2, 5, 20):
            with self.subTest(pieces=pieces):
                split = [(r / pieces, d / pieces) for r, d in BURST_SEQUENCE
                         for _ in range(pieces)]
                value = self.feed(ExponentialMovingRatioRate(0.5), split)
                self.assertGreater(value, previous)
                previous = value
        self.assertAlmostEqual(previous, 10.0, places=6)

    def test_a_constant_rate_is_returned_exactly(self):
        # Debiased, so no zero-initialisation transient to wait out.
        estimator = NormalizedExponentialMovingTimeRate(0.3)
        for duration in (0.1, 7.0, 2.5):
            self.assertAlmostEqual(estimator.update(3.0 * duration, duration), 3.0)

    def test_reset_reproduces_the_sequence(self):
        estimator = NormalizedExponentialMovingTimeRate(0.4)
        first = self.feed(estimator, BURST_SEQUENCE)
        estimator.reset()
        self.assertEqual(self.feed(estimator, BURST_SEQUENCE), first)


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

    def test_reward_weighting_collapses_to_the_ratio_of_emas_when_rewards_are_positive(self):
        # Not a coincidence when the two agents post identical numbers on a
        # positive-reward environment: with w = r the positive branch averages
        # (tau/r)*r = tau against r, so its harmonic mean *is* EMA(r)/EMA(tau),
        # and with no negative or zero rewards the sign mix is that branch alone.
        harmonic = WeightedHarmonicRate(0.3)
        ratio = ExponentialMovingRatioRate(0.3)
        for reward, duration in BURST_SEQUENCE:
            self.assertAlmostEqual(harmonic.update(reward, duration, reward),
                                   ratio.update(reward, duration), places=12)

    def test_a_single_negative_reward_breaks_that_identity(self):
        # The branches are what the harmonic estimator is *for*, so the collapse
        # above is a property of the domain, not of the estimator.
        harmonic = WeightedHarmonicRate(0.3)
        ratio = ExponentialMovingRatioRate(0.3)
        for reward, duration in ((4.0, 1.0), (-2.0, 1.0), (3.0, 2.0)):
            weighted = harmonic.update(reward, duration, reward)
            plain = ratio.update(reward, duration)
        self.assertNotAlmostEqual(weighted, plain)

    def test_is_weighted_harmonic_specialized_to_unit_weight(self):
        sequence = ((4, 1), (-2, 1), (0, 3), (2, 4))
        harmonic = WeightedHarmonicRate(0.3)
        weighted = WeightedHarmonicRate(0.3)
        for reward, duration in sequence:
            h = harmonic.update(reward, duration, reward)
            w = weighted.update(reward, duration, 1.0)
        self.assertNotAlmostEqual(h, w)

    def test_harmonic_is_harmonic(self):
        sequence = ((1, 2), (2, 1))
        harmonic = WeightedHarmonicRate(0.00000000001)
        for reward, duration in sequence:
            r = harmonic.update(reward, duration, 1.0)
        self.assertAlmostEqual(r, 0.8)


class CumulativeAverageTests(unittest.TestCase):
    """The EMA interface with a 1/n gain."""

    def test_it_is_the_running_mean_of_value_times_weight(self):
        average = CumulativeAverage()
        self.assertAlmostEqual(average.update(8.0, 0.5), 4.0)
        self.assertAlmostEqual(average.update(-4.0, 2.0), (4.0 - 8.0) / 2)
        self.assertAlmostEqual(average.update(10.0, 0.0), (4.0 - 8.0 + 0.0) / 3)
        self.assertEqual(average.count, 3)

    def test_reset_clears_the_count_as_well_as_the_value(self):
        average = CumulativeAverage()
        average.update(5.0, 1.0)
        average.reset()
        self.assertEqual((average.value, average.count), (0.0, 0))
        self.assertAlmostEqual(average.update(2.0, 1.0), 2.0)

    def test_rejects_nonfinite_inputs(self):
        average = CumulativeAverage()
        with self.assertRaises(ValueError):
            average.update(float("inf"), 1.0)
        with self.assertRaises(ValueError):
            average.update(1.0, float("nan"))


class CumulativeWeightedHarmonicRateTests(unittest.TestCase):
    """The harmonic estimator with running means where the EMAs were."""

    POSITIVE = ((4.0, 2.0), (1.0, 0.5), (3.0, 1.0), (2.0, 4.0))
    MIXED = ((4.0, 2.0), (-1.0, 0.5), (0.0, 3.0), (3.0, 1.0), (-2.0, 2.0))

    @staticmethod
    def feed(estimator, sequence, weighting="one"):
        value = 0.0
        for reward, duration in sequence:
            weight = reward if weighting == "reward" else 1.0
            value = estimator.update(reward, duration, weight)
        return value

    def test_unit_weight_is_the_harmonic_mean_of_the_rates(self):
        # While every reward is strictly positive: n / sum(tau_i / r_i), the
        # third averaging of the same samples beside sum(r)/sum(tau) and the mean
        # of r_i/tau_i.
        expected = len(self.POSITIVE) / sum(t / r for r, t in self.POSITIVE)
        self.assertAlmostEqual(
            self.feed(CumulativeWeightedHarmonicRate(), self.POSITIVE), expected)

    def test_reward_weighting_collapses_onto_the_cumulative_time_rate(self):
        # The r in (tau/r)*r cancels, so the branch is sum(r)/sum(tau) exactly --
        # the cumulative counterpart of WeightedHarmonicRate collapsing onto
        # ExponentialMovingRatioRate on the same domain.
        harmonic = CumulativeWeightedHarmonicRate()
        cumulative = CumulativeTimeRate()
        for reward, duration in self.POSITIVE:
            self.assertAlmostEqual(harmonic.update(reward, duration, reward),
                                   cumulative.update(reward, duration))

    def test_a_sign_change_puts_the_two_branches_back_to_work(self):
        # Which is the whole reason the estimator has branches: on mixed signs it
        # is no longer the cumulative time rate.
        harmonic = self.feed(CumulativeWeightedHarmonicRate(), self.MIXED, "reward")
        cumulative = self.feed(CumulativeTimeRate(), self.MIXED)
        self.assertNotAlmostEqual(harmonic, cumulative)

    def test_it_is_the_vanishing_gain_limit_of_the_moving_version(self):
        # Same estimator, 1/n in place of beta, so a beta small enough to keep the
        # whole history reproduces it.
        for weighting in ("one", "reward"):
            with self.subTest(weighting=weighting):
                self.assertAlmostEqual(
                    self.feed(WeightedHarmonicRate(1e-9), self.MIXED * 40, weighting),
                    self.feed(CumulativeWeightedHarmonicRate(), self.MIXED * 40,
                              weighting),
                    places=6)

    def test_an_instantaneous_transition_is_accepted(self):
        # tau appears only in the numerator of tau/r, as in the moving version.
        rate = CumulativeWeightedHarmonicRate()
        self.assertTrue(rate.accepts_zero_duration)
        self.assertAlmostEqual(rate.update(2.0, 0.0), 0.0)
        with self.assertRaises(ValueError):
            rate.update(1.0, -1.0)

    def test_all_zero_rewards_return_zero(self):
        rate = CumulativeWeightedHarmonicRate()
        self.assertEqual(rate.update(0.0, 5.0, 3.0), 0.0)
        self.assertEqual(rate.update(0.0, 2.0, -1.0), 0.0)

    def test_reset_reproduces_the_sequence(self):
        rate = CumulativeWeightedHarmonicRate()
        first = self.feed(rate, self.MIXED, "reward")
        rate.reset()
        self.assertEqual(self.feed(rate, self.MIXED, "reward"), first)


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
