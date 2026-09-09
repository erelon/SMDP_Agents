import unittest

from agents.average_rates import (
    CumulativeTimeRate,
    CumulativeWeightedHarmonicRate,
    ExponentialMovingAverage,
    NormalizedEMA,
    WeightedHarmonicRate,
)
from agents.r_learning import ContinuousRLearning
from agents.harmonic_r import (CumulativeHarmonic, CumulativeWeightedHarmonic,
                               Harmonic, WeightedHarmonic)
from agents.smart_r import SMART


def reference_harmonic(sequence, beta, weighted):
    pos_recip = neg_recip = 0.0
    pos_w1 = pos_w2 = neg_w1 = neg_w2 = zero_w = 0.0
    values = []
    for reward, duration in sequence:
        pos, neg, zero = reward > 0, reward < 0, reward == 0
        weight = 1.0 if not weighted else reward
        reciprocal = 0 if zero else duration / reward
        pos_recip = (1 - beta) * pos_recip + beta * reciprocal * pos * weight
        pos_w1 = (1 - beta) * pos_w1 + beta * pos * weight
        pos_w2 = (1 - beta) * pos_w2 + beta * pos
        neg_recip = (1 - beta) * neg_recip + beta * reciprocal * neg * weight
        neg_w1 = (1 - beta) * neg_w1 + beta * neg * weight
        neg_w2 = (1 - beta) * neg_w2 + beta * neg
        h_pos = 0 if pos_recip == 0 else pos_w1 / pos_recip
        h_neg = 0 if neg_recip == 0 else neg_w1 / neg_recip
        zero_w = (1 - beta) * zero_w + beta * zero
        values.append((h_pos * pos_w2 + h_neg * neg_w2) /
                      (pos_w2 + neg_w2 + zero_w))
    return values


class HarmonicTests(unittest.TestCase):
    def assert_agent_rho_matches_estimator(self, agent_class, sequence):
        agent = agent_class("harmonic", [0], rho_learning_rate=0.05)
        for step, (reward, duration) in enumerate(sequence, start=1):
            agent.calc_new_rho(reward, duration, None, None)
            self.assertAlmostEqual(
                agent.rho,
                agent.hma.rho,
                msg=f"{agent_class.__name__} differs at step {step}. HMA is {agent.rho} hma is {agent.hma.rho}\n",
            )
        print(f"HMA of {sequence} is {agent.rho} hma is {agent.hma.rho}\n")

    def assert_both_harmonic_agents_match(self, sequence):
        for agent_class in (Harmonic, WeightedHarmonic):
            with self.subTest(agent=agent_class.__name__):
                self.assert_agent_rho_matches_estimator(agent_class, sequence)

    def test_hma_matches_weighted_harmonic_rate_for_positive_and_zero_rewards(self):
        sequence = [
            (1.0, 2.0), (3.0, 1.0), (0.0, 4.0), (2.0, 3.0),
            (5.0, 2.0), (0.0, 1.0), (4.0, 5.0), (1.5, 2.5),
            (8.0, 4.0), (2.5, 1.5), (6.0, 3.0),
        ]
        self.assert_both_harmonic_agents_match(sequence)

    def test_hma_matches_weighted_harmonic_rate_for_negative_rewards(self):
        sequence = [
            (-1.0, 2.0), (-3.0, 1.0), (-2.0, 4.0), (-5.0, 3.0),
            (-0.5, 2.0), (-4.0, 1.0), (-7.0, 5.0), (-1.5, 2.5),
            (-8.0, 4.0), (-2.5, 1.5), (-6.0, 3.0),
        ]
        self.assert_both_harmonic_agents_match(sequence)

    def test_hma_matches_weighted_harmonic_rate_for_mixed_sign_rewards(self):
        sequence = [
            (2.0, 1.0), (-1.0, 2.0), (4.0, 3.0), (-3.0, 1.0),
            (0.0, 2.0), (1.5, 4.0), (-2.5, 2.5), (6.0, 3.0),
            (-4.0, 1.5), (3.0, 2.0), (-0.5, 1.0),
        ]
        self.assert_both_harmonic_agents_match(sequence)

    def test_hma_matches_weighted_harmonic_rate_for_two_step_sequence(self):
        self.assert_both_harmonic_agents_match([(1.0, 2.0), (2.0, 1.0)])

    def test_ema_rate_matches_weighted_harmonic_rate_for_two_step_sequence(self):
        beta = 0.0001
        agent = Harmonic("harmonic", [0], rho_learning_rate=beta)
        sequence = [(1.0, 2.0), (2.0, 1.0)]
        # e = ExponentialMovingAverage(beta)
        e = NormalizedEMA(beta)
        for step, (reward, duration) in enumerate(sequence, start=1):
            agent.calc_new_rho(reward, duration, None, None)
            e.update((reward / duration), 1.0)
            # e.update((duration/reward), 1.0)

            # self.assertAlmostEqual(
            #     agent.rho,
            #     e.value,
            #     msg=f"{agent.name} differs at step {step} rho is {agent.rho} ema is {e.value}",
            # )
        print(f"ttt HMA of {sequence} is {1 / agent.rho}   EMA is {e.value}")

    def test_ema_rate_matches_harmonic_for_100_step_sequence_high_beta(self):
        sequence = [(1.0, 2.0), (2.0, 1.0)] * 50

        beta = 0.999999999999
        agent = Harmonic(
            "harmonic", [0], rho_learning_rate=beta
        )
        e = ExponentialMovingAverage(beta)
        difference_count = 0

        for step, (reward, duration) in enumerate(sequence, start=1):
            agent.calc_new_rho(reward, duration, None, None)
            e.update(reward / duration, 1.0)

            if round(agent.rho - e.value, 7) != 0:
                difference_count += 1

        print(
            f"beta={beta}: {difference_count} differences "
            f"across {len(sequence)} steps"
        )
        self.assertAlmostEqual(agent.rho, e.value, msg=f"rho is {agent.rho} ema is {e.value}")

    def test_harmonic_converges_to_the_harmonic_not_arithmetic_mean_small_beta(self):
        # The rates alternate between 0.5 and 2.0, whose harmonic mean is 0.8
        # and whose arithmetic mean is 1.25.  A small beta averages over the
        # whole sequence, so the two estimators must not agree.
        sequence = [(1.0, 2.0), (2.0, 1.0)] * 50

        beta = 0.0001
        agent = Harmonic(
            "harmonic", [0], rho_learning_rate=beta
        )
        e = NormalizedEMA(beta)

        for reward, duration in sequence:
            agent.calc_new_rho(reward, duration, None, None)
            e.update(reward / duration, 1.0)

        print(f"beta={beta}: harmonic rho is {agent.rho}, arithmetic ema is {e.value}")
        self.assertAlmostEqual(agent.rho, 0.8, places=3,
                               msg=f"rho is {agent.rho}, harmonic mean is 0.8")
        self.assertAlmostEqual(e.value, 1.25, places=3,
                               msg=f"ema is {e.value}, arithmetic mean is 1.25")
        self.assertNotAlmostEqual(agent.rho, e.value,
                                  msg=f"rho is {agent.rho} ema is {e.value}")

    def test_cumulative_time_rate_matches_weighted_harmonic_for_100_step_sequence_high_beta(self):
        sequence = [(1.0, 2.0), (2.0, 1.0)] * 50

        beta = 0.999
        agent = WeightedHarmonic(
            "wharmonic", [0], rho_learning_rate=beta
        )
        e = CumulativeTimeRate()
        difference_count = 0

        for step, (reward, duration) in enumerate(sequence, start=1):
            agent.calc_new_rho(reward, duration, None, None)
            e.update(reward, duration, 1.0)

            if round(agent.rho - e.rho, 7) != 0:
                difference_count += 1

        print(
            f"beta={beta}: {difference_count} differences "
            f"across {len(sequence)} steps"
        )
        print(f"rho is {agent.rho} cumulative time rate is {e.rho}")
        self.assertNotAlmostEqual(agent.rho, e.rho, msg=f"rho is {agent.rho} cumulative time rate is {e.rho}")

    def test_cumulative_time_rate_matches_weighted_harmonic_for_100_step_sequence_small_beta(self):
        sequence = [(1.0, 2.0), (2.0, 1.0)] * 50

        beta = 0.00000001
        agent = WeightedHarmonic(
            "wharmonic", [0], rho_learning_rate=beta
        )
        e = CumulativeTimeRate()
        difference_count = 0

        for step, (reward, duration) in enumerate(sequence, start=1):
            agent.calc_new_rho(reward, duration, None, None)
            e.update(reward, duration, 1.0)

            if round(agent.rho - e.rho, 7) != 0:
                difference_count += 1

        print(
            f"beta={beta}: {difference_count} differences "
            f"across {len(sequence)} steps"
        )
        print(f"rho is {agent.rho} cumulative time rate is {e.rho}")

        self.assertAlmostEqual(agent.rho, e.rho, msg=f"rho is {agent.rho} cumulative time rate is {e.rho}")

    #    def test_ema_rate_matches_weighted_harmonic_rate_for_two_steo_sequence(self):
    #         beta = 0.8
    #         agent = WeightedHarmonic("wharmonic", [0], rho_learning_rate=beta)
    #         sequence = [(1.0, 2.0), (2.0, 1.0)]
    #         e = ExponentialMovingAverage(beta)
    #         h = WeightedHarmonicRate(beta)
    #         for step, (reward, duration) in enumerate(sequence, start=1):
    #             agent.calc_new_rho(reward, duration, None, None)
    #             e.update((reward/duration), 1.0)
    #             h.update(reward, duration, reward)

    #             self.assertAlmostEqual(
    #                 agent.rho,
    #                 h.value,
    #                 msg=f"{agent.name} differs at step {step} rho is {agent.rho} ema is {h.value}",
    #             )
    #         print(f"HMA of {sequence} is {agent.rho}   EMA is {h.value}")

    def test_unweighted_harmonic_matches_reference_for_signed_and_zero_rewards(self):
        sequence = [(4.0, 2.0), (-2.0, 1.0), (0.0, 3.0), (1.0, 4.0)]
        expected = reference_harmonic(sequence, beta=0.3, weighted=False)
        agent = Harmonic("harmonic", [0], rho_learning_rate=0.3)
        actual = []
        for reward, duration in sequence:
            agent.calc_new_rho(reward, duration, None, None)
            actual.append(agent.rho)
        for observed, wanted in zip(actual, expected):
            self.assertAlmostEqual(observed, wanted)

    def test_weighted_harmonic_matches_current_reward_weighting(self):
        sequence = [(4.0, 2.0), (-2.0, 1.0), (0.0, 3.0), (1.0, 4.0)]
        expected = reference_harmonic(sequence, beta=0.3, weighted=True)
        agent = WeightedHarmonic("weighted", [0], rho_learning_rate=0.3)
        for (reward, duration), wanted in zip(sequence, expected):
            agent.calc_new_rho(reward, duration, None, None)
            self.assertAlmostEqual(agent.rho, wanted)

    def test_instantaneous_transition_is_accepted_and_negative_duration_is_not(self):
        for cls in (Harmonic, WeightedHarmonic):
            with self.subTest(agent=cls.__name__):
                agent = cls("instant", [0], rho_learning_rate=0.3)
                agent.calc_new_rho(2.0, 1.0, None, None)
                warm = agent.rho
                # A zero duration only decays the branch's reciprocal average.
                agent.calc_new_rho(2.0, 0.0, None, None)
                self.assertGreater(agent.rho, warm)
                with self.assertRaises(ValueError):
                    agent.calc_new_rho(2.0, -1.0, None, None)

    def test_all_zero_rewards_produce_zero_rate(self):
        for cls in (Harmonic, WeightedHarmonic):
            agent = cls("zero", [0], rho_learning_rate=0.4)
            agent.calc_new_rho(0.0, 5.0, None, None)
            self.assertEqual(agent.rho, 0.0)

    def test_reset_reproduces_same_harmonic_sequence(self):
        agent = Harmonic("harmonic", [0], rho_learning_rate=0.2)
        sequence = [(2.0, 1.0), (-1.0, 2.0)]
        first = []
        for transition in sequence:
            agent.calc_new_rho(*transition, None, None)
            first.append(agent.rho)
        agent.reset()
        second = []
        for transition in sequence:
            agent.calc_new_rho(*transition, None, None)
            second.append(agent.rho)
        self.assertEqual(first, second)


class HarmonicTargetTests(unittest.TestCase):
    """Every shipped harmonic agent uses R-learning's plain target.

    The divided variant lives in ``agents/experemental_harmonic_r.py`` and is
    covered by its own tests; these guard that none of it leaks back in here.
    """

    ALL = (WeightedHarmonic, CumulativeWeightedHarmonic, Harmonic, CumulativeHarmonic)

    def test_none_of_them_scales_the_advantage(self):
        for agent_class in self.ALL:
            with self.subTest(agent=agent_class.__name__):
                agent = agent_class("harmonic", [0])
                agent.rho = 2.0
                # 5 - 2*2 + 1 = 2, not the divided form's 1.5.
                self.assertAlmostEqual(agent.set_target(5.0, 2.0, 1.0), 2.0)
                self.assertAlmostEqual(
                    agent.set_target(5.0, 2.0, 1.0),
                    ContinuousRLearning.set_target(agent, 5.0, 2.0, 1.0))

    def test_the_target_is_inherited_rather_than_overridden(self):
        for agent_class in self.ALL:
            with self.subTest(agent=agent_class.__name__):
                self.assertNotIn("set_target", vars(agent_class))

    def test_a_negative_rho_does_not_reorder_them(self):
        # The property the plain target has and dividing by a signed rho loses.
        for agent_class in self.ALL:
            with self.subTest(agent=agent_class.__name__):
                agent = agent_class("harmonic", [0])
                plain = ContinuousRLearning("plain", [0])
                agent.rho = plain.rho = -1.5
                pairs = ((-1.0, 1.0), (-10.0, 1.0))
                self.assertEqual([agent.set_target(r, t, 0.0) for r, t in pairs],
                                 [plain.set_target(r, t, 0.0) for r, t in pairs])

    def test_the_weight_is_the_reward_only_for_the_weighted_pair(self):
        expected = {WeightedHarmonic: 4.0, CumulativeWeightedHarmonic: 4.0,
                    Harmonic: 1.0, CumulativeHarmonic: 1.0}
        for agent_class, weight in expected.items():
            with self.subTest(agent=agent_class.__name__):
                agent = agent_class("harmonic", [0])
                agent.calc_new_rho(4.0, 2.0, None, None)
                reference = (CumulativeWeightedHarmonicRate()
                             if "Cumulative" in agent_class.__name__
                             else WeightedHarmonicRate(0.3))
                self.assertAlmostEqual(agent.rho,
                                       reference.update(4.0, 2.0, weight))


class CumulativeHarmonicTests(unittest.TestCase):
    """The same two agents with the whole history behind rho instead of an EMA."""

    POSITIVE = [(1.0, 2.0), (3.0, 1.0), (2.0, 3.0), (5.0, 2.0), (4.0, 5.0)]
    MIXED = [(2.0, 1.0), (-1.0, 2.0), (0.0, 2.0), (4.0, 3.0), (-3.0, 1.0)]

    @staticmethod
    def drive(agent, sequence):
        for reward, duration in sequence:
            agent.calc_new_rho(reward, duration, None, None)
        return agent.rho

    def test_the_unweighted_one_is_the_harmonic_mean_of_the_rates(self):
        agent = CumulativeHarmonic("cumulative", [0])
        expected = len(self.POSITIVE) / sum(t / r for r, t in self.POSITIVE)
        self.assertAlmostEqual(self.drive(agent, self.POSITIVE), expected)

    def test_the_weighted_one_is_smart_while_every_reward_is_positive(self):
        # (tau/r)*r cancels the reward, so the branch is sum(r)/sum(tau) exactly.
        # This is why the pair is worth having: only the unweighted one is a
        # harmonic mean on a positive-reward domain.
        weighted = CumulativeWeightedHarmonic("cumulative_weighted", [0])
        smart = SMART("smart", [0])
        self.assertAlmostEqual(self.drive(weighted, self.POSITIVE),
                               self.drive(smart, self.POSITIVE))
        self.assertNotAlmostEqual(
            self.drive(CumulativeHarmonic("cumulative", [0]), self.POSITIVE),
            smart.rho)

    def test_a_sign_change_separates_the_weighted_one_from_smart(self):
        weighted = self.drive(CumulativeWeightedHarmonic("w", [0]), self.MIXED)
        smart = self.drive(SMART("smart", [0]), self.MIXED)
        self.assertNotAlmostEqual(weighted, smart)

    def test_they_are_the_vanishing_gain_limit_of_the_moving_versions(self):
        for cumulative, moving in ((CumulativeHarmonic, Harmonic),
                                   (CumulativeWeightedHarmonic, WeightedHarmonic)):
            with self.subTest(agent=cumulative.__name__):
                self.assertAlmostEqual(
                    self.drive(cumulative("c", [0]), self.MIXED * 40),
                    self.drive(moving("m", [0], rho_learning_rate=1e-9),
                               self.MIXED * 40),
                    places=6)

    def test_the_gain_is_accepted_and_ignored(self):
        # Every agent in the family takes rho_learning_rate; a cumulative average
        # has no gain to set, so passing wildly different ones changes nothing.
        rhos = {beta: self.drive(CumulativeHarmonic("c", [0],
                                                    rho_learning_rate=beta),
                                 self.MIXED)
                for beta in (0.01, 0.5, 1.0)}
        self.assertEqual(len(set(rhos.values())), 1)

    def test_it_cannot_come_back_down_the_way_the_moving_version_does(self):
        # The cost of not forgetting, and the reason a *smaller* overshoot can be
        # the worse one: after a stretch of fast transitions inflates rho, the EMA
        # is back on a steady rate of 1 within a few transitions and the cumulative
        # average is still far above it hundreds later.
        burst = ([(10.0, 10.0)] + [(1.0, 0.1)] * 10) * 200
        moving, cumulative = Harmonic("m", [0]), CumulativeHarmonic("c", [0])
        for agent in (moving, cumulative):
            self.drive(agent, burst)
        self.assertGreater(moving.rho, 5.0)
        self.assertAlmostEqual(cumulative.rho, 11 / 2, places=3)  # the exact limit

        steady = [(1.0, 1.0)] * 5
        self.assertLess(self.drive(moving, steady), 1.9)
        self.assertGreater(self.drive(cumulative, steady), 5.0)
        self.assertGreater(self.drive(cumulative, [(1.0, 1.0)] * 395), 3.0)

    def test_rho_follows_the_estimator_and_survives_a_reset(self):
        for agent_class in (CumulativeHarmonic, CumulativeWeightedHarmonic):
            with self.subTest(agent=agent_class.__name__):
                agent = agent_class("cumulative", [0])
                first = self.drive(agent, self.MIXED)
                self.assertAlmostEqual(agent.rho, agent.hma.rho)
                agent.reset()
                self.assertEqual(agent.rho, 0.0)
                self.assertEqual(self.drive(agent, self.MIXED), first)


if __name__ == "__main__":
    unittest.main()
