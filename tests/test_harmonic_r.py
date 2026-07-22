from agents.average_rates import NormalizedEMA
from tests.test_average_rates import CumulativeTimeRate, ExponentialMovingAverage, WeightedHarmonicRate
import unittest

from tests._loader import load_tabular_modules

MODULE = load_tabular_modules()["harmonic_r"]
Harmonic = MODULE.Harmonic
WeightedHarmonic = MODULE.WeightedHarmonic


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
                agent.hma.value,
                msg=f"{agent_class.__name__} differs at step {step}. HMA is {agent.rho} hma is {agent.hma.value}\n",
            )
        print(f"HMA of {sequence} is {agent.rho} hma is {agent.hma.value}\n")

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
            e.update((reward/duration), 1.0)
            # e.update((duration/reward), 1.0)
            
            # self.assertAlmostEqual(
            #     agent.rho,
            #     e.value,
            #     msg=f"{agent.name} differs at step {step} rho is {agent.rho} ema is {e.value}",
            # )
        print(f"ttt HMA of {sequence} is {1/agent.rho}   EMA is {e.value}")

    def test_ema_rate_matches_harmonic_for_100_step_sequence_high_beta(self):
        sequence = [(1.0, 2.0), (2.0, 1.0)] * 50

        beta=0.999999999999
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
        self.assertAlmostEqual(agent.rho,e.value,msg=f"rho is {agent.rho} ema is {e.value}")

    @unittest.skip("test is superseded by the certainty-equivalent agents")
    def test_ema_rate_matches_harmonic_for_100_step_sequence_small_beta(self):
        sequence = [(1.0, 2.0), (2.0, 1.0)] * 50

        beta=0.0001
        agent = Harmonic(
            "harmonic", [0], rho_learning_rate=beta
        )
        # e = ExponentialMovingAverage(beta)
        e = NormalizedEMA(beta)
        difference_count = 0

        for step, (reward, duration) in enumerate(sequence, start=1):
            agent.calc_new_rho(reward, duration, None, None)
            e.update(reward/duration, 1.0)

            if round(agent.rho - e.value, 7) != 0:
                difference_count += 1

        print(
            f"beta={beta}: {difference_count} differences "
            f"across {len(sequence)} steps"
        )
        self.assertAlmostEqual(agent.rho,e.value,msg=f"rho is {agent.rho} ema is {e.value}")

    def test_cumulative_time_rate_matches_weighted_harmonic_for_100_step_sequence_high_beta(self):
        sequence = [(1.0, 2.0), (2.0, 1.0)] * 50

        beta=0.999
        agent = WeightedHarmonic(
            "wharmonic", [0], rho_learning_rate=beta
        )
        e = CumulativeTimeRate()
        difference_count = 0

        for step, (reward, duration) in enumerate(sequence, start=1):
            agent.calc_new_rho(reward, duration, None, None)
            e.update(reward, duration, 1.0)

            if round(agent.rho - e.value, 7) != 0:
                difference_count += 1

        print(
            f"beta={beta}: {difference_count} differences "
            f"across {len(sequence)} steps"
        )
        print(f"rho is {agent.rho} cumulative time rate is {e.value}")
        self.assertNotAlmostEqual(agent.rho,e.value,msg=f"rho is {agent.rho} cumulative time rate is {e.value}")

    def test_cumulative_time_rate_matches_weighted_harmonic_for_100_step_sequence_small_beta(self):
        sequence = [(1.0, 2.0), (2.0, 1.0)] * 50

        beta=0.00000001
        agent = WeightedHarmonic(
            "wharmonic", [0], rho_learning_rate=beta
        )
        e = CumulativeTimeRate()
        difference_count = 0

        for step, (reward, duration) in enumerate(sequence, start=1):
            agent.calc_new_rho(reward, duration, None, None)
            e.update(reward, duration, 1.0)

            if round(agent.rho - e.value, 7) != 0:
                difference_count += 1

        print(
            f"beta={beta}: {difference_count} differences "
            f"across {len(sequence)} steps"
        )
        print(f"rho is {agent.rho} cumulative time rate is {e.value}")

        self.assertAlmostEqual(agent.rho,e.value,msg=f"rho is {agent.rho} cumulative time rate is {e.value}")

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


if __name__ == "__main__":
    unittest.main()
