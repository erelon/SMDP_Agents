import unittest

from tests._loader import load_tabular_modules


MODULES = load_tabular_modules()
SMART = MODULES["smart_r"].SMART
RelaxedSMART = MODULES["relaxed_smart"].RelaxedSMART
Harmonic = MODULES["harmonic_r"].Harmonic
WeightedHarmonic = MODULES["harmonic_r"].WeightedHarmonic


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


class SmartTests(unittest.TestCase):
    def test_smart_tracks_cumulative_reward_per_time(self):
        agent = SMART("smart", [0], rho_learning_rate=0.2)
        agent.act("state")

        agent.learn("state", 0, 4.0, "state", 2.0)
        self.assertEqual(agent.rho, 2.0)
        self.assertEqual(agent.step_count, 1)

        agent.learn("state", 0, -1.0, "state", 1.0)
        self.assertEqual(agent.rho, 1.0)
        self.assertEqual((agent.total_reward, agent.total_time), (3.0, 3.0))
        self.assertEqual(agent.step_count, 2)

    def test_smart_zero_total_time_raises(self):
        agent = SMART("smart", [0])
        with self.assertRaises(ZeroDivisionError):
            agent.calc_new_rho(1.0, 0.0, None, None)

    def test_smart_reset_clears_accumulators(self):
        agent = SMART("smart", [0])
        agent.calc_new_rho(2.0, 1.0, None, None)
        agent.reset()
        self.assertEqual((agent.rho, agent.total_reward, agent.total_time, agent.step_count),
                         (0, 0, 0, 0))


class RelaxedSmartTests(unittest.TestCase):
    def test_equal_ema_rates_reduce_to_ratio_of_weighted_sums(self):
        agent = RelaxedSMART("relaxed", [0], rho_learning_rate=0.25)
        agent.calc_new_rho(4.0, 2.0, None, None)
        self.assertAlmostEqual(agent.rho, 2.0)
        agent.calc_new_rho(1.0, 3.0, None, None)
        expected_reward = 0.75 * 1.0 + 0.25 * 1.0
        expected_time = 0.75 * 0.5 + 0.25 * 3.0
        self.assertAlmostEqual(agent.rho_reward, expected_reward)
        self.assertAlmostEqual(agent.rho_time, expected_time)
        self.assertAlmostEqual(agent.rho, expected_reward / expected_time)

    def test_zero_duration_raises_on_first_update(self):
        agent = RelaxedSMART("relaxed", [0], rho_learning_rate=0.5)
        with self.assertRaises(ZeroDivisionError):
            agent.calc_new_rho(0.0, 0.0, None, None)


class HarmonicTests(unittest.TestCase):
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
