import math
import unittest

from tests._loader import load_tabular_modules


MODULES = load_tabular_modules()
MAB_EPSILON = MODULES["mab-epsilon"]
MAB_UCB = MODULES["mab-ucb"]
MAB = MAB_EPSILON.MAB
ContinuesMAB = MAB_EPSILON.ContinuesMAB
UCB = MAB_UCB.UCB
ContinuosUCB = MAB_UCB.ContinuosUCB


class EpsilonGreedyBanditTests(unittest.TestCase):
    def test_mab_uses_sample_mean_per_state_action(self):
        agent = MAB("mab", [0, 1], exploration_rate=0)
        agent.learn("s", 1, 2.0, None, 99)
        agent.learn("s", 1, 4.0, None, 0)
        agent.learn("other", 1, 10.0, None, 1)
        self.assertEqual(agent.q_table["s"][1], 3.0)
        self.assertEqual(agent.q_table["other"][1], 10.0)
        self.assertEqual(agent.eval("s"), 1)

    def test_continuous_mab_uses_reward_per_total_time(self):
        agent = ContinuesMAB("cmab", [0, 1], exploration_rate=0)
        agent.learn("s", 1, 3.0, None, 2.0)
        agent.learn("s", 1, -1.0, None, 2.0)
        self.assertEqual(agent.q_table["s"][1], 0.5)

    def test_continuous_mab_zero_total_time_leaves_previous_value(self):
        agent = ContinuesMAB("cmab", [0], exploration_rate=0)
        agent.learn("s", 0, 1.0, None, 1.0)
        agent.learn("s", 0, -1.0, None, -1.0)
        self.assertEqual(agent.q_table["s"][0], 1.0)

    def test_bandit_reset_clears_tables_and_reseeds(self):
        agent = MAB("mab", [0, 1], exploration_rate=1, seed=6)
        agent.act("s")
        next_random = agent.rng.random()
        agent.learn("s", 0, 1, None, 1)
        agent.reset()
        self.assertEqual(agent.q_table, {})
        self.assertEqual(agent.total_steps, {})
        self.assertNotEqual(agent.rng.random(), next_random)


class UcbTests(unittest.TestCase):
    def test_ucb_initializes_pseudocounts_and_prefers_first_on_tie(self):
        agent = UCB("ucb", [0, 1], exploration_constant=1.0)
        self.assertEqual(agent.act("s"), 0)
        self.assertEqual(agent.total_steps["s"], {0: 1, 1: 1})
        self.assertTrue(all(math.isclose(v, 1e-12) for v in agent.q_table["s"].values()))

    def test_ucb_learns_sample_mean(self):
        agent = UCB("ucb", [0, 1])
        agent.learn("s", 1, 2.0, None, 1)
        agent.learn("s", 1, 6.0, None, 1)
        self.assertEqual(agent.q_table["s"][1], 4.0)
        self.assertEqual(agent.eval("s"), 1)

    def test_continuous_ucb_uses_reward_rate_but_count_stays_at_pseudocount(self):
        agent = ContinuosUCB("cucb", [0, 1])
        agent.act("s")
        agent.learn("s", 1, 6.0, None, 3.0)
        # act() seeds total_time with 1e-12, so the baseline is just below 2.
        self.assertAlmostEqual(agent.q_table["s"][1], 2.0)
        self.assertEqual(agent.total_time["s"][1], 3.000000000001)
        self.assertEqual(agent.total_count["s"][1], 1)

    def test_continuous_ucb_zero_time_raises(self):
        agent = ContinuosUCB("cucb", [0])
        with self.assertRaises(ZeroDivisionError):
            agent.learn("s", 0, 1.0, None, 0.0)


if __name__ == "__main__":
    unittest.main()
