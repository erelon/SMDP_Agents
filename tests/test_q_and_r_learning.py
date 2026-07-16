import unittest

from tests._loader import load_tabular_modules


MODULES = load_tabular_modules()
QLearning = MODULES["q_learning"].QLearning
ContinuousQLearning = MODULES["q_learning"].ContinuousQLearning
RLearning = MODULES["r_learning"].RLearning
ContinuousRLearning = MODULES["r_learning"].ContinuousRLearning


class QLearningTests(unittest.TestCase):
    def test_greedy_action_and_environment_specific_initialization(self):
        class Env:
            def get_available_actions(self, state):
                return [1] if state == "one" else [0, 1]

        agent = ContinuousQLearning("q", [0, 1], env=Env(), exploration_rate=0)
        self.assertEqual(agent.act("one"), 1)
        self.assertEqual(agent.q_table["one"], {1: 0})

    def test_exploration_uses_seeded_rng(self):
        a = QLearning("a", [0, 1, 2], exploration_rate=1, seed=11)
        b = QLearning("b", [0, 1, 2], exploration_rate=1, seed=11)
        self.assertEqual([a.act("s") for _ in range(8)], [b.act("s") for _ in range(8)])

    def test_discrete_q_learning_ignores_supplied_duration(self):
        agent = QLearning("q", [0, 1], learning_rate=0.5, discount_factor=0.5, exploration_rate=0)
        agent.act("s")
        agent.q_table["next"] = {0: 4.0, 1: 2.0}
        agent.learn("s", 0, reward=2.0, next_state="next", time=99.0)
        self.assertAlmostEqual(agent.q_table["s"][0], 2.0)

    def test_continuous_q_discount_uses_fractional_duration(self):
        agent = ContinuousQLearning(
            "cq", [0], learning_rate=0.25, discount_factor=0.25, exploration_rate=0
        )
        agent.act("s")
        agent.q_table["next"] = {0: 8.0}
        agent.learn("s", 0, reward=2.0, next_state="next", time=0.5)
        self.assertAlmostEqual(agent.q_table["s"][0], 1.5)

    def test_zero_duration_means_no_discount(self):
        agent = ContinuousQLearning(
            "cq", [0], learning_rate=1.0, discount_factor=0.2, exploration_rate=0
        )
        agent.act("s")
        agent.q_table["next"] = {0: 3.0}
        agent.learn("s", 0, reward=-1.0, next_state="next", time=0.0)
        self.assertAlmostEqual(agent.q_table["s"][0], 2.0)

    def test_update_records_policy_change(self):
        agent = QLearning("q", [0, 1], learning_rate=1.0, discount_factor=0, exploration_rate=0)
        agent.act("s")
        agent.q_table["s"] = {0: 1.0, 1: 0.0}
        agent.learn("s", 1, reward=3.0, next_state="s", time=1)
        self.assertTrue(agent.policy_changed)
        self.assertEqual(agent.eval("s"), 1)

    def test_reset_clears_q_and_rho_but_currently_does_not_reseed_rng(self):
        agent = ContinuousQLearning("q", [0, 1], seed=3)
        first = agent.rng.random()
        agent.q_table["s"] = {0: 1}
        agent.rho = 4
        agent.reset()
        second = agent.rng.random()
        fresh = ContinuousQLearning("fresh", [0, 1], seed=3)
        fresh.rng.random()
        self.assertEqual(second, fresh.rng.random())
        self.assertNotEqual(first, second)
        self.assertEqual(agent.q_table, {})
        self.assertEqual(agent.rho, 0)


class RLearningTests(unittest.TestCase):
    def test_continuous_target_scales_rho_by_duration(self):
        agent = ContinuousRLearning(
            "r", [0], learning_rate=0.5, rho_learning_rate=0.25, exploration_rate=0
        )
        agent.act("s")
        agent.q_table["next"] = {0: 5.0}
        agent.rho = 2.0
        agent.learn("s", 0, reward=3.0, next_state="next", time=2.0)
        self.assertAlmostEqual(agent.q_table["s"][0], 2.0)
        self.assertAlmostEqual(agent.rho, 3.0)

    def test_discrete_r_learning_ignores_supplied_duration(self):
        agent = RLearning(
            "r", [0], learning_rate=1.0, rho_learning_rate=0.5, exploration_rate=0
        )
        agent.act("s")
        agent.q_table["next"] = {0: 1.0}
        agent.rho = 1.0
        agent.learn("s", 0, reward=3.0, next_state="next", time=12.0)
        self.assertAlmostEqual(agent.q_table["s"][0], 3.0)
        self.assertAlmostEqual(agent.rho, 2.5)

    def test_rho_trick_skips_non_greedy_action(self):
        agent = ContinuousRLearning(
            "r", [0, 1], learning_rate=1.0, rho_learning_rate=0.5,
            exploration_rate=0, with_rho_trick=True
        )
        agent.q_table["s"] = {0: 2.0, 1: 0.0}
        agent.learn("s", 1, reward=4.0, next_state="s", time=1.0)
        self.assertEqual(agent.rho, 0)
        self.assertAlmostEqual(agent.q_table["s"][1], 6.0)

    def test_rho_updates_for_non_greedy_action_when_trick_disabled(self):
        agent = ContinuousRLearning(
            "r", [0, 1], learning_rate=1.0, rho_learning_rate=0.5,
            exploration_rate=0, with_rho_trick=False
        )
        agent.q_table["s"] = {0: 2.0, 1: 0.0}
        agent.learn("s", 1, reward=4.0, next_state="s", time=1.0)
        self.assertAlmostEqual(agent.rho, 3.0)


if __name__ == "__main__":
    unittest.main()
