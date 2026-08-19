import unittest

from agents.r_learning import ContinuousRLearning, RLearning


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
