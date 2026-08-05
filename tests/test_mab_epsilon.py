import unittest

from agents.mab_epsilon import MAB, ContinuesMAB


class RestrictedEnvironment:
    def get_available_actions(self, state):
        return [1] if state == "restricted" else [0, 1]


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

    def test_bandits_tabulate_only_the_available_actions(self):
        # Was: every per-state row was built from the full action_space, so the
        # bandits could pick an action the environment refuses. Now rows hold only
        # the actions available in that state.
        for cls in (MAB, ContinuesMAB):
            with self.subTest(agent=cls.__name__):
                agent = cls("bandit", [0, 1], env=RestrictedEnvironment())
                self.assertEqual(agent.act("restricted"), 1)
                self.assertEqual(sorted(agent.q_table["restricted"]), [1])
                agent.act("other")
                self.assertEqual(sorted(agent.q_table["other"]), [0, 1])

    def test_bandit_exploration_stays_inside_the_available_actions(self):
        for cls in (MAB, ContinuesMAB):
            with self.subTest(agent=cls.__name__):
                agent = cls("bandit", [0, 1], exploration_rate=1.0,
                            env=RestrictedEnvironment())
                self.assertEqual({agent.act("restricted") for _ in range(30)}, {1})

    def test_bandit_learning_tables_stay_restricted(self):
        for cls in (MAB, ContinuesMAB):
            with self.subTest(agent=cls.__name__):
                agent = cls("bandit", [0, 1], env=RestrictedEnvironment())
                agent.learn("restricted", 1, 4.0, "other", 2.0)
                self.assertEqual(sorted(agent.q_table["restricted"]), [1])
                self.assertEqual(agent.eval("restricted"), 1)


if __name__ == "__main__":
    unittest.main()
