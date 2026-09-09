import math
import unittest

from agents.mab_ucb import UCB, ContinuosUCB


class RestrictedEnvironment:
    def get_available_actions(self, state):
        return [1] if state == "restricted" else [0, 1]


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

    def test_ucb_scores_only_the_available_actions(self):
        # Was: the UCB score was computed over the full action_space, so on a
        # restricted state the argmax could be an action the environment refuses.
        for cls, counter in ((UCB, "total_steps"), (ContinuosUCB, "total_count")):
            with self.subTest(agent=cls.__name__):
                agent = cls("ucb", [0, 1], env=RestrictedEnvironment())
                self.assertEqual(agent.act("restricted"), 1)
                self.assertEqual(sorted(agent.q_table["restricted"]), [1])
                self.assertEqual(sorted(getattr(agent, counter)["restricted"]), [1])

    def test_ucb_pseudocounts_only_cover_available_actions(self):
        agent = UCB("ucb", [0, 1], env=RestrictedEnvironment())
        agent.act("restricted")
        self.assertEqual(agent.total_steps["restricted"], {1: 1})
        agent.act("other")
        self.assertEqual(agent.total_steps["other"], {0: 1, 1: 1})


if __name__ == "__main__":
    unittest.main()
