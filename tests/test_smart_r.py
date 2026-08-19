import unittest

from agents.smart_r import SMART


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


if __name__ == "__main__":
    unittest.main()
