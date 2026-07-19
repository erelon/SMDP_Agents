import unittest

from tests._loader import load_tabular_modules

RelaxedSMART = load_tabular_modules()["relaxed_smart"].RelaxedSMART


class RelaxedSmartTests(unittest.TestCase):
    def test_equal_ema_rates_reduce_to_ratio_of_weighted_sums(self):
        agent = RelaxedSMART("relaxed", [0], rho_learning_rate=0.00000001)
        agent.calc_new_rho(4.0, 2.0, None, None)
        self.assertAlmostEqual(agent.rho, 2.0)
        agent.calc_new_rho(1.0, 3.0, None, None)
        expected_reward = 2.5
        expected_time = 2.5
        print(f"agent reward {agent.rho_reward} expected {expected_reward}\n"
              f"agent time {agent.rho_reward} expected {expected_reward}\n"
              f"agent rho {agent.rho} expected {expected_reward/expected_time}\n"
        )
        self.assertAlmostEqual(agent.rho_reward, expected_reward,msg=f"agent reward {agent.rho_reward} expected {expected_reward}")
        self.assertAlmostEqual(agent.rho_time, expected_time,msg=f"agent time {agent.rho_reward} expected {expected_reward}")
        self.assertAlmostEqual(agent.rho, expected_reward / expected_time)

    def test_zero_duration_raises_on_first_update(self):
        agent = RelaxedSMART("relaxed", [0], rho_learning_rate=0.5)
        with self.assertRaises(ZeroDivisionError):
            agent.calc_new_rho(0.0, 0.0, None, None)


if __name__ == "__main__":
    unittest.main()
