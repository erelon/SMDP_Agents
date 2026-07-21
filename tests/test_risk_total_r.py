import importlib
import unittest

from tests._loader import load_tabular_modules


MODULES = load_tabular_modules()
RiskTotalR = importlib.import_module("agents.risk_total_r").RiskTotalR
CumulativePowerMeanRate = importlib.import_module(
    "agents.power_rates"
).CumulativePowerMeanRate
SMART = MODULES["smart_r"].SMART

SEQUENCE = [
    (1.0, 2.0),
    (3.0, 1.0),
    (2.0, 4.0),
    (5.0, 3.0),
    (0.5, 2.0),
    (4.0, 1.0),
    (7.0, 5.0),
    (1.5, 2.5),
    (8.0, 4.0),
    (2.5, 1.5),
    (6.0, 3.0),
]


class RiskTotalRTests(unittest.TestCase):
    def test_rho_matches_owned_estimator_for_multiple_powers(self):
        for p in (-1.0, 0.0, 1.0, 2.0):
            with self.subTest(p=p):
                agent = RiskTotalR(
                    "risk-total", [0], p=p, rate_weight="unit"
                )
                reference = CumulativePowerMeanRate(p)
                for reward, duration in SEQUENCE:
                    agent.calc_new_rho(reward, duration, None, None)
                    self.assertAlmostEqual(
                        agent.rho,
                        reference.update(reward, duration, 1.0),
                    )
                    self.assertEqual(agent.rho, agent.rate.rho)

    def test_constructor_forwards_learning_parameters(self):
        agent = RiskTotalR(
            "risk-total",
            [0],
            p=2,
            learning_rate=0.17,
            exploration_rate=0.23,
            rho_learning_rate=0.41,
            with_rho_trick=False,
        )
        self.assertEqual(agent.p, 2.0)
        self.assertEqual(agent.learning_rate, 0.17)
        self.assertEqual(agent.exploration_rate, 0.23)
        self.assertEqual(agent.rho_learning_rate, 0.41)
        self.assertFalse(agent.with_rho_trick)

    def test_p_one_matches_smart_rho_and_learning_updates(self):
        risk = RiskTotalR(
            "risk-total",
            [0],
            p=1,
            rate_weight="duration",
            learning_rate=0.2,
            exploration_rate=0,
        )
        smart = SMART(
            "smart", [0], learning_rate=0.2, exploration_rate=0
        )
        risk.act("state")
        smart.act("state")

        for reward, duration in SEQUENCE:
            risk.learn("state", 0, reward, "state", duration)
            smart.learn("state", 0, reward, "state", duration)
            self.assertAlmostEqual(risk.rho, smart.rho)
            self.assertEqual(risk.q_table, smart.q_table)
            self.assertEqual(risk.step_count, smart.step_count)

    def test_reset_clears_agent_and_estimator_state(self):
        agent = RiskTotalR("risk-total", [0], p=2)
        agent.calc_new_rho(2.0, 1.0, None, None)
        agent.q_table["state"] = {0: 3.0}
        agent.reset()
        self.assertEqual(agent.rho, 0.0)
        self.assertEqual(agent.rate.rho, 0.0)
        self.assertEqual(agent.observation_count, 0)
        self.assertEqual(agent.total_weight, 0.0)
        self.assertEqual(agent.q_table, {})

    def test_invalid_weight_mode_and_nonpositive_rate_raise(self):
        with self.assertRaises(ValueError):
            RiskTotalR("risk-total", [0], rate_weight="invalid")
        agent = RiskTotalR("risk-total", [0])
        with self.assertRaises(ValueError):
            agent.calc_new_rho(0.0, 1.0, None, None)
        with self.assertRaises(ZeroDivisionError):
            agent.calc_new_rho(1.0, 0.0, None, None)


if __name__ == "__main__":
    unittest.main()
