import importlib
import unittest

from tests._loader import load_tabular_modules


MODULES = load_tabular_modules()
RiskSmoothedR = importlib.import_module("agents.risk_smoothed_r").RiskSmoothedR
NormalizedExponentialPowerMeanRate = importlib.import_module(
    "agents.power_rates"
).NormalizedExponentialPowerMeanRate
Harmonic = MODULES["harmonic_r"].Harmonic
RelaxedSMART = MODULES["relaxed_smart"].RelaxedSMART

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


class RiskSmoothedRTests(unittest.TestCase):
    def test_rho_matches_owned_estimator_for_powers_and_betas(self):
        for beta in (0.0001, 0.8):
            for p in (-1.0, 0.0, 1.0, 2.0):
                with self.subTest(beta=beta, p=p):
                    agent = RiskSmoothedR(
                        "risk-smoothed",
                        [0],
                        p=p,
                        rho_learning_rate=beta,
                        rate_weight="unit",
                    )
                    reference = NormalizedExponentialPowerMeanRate(p, beta)
                    for reward, duration in SEQUENCE:
                        agent.calc_new_rho(reward, duration, None, None)
                        self.assertAlmostEqual(
                            agent.rho,
                            reference.update(reward, duration, 1.0),
                        )
                        self.assertEqual(agent.rho, agent.rate.rho)

    def test_constructor_forwards_learning_parameters(self):
        agent = RiskSmoothedR(
            "risk-smoothed",
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

    def test_p_minus_one_matches_harmonic_rho_and_learning_updates(self):
        for beta in (0.0001, 0.8):
            with self.subTest(beta=beta):
                risk = RiskSmoothedR(
                    "risk-smoothed",
                    [0],
                    p=-1,
                    rate_weight="unit",
                    rho_learning_rate=beta,
                    learning_rate=0.2,
                    exploration_rate=0,
                )
                harmonic = Harmonic(
                    "harmonic",
                    [0],
                    rho_learning_rate=beta,
                    learning_rate=0.2,
                    exploration_rate=0,
                )
                risk.act("state")
                harmonic.act("state")
                for reward, duration in SEQUENCE:
                    risk.learn("state", 0, reward, "state", duration)
                    harmonic.learn("state", 0, reward, "state", duration)
                    self.assertAlmostEqual(risk.rho, harmonic.rho)
                    self.assertEqual(risk.q_table.keys(), harmonic.q_table.keys())
                    self.assertAlmostEqual(
                        risk.q_table["state"][0], harmonic.q_table["state"][0]
                    )
                    self.assertEqual(risk.step_count, harmonic.step_count)

    def test_time_weighted_p_one_matches_relaxed_smart_learning_updates(self):
        for beta in (0.0001, 0.8):
            with self.subTest(beta=beta):
                risk = RiskSmoothedR(
                    "risk-smoothed",
                    [0],
                    p=1,
                    rate_weight="duration",
                    rho_learning_rate=beta,
                    learning_rate=0.2,
                    exploration_rate=0,
                )
                relaxed = RelaxedSMART(
                    "relaxed",
                    [0],
                    rho_learning_rate=beta,
                    learning_rate=0.2,
                    exploration_rate=0,
                )
                risk.act("state")
                relaxed.act("state")
                for reward, duration in SEQUENCE:
                    risk.learn("state", 0, reward, "state", duration)
                    relaxed.learn("state", 0, reward, "state", duration)
                    self.assertAlmostEqual(risk.rho, relaxed.rho)
                    self.assertEqual(risk.q_table.keys(), relaxed.q_table.keys())
                    self.assertAlmostEqual(
                        risk.q_table["state"][0], relaxed.q_table["state"][0]
                    )
                    self.assertEqual(risk.step_count, relaxed.step_count)

    def test_reset_clears_agent_and_estimator_state(self):
        agent = RiskSmoothedR("risk-smoothed", [0], p=0)
        agent.calc_new_rho(2.0, 1.0, None, None)
        agent.q_table["state"] = {0: 3.0}
        agent.reset()
        self.assertEqual(agent.rho, 0.0)
        self.assertEqual(agent.rate.rho, 0.0)
        self.assertEqual(agent.q_table, {})

    def test_invalid_weight_mode_and_nonpositive_rate_raise(self):
        with self.assertRaises(ValueError):
            RiskSmoothedR("risk-smoothed", [0], rate_weight="invalid")
        agent = RiskSmoothedR("risk-smoothed", [0])
        with self.assertRaises(ValueError):
            agent.calc_new_rho(-1.0, 1.0, None, None)
        with self.assertRaises(ZeroDivisionError):
            agent.calc_new_rho(1.0, 0.0, None, None)


if __name__ == "__main__":
    unittest.main()
