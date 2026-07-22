import importlib
import unittest

from tests._loader import load_tabular_modules


load_tabular_modules()
RiskSmoothedR = importlib.import_module("agents.risk_smoothed_r").RiskSmoothedR
NormalizedExponentialPowerMeanRate = importlib.import_module(
    "agents.power_rates"
).NormalizedExponentialPowerMeanRate
crra = importlib.import_module("agents.risk_ce").crra

SEQUENCE = (
    (1.0, 2.0),
    (3.0, 1.0),
    (2.0, 4.0),
    (5.0, 3.0),
    (0.5, 2.0),
    (4.0, 1.0),
)


class RiskSmoothedRTests(unittest.TestCase):
    def test_constructor_maps_theta_to_power_and_forwards_parameters(self):
        agent = RiskSmoothedR(
            "risk-smoothed",
            [0],
            theta=1.5,
            weight_parameter=0.25,
            learning_rate=0.17,
            exploration_rate=0.23,
            rho_learning_rate=0.41,
            with_rho_trick=False,
        )

        self.assertEqual(agent.theta, 1.5)
        self.assertEqual(agent.p, -0.5)
        self.assertEqual(agent.weight_parameter, 0.25)
        self.assertEqual(agent.learning_rate, 0.17)
        self.assertEqual(agent.exploration_rate, 0.23)
        self.assertEqual(agent.rho_learning_rate, 0.41)
        self.assertFalse(agent.with_rho_trick)

    def test_documented_weight_parameter_endpoints(self):
        reward, duration = 6.0, 4.0
        expected = {0: 1.0, 1: reward, -1: duration}
        for parameter, weight in expected.items():
            with self.subTest(parameter=parameter):
                agent = RiskSmoothedR(
                    "risk-smoothed", [0], weight_parameter=parameter
                )
                self.assertEqual(agent.weight(reward, duration), weight)

    def test_rho_matches_owned_power_mean_estimator(self):
        for theta in (-1.0, 0.0, 1.0, 2.0):
            for weight_parameter in (-1, 0, 1):
                with self.subTest(
                    theta=theta, weight_parameter=weight_parameter
                ):
                    beta = 0.3
                    agent = RiskSmoothedR(
                        "risk-smoothed",
                        [0],
                        theta=theta,
                        weight_parameter=weight_parameter,
                        rho_learning_rate=beta,
                    )
                    reference = NormalizedExponentialPowerMeanRate(
                        1.0 - theta, beta
                    )
                    for reward, duration in SEQUENCE:
                        weight = agent.weight(reward, duration)
                        agent.calc_new_rho(reward, duration, None, None)
                        self.assertAlmostEqual(
                            agent.rho,
                            reference.update(reward, duration, weight),
                        )
                        self.assertEqual(agent.rho, agent.rate.rho)

    def test_target_is_weighted_crra_utility_difference(self):
        agent = RiskSmoothedR(
            "risk-smoothed", [0], theta=2.0, weight_parameter=0
        )
        agent.rho = 1.5
        reward, duration, next_q = 8.0, 2.0, 0.75

        expected = crra(reward / duration, 2.0) - crra(1.5, 2.0) + next_q
        self.assertAlmostEqual(agent.set_target(reward, duration, next_q), expected)

    def test_risk_neutral_duration_weight_reduces_to_r_learning_target(self):
        agent = RiskSmoothedR(
            "risk-smoothed", [0], theta=0.0, weight_parameter=-1
        )
        agent.rho = 1.25
        reward, duration, next_q = 7.0, 2.0, 0.5

        self.assertAlmostEqual(
            agent.set_target(reward, duration, next_q),
            reward - agent.rho * duration + next_q,
        )

    def test_reset_clears_agent_and_estimator_state(self):
        agent = RiskSmoothedR("risk-smoothed", [0], theta=1.0)
        agent.calc_new_rho(2.0, 1.0, None, None)
        agent.q_table["state"] = {0: 3.0}
        agent.reset()

        self.assertEqual(agent.rho, 0.0)
        self.assertEqual(agent.rate.rho, 0.0)
        self.assertEqual(agent.q_table, {})

    def test_rejects_invalid_theta_and_transition_values(self):
        for theta in (float("nan"), float("inf"), "invalid"):
            with self.subTest(theta=theta):
                with self.assertRaises(ValueError):
                    RiskSmoothedR("risk-smoothed", [0], theta=theta)

        agent = RiskSmoothedR("risk-smoothed", [0])
        with self.assertRaises(ValueError):
            agent.calc_new_rho(-1.0, 1.0, None, None)
        with self.assertRaises(ZeroDivisionError):
            agent.calc_new_rho(1.0, 0.0, None, None)


if __name__ == "__main__":
    unittest.main()
