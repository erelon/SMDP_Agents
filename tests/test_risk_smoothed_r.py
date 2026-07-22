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
    (7.0, 5.0),
    (1.5, 2.5),
    (8.0, 4.0),
    (2.5, 1.5),
    (6.0, 3.0),
)

S1_ACTIONS = ("seek", "neutral", "averse")
S1_REWARDS = {
    "seek": (2.0, 30.0),
    "neutral": (12.0, 22.0),
    "averse": (16.0, 16.0),
}
SIMPLE_SMDP_SEQUENCE = (
    (2.0, 2.0), (8.0, 1.0),
    (30.0, 2.0), (8.0, 1.0),
    (12.0, 2.0), (8.0, 1.0),
    (22.0, 2.0), (8.0, 1.0),
    (16.0, 2.0), (8.0, 1.0),
)


class SimpleRiskSMDP:
    """Action availability for the two-state SMDP in the Phase 5 plan."""

    def get_available_actions(self, state):
        return ("back",) if state == "s2" else S1_ACTIONS


class RiskSmoothedRTests(unittest.TestCase):
    def assert_selects_action(self, theta, weight_p, expected_action):
        agent = RiskSmoothedR(
            "risk-smoothed",
            [*S1_ACTIONS, "back"],
            env=SimpleRiskSMDP(),
            theta=theta,
            weight_parameter=weight_p,
            rho_learning_rate=0.01,
            learning_rate=0.05,
            exploration_rate=0.0,
            with_rho_trick=False,
        )
        agent.act("s1")
        agent.act("s2")
        agent.calc_new_rho(8.0, 1.0, None, None)

        # Visit every action and both equiprobable outcomes uniformly.
        for step in range(600):
            action = S1_ACTIONS[step % len(S1_ACTIONS)]
            outcome = (step // len(S1_ACTIONS)) % 2
            agent.learn(
                "s1", action, S1_REWARDS[action][outcome], "s2", 2.0
            )
            agent.learn("s2", "back", 8.0, "s1", 1.0)

        self.assertEqual(agent.eval("s1"), expected_action)

    def assert_matches_relaxed_smart_values(
        self, theta, weight_parameter, sequence, beta=0.2
    ):
        agent = RiskSmoothedR(
            "risk-smoothed",
            [0],
            theta=theta,
            weight_parameter=weight_parameter,
            rho_learning_rate=beta,
        )
        reward_ema = 0.0
        duration_ema = 0.0
        normalizer = 0.0

        for reward, duration in sequence:
            reward_ema = (1.0 - beta) * reward_ema + beta * reward
            duration_ema = (1.0 - beta) * duration_ema + beta * duration
            normalizer = (1.0 - beta) * normalizer + beta
            expected = (reward_ema / normalizer) / (duration_ema / normalizer)
            agent.calc_new_rho(reward, duration, None, None)
            self.assertAlmostEqual(agent.rho, expected)

    def test_neutral_agent_selects_neutral_action_on_simple_smdp(self):
        self.assert_selects_action(theta=0.0, weight_p=-1, expected_action="neutral")

    def test_averse_agent_selects_averse_action_on_simple_smdp(self):
        self.assert_selects_action(theta=2.0, weight_p=-1, expected_action="averse")

    def test_seeking_agent_selects_seeking_action_on_simple_smdp(self):
        self.assert_selects_action(theta=-1.0, weight_p=-1, expected_action="seek")

    def test_neutral_time_weight_matches_relaxed_smart_on_simple_smdp(self):
        self.assert_matches_relaxed_smart_values(
            theta=0.0,
            weight_parameter=-1.0,
            sequence=SIMPLE_SMDP_SEQUENCE,
        )

    def test_neutral_time_weight_matches_relaxed_smart_on_long_sequence(self):
        self.assert_matches_relaxed_smart_values(
            theta=0.0, weight_parameter=-1.0, sequence=SEQUENCE
        )

    def test_harmonic_reward_weight_matches_relaxed_smart_on_simple_smdp(self):
        self.assert_matches_relaxed_smart_values(
            theta=2.0,
            weight_parameter=1.0,
            sequence=SIMPLE_SMDP_SEQUENCE,
        )

    def test_harmonic_reward_weight_matches_relaxed_smart_on_long_sequence(self):
        self.assert_matches_relaxed_smart_values(
            theta=2.0, weight_parameter=1.0, sequence=SEQUENCE
        )

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
