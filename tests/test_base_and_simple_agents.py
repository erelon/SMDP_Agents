import unittest

from tests._loader import load_tabular_modules


MODULES = load_tabular_modules()
Agent = MODULES["base"].Agent
Oracle = MODULES["oracle"].Oracle
RandomAgent = MODULES["random_agent"].RandomAgent


class DummyAgent(Agent):
    def act(self, state):
        return self.get_available_actions(state)[0]

    eval = act

    def learn(self, state, action, reward, next_state, time):
        return super().learn(state, action, reward, next_state, time)


class RestrictedEnvironment:
    def get_available_actions(self, state):
        return [1] if state == "restricted" else [0, 1]


class AgentCoreTests(unittest.TestCase):
    def test_action_space_is_required(self):
        with self.assertRaisesRegex(ValueError, "Action space"):
            DummyAgent("missing")

    def test_environment_can_restrict_actions(self):
        agent = DummyAgent("env", [0, 1], env=RestrictedEnvironment())
        self.assertEqual(agent.get_available_actions("restricted"), [1])
        self.assertEqual(agent.get_available_actions("other"), [0, 1])

    def test_convergence_detects_policy_change_for_add_and_assign(self):
        agent = DummyAgent("core", [0, 1])
        agent.q_table["s"] = {0: 2.0, 1: 1.0}
        agent._check_convergence("s", 1, 2.0)
        self.assertTrue(agent.get_policy_changed())
        agent._check_convergence("s", 1, 1.5, assign=True)
        self.assertFalse(agent.get_policy_changed())

    def test_convergence_on_unknown_state_is_noop(self):
        agent = DummyAgent("core", [0])
        self.assertFalse(agent._check_convergence("missing", 0, 5.0))
        self.assertFalse(agent.policy_changed)

    def test_base_reset_restores_seeded_rng_and_state(self):
        agent = DummyAgent("core", [0, 1], seed=7)
        first = agent.rng.random()
        agent.q_table["s"] = {0: 1}
        agent.policy_changed = True
        agent.reset()
        self.assertEqual(agent.rng.random(), first)
        self.assertEqual(agent.q_table, {})
        self.assertFalse(agent.policy_changed)
        self.assertEqual(agent.step_count, 0)

    def test_learn_calls_are_counted_and_reset_clears_count(self):
        agent = DummyAgent("core", [0, 1])
        agent.learn("s", 0, 1, "next", 1)
        agent.learn("next", 1, 2, "end", 1)
        self.assertEqual(agent.step_count, 2)
        agent.reset()
        self.assertEqual(agent.step_count, 0)


class SimpleAgentTests(unittest.TestCase):
    def test_oracle_requires_secret_and_delegates(self):
        with self.assertRaisesRegex(ValueError, "environment secret"):
            Oracle("oracle", [0, 1])
        oracle = Oracle("oracle", [0, 1], env_secret=lambda state: state + 1)
        self.assertEqual(oracle.act(4), 5)
        self.assertEqual(oracle.eval(8), 9)
        self.assertIsNone(oracle.learn(None, None, None, None, None))
        self.assertEqual(oracle.step_count, 1)
        oracle.reset()  # Oracle deliberately does not call super().reset().
        self.assertEqual(oracle.step_count, 0)

    def test_random_agent_reset_restarts_seeded_sequence(self):
        agent = RandomAgent("random", [0, 1, 2])
        first = [agent.act(None) for _ in range(4)]
        agent.reset()
        second = [agent.act(None) for _ in range(4)]
        fresh = RandomAgent("fresh", [0, 1, 2])
        expected_first = [fresh.act(None) for _ in range(4)]
        self.assertEqual(first, expected_first)
        self.assertEqual(second, expected_first)


if __name__ == "__main__":
    unittest.main()
