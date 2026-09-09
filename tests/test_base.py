import unittest

from agents.base import Agent


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

    def test_tabulate_creates_a_row_of_only_the_available_actions(self):
        agent = DummyAgent("tab", [0, 1], env=RestrictedEnvironment())
        table = {}
        self.assertEqual(agent.tabulate(table, "restricted"), {1: 0.0})
        self.assertEqual(agent.tabulate(table, "other"), {0: 0.0, 1: 0.0})

    def test_tabulate_keeps_existing_values_and_extends_the_row(self):
        agent = DummyAgent("tab", [0, 1])
        table = {"s": {0: 5.0}}
        row = agent.tabulate(table, "s", default=9.0)
        self.assertEqual(row, {0: 5.0, 1: 9.0})   # 0 untouched, 1 added
        self.assertIs(row, table["s"])

    def test_tabulate_falls_back_to_the_full_action_space_without_an_env(self):
        agent = DummyAgent("tab", [0, 1, 2])
        self.assertEqual(agent.tabulate({}, "s", default=1.0), {0: 1.0, 1: 1.0, 2: 1.0})

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


if __name__ == "__main__":
    unittest.main()
