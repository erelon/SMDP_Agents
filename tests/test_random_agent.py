import unittest

from agents.random_agent import RandomAgent


class RestrictedEnvironment:
    def get_available_actions(self, state):
        return [1] if state == "restricted" else [0, 1, 2]


class RandomAgentTests(unittest.TestCase):
    def test_random_agent_reset_restarts_seeded_sequence(self):
        agent = RandomAgent("random", [0, 1, 2])
        first = [agent.act(None) for _ in range(4)]
        agent.reset()
        second = [agent.act(None) for _ in range(4)]
        fresh = RandomAgent("fresh", [0, 1, 2])
        expected_first = [fresh.act(None) for _ in range(4)]
        self.assertEqual(first, expected_first)
        self.assertEqual(second, expected_first)

    def test_random_agent_only_draws_available_actions(self):
        # Was: act/eval drew from the full action_space and so could return an
        # action the environment refuses. Now both route through
        # get_available_actions, making this a uniform-over-legal baseline.
        agent = RandomAgent("random", [0, 1, 2], env=RestrictedEnvironment())
        self.assertEqual({agent.act("restricted") for _ in range(30)}, {1})
        self.assertEqual({agent.eval("restricted") for _ in range(30)}, {1})
        self.assertEqual({agent.act("other") for _ in range(60)}, {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
