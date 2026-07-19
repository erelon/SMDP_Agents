import unittest

from tests._loader import load_tabular_modules

RandomAgent = load_tabular_modules()["random_agent"].RandomAgent


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


if __name__ == "__main__":
    unittest.main()
