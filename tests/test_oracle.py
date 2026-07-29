import unittest

from agents.oracle import Oracle


class OracleTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
