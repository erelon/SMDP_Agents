import unittest

from tests._loader import load_tabular_modules


MODULES = load_tabular_modules()
SMART = MODULES["smart_r"].SMART
WeightedHarmonic = MODULES["harmonic_r"].WeightedHarmonic
RelaxedSMART = MODULES["relaxed_smart"].RelaxedSMART


class SMDPCriteriaTests(unittest.TestCase):
    def test_prints_rate_comparison_table_for_short_then_fast_rewards(self):
        sequence = [
            (10.0, 10.0),  # rate = 1
            (1.0, 0.1),    # rate = 10
            (1.0, 0.1),
            (1.0, 0.1),
            (1.0, 0.1),
            (1.0, 0.1),
            (1.0, 0.1),
            (1.0, 0.1),
            (1.0, 0.1),
            (1.0, 0.1),
            (1.0, 0.1),
        ]
        agents = [
            ("SMART", SMART("smart", [0], rho_learning_rate=0.3)),
            (
                "Reward-weighted WeightedHarmonic",
                WeightedHarmonic("weighted_harmonic", [0], rho_learning_rate=0.3),
            ),
            ("Relaxed SMART", RelaxedSMART("relaxed_smart", [0], rho_learning_rate=0.3)),
        ]
        rows = []

        for step, (reward, duration) in enumerate(sequence, start=1):
            row = [step, reward, duration]
            for _, agent in agents:
                agent.calc_new_rho(reward, duration, None, None)
                row.append(agent.rho)
            rows.append(row)

        print()
        print("step | reward | duration | SMART rho | WeightedHarmonic rho | Relaxed SMART rho")
        print("-----|--------|----------|-----------|----------------------|------------------")
        for step, reward, duration, smart_rho, harmonic_rho, relaxed_rho in rows:
            print(
                f"{step:>4} | {reward:>6.3f} | {duration:>8.3f} | "
                f"{smart_rho:>9.6f} | {harmonic_rho:>20.6f} | {relaxed_rho:>16.6f}"
            )

        self.assertEqual(len(rows), len(sequence))


if __name__ == "__main__":
    unittest.main()
