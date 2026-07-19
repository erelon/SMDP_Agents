import importlib.util
import pathlib
import unittest

from tests._loader import load_tabular_modules


AVERAGE_RATES_PATH = pathlib.Path(__file__).resolve().parents[1] / "agents" / "average_rates.py"
AVERAGE_RATES_SPEC = importlib.util.spec_from_file_location("average_rates", AVERAGE_RATES_PATH)
average_rates = importlib.util.module_from_spec(AVERAGE_RATES_SPEC)
AVERAGE_RATES_SPEC.loader.exec_module(average_rates)


MODULES = load_tabular_modules()
SMART = MODULES["smart_r"].SMART
WeightedHarmonic = MODULES["harmonic_r"].WeightedHarmonic
RelaxedSMART = MODULES["relaxed_smart"].RelaxedSMART
NormalizedExponentialMovingTimeRate = average_rates.NormalizedExponentialMovingTimeRate
ExponentialMovingTimeRate = average_rates.ExponentialMovingTimeRate


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
        beta = 0.5
        agents = [
            ("SMART", SMART("smart", [0], rho_learning_rate=beta)),
            (
                "Reward-weighted WeightedHarmonic",
                WeightedHarmonic("weighted_harmonic", [0], rho_learning_rate=beta),
            ),
            ("Relaxed SMART", RelaxedSMART("relaxed_smart", [0], rho_learning_rate=beta)),
            ("EMTimeRate", NormalizedExponentialMovingTimeRate(beta)),
            # ("EMTimeRate", ExponentialMovingTimeRate(beta)),
        ]
        rows = []

        for step, (reward, duration) in enumerate(sequence, start=1):
            row = [step, reward, duration, reward/duration]
            for _, agent in agents:
                if hasattr(agent, "calc_new_rho"):
                    agent.calc_new_rho(reward, duration, None, None)
                else:
                    agent.update(reward, duration, 1.0)
                row.append(agent.rho)
            rows.append(row)

        print()
        print("step | reward | duration |  rate | SMART rho | W.Harmonic rho | R-SMART rho | EMTimeRate")
        print("-----|--------|----------|-------|-----------|----------------|-------------|-----------")
        for step, reward, duration, rate, smart_rho, harmonic_rho, relaxed_rho, em_time_rate in rows:
            print(
                f"{step:>4} | {reward:>6.3f} | {duration:>8.3f} | {rate:>5.2f} | "
                f"{smart_rho:>9.3f} | {harmonic_rho:>14.3f} | "
                f"{relaxed_rho:>11.3f} | {em_time_rate:>9.3f}"
            )

        self.assertEqual(len(rows), len(sequence))

    def test_prints_rate_comparison_table_for_dependent_then_fast_rewards(self):
        sequence = [
            (10.0, 1.0),  # rate = 10
            (1.0, 0.1),    # rate = 10
            (2.0, 0.2),
            (1000.0, 100.0),
            (1.0, 0.1),
            (1.0, 0.1),
            (1.0, 0.1),
            (1.0, 0.1),
            (1.0, 0.1),
            (1.0, 0.1),
            (1.0, 0.1),
        ]
        beta = 0.5
        agents = [
            ("SMART", SMART("smart", [0], rho_learning_rate=beta)),
            (
                "Reward-weighted WeightedHarmonic",
                WeightedHarmonic("weighted_harmonic", [0], rho_learning_rate=beta),
            ),
            ("Relaxed SMART", RelaxedSMART("relaxed_smart", [0], rho_learning_rate=beta)),
            ("EMTimeRate", NormalizedExponentialMovingTimeRate(beta)),
        ]
        rows = []

        for step, (reward, duration) in enumerate(sequence, start=1):
            row = [step, reward, duration, reward/duration]
            for _, agent in agents:
                if hasattr(agent, "calc_new_rho"):
                    agent.calc_new_rho(reward, duration, None, None)
                else:
                    agent.update(reward, duration, 1.0)
                row.append(agent.rho)
            rows.append(row)

        print()
        print("step | reward | duration |  rate | SMART rho | W.Harmonic rho | R-SMART rho | EMTimeRate")
        print("-----|--------|----------|-------|-----------|----------------|-------------|-----------")
        for step, reward, duration, rate, smart_rho, harmonic_rho, relaxed_rho, em_time_rate in rows:
            print(
                f"{step:>4} | {reward:>6.3f} | {duration:>8.3f} | {rate:>5.2f} | "
                f"{smart_rho:>9.6f} | {harmonic_rho:>14.6f} | "
                f"{relaxed_rho:>11.6f} | {em_time_rate:>9.6f}"
            )

        self.assertEqual(len(rows), len(sequence))


if __name__ == "__main__":
    unittest.main()
