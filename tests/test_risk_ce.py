import math
import importlib
import unittest

from tests._loader import load_tabular_modules


load_tabular_modules()
_risk_ce = importlib.import_module("agents.risk_ce")
crra = _risk_ce.crra
crra_invert = _risk_ce.crra_invert


class CRRATests(unittest.TestCase):
    def test_closed_form_special_cases(self):
        rate = 4.0
        self.assertEqual(crra(rate, 0), rate - 1)
        self.assertEqual(crra(rate, 1), math.log(rate))
        self.assertEqual(crra(rate, 2), 1 - 1 / rate)

    def test_inverse_round_trip_across_risk_parameters_and_rates(self):
        for theta in (-2.0, 0.0, 0.5, 1.0, 2.0, 3.0):
            for rate in (0.125, 0.5, 1.0, 3.5, 20.0):
                with self.subTest(theta=theta, rate=rate):
                    self.assertAlmostEqual(
                        crra_invert(crra(rate, theta), theta),
                        rate,
                        places=12,
                    )

    def test_certainty_equivalent_matches_power_means(self):
        rates = (1.0, 2.0, 8.0)
        for theta, expected in (
            (0.0, sum(rates) / len(rates)),
            (1.0, math.prod(rates) ** (1 / len(rates))),
            (2.0, len(rates) / sum(1 / rate for rate in rates)),
        ):
            with self.subTest(theta=theta):
                mean_utility = sum(crra(rate, theta) for rate in rates) / len(rates)
                self.assertAlmostEqual(crra_invert(mean_utility, theta), expected)

    def test_rejects_invalid_rates_parameters_and_inverse_domains(self):
        for rate in (0, -1, float("nan"), float("inf"), "invalid"):
            with self.subTest(rate=rate):
                with self.assertRaises(ValueError):
                    crra(rate, 1.0)

        for theta in (float("nan"), float("inf"), "invalid"):
            with self.subTest(theta=theta):
                with self.assertRaises(ValueError):
                    crra(1.0, theta)

        with self.assertRaises(ValueError):
            crra_invert(-1.0, 0.0)
        with self.assertRaises(ValueError):
            crra_invert(1.0, 2.0)
        with self.assertRaises(ValueError):
            crra_invert(-2.0, 0.5)


if __name__ == "__main__":
    unittest.main()
