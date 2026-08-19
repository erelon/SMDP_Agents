"""Tests for the ``|rho|``-scaled TD target in ``agents/experemental_harmonic_r.py``.

Two claims to hold it to: it really does scale the advantage, and unlike dividing
by a *signed* rho it never reorders two actions -- which is the whole reason for
the absolute value.
"""

import unittest

from agents.experemental_harmonic_r import (AbsRhoScaledTarget,
                                            ExperimentalCumulativeWeightedHarmonic,
                                            ExperimentalWeightedHarmonic)
from agents.harmonic_r import (CumulativeHarmonic, CumulativeWeightedHarmonic,
                               Harmonic, WeightedHarmonic)
from agents.r_learning import ContinuousRLearning


class ExperimentalHarmonicTargetTests(unittest.TestCase):
    """The scaled target, and the properties it does and does not preserve."""

    AGENTS = (ExperimentalWeightedHarmonic, ExperimentalCumulativeWeightedHarmonic)
    #: The plain-target agent each one is otherwise identical to.
    BASES = {ExperimentalWeightedHarmonic: WeightedHarmonic,
             ExperimentalCumulativeWeightedHarmonic: CumulativeWeightedHarmonic}

    def test_it_divides_the_advantage_by_the_magnitude_of_rho(self):
        for agent_class in self.AGENTS:
            with self.subTest(agent=agent_class.__name__):
                agent = agent_class("experimental", [0])
                agent.rho = 2.0
                # (5 - 2*2) / 2 + 1
                self.assertAlmostEqual(agent.set_target(5.0, 2.0, 1.0), 1.5)
                agent.rho = -2.0
                # (5 + 2*2) / 2 + 1 -- the numerator keeps rho's sign, the divisor
                # does not.
                self.assertAlmostEqual(agent.set_target(5.0, 2.0, 1.0), 5.5)

    def test_it_never_reorders_two_actions_whatever_the_sign_of_rho(self):
        # The property dividing by a signed rho loses and this exists to restore:
        # |rho| is a positive constant within one decision, so the ranking is
        # exactly R-learning's at every rho.
        pairs = ((10.0, 10.0), (1.0, 0.1), (-2.0, 3.0), (0.0, 1.0), (-8.0, 0.5))
        for agent_class in self.AGENTS:
            for rho in (2.5, 0.01, -0.01, -1.5, -40.0):
                with self.subTest(agent=agent_class.__name__, rho=rho):
                    agent = agent_class("experimental", [0])
                    plain = ContinuousRLearning("plain", [0])
                    agent.rho = plain.rho = rho
                    scaled = [agent.set_target(r, t, 0.0) for r, t in pairs]
                    unscaled = [plain.set_target(r, t, 0.0) for r, t in pairs]
                    order = lambda v: sorted(range(len(pairs)), key=v.__getitem__)  # noqa: E731
                    self.assertEqual(order(scaled), order(unscaled))

    def test_a_signed_divisor_would_have_reordered_them(self):
        # Guards the test above against being vacuous.
        pairs = ((-1.0, 1.0), (-10.0, 1.0))
        rho = -1.5
        plain = [(r - rho * t) for r, t in pairs]
        signed = [(r - rho * t) / rho for r, t in pairs]
        absolute = [(r - rho * t) / abs(rho) for r, t in pairs]
        self.assertGreater(plain[0], plain[1])
        self.assertLess(signed[0], signed[1])       # inverted
        self.assertGreater(absolute[0], absolute[1])  # not inverted

    def test_it_scales_the_td_error_inversely_with_rho(self):
        # What the change is actually for: the same transition is a bigger update
        # when the rate estimate is small and a smaller one when it is inflated.
        agent = ExperimentalWeightedHarmonic("experimental", [0])
        magnitudes = []
        for rho in (0.5, 1.0, 4.0):
            agent.rho = rho
            magnitudes.append(abs(agent.set_target(10.0, 1.0, 0.0)))
        self.assertEqual(magnitudes, sorted(magnitudes, reverse=True))

    def test_a_zero_rho_falls_back_to_the_plain_target(self):
        # learn() builds the target before calc_new_rho runs, so rho is 0 on the
        # first update of every run and the division would otherwise raise.
        for agent_class in self.AGENTS:
            with self.subTest(agent=agent_class.__name__):
                agent = agent_class("experimental", [0])
                self.assertEqual(agent.rho, 0.0)
                self.assertAlmostEqual(agent.set_target(5.0, 2.0, 1.0), 6.0)
                agent.act("s")
                agent.learn("s", 0, 4.0, "s", 2.0)  # must not raise
                self.assertEqual(agent.step_count, 1)

    def test_rho_is_untouched_and_still_matches_its_plain_counterpart(self):
        # Only the target differs; fed the same transitions the rate is identical.
        sequence = [(2.0, 1.0), (-1.0, 2.0), (0.0, 2.0), (4.0, 3.0), (-3.0, 1.0)]
        for agent_class, base_class in self.BASES.items():
            with self.subTest(agent=agent_class.__name__):
                agent, base = agent_class("e", [0]), base_class("b", [0])
                for reward, duration in sequence:
                    agent.calc_new_rho(reward, duration, None, None)
                    base.calc_new_rho(reward, duration, None, None)
                    self.assertAlmostEqual(agent.rho, base.rho)

    def test_the_shipped_agents_are_unaffected(self):
        # The mixin is applied to two classes only; nothing in harmonic_r.py sees
        # it, including the unweighted agents that subclass the weighted ones.
        for agent_class in (WeightedHarmonic, CumulativeWeightedHarmonic,
                            Harmonic, CumulativeHarmonic):
            with self.subTest(agent=agent_class.__name__):
                self.assertNotIsInstance(agent_class("a", [0]), AbsRhoScaledTarget)
                agent = agent_class("a", [0])
                agent.rho = 2.0
                self.assertAlmostEqual(agent.set_target(5.0, 2.0, 1.0), 2.0)

    def test_reset_and_seeding_behave_like_the_base_agents(self):
        for agent_class in self.AGENTS:
            with self.subTest(agent=agent_class.__name__):
                agent = agent_class("experimental", [0])
                agent.calc_new_rho(4.0, 2.0, None, None)
                self.assertNotEqual(agent.rho, 0.0)
                agent.reset()
                self.assertEqual((agent.rho, agent.step_count), (0.0, 0))


if __name__ == "__main__":
    unittest.main()
