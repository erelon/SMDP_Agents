import math
import unittest

from agents.average_rates import NormalizedEMA
from agents.smart_r import SMART, SmoothedSMART


class SmartTests(unittest.TestCase):
    def test_smart_tracks_cumulative_reward_per_time(self):
        agent = SMART("smart", [0], rho_learning_rate=0.2)
        agent.act("state")

        agent.learn("state", 0, 4.0, "state", 2.0)
        self.assertEqual(agent.rho, 2.0)
        self.assertEqual(agent.step_count, 1)

        agent.learn("state", 0, -1.0, "state", 1.0)
        self.assertEqual(agent.rho, 1.0)
        self.assertEqual((agent.total_reward, agent.total_time), (3.0, 3.0))
        self.assertEqual(agent.step_count, 2)

    def test_smart_zero_total_time_raises(self):
        agent = SMART("smart", [0])
        with self.assertRaises(ZeroDivisionError):
            agent.calc_new_rho(1.0, 0.0, None, None)

    def test_smart_reset_clears_accumulators(self):
        agent = SMART("smart", [0])
        agent.calc_new_rho(2.0, 1.0, None, None)
        agent.reset()
        self.assertEqual((agent.rho, agent.total_reward, agent.total_time, agent.step_count),
                         (0, 0, 0, 0))


class SmoothedSmartTests(unittest.TestCase):
    """SMART's rate, smoothed in elapsed time instead of accumulated forever."""

    def test_the_first_transition_reports_its_own_rate(self):
        agent = SmoothedSMART("smoothed", [0], rho_learning_rate=0.3)
        agent.calc_new_rho(4.0, 2.0, None, None)
        self.assertAlmostEqual(agent.rho, 2.0)

    def test_on_a_unit_clock_it_is_an_ordinary_smoother_of_the_rates(self):
        # lambda = -log(1 - beta) is chosen so that tau = 1 has gain beta. On a
        # unit clock the agent is therefore a debiased EMA of the per-transition
        # rates, which is what makes RelaxedSMART at the same rho_learning_rate
        # the comparable estimator rather than a differently tuned one.
        agent = SmoothedSMART("smoothed", [0], rho_learning_rate=0.5)
        reference = NormalizedEMA(0.5)
        for reward in (1.0, 9.0, -4.0):
            agent.calc_new_rho(reward, 1.0, None, None)
            self.assertAlmostEqual(agent.rho, reference.update(reward, 1.0))
        self.assertAlmostEqual(agent.lambda_, math.log(2.0))

    def test_a_long_transition_forgets_more_than_a_short_one(self):
        # The distinguishing property: the same reward rate arriving over ten
        # time units displaces the old estimate far more than over one tenth.
        def after(duration):
            agent = SmoothedSMART("smoothed", [0], rho_learning_rate=0.5)
            agent.calc_new_rho(0.0, 1.0, None, None)  # rho = 0
            agent.calc_new_rho(10.0 * duration, duration, None, None)
            return agent.rho

        self.assertGreater(after(10.0), after(1.0))
        self.assertGreater(after(1.0), after(0.1))
        # Ten time units at rate 10 all but replace the estimate; a tenth of one
        # barely moves it.
        self.assertAlmostEqual(after(10.0), 10.0, places=2)
        self.assertLess(after(0.1), 2.0)

    def test_splitting_a_transition_does_not_change_rho(self):
        # Segmentation invariance. RelaxedSMART has no such guarantee; its memory
        # is measured in transitions.
        def after(pieces):
            agent = SmoothedSMART("smoothed", [0], rho_learning_rate=0.4)
            for reward, duration in ((10.0, 10.0), (1.0, 0.1), (1.0, 0.1)):
                for _ in range(pieces):
                    agent.calc_new_rho(reward / pieces, duration / pieces, None, None)
            return agent.rho

        self.assertAlmostEqual(after(1), after(4), places=9)
        self.assertAlmostEqual(after(1), after(25), places=9)

    def test_cumulative_totals_are_not_exposed(self):
        # As with RelaxedSMART: the smoothed rate replaces the cumulative one, so
        # SMART's accumulators would describe something this agent never feeds.
        agent = SmoothedSMART("smoothed", [0], rho_learning_rate=0.3)
        agent.act("s")
        agent.learn("s", 0, 4.0, "s", 2.0)
        self.assertEqual(agent.step_count, 1)
        for attribute in ("total_reward", "total_time"):
            with self.subTest(attribute=attribute):
                with self.assertRaises(AttributeError):
                    getattr(agent, attribute)

    def test_reset_clears_the_smoothed_rate(self):
        agent = SmoothedSMART("smoothed", [0], rho_learning_rate=0.3)
        agent.calc_new_rho(2.0, 1.0, None, None)
        agent.reset()
        self.assertEqual((agent.rho, agent.step_count), (0.0, 0))
        agent.calc_new_rho(6.0, 2.0, None, None)
        self.assertAlmostEqual(agent.rho, 3.0)

    def test_zero_duration_raises(self):
        agent = SmoothedSMART("smoothed", [0], rho_learning_rate=0.5)
        with self.assertRaises(ZeroDivisionError):
            agent.calc_new_rho(1.0, 0.0, None, None)


if __name__ == "__main__":
    unittest.main()
