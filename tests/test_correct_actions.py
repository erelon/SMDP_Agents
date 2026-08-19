"""Re-derive the correct-action table in ``examples/correct_actions.py``.

That table is what the trap measurement is scored against, so it must not be able to
drift out of step with the environments. Every entry here is checked by *measuring*
the environment rather than by restating the table: for each action available at the
probe state, the mean immediate reward and the long-run rate of the policy that
always takes it. If someone edits a reward, a duration or a transition probability
so that the correct action or the bait moves, these fail.
"""

import unittest

from examples import correct_actions
from examples.correct_actions import CORRECT, EXTRA_PROBES, CorrectChoice
from examples.envs import ENVS, make


def immediate_reward(env_name, state, action, trials=80):
    """Mean reward of one decision taken from ``state``."""
    total = 0.0
    for seed in range(trials):
        env = make(env_name)
        obs, _ = env.reset(seed=1000 + seed)
        if env.state_of(obs) != state:
            return None  # not reachable as the start state; measured elsewhere
        _, reward, _, _, _ = env.step(action)
        total += reward
    return total / trials


def long_run_rate(env_name, action, steps=4000, seed=1):
    """Rate of the policy that always plays ``action`` where it is legal."""
    env = make(env_name)
    obs, _ = env.reset(seed=seed)
    state = env.state_of(obs)
    reward_sum = tau_sum = 0.0
    for _ in range(steps):
        legal = env.get_available_actions(state)
        obs, reward, terminated, truncated, info = env.step(
            action if action in legal else legal[0])
        reward_sum += reward
        tau_sum += info["tau"]
        state = env.state_of(obs)
        if terminated or truncated:
            obs, _ = env.reset()
            state = env.state_of(obs)
    return reward_sum / tau_sum


class CorrectActionTableTests(unittest.TestCase):
    """The table's structure and its agreement with the environments."""

    def test_every_named_environment_is_registered(self):
        for name in list(CORRECT) + list(EXTRA_PROBES):
            with self.subTest(env=name):
                self.assertIn(name, ENVS)

    def test_every_correct_action_is_legal_at_its_probe_state(self):
        for name, choice in list(CORRECT.items()) + list(EXTRA_PROBES.items()):
            with self.subTest(env=name):
                env = make(name)
                legal = env.get_available_actions(choice.state)
                self.assertIn(choice.resolve(env), legal)
                if choice.bait is not None:
                    self.assertIn(choice.bait, legal)
                    self.assertNotEqual(choice.bait, choice.resolve(env))

    def test_every_entry_states_a_criterion_and_a_reason(self):
        for name, choice in list(CORRECT.items()) + list(EXTRA_PROBES.items()):
            with self.subTest(env=name):
                self.assertTrue(choice.criterion)
                self.assertTrue(choice.note)
                self.assertTrue(choice.source)

    def test_the_correct_action_really_is_the_best_stationary_one(self):
        # Skipped for the two whose criterion is deliberately not the long-run rate:
        # gemini is correct under a ratio of expectations (the time-average picks the
        # other action), and risk under risk-neutrality.
        # sincoslog joins them: its correct action is correct *asymptotically*, and
        # deliberately loses over the horizon a run simulates — that is the trap.
        by_criterion = {"gemini", "risk", "sincoslog", "harmonic_criterion"}
        for name, choice in CORRECT.items():
            if name in by_criterion:
                continue
            with self.subTest(env=name):
                env = make(name)
                expected = choice.resolve(env)
                rates = {a: long_run_rate(name, a)
                         for a in env.get_available_actions(choice.state)}
                self.assertEqual(max(rates, key=rates.get), expected,
                                 f"{name}: measured {rates}")

    def test_the_bait_really_is_more_attractive_than_the_correct_action(self):
        for name, choice in list(CORRECT.items()) + list(EXTRA_PROBES.items()):
            if choice.bait is None:
                continue
            if choice.bait_by != correct_actions.IMMEDIATE_REWARD:
                # A bait that misleads by window rate rather than by immediate
                # reward; checked in SincoslogCorrectActionTests instead.
                continue
            with self.subTest(env=name):
                env = make(name)
                correct = choice.resolve(env)
                bait = immediate_reward(name, choice.state, choice.bait, trials=600)
                good = immediate_reward(name, choice.state, correct, trials=600)
                if bait is None or good is None:
                    self.skipTest(f"{choice.state!r} is not {name}'s start state")
                self.assertGreater(bait, good,
                                   f"{name}: bait {bait} is not more attractive "
                                   f"than the correct action's {good}")

    def test_the_environments_without_a_bait_are_not_traps(self):
        for name, choice in CORRECT.items():
            if choice.bait is not None:
                continue
            with self.subTest(env=name):
                env = make(name)
                correct = choice.resolve(env)
                others = [a for a in env.get_available_actions(choice.state)
                          if a != correct]
                good = immediate_reward(name, choice.state, correct, trials=600)
                if good is None:
                    self.skipTest(f"{choice.state!r} is not {name}'s start state")
                for action in others:
                    rival = immediate_reward(name, choice.state, action, trials=600)
                    self.assertGreaterEqual(good, rival,
                                            f"{name}: action {action} is more "
                                            f"attractive, so this IS a trap")

    def test_the_traps_are_the_ones_we_expect(self):
        # risk is deliberately absent: its correct action also pays best immediately
        # (mean 17 against 16), so the pull toward `averse` is its zero variance, not
        # a reward ordering, and this measurement cannot see it.
        self.assertEqual(sorted(correct_actions.traps()),
                         ["gemini", "hell_or_heaven", "ratio_vs_step_rate",
                          "sincoslog", "two_path", "uneven_cycling"])
        # two_states_uneven's trap is at its *second* state, not the start state.
        self.assertTrue(EXTRA_PROBES["two_states_uneven"].is_trap)


class SincoslogCorrectActionTests(unittest.TestCase):
    """The correct arm is right only beyond the horizon a run can see."""

    def test_the_correct_action_is_the_sources_and_never_changes(self):
        # robustness_table.py:55 CORRECT_ACTION = 1, the sin/log arm, for the whole
        # sweep. Adopted as the criterion: its rate grows exponentially where the
        # ramp's grows linearly, so it wins for every log_scale *eventually*.
        for log_scale in (1e-5, 1e-3, 5e-3, 1e-1):
            with self.subTest(log_scale=log_scale):
                env = make("sincoslog", log_scale=log_scale)
                self.assertEqual(CORRECT["sincoslog"].resolve(env),
                                 correct_actions.SINCOSLOG_SOURCE_CONSTANT)

    def test_the_bait_is_the_ramp_and_it_wins_inside_the_window(self):
        # The trap: over the episode the ramp earns several times the correct arm's
        # rate, so an agent judging only what it can measure takes the ramp.
        choice = CORRECT["sincoslog"]
        self.assertEqual(choice.bait, 0)
        self.assertEqual(choice.bait_by, correct_actions.WINDOW_RATE)
        ramp = long_run_rate("sincoslog", 0, steps=1_000)
        arm = long_run_rate("sincoslog", 1, steps=1_000)
        self.assertGreater(ramp, arm)
        self.assertGreater(ramp / arm, 3.0)

    def test_the_bait_pays_less_than_the_correct_arm_on_the_first_decision(self):
        # Which is exactly why an immediate-reward notion of "bait" misses this one.
        ramp = immediate_reward("sincoslog", "s1", 0, trials=200)
        arm = immediate_reward("sincoslog", "s1", 1, trials=200)
        self.assertLess(ramp, arm)
        self.assertGreater(arm / ramp, 100.0)

    def test_the_overtake_count_reproduces_a_direct_simulation(self):
        # Closed form with the oscillation dropped (offset 10 >> amplitude 1); these
        # three were confirmed against a simulation of the source's own processes,
        # whose s2 -> s1 leg paid nothing.
        for log_scale, expected in ((1e-5, 867_213), (1.4e-5, 596_189),
                                    (0.000127, 48_570)):
            with self.subTest(log_scale=log_scale, return_reward=0.0):
                self.assertEqual(
                    correct_actions.visits_to_overtake(log_scale, return_reward=0.0),
                    expected)

    def test_paying_for_the_return_leg_moves_the_overtake_count_slightly(self):
        # This repo pays 1 for s2 -> s1, which adds n to both arms' cumulative
        # reward. It shifts the crossover a little later -- the ramp's own reward
        # starts near zero, so a flat 1 per visit helps it proportionally more --
        # but nowhere near enough to change which configurations are traps.
        env = make("sincoslog")
        self.assertEqual(correct_actions.sincoslog_return_reward(env), 1.0)
        for log_scale, unpaid, paid in ((1e-5, 867_213, 867_218),
                                        (0.000127, 48_570, 48_577),
                                        (1e-3, 3_994, 4_005)):
            with self.subTest(log_scale=log_scale):
                self.assertEqual(
                    correct_actions.visits_to_overtake(log_scale, return_reward=0.0),
                    unpaid)
                self.assertEqual(correct_actions.visits_to_overtake(log_scale), paid)
                self.assertGreater(paid, unpaid)

    def test_the_payoff_is_far_beyond_the_horizon_for_most_of_the_sweep(self):
        from examples.envs.configs import SINCOSLOG_LOG_SCALES
        horizon = correct_actions.SINCOSLOG_VISITS_PER_EPISODE
        beyond = [ls for ls in SINCOSLOG_LOG_SCALES
                  if correct_actions.visits_to_overtake(ls) > horizon]
        # Two thirds of the swept range hides the answer past the horizon; the rest
        # resolves inside it and so is not a trap at all.
        self.assertEqual(len(beyond), 20)
        self.assertEqual(len(SINCOSLOG_LOG_SCALES), 30)

    def test_the_slowest_configuration_is_hopeless_by_design(self):
        visits = correct_actions.visits_to_overtake(1e-5)
        episodes = visits / correct_actions.SINCOSLOG_VISITS_PER_EPISODE
        self.assertGreater(episodes, 1_500)  # against the 10 a run performs

    def test_the_detail_line_states_the_gap(self):
        detail = CORRECT["sincoslog"].detail(make("sincoslog", log_scale=1e-3))
        self.assertIn("visits", detail)
        self.assertIn("500", detail)


if __name__ == "__main__":
    unittest.main()
