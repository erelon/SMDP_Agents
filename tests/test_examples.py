"""Tests for the example environments in ``examples/envs``.

Three kinds of check:

* **Contract** — every registered environment is put through
  ``check_smdp_env``: holding times finite and non-negative, ``info`` complete,
  states hashable, the clock consistent with the holding times, and the whole
  trajectory reproducible from its seed.
* **Analytic** — the environments built to demonstrate a specific numeric claim
  are asserted against that claim, both from their transition tables directly and
  by sampling. If someone edits a reward and breaks the counterexample, these
  fail rather than the claim quietly becoming false.
* **Behavioural** — the properties the environments exist to have: whack-a-mole's
  MDP variant really does hold τ ≡ 1, the drifting worlds really do change which
  action is best, action masks really do exclude the illegal moves.
"""

import statistics
import unittest

from agents.average_rates import (CumulativeTimeRate, ExponentialMovingRatioRate,
                                  NormalizedExponentialMovingTimeRate)
from examples.envs import (ENVS, FAMILIES, check_smdp_env,
                           heuristic_policy, make)
from examples.envs.base import EnvContractError, SMDPEnv
from examples.envs.configs import (AVERSE, NEUTRAL, RISK_ACTION_NAMES, SEEK,
                                   feinberg_three_state, gemini_three_state,
                                   ratio_vs_step_rate, risk_three_actions,
                                   self_similar_margin,
                                   sincoslog_self_similar)
from examples.envs.distributions import make_reward
from examples.envs.tabular import SMDPConfig, TabularSMDPEnv, Transition
from examples.envs.two_path import ACTION_A, ACTION_B, STATE0, TwoPathEnv
from examples.envs.whack_a_mole import WhackAMoleMDP, WhackAMoleSMDP

A, B = 0, 1


# ------------------------------------------------------------------ helpers
def outcomes(config: SMDPConfig, state, action):
    """The ``(prob, reward, duration)`` triples of a constant transition."""
    triples = []
    for t in config.transitions[(state, action)]:
        triples.append((t.constant("prob"), t.constant("reward"), t.constant("duration")))
    if any(None in triple for triple in triples):
        raise AssertionError(f"({state!r}, {action!r}) is not a constant transition")
    return triples


def ratio_of_expectations(config, state, action):
    """``E[r] / E[tau]`` over one transition's outcome distribution."""
    triples = outcomes(config, state, action)
    return (sum(p * r for p, r, _ in triples) / sum(p * d for p, _, d in triples))


def mean_of_rates(config, state, action):
    """``E[r / tau]`` over one transition's outcome distribution."""
    return sum(p * r / d for p, r, d in outcomes(config, state, action))


def run(env: SMDPEnv, policy, steps, seed=0, reset_between=True):
    """Drive ``env`` for ``steps`` decisions; return ``(rewards, taus)``."""
    obs, _ = env.reset(seed=seed)
    state = env.state_of(obs)
    rewards, taus = [], []
    for _ in range(steps):
        action = policy(env, state)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        taus.append(info["tau"])
        state = env.state_of(obs)
        if terminated or truncated:
            if not reset_between:
                break
            obs, _ = env.reset(seed=seed)
            state = env.state_of(obs)
    return rewards, taus


def rate(rewards, taus):
    return sum(rewards) / sum(taus)


def fixed(action):
    """A policy that always plays ``action`` where legal, else the first legal one."""
    def policy(env, state):
        available = env.get_available_actions(state)
        return action if action in available else available[0]
    return policy


def trajectory_totals(env, policy, trajectories, steps, seed=0):
    """``(reward, time)`` totals for several independent trajectories.

    Seeds once and then re-resets without a seed, so every trajectory draws from
    one well-mixed stream. Seeding each trajectory from a consecutive integer
    instead would make the *first* draw of each — which is the branch these
    environments hinge on — correlated across trajectories.
    """
    totals = []
    for i in range(trajectories):
        obs, _ = env.reset(seed=seed if i == 0 else None)
        state = env.state_of(obs)
        reward_sum = tau_sum = 0.0
        for _ in range(steps):
            obs, reward, terminated, truncated, info = env.step(policy(env, state))
            reward_sum += reward
            tau_sum += info["tau"]
            state = env.state_of(obs)
            if terminated or truncated:
                break
        totals.append((reward_sum, tau_sum))
    return totals


def time_average(totals):
    """The mean of the per-trajectory rates."""
    return statistics.fmean(r / t for r, t in totals)


def pooled_rate(totals):
    """Total reward over total time — the ratio of expectations."""
    return sum(r for r, _ in totals) / sum(t for _, t in totals)


# ------------------------------------------------------------------- contract
class ExampleEnvContractTests(unittest.TestCase):
    """Every registered environment obeys the SMDP contract."""

    def test_every_registered_env_obeys_the_contract(self):
        for name in ENVS:
            with self.subTest(env=name):
                summary = check_smdp_env(make(name), steps=200, seed=3)
                self.assertGreater(summary["steps"], 0)
                self.assertGreater(summary["elapsed"], 0.0)

    def test_registry_entries_describe_themselves(self):
        for name, spec in ENVS.items():
            with self.subTest(env=name):
                self.assertTrue(spec.describe(), f"{name} has no description")
                self.assertIn(spec.family, FAMILIES)

    def test_unknown_env_name_lists_the_registry(self):
        with self.assertRaises(KeyError) as ctx:
            make("no_such_env")
        self.assertIn("sincoslog", str(ctx.exception))

    def test_contract_check_rejects_a_missing_holding_time(self):
        class NoTau(TwoPathEnv):
            def step(self, action):
                obs, reward, terminated, truncated, info = super().step(action)
                return obs, reward, terminated, truncated, {}

        with self.assertRaises(EnvContractError) as ctx:
            check_smdp_env(NoTau(), steps=5)
        self.assertIn("missing 'tau'", str(ctx.exception))

    def test_contract_check_rejects_an_unhashable_state(self):
        class ListState(TwoPathEnv):
            def state_of(self, obs):
                return [obs]

        with self.assertRaises(EnvContractError) as ctx:
            check_smdp_env(ListState(), steps=5)
        self.assertIn("unhashable", str(ctx.exception))

    def test_contract_check_rejects_a_clock_that_disagrees_with_tau(self):
        class BadClock(TwoPathEnv):
            def step(self, action):
                obs, reward, terminated, truncated, info = super().step(action)
                info["time"] = info["time"] + 5.0
                return obs, reward, terminated, truncated, info

        with self.assertRaises(EnvContractError) as ctx:
            check_smdp_env(BadClock(), steps=5)
        self.assertIn("disagrees with the sum of holding times", str(ctx.exception))

    def test_contract_check_rejects_an_irreproducible_env(self):
        class Drifting(TwoPathEnv):
            counter = 0.0

            def step(self, action):
                obs, reward, terminated, truncated, info = super().step(action)
                Drifting.counter += 1.0
                return obs, reward + Drifting.counter, terminated, truncated, info

        with self.assertRaises(EnvContractError) as ctx:
            check_smdp_env(Drifting(), steps=5)
        self.assertIn("not reproducible", str(ctx.exception))


# ----------------------------------------------------------- criterion claims
class CriterionEnvTests(unittest.TestCase):
    """The counterexamples really do separate the candidate criteria."""

    def test_gemini_pays_10_by_time_average_and_1_by_ratio_of_expectations(self):
        config = gemini_three_state()
        self.assertAlmostEqual(ratio_of_expectations(config, "s1", A), 1.0)

        # max_steps=None: the criterion is asymptotic, so the registry's
        # episode cap (which exists so an agent sees s1 more than once) must
        # be lifted to measure it.
        totals = trajectory_totals(make("gemini", max_steps=None), fixed(A),
                                   trajectories=400, steps=800, seed=11)
        # Each trajectory locks into one of the two absorbing loops.
        for reward_sum, tau_sum in totals:
            realised = reward_sum / tau_sum
            self.assertTrue(realised > 19.0 or realised < 0.1,
                            f"unexpected trajectory rate {realised}")
        self.assertAlmostEqual(time_average(totals), 10.0, delta=1.5)
        # The pooled denominator is dominated by the slow branch (19 time units
        # against 1), so a couple of percent of imbalance in the branch split
        # moves this several percent — hence the loose delta on an exact 1.0.
        self.assertAlmostEqual(pooled_rate(totals), 1.0, delta=0.35)
        self.assertGreater(time_average(totals) / pooled_rate(totals), 5.0)

    def test_gemini_action_b_pays_4_under_both_criteria(self):
        config = gemini_three_state()
        self.assertAlmostEqual(ratio_of_expectations(config, "s1", B), 4.0)
        self.assertAlmostEqual(mean_of_rates(config, "s1", B), 4.0)

        rewards, taus = run(make("gemini"), fixed(B), steps=500, seed=1)
        self.assertAlmostEqual(rate(rewards, taus), 4.0)

    def test_feinberg_is_7_point_5_by_time_average_and_6_667_by_ratio(self):
        totals = trajectory_totals(make("feinberg", max_steps=None), fixed(A),
                                   trajectories=600, steps=800, seed=11)
        for reward_sum, tau_sum in totals:
            realised = reward_sum / tau_sum
            self.assertTrue(abs(realised - 10.0) < 0.1 or abs(realised - 5.0) < 0.1,
                            f"unexpected trajectory rate {realised}")
        # Averaging the per-trajectory rates gives the time-average, 7.5;
        # pooling reward and time first gives the ratio of expectations, 6.667.
        self.assertAlmostEqual(time_average(totals), 7.5, delta=0.25)
        self.assertAlmostEqual(pooled_rate(totals), 10.0 / 1.5, delta=0.15)

    def test_ratio_and_step_rate_rank_the_jackpot_oppositely(self):
        config = ratio_vs_step_rate()
        self.assertAlmostEqual(ratio_of_expectations(config, "s0", A), 1.99 / 990.01)
        self.assertAlmostEqual(mean_of_rates(config, "s0", A), 1.00099)
        # Action b sits between the two, so the criteria disagree about the order.
        self.assertAlmostEqual(ratio_of_expectations(config, "s0", B), 1.0)
        self.assertLess(ratio_of_expectations(config, "s0", A),
                        ratio_of_expectations(config, "s0", B))
        self.assertGreater(mean_of_rates(config, "s0", A),
                           mean_of_rates(config, "s0", B))

    def test_the_multichain_configs_restart_far_sooner_than_their_source(self):
        # The source protocol (PythonProject3) caps an episode at 10,000
        # decisions, which the unichain configs inherit. The multichain ones need
        # a much shorter cap: both loops absorb, so a long episode gives an agent
        # one decision and then thousands of consequences, and every agent ends up
        # scoring the same.
        # The multichain configs get a cap far below their source protocol's.
        for name, cap in (("gemini", 50), ("feinberg", 50),
                          ("hell_or_heaven", 500), ("bonus", 50)):
            with self.subTest(env=name):
                self.assertEqual(make(name).max_steps, cap)
        # The rest inherit the episode length their source used: 1,000 steps for
        # the sincoslog family and the ambient configs, 10,000 for the one config
        # `run_smdp_experiment.py` actually has live, 20 for the compounding one.
        for name, cap in (("sincoslog", 1_000), ("risk", 1_000),
                          ("schwartz", 1_000), ("ratio_vs_step_rate", 10_000),
                          ("non_stationary", 20)):
            with self.subTest(env=name):
                self.assertEqual(make(name).max_steps, cap)

    def test_refusing_the_jackpot_needs_a_long_enough_episode(self):
        # Taking it is worth (jackpot - L + 1)/L and refusing (L - 1)/L, so the
        # registry's episode length has to stay above jackpot/2 + 1 or the
        # environment says the opposite of what it is for.
        env = make("hell_or_heaven")
        jackpot = 100.0
        self.assertGreater(env.max_steps, jackpot / 2 + 1)
        take = rate(*run(env, fixed(B), steps=env.max_steps, seed=0,
                         reset_between=False))
        refuse = rate(*run(make("hell_or_heaven"), fixed(A), steps=env.max_steps,
                           seed=0, reset_between=False))
        self.assertGreater(refuse, take)
        self.assertAlmostEqual(refuse, 0.998, delta=0.001)
        self.assertAlmostEqual(take, -0.798, delta=0.001)

    def test_the_jackpot_disagreement_shows_up_in_samples(self):
        # The default p=0.01 needs impractically many samples to measure, so the
        # sampled check uses an even-odds variant of the same shape.
        build = lambda: TabularSMDPEnv(  # noqa: E731
            ratio_vs_step_rate(p=0.5), name="jackpot")
        rewards, taus = run(build(), fixed(A), steps=6000, seed=5)
        pooled = rate(rewards, taus)
        per_step = statistics.fmean(r / t for r, t in zip(rewards, taus))

        self.assertAlmostEqual(pooled, 50.5 / 500.5, delta=0.02)
        self.assertAlmostEqual(per_step, 50.0005, delta=2.0)
        self.assertLess(pooled, 1.0)        # ratio of expectations prefers b
        self.assertGreater(per_step, 1.0)   # mean of rates prefers a

    def test_the_burst_option_is_worth_20_over_11_and_the_steady_action_1(self):
        rewards, taus = run(make("high_time_variance"), fixed(A), steps=1_100, seed=0)
        self.assertAlmostEqual(rate(rewards, taus), 20 / 11, places=9)
        self.assertEqual(len(set(taus)), 2)          # one slow step, ten fast ones
        self.assertAlmostEqual(max(taus) / min(taus), 100.0)
        steady_rewards, steady_taus = run(make("high_time_variance"), fixed(B),
                                          steps=100, seed=0)
        self.assertAlmostEqual(rate(steady_rewards, steady_taus), 1.0)
        self.assertEqual(set(steady_taus), {1.0})    # no holding-time variance

    def test_half_the_cycles_time_is_one_eleventh_of_its_transitions(self):
        # This is the whole environment: the split into transitions carries no
        # information about where the time goes, so an estimator that forgets per
        # transition weights the burst ten times too heavily.
        _, taus = run(make("high_time_variance"), fixed(A), steps=11, seed=0)
        self.assertEqual(len(taus), 11)
        self.assertAlmostEqual(taus[0] / sum(taus), 10 / 11)
        self.assertAlmostEqual(sum(taus[1:]) / sum(taus), 1 / 11)

    def test_the_criteria_disagree_about_the_burst_by_a_factor_of_five(self):
        rewards, taus = run(make("high_time_variance"), fixed(A), steps=1_100, seed=0)
        time_average = rate(rewards, taus)
        mean_of_transition_rates = statistics.fmean(r / t for r, t in zip(rewards, taus))
        self.assertAlmostEqual(time_average, 20 / 11, places=9)
        self.assertAlmostEqual(mean_of_transition_rates, 101 / 11, places=9)
        self.assertGreater(mean_of_transition_rates / time_average, 5.0)

    def test_the_option_survives_only_while_rho_stays_under_1_point_9(self):
        # Under R-learning the option is preferred while
        # (R_cycle - rho*T_cycle) > (r_steady - rho*tau_steady), and the
        # environment is built so that threshold sits just 4.5% above the option's
        # own true rate.
        config = ENVS["high_time_variance"].build().config
        cycle_reward = cycle_time = 0.0
        state = "s0"
        while True:
            transition, = config.transitions[(state, A)]
            cycle_reward += transition.reward()
            cycle_time += transition.duration()
            state = transition.next_state()
            if state == "s0":
                break
        steady, = config.transitions[("s0", B)]
        threshold = ((cycle_reward - steady.reward())
                     / (cycle_time - steady.duration()))
        self.assertAlmostEqual(threshold, 1.9)
        self.assertGreater(threshold, cycle_reward / cycle_time)
        self.assertLess(threshold / (cycle_reward / cycle_time), 1.05)

    def test_the_estimators_overshoot_in_the_order_the_config_claims(self):
        # The rho each estimator averages over the transitions of the always-a
        # cycle -- which is the rho the updates actually see. Deterministic, and
        # the ordering holds at every beta: forgetting in seconds overshoots less
        # than forgetting in transitions, and both overshoot.
        rewards, taus = run(make("high_time_variance"), fixed(A), steps=4_400, seed=0)
        stream = list(zip(rewards, taus))
        truth = rate(rewards, taus)

        def settled_mean(estimator):
            values = [estimator.update(r, t) for r, t in stream]
            return statistics.fmean(values[len(values) // 2:])

        for beta in (0.3, 0.2, 0.1, 0.05):
            with self.subTest(beta=beta):
                cumulative = settled_mean(CumulativeTimeRate())
                timed = settled_mean(NormalizedExponentialMovingTimeRate(beta))
                ratio = settled_mean(ExponentialMovingRatioRate(beta))
                self.assertAlmostEqual(cumulative, truth, places=2)
                self.assertLess(truth, timed)
                self.assertLess(timed, ratio)

        # At the registered environment's default beta the gap is worth stating.
        self.assertAlmostEqual(
            settled_mean(NormalizedExponentialMovingTimeRate(0.3)), 2.488, places=3)
        self.assertAlmostEqual(
            settled_mean(ExponentialMovingRatioRate(0.3)), 2.868, places=3)

    def test_the_estimators_agree_again_once_the_gain_is_small(self):
        # The disagreement is a fixed-gain effect, not an asymptotic bias.
        rewards, taus = run(make("high_time_variance"), fixed(A), steps=22_000, seed=0)
        for estimator in (NormalizedExponentialMovingTimeRate(1e-4),
                          ExponentialMovingRatioRate(1e-4)):
            with self.subTest(estimator=type(estimator).__name__):
                value = 0.0
                for reward, tau in zip(rewards, taus):
                    value = estimator.update(reward, tau)
                self.assertAlmostEqual(value, 20 / 11, places=2)


# ------------------------------------------------------------------ risk
class RiskEnvTests(unittest.TestCase):
    """Each risk attitude has its own strict optimum among the three gambles."""

    def _moments(self, config, action):
        triples = outcomes(config, "s1", action)
        mean = sum(p * r for p, r, _ in triples)
        var = sum(p * (r - mean) ** 2 for p, r, _ in triples)
        return mean, var

    def test_neutral_has_the_highest_mean(self):
        config = risk_three_actions()
        means = {a: self._moments(config, a)[0] for a in (SEEK, NEUTRAL, AVERSE)}
        self.assertAlmostEqual(means[NEUTRAL], 17.0)
        self.assertAlmostEqual(means[SEEK], 16.0)
        self.assertAlmostEqual(means[AVERSE], 16.0)
        self.assertGreater(means[NEUTRAL], means[SEEK])

    def test_seek_and_averse_tie_on_mean_and_differ_only_in_spread(self):
        config = risk_three_actions()
        seek_mean, seek_var = self._moments(config, SEEK)
        averse_mean, averse_var = self._moments(config, AVERSE)
        self.assertAlmostEqual(seek_mean, averse_mean)
        self.assertAlmostEqual(averse_var, 0.0)
        self.assertAlmostEqual(seek_var, 14.0 ** 2)

    def test_spread_increases_from_averse_through_neutral_to_seeking(self):
        config = risk_three_actions()
        variances = [self._moments(config, a)[1] for a in (AVERSE, NEUTRAL, SEEK)]
        self.assertEqual(variances, sorted(variances))

    def test_the_realised_rates_are_8_8_and_8_and_a_third(self):
        expected = {SEEK: 8.0, AVERSE: 8.0, NEUTRAL: 25.0 / 3.0}
        for action, target in expected.items():
            with self.subTest(action=RISK_ACTION_NAMES[action]):
                rewards, taus = run(make("risk"), fixed(action), steps=8000, seed=2)
                self.assertAlmostEqual(rate(rewards, taus), target, delta=0.2)


# ----------------------------------------------------------------- drift
class DriftEnvTests(unittest.TestCase):
    """The non-stationary worlds change which action is best."""

    def test_growing_reward_with_faster_growing_time_decays_in_rate(self):
        ramp = rate(*run(make("non_stationary"), fixed(A), steps=2000, seed=1))
        steady = rate(*run(make("non_stationary"), fixed(B), steps=2000, seed=1))
        self.assertLess(ramp, steady)
        self.assertAlmostEqual(steady, 5.0, delta=0.1)  # 10 per 2 time units

    def test_the_ramp_is_capped_so_the_clock_stays_bounded(self):
        _, taus = run(make("non_stationary"), fixed(A), steps=2000, seed=1)
        # 12 doublings, so no single holding time may exceed 2 ** 12.
        self.assertLessEqual(max(taus), 2.0 ** 12)

    def test_sincoslog_leader_changes_between_early_and_late(self):
        early_a = rate(*run(make("sincoslog"), fixed(A), steps=20, seed=1))
        early_b = rate(*run(make("sincoslog"), fixed(B), steps=20, seed=1))
        self.assertLess(early_a, early_b)

        late_a = rate(*run(make("sincoslog"), fixed(A), steps=4000, seed=1))
        late_b = rate(*run(make("sincoslog"), fixed(B), steps=4000, seed=1))
        self.assertGreater(late_a, late_b)

    def test_a_larger_log_scale_flips_the_late_winner(self):
        build = lambda ls: TabularSMDPEnv(  # noqa: E731
            __import__("examples.envs.configs", fromlist=["sincoslog"])
            .sincoslog(log_scale=ls), name="sincoslog")
        late_a = rate(*run(build(1e-2), fixed(A), steps=2000, seed=1))
        late_b = rate(*run(build(1e-2), fixed(B), steps=2000, seed=1))
        self.assertGreater(late_b, late_a)

    def test_the_self_similar_margin_is_closed_form_and_envelope_independent(self):
        # The whole point of the rebuild: M does not decay over an episode, so the
        # decision keeps a constant 6.7% margin instead of collapsing to a knife
        # edge the way the legacy env's does.
        self.assertAlmostEqual(self_similar_margin(), 1.0 / 15.0)          # +0.0667
        self.assertAlmostEqual(self_similar_margin(a_r=60.0), -1.0 / 15.0)  # -0.0667
        # It depends on the geometry only -- no envelope term appears in it.
        for s in (1e-4, 1e-2, 3e-2):
            with self.subTest(s=s):
                env = TabularSMDPEnv(sincoslog_self_similar(s=s), name="ss")
                self.assertEqual(sorted(env.config.states), ["s1", "s2"])
        self.assertEqual(self_similar_margin(o_d=0.5), float("inf"))  # not the long arm

    def test_a_r_alone_flips_which_arm_is_optimal(self):
        # sincoslog_ss_paid and sincoslog_ss_short differ in a_r and nothing else,
        # and that single number moves the optimum from the long arm to the short
        # one. It is what bounds the harmonic family's advantage.
        rates = {}
        for name in ("sincoslog_ss_paid", "sincoslog_ss_short"):
            rates[name] = {a: rate(*run(make(name), fixed(a), steps=2000, seed=1))
                           for a in (A, B)}
        self.assertGreater(rates["sincoslog_ss_paid"][B], rates["sincoslog_ss_paid"][A])
        self.assertGreater(rates["sincoslog_ss_short"][A], rates["sincoslog_ss_short"][B])
        self.assertEqual(make("sincoslog_ss_paid").config.transitions[("s1", B)][0]
                         .constant("reward"), None)  # both arms are processes

    def test_paying_the_return_leg_removes_every_zero_reward(self):
        # The control that separates a real result from the legacy encoding
        # artefact: with the leg paid there is no zero reward anywhere, so nothing
        # can be attributed to how the harmonic estimators treat one.
        for name, expect_zeros in (("sincoslog_ss", True),
                                   ("sincoslog_ss_paid", False),
                                   ("sincoslog_ss_short", False)):
            with self.subTest(env=name):
                rewards, _ = run(make(name), fixed(B), steps=600, seed=1)
                self.assertEqual(any(r == 0.0 for r in rewards), expect_zeros)

    def test_the_self_similar_envelope_advances_on_every_decision(self):
        # Exogenous drift: both arms' rewards *and* both durations share one clock,
        # so the two fixed policies are comparable. The legacy config leaves the
        # durations unhooked, which is part of why its arms are not.
        env = make("sincoslog_ss")
        env.reset(seed=1)
        for _ in range(40):                      # advance using arm B only
            env.step(B if env.state == "s1" else A)
        first_a = None
        while first_a is None:
            _, reward, _, _, _ = env.step(A if env.state == "s1" else A)
            if env.state == "s2":
                first_a = reward
        # a_r is 40 at k=0; after ~20 shared decisions the envelope has lifted it.
        self.assertGreater(first_a, 40.0)

    def test_hooked_rewards_drift_even_when_the_other_action_is_taken(self):
        # Action a's reward is a ramp indexed by *decisions*, not by a-choices, so
        # playing b for a while still advances it.
        env = make("sincoslog")
        env.reset(seed=1)
        for _ in range(40):
            env.step(B if env.state == "s1" else A)
        first_a_reward = None
        while first_a_reward is None:
            _, reward, _, _, _ = env.step(A)
            if env.state == "s2":
                first_a_reward = reward
        self.assertGreater(first_a_reward, 0.5)  # 0.05 per decision, ~20 elapsed


# ------------------------------------------------------------- whack-a-mole
class WhackAMoleEnvTests(unittest.TestCase):
    """The MDP/SMDP pair differs in exactly one thing: whether τ varies."""

    def test_the_mdp_variant_holds_tau_at_one(self):
        for name in [n for n in ENVS if n.startswith("wam_mdp")]:
            with self.subTest(env=name):
                env = make(name)
                self.assertTrue(env.is_mdp)
                _, taus = run(env, lambda e, s: e.get_available_actions(s)[0],
                              steps=100, seed=1)
                self.assertEqual(set(taus), {1.0})

    def test_the_smdp_variant_charges_travel_time(self):
        env = make("wam_smdp_until_whacked_whack")
        self.assertFalse(env.is_mdp)
        env.reset(seed=1)
        # A jump from cell 0 to cell 8 crosses the 3x3 grid diagonally.
        _, _, _, _, info = env.step(8)
        self.assertAlmostEqual(info["tau"], 2.0 * 2.0 ** 0.5)
        _, _, _, _, info = env.step(env.action_space.n - 1)  # whack in place
        self.assertAlmostEqual(info["tau"], env.whack_time)

    def test_step_downed_is_rejected_on_the_smdp_variant(self):
        with self.assertRaises(ValueError) as ctx:
            WhackAMoleSMDP(reward_mode="step_downed")
        self.assertIn("MDP-only", str(ctx.exception))
        self.assertNotIn("wam_smdp_until_whacked_step_downed", ENVS)

    def test_the_mask_drops_off_grid_king_moves(self):
        env = WhackAMoleMDP(rows=3, cols=3)
        env.reset(seed=0)
        # From the top-left corner only 4 of the 9 king moves stay on the grid.
        state = (tuple([0] * 9), 0)
        self.assertEqual(env.get_available_actions(state), [4, 5, 7, 8, 9])
        # From the centre every move is legal.
        centre = (tuple([0] * 9), 4)
        self.assertEqual(env.get_available_actions(centre), list(range(10)))

    def test_the_mask_drops_the_illegal_self_move(self):
        env = WhackAMoleSMDP(rows=3, cols=3)
        env.reset(seed=0)
        state = (tuple([0] * 9), 3)
        available = env.get_available_actions(state)
        self.assertNotIn(3, available)
        self.assertEqual(len(available), 9)  # 8 other cells plus whack

    def test_whack_downed_pays_more_on_a_cleaner_board(self):
        env = make("wam_mdp_until_whacked_whack_downed")
        env.reset(seed=0)
        env.moles[:] = False
        env.moles[env.agent_pos] = True
        _, reward, _, _, info = env.step(9)
        self.assertTrue(info["whack_success"])
        self.assertEqual(reward, 9.0)  # all nine holes down after the whack

    def test_the_heuristic_beats_a_fixed_policy(self):
        for name in ("wam_mdp_until_whacked_whack", "wam_smdp_until_whacked_whack"):
            with self.subTest(env=name):
                env = make(name)
                smart = rate(*run(env, heuristic_policy, steps=3000, seed=1))
                idle = rate(*run(env, fixed(env.action_space.n - 1),
                                 steps=3000, seed=1))
                self.assertGreater(smart, idle)


# -------------------------------------------------------------------- rates
class RateEnvTests(unittest.TestCase):
    """Reward earned at a rate over a random duration, so the two are coupled."""

    def test_reward_stays_inside_the_floor_and_the_holding_time(self):
        for name in ("stateless", "two_states_uneven", "two_states_even"):
            with self.subTest(env=name):
                env = make(name)
                obs, _ = env.reset(seed=4)
                for _ in range(400):
                    action = int(env.np_random.integers(0, 2))
                    obs, reward, _, _, info = env.step(action)
                    self.assertGreaterEqual(reward, env.interval[0])
                    self.assertLessEqual(reward, info["tau"])

    def test_reward_tracks_the_holding_time_when_it_scales_with_it(self):
        scaled, _ = self._samples("two_states_uneven")
        flat, _ = self._samples("two_states_even")
        self.assertGreater(scaled, 0.9)   # reward is essentially tau * rate
        self.assertLess(flat, 0.2)        # flat reward, so no relationship

    def _samples(self, name, steps=2000):
        """Correlation between reward and holding time, and the realised rate."""
        env = make(name)
        obs, _ = env.reset(seed=4)
        rewards, taus = [], []
        for _ in range(steps):
            obs, reward, _, _, info = env.step(0)
            rewards.append(reward)
            taus.append(info["tau"])
        return statistics.correlation(rewards, taus), rate(rewards, taus)

    def test_escaping_the_poor_state_beats_milking_it(self):
        always_zero = rate(*run(make("two_states_uneven"), fixed(0),
                                steps=60_000, seed=1))
        escape = rate(*run(make("two_states_uneven"),
                           lambda e, s: 0 if s == 0 else 1, steps=60_000, seed=1))
        self.assertAlmostEqual(always_zero, 0.667, delta=0.02)
        self.assertAlmostEqual(escape, 0.693, delta=0.02)
        self.assertGreater(escape, always_zero)

    def test_the_secret_policy_is_the_best_of_the_four(self):
        env = make("two_states_uneven")
        secret = env.secret()
        policies = {(a0, a1): (lambda e, s, a0=a0, a1=a1: a0 if s == 0 else a1)
                    for a0 in (0, 1) for a1 in (0, 1)}
        rates = {key: rate(*run(make("two_states_uneven"), policy,
                                steps=60_000, seed=1))
                 for key, policy in policies.items()}
        best = max(rates, key=rates.get)
        self.assertEqual(best, (secret(0), secret(1)))

    def test_the_stateless_env_prefers_the_faster_rate(self):
        first = rate(*run(make("stateless"), fixed(0), steps=40_000, seed=1))
        second = rate(*run(make("stateless"), fixed(1), steps=40_000, seed=1))
        self.assertGreater(first, second)
        self.assertEqual(make("stateless").secret()(0), 0)

    def test_the_even_env_pays_a_flat_two_or_one(self):
        env = make("two_states_even")
        obs, _ = env.reset(seed=4)
        seen = set()
        for _ in range(200):
            state = obs
            obs, reward, _, _, info = env.step(state)  # always match the state
            if info["tau"] > 2.0:
                seen.add(reward)
        self.assertEqual(seen, {2.0})
        matched = rate(*run(make("two_states_even"), lambda e, s: s,
                            steps=40_000, seed=1))
        mismatched = rate(*run(make("two_states_even"), lambda e, s: 1 - s,
                               steps=40_000, seed=1))
        self.assertAlmostEqual(matched / mismatched, 2.0, delta=0.05)

    def test_the_cycle_phase_flips_on_schedule_and_swaps_the_best_action(self):
        env = make("uneven_cycling", cycle=5)
        env.reset(seed=1)
        self.assertEqual(env.phase, 0)
        self.assertGreater(env.rate(0, 0), env.rate(0, 1))
        for _ in range(5):
            env.step(0)
        self.assertEqual(env.phase, 1)
        self.assertLess(env.rate(0, 0), env.rate(0, 1))
        for _ in range(5):
            env.step(0)
        self.assertEqual(env.phase, 0)

    def test_the_cycle_phase_is_not_observable(self):
        env = make("uneven_cycling", cycle=5)
        obs, _ = env.reset(seed=1)
        observations = {obs}
        for _ in range(60):
            obs, _, _, _, _ = env.step(0)
            observations.add(obs)
        # Two observable states only: nothing reveals the phase.
        self.assertEqual(observations, {0, 1})
        self.assertEqual(env.observation_space.n, 2)

    def test_the_latent_cycle_ignores_the_state(self):
        env = make("uneven_latent_cycling", cycle=50)
        env.reset(seed=1)
        self.assertEqual(env.rate(0, 0), env.rate(1, 0))
        self.assertEqual(env.rate(0, 1), env.rate(1, 1))
        self.assertEqual(env.secret()(0), 0)
        for _ in range(50):
            env.step(0)
        self.assertEqual(env.secret()(0), 1)

    def test_the_random_shift_stays_in_range_and_keeps_the_ordering(self):
        env = make("shifting_uneven", shift_steps=10, shift_min=0.1, shift_max=1.0)
        env.reset(seed=1)
        seen = set()
        for _ in range(500):
            env.step(0)
            seen.add(round(env.shift_constant, 9))
            self.assertGreaterEqual(env.shift_constant, 0.1)
            self.assertLessEqual(env.shift_constant, 1.0)
            # A common multiplier cannot reorder the actions.
            self.assertGreater(env.rate(0, 0), env.rate(0, 1))
        self.assertGreater(len(seen), 10)

    def test_the_exponential_slope_decays_monotonically(self):
        env = make("slope_shifting_uneven", shift_steps=10, slope=0.9,
                   scale_mode="exp")
        env.reset(seed=1)
        scales = []
        for _ in range(200):
            env.step(0)
            scales.append(env.shift_constant)
        self.assertEqual(scales, sorted(scales, reverse=True))
        self.assertAlmostEqual(scales[-1], 0.9 ** 20)

    def test_the_sinusoidal_mode_oscillates_around_its_base(self):
        env = make("slope_shifting_uneven", scale_mode="sinusoidal",
                   apply_every_step=True, period=40, amplitude=0.5, base=1.0)
        env.reset(seed=1)
        scales = []
        for _ in range(80):
            env.step(0)
            scales.append(env.shift_constant)
        self.assertAlmostEqual(max(scales), 1.5, delta=0.01)
        self.assertAlmostEqual(min(scales), 0.5, delta=0.01)

    def test_the_slope_modes_demand_their_parameters(self):
        for kwargs, expected in (
            (dict(scale_mode="toward_target"), "target is required"),
            (dict(scale_mode="logistic"), "target is required"),
            (dict(scale_mode="logistic", target=-1.0), "must be > 0"),
            (dict(scale_mode="sinusoidal", amplitude=0.5), "period must be > 0"),
            (dict(scale_mode="sinusoidal", period=10), "amplitude must be non-zero"),
            (dict(scale_mode="nonsense"), "scale_mode must be one of"),
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError) as ctx:
                    make("slope_shifting_uneven", **kwargs)
                self.assertIn(expected, str(ctx.exception))

    def test_the_shifting_variants_transpose_state_one(self):
        # Inherited from the original files, which disagreed; pinned so the
        # difference stays deliberate.
        plain = make("two_states_uneven")
        shifting = make("shifting_uneven")
        self.assertGreater(plain.rate(1, 0), plain.rate(1, 1))
        self.assertLess(shifting.rate(1, 0), shifting.rate(1, 1))

    def test_an_invalid_interval_or_noise_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            make("two_states_uneven", interval=(0.0, 10.0))
        self.assertIn("0 < min <= max", str(ctx.exception))
        with self.assertRaises(ValueError):
            make("two_states_uneven", noise=-1.0)
        with self.assertRaises(ValueError):
            make("two_states_uneven", maxp=0.0)


# ------------------------------------------------------------------- market
class MarketEnvTests(unittest.TestCase):
    """Real BTC bars, with the holding time derived from the same return."""

    def test_the_committed_slice_is_present_and_shaped(self):
        env = make("btc_market")
        self.assertGreater(len(env.returns), 30_000)
        self.assertEqual(env.observation_space.n, 27)  # 3 signs ** 3 bars
        self.assertEqual(env.action_space.n, 2)

    def test_holding_times_stay_inside_one_minus_the_window(self):
        env = make("btc_market", time_window=(0.1, 0.75))
        _, taus = run(env, fixed(1), steps=2000, seed=1)
        self.assertGreaterEqual(min(taus), 0.25 - 1e-9)
        self.assertLessEqual(max(taus), 0.90 + 1e-9)

    def test_the_state_is_the_previous_signs_with_no_lookahead(self):
        env = make("btc_market", random_start=False)
        obs, _ = env.reset(seed=1)
        for _ in range(50):
            bar = env.t
            # Oldest first, exactly the original's tuple(sign(returns[i-k:i])).
            expected = tuple(int(s) for s in env.signs[bar - env.state_size:bar])
            self.assertEqual(env.state_of(obs), expected)
            obs, _, _, truncated, _ = env.step(1)
            if truncated:
                break

    def test_the_oracle_ceiling_beats_both_fixed_positions(self):
        rates = {}
        for label, policy in (("long", fixed(1)), ("short", fixed(0))):
            rates[label] = rate(*run(make("btc_market"), policy, steps=20_000, seed=1))
        env = make("btc_market")
        secret = env.secret()
        rates["oracle"] = rate(*run(env, lambda e, s: secret(s),
                                    steps=20_000, seed=1))
        self.assertGreater(rates["oracle"], rates["long"])
        self.assertGreater(rates["oracle"], rates["short"])
        # Positions are +-1 and the reward is position * return * tau, so the two
        # fixed policies are exact mirror images.
        self.assertAlmostEqual(rates["long"], -rates["short"], places=9)

    def test_the_random_dependency_decouples_duration_from_the_return(self):
        coupled = self._duration_coupling("linear")
        decoupled = self._duration_coupling("random")
        self.assertGreater(abs(coupled), 0.2)
        self.assertLess(abs(decoupled), 0.06)

    def _duration_coupling(self, dependency, steps=4000):
        """Correlation between the bar's return and its holding time."""
        env = make("btc_market", dependency=dependency)
        env.reset(seed=2)
        returns, taus = [], []
        for _ in range(steps):
            bar = env.t
            _, _, _, truncated, info = env.step(1)
            returns.append(float(env.returns[bar]))
            taus.append(info["tau"])
            if truncated:
                env.reset()
        return statistics.correlation(returns, taus)

    def test_relative_and_absolute_returns_differ_by_orders_of_magnitude(self):
        relative = make("btc_market", percentage=True)
        absolute = make("btc_market", percentage=False)
        self.assertLess(abs(relative.returns).max(), 1.0)      # fractions
        self.assertGreater(abs(absolute.returns).max(), 100.0)  # dollars

    def test_running_off_the_end_truncates_rather_than_terminating(self):
        env = make("btc_market", random_start=False)
        env.reset(seed=1)
        env.t = len(env.returns) - 3
        for _ in range(5):
            _, _, terminated, truncated, info = env.step(1)
            self.assertFalse(terminated)
            if truncated:
                self.assertTrue(info["exhausted"])
                return
        self.fail("the environment never reported the end of the sample")

    def test_invalid_configuration_is_rejected(self):
        for kwargs, expected in (
            (dict(state_size=0), "state_size must be >= 1"),
            (dict(dependency="wishful"), "dependency must be one of"),
            (dict(time_window=(0.5, 0.2)), "time_window must satisfy"),
            (dict(time_window=(0.1, 1.0)), "time_window must satisfy"),
            (dict(positions=(1.0,)), "at least two positions"),
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError) as ctx:
                    make("btc_market", **kwargs)
                self.assertIn(expected, str(ctx.exception))

    def test_a_missing_data_file_says_how_to_build_it(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            make("btc_market", data_path="/nonexistent/btc.csv.gz")
        self.assertIn("make_btc_slice.py", str(ctx.exception))


# ------------------------------------------------------------------ tabular
class TabularEnvTests(unittest.TestCase):
    """The finite-SMDP engine validates its configs and samples them correctly."""

    def _config(self, **overrides):
        kwargs = dict(
            states=["s1", "s2"],
            actions=[A],
            transitions={("s1", A): [Transition("s2", 1.0, 1.0, 1.0)],
                         ("s2", A): [Transition("s1", 1.0, 1.0, 1.0)]},
            start_state="s1",
        )
        kwargs.update(overrides)
        return SMDPConfig(**kwargs)

    def test_probabilities_must_sum_to_one(self):
        with self.assertRaises(ValueError) as ctx:
            self._config(transitions={
                ("s1", A): [Transition("s2", 0.5, 1.0, 1.0),
                            Transition("s1", 0.2, 1.0, 1.0)],
                ("s2", A): [Transition("s1", 1.0, 1.0, 1.0)],
            })
        self.assertIn("sum to 0.7", str(ctx.exception))

    def test_a_transition_to_an_unlisted_state_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self._config(transitions={
                ("s1", A): [Transition("typo", 1.0, 1.0, 1.0)],
                ("s2", A): [Transition("s1", 1.0, 1.0, 1.0)],
            })
        self.assertIn("unlisted state 'typo'", str(ctx.exception))

    def test_a_negative_holding_time_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self._config(transitions={
                ("s1", A): [Transition("s2", 1.0, 1.0, -1.0)],
                ("s2", A): [Transition("s1", 1.0, 1.0, 1.0)],
            })
        self.assertIn("non-negative", str(ctx.exception))

    def test_a_dead_end_state_must_be_declared_terminal(self):
        with self.assertRaises(ValueError) as ctx:
            self._config(states=["s1", "s2", "s3"])
        self.assertIn("'s3'", str(ctx.exception))
        # Declaring it terminal is accepted.
        self._config(states=["s1", "s2", "s3"], terminal_states=["s3"])

    def test_an_unavailable_action_raises_rather_than_sampling_something_else(self):
        env = make("risk")
        env.reset(seed=0)
        env.step(SEEK)  # now in s2, where only "back" is legal
        with self.assertRaises(ValueError) as ctx:
            env.step(SEEK)
        self.assertIn("not available in state 's2'", str(ctx.exception))

    def test_sampling_honours_the_stated_probabilities(self):
        config = self._config(transitions={
            ("s1", A): [Transition("s2", 0.25, 1.0, 1.0),
                        Transition("s1", 0.75, 2.0, 1.0)],
            ("s2", A): [Transition("s1", 1.0, 0.0, 1.0)],
        })
        env = TabularSMDPEnv(config, name="weighted")
        env.reset(seed=7)
        first_branch = 0
        trials = 0
        for _ in range(8000):
            if env.state != "s1":
                env.step(A)
                continue
            _, reward, _, _, _ = env.step(A)
            trials += 1
            first_branch += reward == 1.0
        self.assertAlmostEqual(first_branch / trials, 0.25, delta=0.02)

    def test_info_reports_the_sampled_branch_probability(self):
        env = make("gemini")
        env.reset(seed=3)
        _, _, _, _, info = env.step(A)
        self.assertAlmostEqual(info["prob"], 0.5)

    def test_state_labels_are_what_the_agents_key_on(self):
        env = make("gemini")
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs, 0)                 # the observation is an index
        self.assertEqual(env.state_of(obs), "s1")  # the state is the label

    def test_reseeding_varies_the_reward_stream_across_seeds(self):
        first = run(make("sincoslog"), fixed(A), steps=50, seed=1)[1]
        again = run(make("sincoslog"), fixed(A), steps=50, seed=1)[1]
        other = run(make("sincoslog"), fixed(A), steps=50, seed=2)[1]
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)

    def test_reseeding_can_be_switched_off(self):
        first = run(make("sincoslog", reseed_processes=False), fixed(A),
                    steps=50, seed=1)[1]
        other = run(make("sincoslog", reseed_processes=False), fixed(A),
                    steps=50, seed=2)[1]
        self.assertEqual(first, other)

    def test_a_shared_process_is_reset_once_not_once_per_transition(self):
        shared = make_reward("linear", start=0.0, step=1.0)
        config = SMDPConfig(
            states=["s1", "s2"],
            actions=[A, B],
            transitions={("s1", A): [Transition("s2", 1.0, shared, 1.0)],
                         ("s1", B): [Transition("s2", 1.0, shared, 1.0)],
                         ("s2", A): [Transition("s1", 1.0, 0.0, 1.0)]},
            start_state="s1",
        )
        self.assertEqual(len(config.processes()), 1)
        env = TabularSMDPEnv(config, name="shared")
        env.reset(seed=0)
        rewards = []
        for _ in range(3):
            _, reward, _, _, _ = env.step(A)
            rewards.append(reward)
            env.step(A)
        self.assertEqual(rewards, [1.0, 2.0, 3.0])

    def test_is_mdp_is_set_from_the_holding_times(self):
        self.assertTrue(make("gemini").config.is_mdp is False)  # tau in {1, 19}
        self.assertTrue(make("hell_or_heaven").is_mdp)


# ----------------------------------------------------------------- two-path
class TwoPathEnvTests(unittest.TestCase):
    """The horizon fixture: 101 down the patient path, 100 down the greedy one."""

    def test_the_patient_path_totals_101(self):
        env = TwoPathEnv()
        rewards, taus = run(env, fixed(ACTION_A), steps=2, seed=0,
                            reset_between=False)
        self.assertEqual(sum(rewards), 101.0)
        self.assertEqual(sum(taus), 2.0)

    def test_the_greedy_path_totals_100(self):
        env = TwoPathEnv()
        rewards, _ = run(env, fixed(ACTION_B), steps=2, seed=0, reset_between=False)
        self.assertEqual(sum(rewards), 100.0)

    def test_the_second_decision_is_forced(self):
        env = TwoPathEnv()
        env.reset(seed=0)
        env.step(ACTION_B)
        self.assertEqual(env.get_available_actions(), [ACTION_B])
        with self.assertRaises(ValueError) as ctx:
            env.step(ACTION_A)
        self.assertIn("invalid in", str(ctx.exception))

    def test_the_secret_policy_takes_the_patient_path(self):
        env = TwoPathEnv()
        secret = env.secret()
        self.assertEqual(secret(STATE0), ACTION_A)
        rewards, _ = run(env, lambda e, s: secret(s), steps=2, seed=0,
                         reset_between=False)
        self.assertEqual(sum(rewards), 101.0)

    def test_reaching_a_terminal_state_terminates(self):
        env = TwoPathEnv()
        env.reset(seed=0)
        _, _, terminated, _, _ = env.step(ACTION_A)
        self.assertFalse(terminated)
        _, _, terminated, _, _ = env.step(ACTION_A)
        self.assertTrue(terminated)


if __name__ == "__main__":
    unittest.main()
