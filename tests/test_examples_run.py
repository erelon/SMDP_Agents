"""Tests for the example runner, report and plots (``examples/{run,make_*}.py``).

The environments themselves are covered by ``tests/test_examples.py``; this file
covers the harness around them — that the budget is honoured, that agents are
constructed with only the arguments they accept, that the greedy evaluation really
does not learn, that illegal-action substitution is counted and surfaced, and that
the report's aggregation survives diverged and tied results.
"""

import contextlib
import inspect
import io
import json
import math
import os
import shutil
import tempfile
import unittest
from random import Random

from examples import make_plots, make_report, run, source_settings
from examples.envs import ENVS, make
from examples.run import (AGENTS, ORACLE, agent_names, build_agent, collect,
                          evaluate, parse_args, resolve_budget, resolve_evaluation,
                          run_job, safe_eval, train, write)


def job(**overrides):
    base = dict(env="gemini", agent="QLearning", seed=1, budget=400.0,
                budget_unit="time", hp={}, epsilon_schedule=None,
                warmup_frac=0.5, curve_points=8, max_steps=100_000,
                eval_budget=80.0)
    base.update(overrides)
    return base


class AgentConstructionTests(unittest.TestCase):
    """Agents take different constructor arguments; the runner must respect that."""

    def test_every_agent_in_the_zoo_can_be_built(self):
        env = make("risk")
        for name in AGENTS:
            with self.subTest(agent=name):
                agent = build_agent(name, env, seed=3)
                self.assertEqual(agent.seed, 3)
                self.assertEqual(agent.action_space, env.action_list)

    def test_random_agent_takes_neither_env_nor_seed_yet_is_seeded(self):
        # RandomAgent.__init__ accepts only (name, action_space), so the runner
        # must seed it afterwards rather than through the constructor.
        first = build_agent("RandomAgent", make("risk"), seed=7)
        second = build_agent("RandomAgent", make("risk"), seed=7)
        other = build_agent("RandomAgent", make("risk"), seed=8)
        draws = lambda a: [a.act("s1") for _ in range(40)]  # noqa: E731
        self.assertEqual(draws(first), draws(second))
        self.assertNotEqual(draws(first), draws(other))

    def test_the_oracle_is_offered_only_where_a_secret_exists(self):
        self.assertIn(ORACLE, agent_names(make("two_path")))
        self.assertNotIn(ORACLE, agent_names(make("gemini")))

    def test_building_an_oracle_without_a_secret_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            build_agent(ORACLE, make("gemini"), seed=1)
        self.assertIn("no secret", str(ctx.exception))

    def test_the_oracle_follows_the_environment_secret(self):
        env = make("two_path")
        agent = build_agent(ORACLE, env, seed=1)
        obs, _ = env.reset(seed=1)
        self.assertEqual(agent.act(env.state_of(obs)), env.secret()(obs))


class TrainingLoopTests(unittest.TestCase):
    """The budget is in time units, and the curve is what the plots read."""

    def test_the_run_stops_on_the_time_budget_not_a_step_count(self):
        env = make("gemini", max_steps=None)
        agent = build_agent("QLearning", env, seed=1)
        result = train(agent, env, budget=500.0, seed=1, curve_points=0)
        self.assertGreaterEqual(result["elapsed"], 500.0)
        # gemini's holding times are 1 and 19, so 500 time units is far fewer
        # than 500 decisions.
        self.assertLess(result["steps"], 500)

    def test_the_step_cap_bounds_a_run_with_tiny_holding_times(self):
        env = make("gemini", max_steps=None)
        agent = build_agent("QLearning", env, seed=1)
        result = train(agent, env, budget=1e12, seed=1, curve_points=0,
                       max_steps=250)
        self.assertEqual(result["steps"], 250)

    def test_the_curve_carries_time_reward_and_rho(self):
        env = make("risk")
        agent = build_agent("Harmonic", env, seed=1)
        result = train(agent, env, budget=600.0, seed=1, curve_points=6)
        self.assertGreaterEqual(len(result["curve"]), 5)
        for point in result["curve"]:
            self.assertEqual(len(point), 3)
        times = [point[0] for point in result["curve"]]
        rewards = [point[1] for point in result["curve"]]
        self.assertEqual(times, sorted(times))
        self.assertEqual(rewards, sorted(rewards))  # risk pays only positives
        self.assertAlmostEqual(result["curve"][-1][2], result["rho"], places=6)

    def test_the_window_excludes_the_warmup(self):
        env = make("risk")
        agent = build_agent("SMART", env, seed=1)
        result = train(agent, env, budget=2000.0, seed=1, warmup_frac=0.5,
                       curve_points=0)
        self.assertGreater(result["window_rate"], 0.0)
        self.assertNotAlmostEqual(result["window_rate"], result["lifetime_rate"],
                                  places=9)

    def test_the_smoothers_split_on_high_time_variance(self):
        # What high_time_variance exists to show, end to end. The cumulative rate
        # is exact and keeps the option on every seed; the two fixed-gain
        # smoothers both overshoot the 1.9 threshold sometimes and drop it, and
        # the one that forgets per unit time drops it less often than the one
        # that forgets per transition. Pooled over seeds, not per seed: which
        # individual seeds flip is noisy.
        kept = {}
        for name in ("SMART", "SmoothedSMART", "RelaxedSMART"):
            kept[name] = 0
            for seed in range(1, 13):
                env = make("high_time_variance")
                agent = build_agent(name, env, seed)
                train(agent, env, budget=20_000, seed=seed, unit="steps",
                      curve_points=0)
                choice = safe_eval(agent, "s0", Random(seed),
                                   env.get_available_actions("s0"))
                kept[name] += choice == 0
        self.assertEqual(kept["SMART"], 12)
        self.assertGreater(kept["SmoothedSMART"], kept["RelaxedSMART"])

    def test_no_agent_needs_its_actions_overridden(self):
        # Was: the bandits and RandomAgent drew from the full action space, so on
        # risk (whose s2 allows only "back") the runner had to substitute a legal
        # action for up to half their decisions. Now every agent routes its choice
        # through get_available_actions, so nothing is ever overridden.
        for name in AGENTS:
            with self.subTest(agent=name):
                env = make("risk")
                result = train(build_agent(name, env, seed=1), env, budget=900.0,
                               seed=1, curve_points=0)
                self.assertEqual(result["illegal_frac"], 0.0)

    def test_the_substitution_safety_net_still_counts_a_bad_choice(self):
        class Stubborn:
            """An agent that ignores legality entirely."""

            rho = 0.0
            q_table = {}

            def act(self, state):
                return 0  # "seek", illegal in s2

            def learn(self, *args):
                pass

        env = make("risk")
        result = train(Stubborn(), env, budget=900.0, seed=1, curve_points=0)
        self.assertGreater(result["illegal_frac"], 0.3)

    def test_an_episodic_environment_is_restarted_and_counted(self):
        env = make("two_path")
        agent = build_agent("QLearning", env, seed=1)
        result = train(agent, env, budget=100.0, seed=1, curve_points=0)
        self.assertGreater(result["resets"], 10)

    def test_evaluation_does_not_learn(self):
        env = make("risk")
        agent = build_agent("QLearning", env, seed=1)
        train(agent, env, budget=600.0, seed=1, curve_points=0)
        before = json.dumps({str(k): v for k, v in agent.q_table.items()},
                            sort_keys=True)
        steps_before = agent.step_count

        result = evaluate(agent, make("risk"), budget=600.0, seed=99)
        after = json.dumps({str(k): v for k, v in agent.q_table.items()},
                           sort_keys=True)
        self.assertEqual(before, after)
        self.assertEqual(agent.step_count, steps_before)
        self.assertGreater(result["eval_steps"], 0)

    def test_evaluation_tolerates_unseen_states(self):
        # A fresh agent has an empty table, so every state is unseen.
        env = make("wam_smdp_until_whacked_whack")
        agent = build_agent("QLearning", env, seed=1)
        result = evaluate(agent, env, budget=200.0, seed=1)
        self.assertGreater(result["eval_steps"], 0)


class JobTests(unittest.TestCase):
    """One (environment, agent, seed) cell, and the files it lands in."""

    def test_a_job_returns_the_expected_fields(self):
        result = run_job(job())
        for key in ("lifetime_rate", "window_rate", "eval_rate", "rho", "steps",
                    "elapsed", "resets", "illegal_frac", "states", "curve",
                    "env", "agent", "seed", "wallclock"):
            self.assertIn(key, result)
        self.assertEqual((result["env"], result["agent"], result["seed"]),
                         ("gemini", "QLearning", 1))

    def test_skipping_the_greedy_run_omits_its_fields(self):
        result = run_job(job(eval_budget=0.0))
        self.assertNotIn("eval_rate", result)

    def test_results_group_by_environment_and_sort_by_seed(self):
        args = parse_args(["--seeds", "2"])
        results = [run_job(job(seed=seed, agent=agent, curve_points=2))
                   for seed in (2, 1) for agent in ("QLearning", "SMART")]
        grouped = collect(results, ["gemini"], [1, 2], args)
        self.assertEqual(set(grouped["gemini"]["agents"]), {"QLearning", "SMART"})
        self.assertEqual([r["seed"] for r in grouped["gemini"]["agents"]["SMART"]],
                         [1, 2])
        self.assertEqual(grouped["gemini"]["family"], "criterion")

    def test_written_files_reload_as_json(self):
        args = parse_args(["--seeds", "1"])
        grouped = collect([run_job(job(curve_points=2))], ["gemini"], [1], args)
        directory = tempfile.mkdtemp()
        try:
            paths = write(grouped, directory)
            self.assertEqual(len(paths), 1)
            with open(paths[0]) as handle:
                reloaded = json.load(handle)
            self.assertEqual(reloaded["env"], "gemini")
            self.assertEqual(make_report.load(directory)[0]["env"], "gemini")
        finally:
            shutil.rmtree(directory)


def record(env="demo", family="criterion", agents=None, budget=1000.0,
           seeds=(1, 2, 3)):
    return {"env": env, "family": family, "budget": budget, "seeds": list(seeds),
            "warmup_frac": 0.5, "eval_frac": 0.2, "note": "a note",
            "agents": agents or {}}


def runs(values, key="lifetime_rate", illegal=0.0, rho=1.0, curve=None):
    return [{key: value, "rho": rho, "states": 3, "resets": 0,
             "illegal_frac": illegal, "window_rate": value,
             "eval_rate": value, "seed": i + 1,
             "curve": curve or [[1.0, value, rho], [2.0, 2 * value, rho]]}
            for i, value in enumerate(values)]


class ReportAggregationTests(unittest.TestCase):
    """Aggregation has to survive diverged, tied and substituted results."""

    def test_a_non_finite_value_is_skipped_rather_than_poisoning_the_mean(self):
        sample = [{"rho": 1.0}, {"rho": float("nan")}, {"rho": 3.0}]
        mean, _ = make_report.summarise(sample, "rho")
        self.assertAlmostEqual(mean, 2.0)
        self.assertEqual(make_report.diverged(sample, "rho"), 1)

    def test_all_non_finite_reports_nan_and_formats_as_a_dash(self):
        sample = [{"rho": float("inf")}, {"rho": float("nan")}]
        mean, _ = make_report.summarise(sample, "rho")
        self.assertTrue(math.isnan(mean))
        self.assertEqual(make_report.fmt(mean), "—")

    def test_the_bandits_are_excluded_but_random_agent_is_not(self):
        heavy = runs([1.0, 1.0], illegal=0.5)
        self.assertTrue(make_report.distorted("UCB", heavy))
        self.assertTrue(make_report.distorted("EpsilonGreedyMAB", heavy))
        self.assertFalse(make_report.distorted("RandomAgent", heavy))
        self.assertFalse(make_report.distorted("UCB", runs([1.0], illegal=0.0)))

    def test_an_excluded_agent_cannot_win(self):
        data = record(agents={"UCB": runs([9.0, 9.0], illegal=0.9),
                              "SMART": runs([1.0, 1.0])})
        report = make_report.build([data], "lifetime_rate")
        self.assertIn("SMART", report.split("## Win tally")[0])
        self.assertIn("| UCB | `demo` | 90% | yes |", report)

    def test_a_three_way_tie_is_named_rather_than_broken(self):
        tied = record(agents={name: runs([2.0, 2.0]) for name in ("A", "B", "C")})
        report = make_report.build([tied], "lifetime_rate")
        self.assertIn("3-way tie", report)
        # A tie contributes no win to the tally.
        self.assertIn("No comparable results.", report.split("## Win tally")[1])

    def test_a_clear_winner_is_named_and_tallied(self):
        data = record(agents={"Harmonic": runs([5.0, 5.1]),
                              "QLearning": runs([1.0, 1.1]),
                              "SMART": runs([0.5, 0.6])})
        report = make_report.build([data], "lifetime_rate")
        self.assertIn("**Harmonic**", report)
        self.assertIn("| Harmonic | 1 |", report)

    def test_a_diverged_rho_is_called_out(self):
        data = record(agents={"ContinuousRLearning":
                              runs([1.0, 1.0], rho=float("nan"))})
        report = make_report.build([data], "lifetime_rate")
        self.assertIn("Diverged rho estimates", report)
        self.assertIn("diverged 2/2", report)

    def test_no_results_is_reported_not_crashed(self):
        self.assertIn("No result files found", make_report.build([], "lifetime_rate"))

    def test_records_are_ordered_by_family(self):
        directory = tempfile.mkdtemp()
        try:
            for env, family in (("z", "whack_a_mole"), ("a", "criterion")):
                with open(os.path.join(directory, f"{env}.json"), "w") as handle:
                    json.dump(record(env=env, family=family,
                                     agents={"SMART": runs([1.0])}), handle)
            loaded = make_report.load(directory)
            self.assertEqual([r["family"] for r in loaded],
                             ["criterion", "whack_a_mole"])
        finally:
            shutil.rmtree(directory)


class PlotTests(unittest.TestCase):
    """The curve maths, and that figures actually get written."""

    def test_segment_rates_are_per_interval_not_cumulative(self):
        curve = [[10.0, 10.0, 0.0], [20.0, 40.0, 0.0], [30.0, 40.0, 0.0]]
        times, rates = make_plots.segment_rates(curve)
        self.assertEqual(times, [10.0, 20.0, 30.0])
        self.assertEqual(rates, [1.0, 3.0, 0.0])

    def test_a_zero_length_segment_is_dropped(self):
        times, rates = make_plots.segment_rates([[5.0, 5.0, 0.0], [5.0, 9.0, 0.0]])
        self.assertEqual(times, [5.0])
        self.assertEqual(rates, [1.0])

    def test_the_rho_trace_ignores_curves_recorded_without_it(self):
        times, rhos = make_plots.rho_trace([[1.0, 1.0, 7.0], [2.0, 2.0, 6.5]])
        self.assertEqual((times, rhos), ([1.0, 2.0], [7.0, 6.5]))
        self.assertEqual(make_plots.rho_trace([[1.0, 1.0], [2.0, 2.0]]), ([], []))

    def test_figures_are_written_for_a_record(self):
        data = record(agents={"Harmonic": runs([5.0, 5.1]),
                              "SMART": runs([1.0, 1.1])})
        directory = tempfile.mkdtemp()
        try:
            with open(os.path.join(directory, "demo.json"), "w") as handle:
                json.dump(data, handle)
            out = os.path.join(directory, "plots")
            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = make_plots.main(["--results", directory, "--out", out])
            self.assertEqual(exit_code, 0)
            written = sorted(os.listdir(out))
            self.assertIn("demo_learning.png", written)
            self.assertIn("demo_rho.png", written)
            self.assertIn("family_criterion.png", written)
        finally:
            shutil.rmtree(directory)

    def test_an_agent_with_no_rho_is_omitted_from_the_rho_panel(self):
        data = record(agents={"QLearning": runs([1.0, 1.0], rho=0.0)})
        directory = tempfile.mkdtemp()
        try:
            with open(os.path.join(directory, "demo.json"), "w") as handle:
                json.dump(data, handle)
            out = os.path.join(directory, "plots")
            with contextlib.redirect_stdout(io.StringIO()):
                make_plots.main(["--results", directory, "--out", out])
            # Nothing to draw on the rho panel, so no rho figure at all.
            self.assertNotIn("demo_rho.png", os.listdir(out))
            self.assertIn("demo_learning.png", os.listdir(out))
        finally:
            shutil.rmtree(directory)

    def test_missing_results_are_reported(self):
        directory = tempfile.mkdtemp()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(make_plots.main(["--results", directory]), 1)
        finally:
            shutil.rmtree(directory)


if __name__ == "__main__":
    unittest.main()


class SourceSettingsTests(unittest.TestCase):
    """Budgets and hyperparameters must match the sources and actually apply."""

    def test_every_registered_env_names_a_known_source(self):
        for name, spec in ENVS.items():
            with self.subTest(env=name):
                self.assertIn(spec.source, source_settings.PROTOCOL)

    def test_each_env_declares_exactly_one_budget_unit(self):
        for name, spec in ENVS.items():
            with self.subTest(env=name):
                declared = [spec.budget is not None, spec.budget_steps is not None]
                self.assertEqual(sum(declared), 1,
                                 f"{name} must declare a time budget or a step "
                                 f"budget, not both or neither")

    def test_whack_a_mole_is_the_only_time_bounded_family(self):
        # Its source (whackAmole/run_all.py) bounds by simulated time; the other
        # three bound by decisions.
        for name, spec in ENVS.items():
            with self.subTest(env=name):
                if spec.family == "whack_a_mole":
                    self.assertIsNotNone(spec.budget)
                else:
                    self.assertIsNotNone(spec.budget_steps)

    def test_the_whack_a_mole_budgets_match_the_source_table(self):
        # whackAmole/benchmark.py::CONFIG_BUDGET
        self.assertEqual(ENVS["wam_mdp_until_whacked_whack"].budget, 100_000)
        self.assertEqual(ENVS["wam_smdp_until_whacked_whack"].budget, 200_000)
        self.assertEqual(ENVS["wam_smdp_up_down_whack_downed"].budget, 600_000)

    def test_the_step_budgets_match_their_sources(self):
        # PythonProject3/main.py runs sincoslog as 10 episodes x 1,000 steps, and
        # the engine resets every drifting process on reset, so 1,000 steps is also
        # the drift horizon.
        self.assertEqual(ENVS["sincoslog"].budget_steps, 10_000)
        self.assertEqual(ENVS["sincoslog"].build().max_steps, 1_000)
        # run_smdp_experiment.py: 100 x 10,000, and the only config live there.
        self.assertEqual(ENVS["ratio_vs_step_rate"].budget_steps, 1_000_000)
        self.assertEqual(ENVS["ratio_vs_step_rate"].build().max_steps, 10_000)
        # TimeBasedAgentsComparer/configs/config.yaml: 1,000 decisions.
        self.assertEqual(ENVS["stateless"].budget_steps, 1_000)
        # BtcSwarm: one pass through the price series.
        self.assertEqual(ENVS["btc_market"].budget_steps, 32_900)

    def test_the_unattributed_environments_are_marked_as_such(self):
        # The source has these commented out at its only call site — and two_path
        # and high_time_variance have no source at all — so their budget is this
        # repo's choice and must not read as provenance.
        for name in ("gemini", "feinberg", "hell_or_heaven", "bonus", "schwartz",
                     "risk", "non_stationary", "two_path", "high_time_variance"):
            with self.subTest(env=name):
                self.assertFalse(ENVS[name].attributed)
        for name in ("sincoslog", "ratio_vs_step_rate", "stateless", "btc_market",
                     "wam_smdp_up_down_whack"):
            with self.subTest(env=name):
                self.assertTrue(ENVS[name].attributed)

    def test_the_counterexample_uses_the_source_probability(self):
        # run_smdp_experiment.py:68 -> counterexample_ratio_vs_expected_rate(p=0.1)
        env = ENVS["ratio_vs_step_rate"].build()
        probs = sorted(t.constant("prob")
                       for t in env.config.transitions[("s0", 0)])
        self.assertEqual(probs, [0.1, 0.9])

    def test_the_shifting_env_uses_the_source_shift_range(self):
        # configs/config.yaml:24-29 -> _shift_max: 2.0, not the class default of 1.0
        self.assertEqual(ENVS["shifting_uneven"].build().shift_max, 2.0)

    def test_the_epsilon_schedule_is_applied_only_where_the_source_had_one(self):
        self.assertIsNotNone(ENVS["stateless"].epsilon_schedule)
        # BtcSwarm's epsilon_decay is a dead parameter, read nowhere in that repo.
        self.assertIsNone(ENVS["btc_market"].epsilon_schedule)
        self.assertIsNone(ENVS["sincoslog"].epsilon_schedule)

    def test_the_epsilon_schedule_decays_across_the_run(self):
        schedule = ENVS["stateless"].epsilon_schedule
        env = make("stateless")
        agent = build_agent("SMART", env, seed=1,
                            hyperparameters=ENVS["stateless"].hyperparameters("SMART"))
        self.assertEqual(agent.exploration_rate, 0.1)  # from the config
        result = train(agent, env, budget=1_000, seed=1, curve_points=0,
                       unit="steps", epsilon_schedule=schedule)
        # Starts at the schedule's start and finishes at its end, not at the
        # constructed value.
        self.assertAlmostEqual(result["final_exploration_rate"], 0.0, places=3)

    def test_evaluation_budgets_come_from_the_source_where_it_ran_one(self):
        args = parse_args([])
        # whackAmole used a fixed 20,000, not a fraction.
        wam = ENVS["wam_mdp_up_down_whack"]
        self.assertEqual(resolve_evaluation(wam, 600_000.0, args), 20_000.0)
        # TimeBased used eval_steps == episodes, i.e. 100% of training.
        self.assertEqual(resolve_evaluation(ENVS["stateless"], 1_000.0, args), 1_000.0)
        # PythonProject3 ran none, so the runner's own fraction applies.
        self.assertEqual(resolve_evaluation(ENVS["sincoslog"], 10_000.0, args), 2_000.0)
        self.assertEqual(resolve_evaluation(ENVS["sincoslog"], 10_000.0,
                                        parse_args(["--no-eval"])), 0.0)

    def test_every_hyperparameter_is_accepted_by_its_agent(self):
        # A typo here would otherwise be silently swallowed by **kwargs.
        for env_name, spec in ENVS.items():
            for agent_name, kwargs in spec.agent_kwargs.items():
                with self.subTest(env=env_name, agent=agent_name):
                    self.assertIn(agent_name, AGENTS)
                    cls = AGENTS[agent_name][0]
                    accepted = set()
                    for klass in cls.__mro__:
                        accepted |= set(inspect.signature(
                            klass.__init__).parameters) if hasattr(
                            klass, "__init__") else set()
                    self.assertLessEqual(set(kwargs), accepted)

    def test_the_hyperparameters_reach_the_constructed_agent(self):
        env = make("stateless")
        agent = build_agent("Harmonic", env, seed=1,
                            hyperparameters=ENVS["stateless"].hyperparameters("Harmonic"))
        # TimeBasedAgentsComparer's HarmonicROLAgent block.
        self.assertEqual(agent.learning_rate, 0.3)
        self.assertEqual(agent.rho_learning_rate, 0.3)
        self.assertEqual(agent.exploration_rate, 0.1)

    def test_the_sources_really_do_disagree_about_hyperparameters(self):
        # The whole reason for this module: one global default would reproduce none
        # of the source experiments.
        harmonic = {name: ENVS[name].hyperparameters("Harmonic")["rho_learning_rate"]
                    for name in ("gemini", "stateless", "btc_market",
                                 "wam_smdp_up_down_whack")}
        self.assertEqual(len(set(harmonic.values())), len(harmonic), harmonic)

    def test_an_unknown_hyperparameter_is_refused(self):
        env = make("risk")
        with self.assertRaises(TypeError) as ctx:
            build_agent("RandomAgent", env, seed=1,
                        hyperparameters={"nonsense": 1.0})
        self.assertIn("nonsense", str(ctx.exception))

    def test_an_unknown_source_or_config_is_refused(self):
        with self.assertRaises(KeyError):
            source_settings.settings("gemini", "no_such_source")
        with self.assertRaises(KeyError):
            source_settings.whack_a_mole_hp("wam_mdp_nonexistent_whack")

    def test_resolve_budget_honours_the_units_and_the_override(self):
        args = parse_args([])
        self.assertEqual(resolve_budget(ENVS["gemini"], args), (100_000.0, "steps"))
        self.assertEqual(resolve_budget(ENVS["wam_mdp_up_down_whack"], args),
                         (600_000.0, "time"))
        scaled = parse_args(["--budget-scale", "0.5"])
        self.assertEqual(resolve_budget(ENVS["gemini"], scaled), (50_000.0, "steps"))
        # --budget replaces the amount but not the unit.
        forced = parse_args(["--budget", "77"])
        self.assertEqual(resolve_budget(ENVS["gemini"], forced), (77.0, "steps"))
        self.assertEqual(resolve_budget(ENVS["wam_mdp_up_down_whack"], forced),
                         (77.0, "time"))

    def test_a_step_budget_bounds_by_decisions(self):
        env = make("gemini", max_steps=None)
        agent = build_agent("QLearning", env, seed=1)
        result = train(agent, env, budget=300, seed=1, curve_points=0, unit="steps")
        self.assertEqual(result["steps"], 300)
        # The same amount as a time budget buys far fewer decisions, since gemini's
        # holding times are 1 and 19.
        env = make("gemini", max_steps=None)
        timed = train(build_agent("QLearning", env, seed=1), env, budget=300,
                      seed=1, curve_points=0, unit="time")
        self.assertLess(timed["steps"], 300)

    def test_an_unknown_budget_unit_is_refused(self):
        env = make("gemini")
        with self.assertRaises(ValueError) as ctx:
            train(build_agent("QLearning", env, seed=1), env, budget=10, seed=1,
                  unit="furlongs")
        self.assertIn("must be 'time' or 'steps'", str(ctx.exception))


class WorkerPoolTests(unittest.TestCase):
    """Workers must not be forked from a torch-loaded parent."""

    def test_the_worker_context_is_never_fork(self):
        # Forking after `import agents` (which imports torch) leaves the children
        # running several times slower than the same job standalone — 4.3x with
        # four workers, worse with more.
        self.assertNotIn("fork", run.START_METHODS)
        self.assertIn(run.worker_context().get_start_method(), run.START_METHODS)
