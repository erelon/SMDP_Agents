"""Run the tabular agents on the example environments and record the results.

    python -m examples.run --list
    python -m examples.run --env sincoslog --seeds 4
    python -m examples.run --family whack_a_mole --seeds 8
    python -m examples.run --all --seeds 8 --jobs 12

Protocol
--------
Every environment here is a *continuing* task: ``reset`` does not start a new
problem, it just re-zeros one. So the number of episodes is meaningless and the
runner uses **one trajectory per (environment, agent, seed)**. Episodic or
absorbing environments are auto-reset when they end and the trajectory continues.

The budget, its unit, and every agent's hyperparameters come from the environment's
**source repository** — see :mod:`examples.source_settings`; nothing here is tuned
by this repo. The unit varies because the sources disagree: whack-a-mole is bounded
by simulated *time*, which is the fair unit when holding times vary (an agent taking
fewer, longer actions must not be handed more experience), while the other three
sources bound by *decisions*. Each result records which unit was used.

Three metrics, all reward per unit time:

``lifetime_rate``
    Total reward over total time for the whole run. This is an *online* measure:
    it charges an agent for its learning transient and for its permanent
    ε-exploration, so a non-learning baseline pays no transient and can look
    competitive when the ceiling is low. It is the default comparison.
``window_rate``
    The same, over the post-warmup tail only (``--warmup-frac``). Shows the
    learned policy without the early transient.
``greedy_rate``
    A separate run on a fresh environment with exploration and learning off.
    The cleanest read on the quality of the policy that was learned, and the one
    to quote when the transient is not the question.

Each result also carries a learning curve — ``(time, cumulative reward, rho)`` at
``--curve-points`` checkpoints — which ``make_plots.py`` turns into rate-versus-
time and rho-versus-time curves.

Agents
------
The thirteen tabular agents, plus ``Oracle`` on the environments that expose a
``secret``. The deep and PPO agents are out of scope here: they need a network
per observation space, which is a different experiment.

Every agent routes its choice through ``get_available_actions``, so none of them
should ever propose an illegal action. The runner still checks, substitutes a
uniformly random legal action if one appears, and counts it in ``illegal_frac`` —
a non-zero value there means an agent regressed, and ``make_report.py`` flags it.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import multiprocessing
import time as wallclock
from concurrent.futures import ProcessPoolExecutor, as_completed
from random import Random
from typing import Any, Dict, List, Optional, Sequence

# One thread per worker, set before anything imports torch or numpy's BLAS. None of
# the tabular agents use a thread pool, so capping it costs nothing and keeps 28
# workers from each opening a pool sized for the whole machine.
for _threads in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_threads, "1")

from agents import (MAB, SMART, UCB, ContinuesMAB, ContinuosUCB,  # noqa: E402
                    ContinuousQLearning, ContinuousRLearning, Harmonic, Oracle,
                    QLearning, RandomAgent, RelaxedSMART, RLearning,
                    WeightedHarmonic)

try:  # belt and braces: fork inherits an already-initialised pool
    import torch

    torch.set_num_threads(1)
except ImportError:  # pragma: no cover - torch is a hard dependency of agents/
    pass

from . import correct_actions, source_settings
from .envs import ENVS, FAMILIES, EnvSpec, make

#: name -> (class, constructor kwargs). Every one is table-based.
AGENTS: Dict[str, tuple] = {
    "QLearning": (QLearning, {}),
    "ContinuousQLearning": (ContinuousQLearning, {}),
    "RLearning": (RLearning, {}),
    "ContinuousRLearning": (ContinuousRLearning, {}),
    "SMART": (SMART, {}),
    "RelaxedSMART": (RelaxedSMART, {}),
    "Harmonic": (Harmonic, {}),
    "WeightedHarmonic": (WeightedHarmonic, {}),
    "MAB": (MAB, {}),
    "ContinuesMAB": (ContinuesMAB, {}),
    "UCB": (UCB, {}),
    "ContinuosUCB": (ContinuosUCB, {}),
    "RandomAgent": (RandomAgent, {}),
}

#: Only built where the environment provides ``secret()``.
ORACLE = "Oracle"

#: Fallback time budget for a spec that declares none.
DEFAULT_BUDGET = 100_000
#: Greedy-evaluation budget, as a fraction of training, for the sources that ran no
#: greedy evaluation of their own.
DEFAULT_GREEDY_FRAC = 0.2
#: Hard cap on decisions, so an environment with tiny holding times cannot spin.
DEFAULT_MAX_STEPS = 2_000_000
DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "results")


# --------------------------------------------------------------- construction
def agent_names(env) -> List[str]:
    """The agents to run on ``env`` — the zoo, plus Oracle when there is a secret."""
    names = list(AGENTS)
    if env.secret() is not None:
        names.append(ORACLE)
    return names


def _accepted_arguments(cls) -> set:
    """Every keyword any ``__init__`` in ``cls``'s MRO names."""
    accepted = set()
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is not None:
            accepted |= set(inspect.signature(init).parameters)
    return accepted


def build_agent(name: str, env, seed: int,
                hyperparameters: Optional[Dict[str, Any]] = None):
    """Construct one agent, passing only the arguments its class accepts.

    ``hyperparameters`` are the source repository's settings for this environment
    (see :mod:`examples.source_settings`); anything absent keeps the library
    default.
    """
    if name == ORACLE:
        secret = env.secret()
        if secret is None:
            raise ValueError(f"{type(env).__name__} exposes no secret for the Oracle")
        agent = Oracle(name, action_space=env.action_list, env_secret=secret)
    else:
        cls, defaults = AGENTS[name]
        kwargs = {**defaults, **(hyperparameters or {})}
        params = inspect.signature(cls.__init__).parameters
        # Checked against the whole MRO, not just this class: every agent forwards
        # **kwargs up to Agent.__init__, which accepts and ignores what it does not
        # recognise, so a typo in the hyperparameter tables would otherwise be
        # applied silently as a default.
        unknown = sorted(set(kwargs) - _accepted_arguments(cls))
        if unknown:
            raise TypeError(
                f"{name} does not accept {unknown}; check the hyperparameters for "
                f"this environment in examples/source_settings.py")
        takes_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD
                           for p in params.values())
        extra = {key: value for key, value in (("env", env), ("seed", seed))
                 if key in params or takes_kwargs}
        agent = cls(name, action_space=env.action_list, **kwargs, **extra)
    agent.set_seed(seed)
    agent.reset()
    return agent


def safe_eval(agent, state, rng: Random, legal: Sequence[int]) -> int:
    """The greedy action, tolerating states never seen during training."""
    if hasattr(agent, "initialize_table"):
        agent.initialize_table(state)
    try:
        action = agent.eval(state)
    except (KeyError, ValueError):
        return rng.choice(list(legal))
    return action


def _spent(steps: int, clock: float, unit: str) -> float:
    """How much of the budget has been consumed, in its own unit."""
    return steps if unit == "steps" else clock


# ------------------------------------------------------------------ the loop
def train(agent, env, budget: float, seed: int, warmup_frac: float = 0.5,
          curve_points: int = 40, max_steps: int = DEFAULT_MAX_STEPS,
          unit: str = "time",
          epsilon_schedule: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One continuing trajectory of ``budget`` ``unit``, learning throughout.

    ``unit`` is ``"time"`` or ``"steps"``, whichever the environment's source
    repository used to bound its runs. ``epsilon_schedule`` is that source's decay,
    applied per decision — see :mod:`examples.source_settings`.
    """
    if unit not in ("time", "steps"):
        raise ValueError(f"unit must be 'time' or 'steps', got {unit!r}")
    rng = Random(seed ^ 0x5EED)
    obs, _ = env.reset(seed=seed)
    state = env.state_of(obs)

    total_reward = clock = 0.0
    warmup_time = budget * warmup_frac
    warm_reward = warm_clock = 0.0
    steps = illegal = resets = 0
    curve: List[List[float]] = []
    next_mark = budget / curve_points if curve_points else float("inf")

    while _spent(steps, clock, unit) < budget and steps < max_steps:
        if epsilon_schedule is not None and hasattr(agent, "exploration_rate"):
            # Applied per decision, as the source does, and only to agents that
            # have an exploration rate at all (UCB and the Oracle do not).
            agent.exploration_rate = source_settings.epsilon_at(
                epsilon_schedule, _spent(steps, clock, unit) / budget)
        legal = env.get_available_actions(state)
        action = agent.act(state)
        if action not in legal:
            # No shipped agent should reach this; see the module docstring.
            illegal += 1
            action = rng.choice(list(legal))

        obs, reward, terminated, truncated, info = env.step(action)
        next_state = env.state_of(obs)
        agent.learn(state, action, reward, next_state, info["tau"])

        total_reward += reward
        clock += info["tau"]
        steps += 1
        if _spent(steps, clock, unit) > warmup_time:
            warm_reward += reward
            warm_clock += info["tau"]
        if _spent(steps, clock, unit) >= next_mark:
            # (clock, cumulative reward, the agent's own rate estimate). The third
            # entry is what make_plots draws for the criterion environments, where
            # what rho converges to is the whole measurement.
            curve.append([round(clock, 4), round(total_reward, 4),
                          round(float(getattr(agent, "rho", 0.0)), 6)])
            next_mark += budget / curve_points

        if terminated or truncated:
            obs, _ = env.reset()
            state = env.state_of(obs)
            resets += 1
        else:
            state = next_state

    return {
        "final_exploration_rate": float(getattr(agent, "exploration_rate", 0.0)),
        "lifetime_rate": total_reward / clock if clock else 0.0,
        "window_rate": warm_reward / warm_clock if warm_clock else 0.0,
        "steps": steps,
        "elapsed": clock,
        "resets": resets,
        "illegal_frac": illegal / steps if steps else 0.0,
        "rho": float(getattr(agent, "rho", 0.0)),
        "states": len(getattr(agent, "q_table", {}) or {}),
        "curve": curve,
    }


def greedy(agent, env, budget: float, seed: int,
           max_steps: int = DEFAULT_MAX_STEPS,
           unit: str = "time") -> Dict[str, Any]:
    """Rate of the learned policy: a fresh run with no exploration and no learning."""
    rng = Random(seed ^ 0xC0FFEE)
    obs, _ = env.reset(seed=seed)
    state = env.state_of(obs)

    total_reward = clock = 0.0
    steps = illegal = 0
    while _spent(steps, clock, unit) < budget and steps < max_steps:
        legal = env.get_available_actions(state)
        action = safe_eval(agent, state, rng, legal)
        if action not in legal:
            illegal += 1
            action = rng.choice(list(legal))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        clock += info["tau"]
        steps += 1
        if terminated or truncated:
            obs, _ = env.reset()
            state = env.state_of(obs)
        else:
            state = env.state_of(obs)

    return {"greedy_rate": total_reward / clock if clock else 0.0,
            "greedy_steps": steps,
            "greedy_illegal_frac": illegal / steps if steps else 0.0}


def probe_choices(agent, env, env_name: str, seed: int) -> Dict[str, Any]:
    """What the trained agent would greedily do at each environment's decision state.

    This is the measurement the environments' own source uses on the trap
    environments — greedy action at a target state, scored against a known-correct
    one — rather than a reward rate, which on a trap rewards taking the bait. See
    :mod:`examples.correct_actions`.
    """
    probes = correct_actions.choices(env_name)
    if not probes:
        return {}
    rng = Random(seed ^ 0xB0BA)
    # Only the *observation* is stored — which action the agent settled on. Whether
    # that was correct is derived at report time against the current table, so
    # revising a criterion does not silently leave stale verdicts in old results.
    return {"choices": {f"choice_{state}": int(safe_eval(agent, state, rng,
                                                        env.get_available_actions(state)))
                        for state in probes}}


def run_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """One (environment, agent, seed) cell. Top-level so it can be pickled."""
    env_name, agent_name, seed = job["env"], job["agent"], job["seed"]
    env_kwargs = job.get("env_kwargs") or {}
    started = wallclock.perf_counter()

    env = make(env_name, **env_kwargs)
    agent = build_agent(agent_name, env, seed, job.get("hp"))
    unit = job["budget_unit"]
    result = train(agent, env, budget=job["budget"], seed=seed,
                   warmup_frac=job["warmup_frac"],
                   curve_points=job["curve_points"], max_steps=job["max_steps"],
                   unit=unit, epsilon_schedule=job.get("epsilon_schedule"))

    if job["greedy_budget"] > 0:
        # A fresh environment and a different seed, so the greedy measurement is
        # not the training trajectory's continuation.
        eval_env = make(env_name, **env_kwargs)
        result.update(greedy(agent, eval_env, budget=job["greedy_budget"],
                             seed=seed + 10_000, max_steps=job["max_steps"],
                             unit=unit))

    result.update(probe_choices(agent, env, env_name, seed))
    result.update(env=env_name, agent=agent_name, seed=seed, hp=job.get("hp") or {},
                  wallclock=round(wallclock.perf_counter() - started, 3))
    return result


# -------------------------------------------------------------------- driver
def resolve_budget(spec: EnvSpec, args) -> tuple:
    """``(budget, unit)`` for one environment, honouring its source's unit.

    A spec declares either ``budget`` (simulated time) or ``budget_steps``
    (decisions) — whichever its source repository used. ``--budget`` overrides the
    amount but not the unit; ``--budget-scale`` multiplies it.
    """
    if spec.budget_steps is not None:
        amount, unit = spec.budget_steps, "steps"
    elif spec.budget is not None:
        amount, unit = spec.budget, "time"
    else:
        amount, unit = DEFAULT_BUDGET, "time"
    if args.budget:
        amount = args.budget
    return float(amount) * args.budget_scale, unit


def resolve_greedy(spec: EnvSpec, budget: float, args) -> float:
    """The greedy-evaluation budget: the source's if it ran one, else a fraction."""
    if args.no_greedy:
        return 0.0
    greedy = spec.greedy if args.greedy_frac is None else args.greedy_frac
    return source_settings.greedy_budget(greedy, budget,
                                         fallback=DEFAULT_GREEDY_FRAC)


def build_jobs(env_names: Sequence[str], seeds: Sequence[int], args) -> List[Dict]:
    jobs = []
    for env_name in env_names:
        spec: EnvSpec = ENVS[env_name]
        budget, unit = resolve_budget(spec, args)
        greedy_budget = resolve_greedy(spec, budget, args)
        probe = make(env_name)
        for agent_name in agent_names(probe):
            if args.agents and agent_name not in args.agents:
                continue
            hp = spec.hyperparameters(agent_name)
            for seed in seeds:
                jobs.append(dict(env=env_name, agent=agent_name, seed=seed,
                                 budget=budget, budget_unit=unit, hp=hp,
                                 epsilon_schedule=spec.epsilon_schedule,
                                 warmup_frac=args.warmup_frac,
                                 curve_points=args.curve_points,
                                 max_steps=args.max_steps,
                                 greedy_budget=greedy_budget))
    return jobs


def collect(results: Sequence[Dict], env_names: Sequence[str], seeds: Sequence[int],
            args) -> Dict[str, Dict]:
    """Group the flat results into one record per environment."""
    grouped: Dict[str, Dict] = {}
    for env_name in env_names:
        spec = ENVS[env_name]
        budget, unit = resolve_budget(spec, args)
        grouped[env_name] = {
            "env": env_name,
            "family": spec.family,
            "note": spec.describe(),
            "budget": budget,
            "budget_unit": unit,
            "greedy_budget": resolve_greedy(spec, budget, args),
            "greedy_from_source": spec.greedy is not None and args.greedy_frac is None,
            "attributed": spec.attributed,
            "source_seeds": spec.source_seeds,
            "epsilon_schedule": spec.epsilon_schedule,
            "source": source_settings.PROTOCOL[spec.source]["source"],
            "hyperparameters": {name: spec.hyperparameters(name)
                                for name in spec.agent_kwargs},
            "seeds": list(seeds),
            "warmup_frac": args.warmup_frac,
            "agents": {},
        }
    for result in results:
        agents = grouped[result["env"]]["agents"]
        agents.setdefault(result["agent"], []).append(result)
    for record in grouped.values():
        for runs in record["agents"].values():
            runs.sort(key=lambda r: r["seed"])
    return grouped


def write(grouped: Dict[str, Dict], out_dir: str) -> List[str]:
    """One compact JSON per environment.

    Compact rather than indented: the learning curves are thousands of triples per
    file, and one number per line quadruples the size for no readability gain — the
    human-facing view is ``results/REPORT.md``.
    """
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for env_name, record in grouped.items():
        path = os.path.join(out_dir, f"{env_name}.json")
        with open(path, "w") as handle:
            json.dump(record, handle, separators=(",", ":"), sort_keys=True)
        written.append(path)
    return written


def resolve_envs(args) -> List[str]:
    chosen: List[str] = []
    if args.all:
        chosen = list(ENVS)
    for family in args.family or []:
        if family not in FAMILIES:
            raise SystemExit(f"unknown family {family!r}; known: {', '.join(FAMILIES)}")
        chosen += [n for n, s in ENVS.items() if s.family == family]
    for name in args.env or []:
        if name not in ENVS:
            raise SystemExit(f"unknown environment {name!r}; "
                             f"try --list")
        chosen.append(name)
    if not chosen:
        raise SystemExit("nothing to run: pass --env, --family or --all (or --list)")
    return list(dict.fromkeys(chosen))


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", action="append", help="environment name (repeatable)")
    parser.add_argument("--family", action="append",
                        help=f"whole family (repeatable): {', '.join(FAMILIES)}")
    parser.add_argument("--all", action="store_true", help="every environment")
    parser.add_argument("--agents", nargs="+", help="restrict to these agents")
    parser.add_argument("--seeds", type=int, default=8,
                        help="number of seeds, 1..N (default 8)")
    parser.add_argument("--budget", type=float,
                        help="time budget, overriding every per-environment default")
    parser.add_argument("--budget-scale", type=float, default=1.0,
                        help="multiply each environment's default budget")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                        help="hard cap on decisions per run")
    parser.add_argument("--warmup-frac", type=float, default=0.5,
                        help="fraction of the budget excluded from window_rate")
    parser.add_argument("--greedy-frac", type=float, default=None,
                        help="override the greedy evaluation budget, as a fraction "
                             "of the training one (default: the source's own, or "
                             f"{DEFAULT_GREEDY_FRAC} where the source ran none)")
    parser.add_argument("--no-greedy", action="store_true",
                        help="skip the greedy evaluation run")
    parser.add_argument("--curve-points", type=int, default=40,
                        help="learning-curve checkpoints per run (0 to disable)")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="worker processes")
    parser.add_argument("--out", default=DEFAULT_RESULTS_DIR,
                        help="directory for the per-environment JSON")
    parser.add_argument("--list", action="store_true",
                        help="list the environments and exit")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.list:
        from .envs import describe
        print(describe())
        print(f"\n{len(ENVS)} environments, {len(AGENTS)} agents "
              f"(+ Oracle where a secret exists)")
        return 0

    env_names = resolve_envs(args)
    seeds = list(range(1, args.seeds + 1))
    jobs = build_jobs(env_names, seeds, args)
    if not args.quiet:
        print(f"{len(jobs)} runs: {len(env_names)} environments x agents x "
              f"{len(seeds)} seeds, {args.jobs} workers", file=sys.stderr)

    started = wallclock.perf_counter()
    results: List[Dict] = []
    if args.jobs <= 1:
        for i, job in enumerate(jobs, 1):
            results.append(run_job(job))
            _progress(i, len(jobs), started, args.quiet)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs,
                                 mp_context=worker_context()) as pool:
            futures = [pool.submit(run_job, job) for job in jobs]
            for i, future in enumerate(as_completed(futures), 1):
                results.append(future.result())
                _progress(i, len(jobs), started, args.quiet)

    paths = write(collect(results, env_names, seeds, args), args.out)
    if not args.quiet:
        print(f"\nwrote {len(paths)} files to {args.out}", file=sys.stderr)
    return 0


#: Start methods to try, in order. **Never ``fork``**: ``import agents`` pulls in
#: torch, and forking a process that has torch loaded leaves the children in a state
#: where they run several times slower than the same work in a standalone process —
#: measured at 4.3x with only four workers, and it gets worse as workers are added.
#: ``forkserver`` forks from a clean intermediary instead, which costs one extra
#: process and restores full speed.
START_METHODS = ("forkserver", "spawn")


def worker_context():
    """A multiprocessing context whose workers are not forked from this process."""
    available = multiprocessing.get_all_start_methods()
    for method in START_METHODS:
        if method in available:
            return multiprocessing.get_context(method)
    raise RuntimeError(
        f"none of {START_METHODS} is available (have {available}); forking from a "
        f"torch-loaded parent would silently run several times slower")


def _progress(done: int, total: int, started: float, quiet: bool) -> None:
    if quiet:
        return
    spent = wallclock.perf_counter() - started
    eta = spent / done * (total - done)
    print(f"\r  {done}/{total} runs  {spent:6.1f}s elapsed  {eta:6.1f}s left",
          end="", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
