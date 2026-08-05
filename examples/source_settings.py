"""The training budgets and hyperparameters each environment's source repo used.

Nothing here is tuned by this repository. Every number is copied from the script or
config file that ran the original experiment, and where a source does not pin a
number this module says so rather than inventing one.

Four things vary by source, and all four matter:

**The budget's unit.** ``whackAmole`` bounds a run by simulated *time*, which is the
fair unit when holding times vary — an agent taking fewer, longer actions must not
be handed more experience. The other three bound by *decisions*. An
:class:`~examples.envs.EnvSpec` therefore carries either ``budget`` or
``budget_steps``, never both.

**The episode length.** This is not cosmetic for the drifting environments: the
finite-SMDP engine resets every reward and duration process on ``reset``, so the
episode length *is* the drift horizon. ``sincoslog`` run as 10 episodes of 1,000
steps is a different environment from the same config run as one 10,000-step
trajectory — in the second the sin/log multiplier compounds without ever restarting.

**The greedy-evaluation budget.** One source disabled it, one used 100% of the
training budget, one used a fixed 20,000, one never had one.

**The hyperparameters, per agent per environment.** These differ by two orders of
magnitude between sources — ``rho_learning_rate`` is 0.003 in one config and 0.3 in
another, and the market work used ``learning_rate=0.001`` where the tabular work
used 0.2. Running everything at one global default would reproduce none of them.

Sources
-------
``PythonProject3/main.py`` (``python_project_3_main``)
    Runs the ``SMDPConfigFactory`` sin/cos-log family. **10 episodes x 1,000
    steps**, ``lr=0.2, er=0.2, beta=0.2``. Four agents: the two Harmonics, SMART and
    the EMA variant. No seed loop — one run per (config, agent), with the episode
    index used as the environment seed.
``PythonProject3/run_smdp_experiment.py`` (``python_project_3_counterexample``)
    100 episodes x 10,000 steps, ``lr=0.2, er=0.1, beta=0.01``. Only *one* config is
    live at HEAD: ``counterexample_ratio_vs_expected_rate(p=0.1)``.
``PythonProject3`` ambient (``python_project_3_unattributed``)
    The other hardcoded configs — gemini, feinberg, hell-or-heaven, bonus, schwartz
    — are all commented out at the single call site and have **no attributable
    protocol**. Git history shows the budget changing wildly whenever one was last
    active (gemini at 100x100, 5000x100 and 200x1000; bonus at 100x200, 10000x20 and
    1000x500; feinberg at 1000x500), and hell-or-heaven and schwartz were never
    uncommented in any commit. They get the ambient
    ``run_smdp_experiment.py`` hyperparameters and a budget chosen here, flagged as
    such by :data:`UNATTRIBUTED`.
``TimeBasedAgentsComparer/configs/config.yaml`` (``time_based``)
    **1,000 decisions** — one "episode" there is a single decision — then 1,000
    greedy decisions, over 30 seeds. Hyperparameters are per agent, and epsilon
    decays linearly from 0.3 to 0 across the whole run.
``whackAmole/benchmark.py`` + ``results/best_hp.json`` (``whack_a_mole``)
    100k-600k **time units** per configuration from that repo's convergence study,
    30 seeds, ``warmup_frac=0.5``, and per-(configuration, agent) hyperparameters
    from its own grid search over 3 seeds. Its canonical run *disabled* greedy
    evaluation, so its reports have no ``greedy_rate``.
``BtcSwarm/default-config.json`` (``btc_swarm``)
    One pass over the price series, 30 seeds, ``lr=0.001, beta=0.1, er=0.2``
    constant. No warmup and no greedy evaluation.

Deviations from the sources, all deliberate and all also noted at the point of use:

* The multichain configs restart far sooner than any source episode length. Both
  their loops absorb, so at a 10,000-step episode an agent gets one decision and
  9,999 consequences and every agent scores the same.
* ``btc_market`` uses relative rather than absolute returns and couples the holding
  time to the return; the source used absolute dollars and drew the holding time at
  random, independently of the return.
* Seed counts here are set on the command line, not by the source. Every source
  above used 30 (or, for ``PythonProject3``, one).
* ``epsilon_decay`` and ``tau`` in ``BtcSwarm``'s config are dead parameters — read
  nowhere in that code — so no schedule is applied for it.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
WHACK_A_MOLE_HP = os.path.join(HERE, "data", "whack_a_mole_hp.json")

# --------------------------------------------------------------------- sources
#: ``PythonProject3/main.py:143-146`` — the sin/cos-log family.
PP3_MAIN_HP = {
    "learning_rate": 0.2, "exploration_rate": 0.2, "rho_learning_rate": 0.2,
    "with_rho_trick": True,
}
#: ``PythonProject3/run_smdp_experiment.py:82-85`` — the counterexample, and the
#: ambient setting for the configs that have no protocol of their own.
PP3_EXPERIMENT_HP = {
    "learning_rate": 0.2, "exploration_rate": 0.1, "rho_learning_rate": 0.01,
    "with_rho_trick": True,
}


def _pp3(shared: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """One shared setting applied to every agent, as those scripts do.

    They ran only the rate-based agents (and, commented out, Q-learning); the
    bandits keep the library defaults. ``discount_factor`` is never passed and is
    unused by the R-learning family anyway, since ``set_target`` returns
    ``(r - rho*tau) + next_q``.
    """
    rate_agents = ("RLearning", "ContinuousRLearning", "SMART", "RelaxedSMART",
                   "Harmonic", "WeightedHarmonic")
    plain = {"learning_rate": shared["learning_rate"],
             "exploration_rate": shared["exploration_rate"]}
    table: Dict[str, Dict[str, Any]] = {"QLearning": dict(plain),
                                        "ContinuousQLearning": dict(plain)}
    for name in rate_agents:
        table[name] = dict(shared)
    return table


#: ``TimeBasedAgentsComparer/configs/config.yaml``, per agent. The class mapping is
#: ``HarmonicRLAgent`` -> ``WeightedHarmonic`` (its HMA weight is the reward) and
#: ``HarmonicROLAgent`` -> ``Harmonic`` (weight 1), matching this library's split.
#: Agents marked (disabled) are ``enabled: false`` there — their settings are read
#: from the disabled block, so they are an extension of the source, not part of it.
TIME_BASED = {
    "QLearning": {"learning_rate": 0.1, "discount_factor": 0.99,          # disabled
                  "exploration_rate": 0.2},
    "ContinuousQLearning": {"learning_rate": 0.1, "discount_factor": 0.99,
                            "exploration_rate": 0.15, "_lambda": 1.0},
    "RLearning": {"learning_rate": 0.3, "exploration_rate": 0.1,
                  "rho_learning_rate": 0.03, "with_rho_trick": True},
    "ContinuousRLearning": {"learning_rate": 0.2, "exploration_rate": 0.1,  # disabled
                            "rho_learning_rate": 0.03, "with_rho_trick": True},
    "SMART": {"learning_rate": 0.3, "exploration_rate": 0.1,
              "rho_learning_rate": 0.3, "with_rho_trick": True},
    "RelaxedSMART": {"learning_rate": 0.3, "exploration_rate": 0.1,
                     "rho_learning_rate": 0.05, "with_rho_trick": True},
    "WeightedHarmonic": {"learning_rate": 0.3, "exploration_rate": 0.1,
                         "rho_learning_rate": 0.003, "with_rho_trick": True},
    "Harmonic": {"learning_rate": 0.3, "exploration_rate": 0.1,
                 "rho_learning_rate": 0.3, "with_rho_trick": True},
    "MAB": {"learning_rate": 0.1, "exploration_rate": 0.1},                # disabled
    "ContinuesMAB": {"learning_rate": 0.1, "exploration_rate": 0.1},       # disabled
    "UCB": {"exploration_constant": 5.0},
    "ContinuosUCB": {"exploration_constant": 5.0},
}

#: Agents ``TimeBasedAgentsComparer``'s canonical config has ``enabled: false``.
TIME_BASED_DISABLED = ("QLearning", "ContinuousRLearning", "MAB", "ContinuesMAB")

#: ``BtcSwarm/default-config.json``'s ``agent_params``. It ran three agents:
#: ``SMARTRLAgent`` (SMART), ``HarmonicRLAgent`` (WeightedHarmonic) and
#: ``SMARTEMARLAgent`` (RelaxedSMART). The same learning rate, beta and epsilon are
#: extended to the rest of the zoo so the family is measured on one setting.
#: ``discount_factor=0.99`` is in that config but dead for those three agents, since
#: ``ContinuousRLAgent.set_target`` overrides the only place it is used.
BTC_SWARM = {
    "QLearning": {"learning_rate": 0.001, "discount_factor": 0.99,
                  "exploration_rate": 0.2},
    "ContinuousQLearning": {"learning_rate": 0.001, "discount_factor": 0.99,
                            "exploration_rate": 0.2},
    "RLearning": {"learning_rate": 0.001, "exploration_rate": 0.2,
                  "rho_learning_rate": 0.1, "with_rho_trick": True},
    "ContinuousRLearning": {"learning_rate": 0.001, "exploration_rate": 0.2,
                            "rho_learning_rate": 0.1, "with_rho_trick": True},
    "SMART": {"learning_rate": 0.001, "exploration_rate": 0.2,
              "rho_learning_rate": 0.1, "with_rho_trick": True},
    "RelaxedSMART": {"learning_rate": 0.001, "exploration_rate": 0.2,
                     "rho_learning_rate": 0.1, "with_rho_trick": True},
    "Harmonic": {"learning_rate": 0.001, "exploration_rate": 0.2,
                 "rho_learning_rate": 0.1, "with_rho_trick": True},
    "WeightedHarmonic": {"learning_rate": 0.001, "exploration_rate": 0.2,
                         "rho_learning_rate": 0.1, "with_rho_trick": True},
    "MAB": {"exploration_rate": 0.2},
    "ContinuesMAB": {"exploration_rate": 0.2},
}

#: Sources whose budget for an environment is *not* recorded anywhere, so the number
#: below is this repo's choice. Surfaced in the report so it is never mistaken for
#: provenance.
UNATTRIBUTED = ("python_project_3_unattributed", "none")

#: Protocol per source. ``greedy`` is the evaluation budget: a float is a fraction of
#: the training budget, an int is an absolute amount in the same unit, and ``None``
#: means the source ran no greedy evaluation at all — the runner then measures one
#: anyway, since it runs on a fresh environment with learning off and so cannot
#: affect the trained agent, but the report marks it as not from the source.
PROTOCOL: Dict[str, Dict[str, Any]] = {
    "python_project_3_main": {
        "budget_steps": 10_000, "episode_steps": 1_000, "greedy": None,
        "seeds": 1, "hp": _pp3(PP3_MAIN_HP), "epsilon_schedule": None,
        "source": "PythonProject3/main.py (10 episodes x 1,000 steps)",
    },
    "python_project_3_counterexample": {
        "budget_steps": 1_000_000, "episode_steps": 10_000, "greedy": None,
        "seeds": 1, "hp": _pp3(PP3_EXPERIMENT_HP), "epsilon_schedule": None,
        "source": "PythonProject3/run_smdp_experiment.py (100 x 10,000 steps)",
    },
    "python_project_3_unattributed": {
        # No source protocol; see the module docstring. 100 episodes of 1,000 steps
        # is the middle of the range git history shows for these configs.
        "budget_steps": 100_000, "episode_steps": 1_000, "greedy": None,
        "seeds": 1, "hp": _pp3(PP3_EXPERIMENT_HP), "epsilon_schedule": None,
        "source": "PythonProject3 (no attributable protocol; budget chosen here)",
    },
    "time_based": {
        "budget_steps": 1_000, "episode_steps": None, "greedy": 1_000,
        "seeds": 30, "hp": TIME_BASED,
        # configs/config.yaml:12-16 — decay_episodes equals episodes, so the run
        # finishes greedy.
        "epsilon_schedule": {"start": 0.3, "end": 0.0, "decay_fraction": 1.0},
        "source": "TimeBasedAgentsComparer/configs/config.yaml (1,000 decisions)",
    },
    "whack_a_mole": {
        "budget_steps": None, "episode_steps": None, "greedy": 20_000,
        "seeds": 30, "hp": None, "epsilon_schedule": None,
        "source": "whackAmole/benchmark.py + results/best_hp.json",
    },
    "btc_swarm": {
        # One pass over the series; the exact count comes from the data length.
        "budget_steps": 32_900, "episode_steps": None, "greedy": None,
        "seeds": 30, "hp": BTC_SWARM, "epsilon_schedule": None,
        "source": "BtcSwarm/default-config.json (one pass over the series)",
    },
    "none": {
        "budget_steps": 20_000, "episode_steps": None, "greedy": 0.2,
        "seeds": 1, "hp": {}, "epsilon_schedule": None,
        "source": "this repository (no benchmark source)",
    },
}

#: ``whackAmole/benchmark.py::CONFIG_BUDGET``, in time units, from that repo's
#: convergence study: ``up_down`` with a whack-triggered reward needs roughly six
#: times as long to settle as the rest.
WHACK_A_MOLE_BUDGET = {
    "wam_mdp_until_whacked_whack": 100_000,
    "wam_mdp_until_whacked_whack_downed": 100_000,
    "wam_mdp_until_whacked_step_downed": 100_000,
    "wam_mdp_up_down_whack": 600_000,
    "wam_mdp_up_down_whack_downed": 600_000,
    "wam_mdp_up_down_step_downed": 100_000,
    "wam_smdp_until_whacked_whack": 200_000,
    "wam_smdp_until_whacked_whack_downed": 200_000,
    "wam_smdp_up_down_whack": 600_000,
    "wam_smdp_up_down_whack_downed": 600_000,
}

_whack_a_mole_hp: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None


def whack_a_mole_hp(env_name: str) -> Dict[str, Dict[str, Any]]:
    """That repo's tuned hyperparameters for one whack-a-mole configuration.

    Read from ``data/whack_a_mole_hp.json``, a verbatim copy of its
    ``results/best_hp.json`` — a grid search scored on mean ``lifetime_rate`` over
    3 seeds at each configuration's own budget. Its keys are the configuration tags
    (``SMDP_up_down_whack``); ours prefix them with ``wam_`` and lowercase them.
    """
    global _whack_a_mole_hp
    if _whack_a_mole_hp is None:
        with open(WHACK_A_MOLE_HP) as handle:
            _whack_a_mole_hp = json.load(handle)
    tag = env_name[len("wam_"):]
    kind, rest = tag.split("_", 1)
    key = f"{kind.upper()}_{rest}"
    if key not in _whack_a_mole_hp:
        raise KeyError(f"no recorded hyperparameters for {env_name!r} (looked for "
                       f"{key!r} in {os.path.basename(WHACK_A_MOLE_HP)})")
    return _whack_a_mole_hp[key]


def settings(env_name: str, source: str) -> Dict[str, Any]:
    """The budget, episode cap, greedy budget, hyperparameters and provenance.

    Returns ``budget``, ``budget_steps``, ``episode_steps``, ``greedy``,
    ``epsilon_schedule``, ``hp`` (agent name -> kwargs), ``source_seeds`` and
    ``source``. Exactly one of ``budget`` and ``budget_steps`` is set.
    """
    if source not in PROTOCOL:
        raise KeyError(f"unknown source {source!r}; known: {', '.join(PROTOCOL)}")
    protocol = PROTOCOL[source]
    common = {"episode_steps": protocol["episode_steps"],
              "greedy": protocol["greedy"],
              "epsilon_schedule": protocol["epsilon_schedule"],
              "source_seeds": protocol["seeds"],
              "source": protocol["source"],
              "attributed": source not in UNATTRIBUTED}
    if source == "whack_a_mole":
        budget = WHACK_A_MOLE_BUDGET.get(env_name)
        if budget is None:
            raise KeyError(f"no recorded budget for {env_name!r}")
        return {**common, "budget": budget, "budget_steps": None,
                "hp": whack_a_mole_hp(env_name)}
    return {**common, "budget": None, "budget_steps": protocol["budget_steps"],
            "hp": dict(protocol["hp"] or {})}


def epsilon_at(schedule: Optional[Dict[str, Any]], progress: float) -> Optional[float]:
    """The scheduled exploration rate at ``progress`` in ``[0, 1]`` through a run.

    Linear from ``start`` to ``end`` over ``decay_fraction`` of the budget, then held
    at ``end`` — the shape ``TimeBasedAgentsComparer``'s
    ``_build_epsilon_scheduler`` produces. ``None`` when the source has no schedule,
    in which case the agent's constructed exploration rate stands.
    """
    if not schedule:
        return None
    span = max(float(schedule.get("decay_fraction", 1.0)), 1e-12)
    fraction = min(max(progress, 0.0) / span, 1.0)
    start, end = float(schedule["start"]), float(schedule["end"])
    return start + (end - start) * fraction


def greedy_budget(greedy: Any, budget: float, fallback: float = 0.2) -> float:
    """Resolve a ``greedy`` protocol entry against a training budget.

    ``None`` (the source ran no greedy evaluation) falls back to ``fallback`` as a
    fraction of the training budget. A float is a fraction, an int an absolute
    amount in the budget's own unit.
    """
    if greedy is None:
        return budget * float(fallback)
    if isinstance(greedy, bool):  # guard: bools are ints in Python
        raise TypeError("greedy must be a number or None, not a bool")
    if isinstance(greedy, float):
        return budget * greedy
    return float(greedy)
