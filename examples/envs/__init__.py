"""The registry of example environments.

    from examples.envs import ENVS, make, names

    env = make("sincoslog")                 # or any name in names()
    env = make("wam_smdp_up_down_whack", max_time=50_000)

Every entry obeys the contract in :mod:`examples.envs.base`: a Gymnasium
environment reporting the holding time of each action in ``info["tau"]``, plus
``state_of`` for a hashable state, ``get_available_actions`` for legality, and
``secret`` where the optimal policy is known. ``make`` returns *unwrapped*
instances, so those hooks are reachable directly.

Environments are grouped by *family*, which is what the report aggregates over:

``criterion``
    Small graphs where the candidate definitions of average reward per unit time
    disagree, or where the estimators of one definition do; the docstrings state
    the competing values.
``risk``
    Same mean rate, different spread.
``horizon``
    Whether an agent looks far enough ahead to refuse a poisoned jackpot.
``drift``
    Non-stationary rewards and holding times, where the best action changes.
``rates``
    Reward earned at a rate over a random duration, so reward and holding time
    are strongly correlated and only the rate is worth comparing.
``whack_a_mole``
    The grid task, under a unit-time control (an MDP) and a distance-time control
    (an SMDP), so the pair isolates whether holding times vary.
``market``
    Real hourly Bitcoin bars: a messy, non-stationary reward stream whose holding
    times are derived from the same returns that decide the reward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .. import source_settings
from . import configs
from .base import SMDPEnv, check_smdp_env
from .btc_market import BtcMarketEnv
from .tabular import SMDPConfig, TabularSMDPEnv, Transition
from .two_path import TwoPathEnv
from .two_states import (CyclingUnevenTwoStates, EvenTwoStates,
                         LatentCyclingUnevenTwoStates, RateEnv,
                         ShiftingUnevenTwoStates, SlopeShiftingUnevenTwoStates,
                         StatelessRates, UnevenTwoStates)
from .whack_a_mole import (MOLE_DYNAMICS, REWARD_MODES, WhackAMoleMDP,
                           WhackAMoleSMDP, heuristic_policy, register_gym_ids)

#: The families an :class:`EnvSpec` may belong to, in the order the report groups
#: them. See the module docstring for what each one is testing.
FAMILIES = ("criterion", "risk", "horizon", "drift", "rates", "whack_a_mole",
            "market")

__all__ = [
    "ENVS", "EnvSpec", "FAMILIES", "make", "names", "specs", "describe",
    "SMDPEnv", "check_smdp_env",
    "SMDPConfig", "TabularSMDPEnv", "Transition",
    "TwoPathEnv", "WhackAMoleMDP", "WhackAMoleSMDP", "BtcMarketEnv",
    "RateEnv", "StatelessRates", "EvenTwoStates", "UnevenTwoStates",
    "CyclingUnevenTwoStates", "LatentCyclingUnevenTwoStates",
    "ShiftingUnevenTwoStates", "SlopeShiftingUnevenTwoStates",
    "heuristic_policy", "register_gym_ids",
]


@dataclass(frozen=True)
class EnvSpec:
    """How to build one registered environment, and what it is for."""

    name: str
    family: str
    build: Callable[..., SMDPEnv]
    note: str = ""
    #: Training budget in *time units*, for the sources that bound a run that way.
    #: Mutually exclusive with ``budget_steps``.
    budget: Optional[int] = None
    #: Training budget in *decisions*, for the sources that bound a run that way.
    #: See :mod:`examples.source_settings` for why the unit varies.
    budget_steps: Optional[int] = None
    #: Which source repository's protocol and hyperparameters apply; a key of
    #: ``source_settings.PROTOCOL``.
    source: str = "none"
    #: Post-training evaluation budget: a float is a fraction of the training budget, an int
    #: an absolute amount in the same unit, ``None`` if the source ran none (the
    #: runner then measures one anyway and the report says so).
    evaluation: Any = None
    #: Linear epsilon schedule the source applied, or ``None`` to hold the agent's
    #: constructed exploration rate fixed.
    epsilon_schedule: Optional[Dict[str, Any]] = None
    #: How many seeds the source itself used, for the report to note.
    source_seeds: int = 1
    #: Which metric this environment should be *compared* on, overriding the
    #: report's global choice. ``None`` uses the global one. Whack-a-mole sets
    #: ``eval_rate``: its runs are 100k-600k time units of permanent epsilon
    #: exploration, so a lifetime average is dominated by exploration cost rather
    #: than by the quality of the policy learned.
    metric: Optional[str] = None
    #: False when this repo chose the budget because the source pins none.
    attributed: bool = True
    #: agent name -> constructor kwargs, from that source. Agents absent from the
    #: mapping keep the library defaults.
    agent_kwargs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tags: tuple = field(default_factory=tuple)

    def hyperparameters(self, agent: str) -> Dict[str, Any]:
        """This source's constructor kwargs for ``agent``, or ``{}`` for defaults."""
        return dict(self.agent_kwargs.get(agent, {}))

    def __call__(self, **kwargs) -> SMDPEnv:
        return self.build(**kwargs)

    def describe(self) -> str:
        """A one-line description, taken from the config when not given here."""
        if self.note:
            return self.note
        env = self.build()
        note = getattr(getattr(env, "config", None), "note", "")
        if note:
            return note
        doc = (type(env).__doc__ or "").strip()
        return doc.splitlines()[0] if doc else ""


def _tabular(name: str, family: str, config_fn: Callable[..., SMDPConfig],
             source: str = "python_project_3", episode_steps: Optional[int] = None,
             tags: tuple = (), **config_defaults) -> EnvSpec:
    """Register a finite SMDP given by a config builder.

    The budget and per-agent hyperparameters come from ``source``; see
    :mod:`examples.source_settings`. ``episode_steps`` overrides that source's
    per-episode decision cap, which the runner turns into a restart. The multichain
    configs need a cap: once an action has committed you to an absorbing loop the
    start state is never seen again, so without restarts an agent gets exactly one
    decision and every agent scores the same.
    """
    setup = source_settings.settings(name, source)
    cap = setup["episode_steps"] if episode_steps is None else episode_steps

    def build(*, max_steps: Optional[int] = cap,
              max_time: Optional[float] = None,
              render_mode: Optional[str] = None, reseed_processes: bool = True,
              **config_kwargs: Any) -> SMDPEnv:
        cfg = config_fn(**{**config_defaults, **config_kwargs})
        return TabularSMDPEnv(cfg, name=name, max_steps=max_steps,
                              max_time=max_time, render_mode=render_mode,
                              reseed_processes=reseed_processes)

    return EnvSpec(name=name, family=family, build=build, budget=setup["budget"],
                   budget_steps=setup["budget_steps"], source=source,
                   agent_kwargs=setup["hp"], evaluation=setup["evaluation"],
                   epsilon_schedule=setup["epsilon_schedule"],
                   source_seeds=setup["source_seeds"],
                   attributed=setup["attributed"], tags=tags)


def _rate(name: str, cls, note: str = "", **env_defaults: Any) -> EnvSpec:
    """Register one of the rate-coupled environments from ``two_states.py``."""
    setup = source_settings.settings(name, "time_based")

    def build(**kwargs: Any) -> SMDPEnv:
        return cls(name=name, **{**env_defaults, **kwargs})

    return EnvSpec(name=name, family="rates", build=build, note=note,
                   budget=setup["budget"], budget_steps=setup["budget_steps"],
                   source="time_based", agent_kwargs=setup["hp"],
                   evaluation=setup["evaluation"],
                   epsilon_schedule=setup["epsilon_schedule"],
                   source_seeds=setup["source_seeds"],
                   attributed=setup["attributed"])


def _whack_a_mole(kind: str, mole_dynamics: str, reward_mode: str) -> EnvSpec:
    """Register one cell of the whack-a-mole config matrix."""
    cls = WhackAMoleMDP if kind == "mdp" else WhackAMoleSMDP
    name = f"wam_{kind}_{mole_dynamics}_{reward_mode}"

    def build(*, rows: int = 3, cols: int = 3, mole_up_prob: float = 0.2,
              mole_down_prob: float = 0.1, whack_time: float = 1.0,
              max_steps: Optional[int] = None, max_time: Optional[float] = None,
              render_mode: Optional[str] = None, **kwargs: Any) -> SMDPEnv:
        return cls(rows=rows, cols=cols, mole_dynamics=mole_dynamics,
                   reward_mode=reward_mode, mole_up_prob=mole_up_prob,
                   mole_down_prob=mole_down_prob, whack_time=whack_time,
                   max_steps=max_steps, max_time=max_time,
                   render_mode=render_mode, **kwargs)

    dynamics = ("moles stay up until whacked" if mole_dynamics == "until_whacked"
                else "moles rise and fall on their own")
    payout = {"whack": "+1 per whack",
              "whack_downed": "per whack, the number of down holes",
              "step_downed": "the number of down holes every step"}[reward_mode]
    clock = "unit holding times (an MDP)" if kind == "mdp" else \
            "distance-based holding times (an SMDP)"
    setup = source_settings.settings(name, "whack_a_mole")
    return EnvSpec(name=name, family="whack_a_mole", build=build,
                   budget=setup["budget"], budget_steps=setup["budget_steps"],
                   source="whack_a_mole", agent_kwargs=setup["hp"],
                   evaluation=setup["evaluation"],
                   epsilon_schedule=setup["epsilon_schedule"],
                   source_seeds=setup["source_seeds"],
                   attributed=setup["attributed"],
                   tags=(kind, mole_dynamics, reward_mode), metric="eval_rate",
                   note=f"3x3 grid, {clock}; {dynamics}; {payout}")


def _parse_whack_a_mole(name: str):
    """``wam_smdp_up_down_whack_downed`` -> ``("smdp", "up_down", "whack_downed")``."""
    tag = name[len("wam_"):]
    kind, rest = tag.split("_", 1)
    for dynamics in MOLE_DYNAMICS:
        if rest.startswith(dynamics + "_"):
            return kind, dynamics, rest[len(dynamics) + 1:]
    raise ValueError(f"cannot parse a whack-a-mole name from {name!r}")


def _build_registry() -> Dict[str, EnvSpec]:
    # Budgets, episode lengths, evaluation budgets and hyperparameters all come from
    # each environment's source repository via `source_settings`. Nothing here is
    # tuned by this repo, and where a source pins no budget the spec is marked
    # unattributed rather than presented as provenance.
    entries: List[EnvSpec] = [
        # --- criterion counterexamples ---
        # `ratio_vs_step_rate` is the one config `run_smdp_experiment.py` actually
        # has live, at p=0.1 — the only protocol in that repo asserted by code
        # rather than by git archaeology.
        _tabular("ratio_vs_step_rate", "criterion", configs.ratio_vs_step_rate,
                 source="python_project_3_counterexample", p=0.1),
        # gemini and feinberg are commented out at that call site and have no
        # attributable protocol. Both are multichain, so they also need an episode
        # far shorter than any the source used: both loops absorb, so at the
        # source's 1,000-step episode an agent gets one decision and 999
        # consequences and every agent scores the same.
        _tabular("gemini", "criterion", configs.gemini_three_state,
                 source="python_project_3_unattributed", episode_steps=50),
        # feinberg offers a single action, so every agent scores the same rate by
        # construction; the measurement of interest is the rho each one converges
        # to (7.5 for a time-average, 6.667 for a ratio of expectations).
        _tabular("feinberg", "criterion", configs.feinberg_three_state,
                 source="python_project_3_unattributed", episode_steps=50),
        # high_time_variance is this repository's own, built from the worked
        # example behind `SmoothedSMART` (see the README), so it has no source
        # protocol at all and takes the `none` one: a step budget
        # chosen here, and the agents' library-default hyperparameters. The
        # defaults matter for once — the estimators it separates only part company
        # at a fixed gain, and beta=0.3 is the default the rate agents carry.
        _tabular("high_time_variance", "criterion", configs.high_time_variance,
                 source="none"),
        # harmonic_criterion isolates the third averaging, 1/E[tau/r]; also
        # this repo's own, so it takes the `none` protocol.
        _tabular("harmonic_criterion", "criterion", configs.harmonic_criterion,
                 source="none"),
        # --- risk: from the risk branch's test fixture, not a benchmark ---
        _tabular("risk", "risk", configs.risk_three_actions,
                 source="python_project_3_unattributed"),
        # --- horizon ---
        # 500-decision episodes: long enough that refusing the jackpot stays
        # optimal (see the config docstring's threshold).
        _tabular("hell_or_heaven", "horizon", configs.hell_or_heaven,
                 source="python_project_3_unattributed", episode_steps=500),
        _tabular("bonus", "horizon", configs.bonus_unichain,
                 source="python_project_3_unattributed", episode_steps=50),
        _tabular("schwartz", "horizon", configs.schwartz_loop,
                 source="python_project_3_unattributed"),
        # --- drift ---
        # sincoslog: `main.py` runs 10 episodes of 1,000 steps, and the engine
        # resets every reward and duration process on reset, so 1,000 steps *is*
        # the drift horizon. Running it as one long trajectory instead compounds
        # the sin/log multiplier without bound and is a different environment.
        _tabular("sincoslog", "drift", configs.sincoslog,
                 source="python_project_3_main"),
        # non_stationary was last live at 100 episodes x 20 steps — short because
        # its reward and duration compound per visit. The 20-step episode, not the
        # config's own `ramp_visits` guard, is what bounds the clock here.
        # sincoslog with the empty s2 -> s1 leg folded away: same physical process,
        # no zero-reward transitions, and the bait's strength untouched. The clean
        # control for whether the harmonic family's advantage was the encoding.
        _tabular("sincoslog_folded", "drift", configs.sincoslog,
                 source="python_project_3_main", fold_return=True,
                 return_reward=0.0),
        # The self-similar rebuild: both arms on one exogenous envelope, so the
        # relative margin stays at 0.0667 instead of collapsing. s=1e-2 is the
        # speed at which the estimators come apart; see the config docstring.
        _tabular("sincoslog_ss", "drift", configs.sincoslog_self_similar,
                 source="python_project_3_main", s=0.01),
        # Two controls at the speed where the estimators come apart, s=0.03. They
        # differ from each other in `a_r` alone, which flips the sign of the margin
        # and so which arm is optimal -- that pair is what bounds the harmonic
        # family's advantage to "long arm optimal" rather than "better". Both pay
        # the s2 -> s1 leg, so neither contains a zero reward and neither can be
        # explained by the encoding artefact that produced the legacy result.
        _tabular("sincoslog_ss_paid", "drift", configs.sincoslog_self_similar,
                 source="python_project_3_main", s=0.03, return_reward=20.0),
        _tabular("sincoslog_ss_short", "drift", configs.sincoslog_self_similar,
                 source="python_project_3_main", s=0.03, return_reward=20.0,
                 a_r=60.0),
        _tabular("non_stationary", "drift", configs.non_stationary_unichain,
                 source="python_project_3_unattributed", episode_steps=20),
        # ripening_bait is this repository's own. One state, two self-loops, so it
        # has no episode structure to speak of and runs as one long trajectory:
        # the bait's cycle *is* the horizon, and resetting would restart the drift.
        _tabular("ripening_bait", "drift", configs.ripening_bait, source="none",
                 episode_steps=None),
        # rotting_bait is the 10x-rate-dispersion companion; see its docstring.
        _tabular("rotting_bait", "drift", configs.rotting_bait, source="none",
                 episode_steps=None),
        # --- horizon, episodic. A repo-local fixture with no benchmark source. ---
        EnvSpec(name="two_path", family="horizon",
                build=lambda **kw: TwoPathEnv(**kw),
                budget_steps=source_settings.PROTOCOL["none"]["budget_steps"],
                source="none", evaluation=source_settings.PROTOCOL["none"]["evaluation"],
                attributed=False,
                note="two 2-step paths, totals 101 versus 100; probes the horizon"),
        # --- reward earned at a rate over a random duration ---
        # config.yaml only ever ran the shifting variant; the rest of the family
        # shares its protocol. Its `_shift_max` is 2.0, not the class default.
        _rate("shifting_uneven", ShiftingUnevenTwoStates, shift_max=2.0,
              note="all rates rescaled by a factor jumping in [0.1, 2] every 50 "
                   "decisions; the optimal policy never changes"),
        _rate("stateless", StatelessRates,
              note="one state, rates 0.6 versus 0.55, noise 100: pure discrimination"),
        _rate("two_states_even", EvenTwoStates,
              note="flat reward 2 versus 1 against a duration spanning 1-500"),
        _rate("two_states_uneven", UnevenTwoStates,
              note="state 0 pays best; escaping state 1 fast beats milking it "
                   "(0.693 versus 0.667)"),
        _rate("uneven_cycling", CyclingUnevenTwoStates,
              note="state 0's better action flips every 5 decisions, phase latent"),
        _rate("uneven_latent_cycling", LatentCyclingUnevenTwoStates,
              note="the best action depends only on an unobservable 50-step phase"),
        _rate("slope_shifting_uneven", SlopeShiftingUnevenTwoStates,
              note="all rates decaying deterministically (exp, slope 0.99 every "
                   "50 decisions)"),
        # --- real market data: one pass through the 32,928-bar slice ---
        EnvSpec(name="btc_market", family="market",
                build=lambda **kw: BtcMarketEnv(**kw),
                budget_steps=source_settings.PROTOCOL["btc_swarm"]["budget_steps"],
                source="btc_swarm",
                agent_kwargs=source_settings.BTC_SWARM,
                evaluation=source_settings.PROTOCOL["btc_swarm"]["evaluation"],
                source_seeds=source_settings.PROTOCOL["btc_swarm"]["seeds"],
                note="long or short on hourly BTC bars; the holding time is "
                     "derived from the same return that pays the reward"),
    ]
    for name in source_settings.WHACK_A_MOLE_BUDGET:
        kind, dynamics, reward = _parse_whack_a_mole(name)
        entries.append(_whack_a_mole(kind, dynamics, reward))
    return {spec.name: spec for spec in entries}


#: name -> :class:`EnvSpec` for every registered environment.
ENVS: Dict[str, EnvSpec] = _build_registry()


def names(family: Optional[str] = None) -> List[str]:
    """Registered environment names, optionally restricted to one family."""
    return [n for n, s in ENVS.items() if family is None or s.family == family]


def specs(family: Optional[str] = None) -> List[EnvSpec]:
    """Registered specs, optionally restricted to one family."""
    return [s for s in ENVS.values() if family is None or s.family == family]


def make(name: str, **kwargs) -> SMDPEnv:
    """Build the registered environment ``name``.

    Keyword arguments go to the environment (``max_time``, ``render_mode``, …) or,
    for the finite SMDPs, to the config builder (``log_scale``, ``jackpot``, …).
    """
    try:
        spec = ENVS[name]
    except KeyError:
        raise KeyError(
            f"Unknown environment {name!r}. Registered: {', '.join(sorted(ENVS))}"
        ) from None
    return spec(**kwargs)


def describe(family: Optional[str] = None) -> str:
    """A printable table of the registry."""
    chosen = specs(family)
    width = max((len(s.name) for s in chosen), default=0)
    lines = []
    for spec in sorted(chosen, key=lambda s: (s.family, s.name)):
        lines.append(f"{spec.name:<{width}}  [{spec.family}]  {spec.describe()}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
