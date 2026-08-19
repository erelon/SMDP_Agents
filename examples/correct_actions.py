"""Which action is *correct* in each environment, and which one is the bait.

Ranking a trap environment by its reward rate measures the wrong thing. Several of
these environments are built so that the action an agent should settle on is the one
that looks **worse** during the window in which it is learning — that is the whole
point of them — so an agent that scores a high ``lifetime_rate`` may simply have
taken the bait and been paid for it, while the agent that got the answer right paid
for the privilege.

The environments' own source measures this directly rather than by rate:
``PythonProject3/robustness_table.py`` trains, takes the **greedy action at a target
state**, scores ``chosen == CORRECT_ACTION`` as a binary, and reports the fraction of
hyperparameter combinations that converged to the correct policy. This module is that
idea generalised across the whole set.

Each entry names:

``state``
    The decision state to probe after training. Probing a specific state matters:
    most of these environments have exactly one state where a real choice exists.
``action``
    The correct action there — an int, or a callable ``(env) -> int`` for the
    environments where it depends on how the environment was parameterised.
``criterion``
    **What the action is correct under.** This is not decoration. In ``gemini`` the
    correct action is 1 under a ratio of expectations and 0 under a time-average, so
    an unlabelled "optimal action" would be meaningless. Recording the criterion is
    what makes the measurement interpretable.
``bait``
    The action that is more attractive during learning, or ``None`` where none is.
    Lets the report say how often an agent was actively fooled rather than merely
    wrong.
``bait_by``
    *How* the bait misleads: by paying more on the spot (``IMMEDIATE_REWARD``) or by
    earning a better rate over the whole window the agent can measure
    (``WINDOW_RATE``). ``sincoslog`` needs the second — its bait pays less on the
    first decision but wins over the episode — and conflating the two is how the
    trap catches an evaluator.
``note`` / ``source``
    Why, and where the answer comes from.

The values here were established by measuring each environment: for every action
available at the probe state, the mean immediate reward and rate of the first
decision, against the long-run rate of the policy that always takes it. An entry is
a trap exactly when those two disagree. ``tests/test_correct_actions.py`` re-derives
them, so a change to a reward or a duration that moves the correct action fails a
test instead of silently invalidating the table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, Optional, Union

#: Criteria an action can be correct under.
TIME_AVERAGE = "time-average"
RATIO_OF_EXPECTATIONS = "ratio-of-expectations"
MEAN_OF_RATES = "mean-of-per-transition-rates"
MEAN_TIME_PER_REWARD = "mean time per unit reward (1/E[tau/r])"
LONG_RUN_RATE = "long-run rate of the always-this-action policy"
RISK_NEUTRAL = "risk-neutral (highest mean)"
BEST_STATIONARY = "best stationary action (the deciding state is unobservable)"
ASYMPTOTIC_RATE = "asymptotic rate (correct beyond the simulated horizon)"

#: How a bait misleads.
IMMEDIATE_REWARD = "pays more on the first decision"
WINDOW_RATE = "earns a better rate over the horizon the agent can measure"


@dataclass(frozen=True)
class CorrectChoice:
    """The correct action at one state, and what makes it correct."""

    state: Hashable
    action: Union[int, Callable[[Any], int]]
    criterion: str
    bait: Optional[int] = None
    bait_by: str = IMMEDIATE_REWARD
    note: str = ""
    source: str = ""
    #: Optional extra text for the report, computed from the built environment.
    detail: Optional[Callable[[Any], str]] = None

    def resolve(self, env) -> int:
        """The correct action for this particular environment instance."""
        return self.action(env) if callable(self.action) else self.action

    @property
    def is_trap(self) -> bool:
        return self.bait is not None


def sincoslog_log_scale(env) -> Optional[float]:
    """The ``log_scale`` a built sincoslog environment was parameterised with."""
    for transition in env.config.transitions[("s1", 1)]:
        reward = transition.specs[2]  # the sin_log reward, whose exponent is log_scale
        log_scale = getattr(getattr(reward, "dist", None), "log_scale", None)
        if log_scale is not None:
            return float(log_scale)
    return None


def sincoslog_return_reward(env) -> float:
    """What the ``s2 -> s1`` leg of a built sincoslog environment pays.

    Every visit to ``s1`` takes that leg exactly once, so it enters the cumulative
    rate of *both* arms and :func:`visits_to_overtake` needs it.
    """
    transition, = env.config.transitions[("s2", 0)]
    return float(transition.constant("reward"))


def visits_to_overtake(log_scale: float, cap: int = 10_000_000,
                       return_reward: float = 1.0) -> Optional[int]:
    """How many ``s1`` visits before the sin/log arm's cumulative rate beats the ramp.

    This is the number that makes sincoslog a trap. The ramp's cumulative rate grows
    linearly in the visit count (``0.05 n(n+1)/2`` reward over ``2n`` time, so about
    ``0.0125 n``); the sin/log arm's grows exponentially, as
    ``10 ** (n * log_scale / 2)``, because its reward's exponent climbs twice as fast
    as its holding time's. Exponential wins eventually, but "eventually" is far
    beyond any horizon a run simulates at the slow end of the sweep.

    ``return_reward`` is what the ``s2 -> s1`` leg pays, which every visit takes
    exactly once, so it adds ``n * return_reward`` to *both* numerators. It is not a
    wash: it lifts the ramp's rate off the floor early -- the ramp's own reward
    starts near zero -- and so moves the crossover. Pass the value the environment
    was built with; :func:`sincoslog_return_reward` reads it off a built one.

    Computed in closed form with the oscillation dropped, which is safe because the
    offset of 10 dominates the amplitude of 1. ``None`` if the arm does not overtake
    within ``cap``.
    """
    if log_scale <= 0:
        return None
    q_r = 10.0 ** log_scale
    q_t = 10.0 ** (log_scale / 2.0)

    def ahead(n: int) -> bool:
        try:
            ramp = (0.05 * n * (n + 1) / 2.0 + n * return_reward) / (2.0 * n)
            reward = (10.0 * (q_r ** (n + 1) - q_r) / (q_r - 1.0)
                      + n * return_reward)
            duration = 10.0 * (q_t ** (n + 1) - q_t) / (q_t - 1.0) + n
            return reward / duration > ramp
        except OverflowError:  # the exponential has run away, so it is ahead
            return True

    # There are two crossings, not one. The arm starts *ahead* — 10.87 over a
    # holding time of 10.55 is a rate of 1.03, against the ramp's first 0.05 over 2 —
    # then the ramp overtakes around visit 73, and only much later does the
    # exponential re-overtake for good. It is that last crossing we want, so first
    # find a visit where the ramp leads.
    behind = None
    probe = 1
    while probe < cap:
        if not ahead(probe):
            behind = probe
            break
        probe *= 2
    if behind is None:
        return 0  # the arm is ahead from the first visit and never loses the lead
    low, high = behind, behind
    while high < cap and not ahead(high):
        low, high = high, high * 2
    if not ahead(high):
        return None
    while low < high:
        mid = (low + high) // 2
        if ahead(mid):
            high = mid
        else:
            low = mid + 1
    return low


#: Visits an episode provides: ``max_steps`` of 1,000 alternating s1 -> s2 -> s1.
SINCOSLOG_VISITS_PER_EPISODE = 500


def _sincoslog_correct(env) -> int:
    """Always the sin/log arm — action 1 — on the asymptotic criterion.

    This is the source's own label (``robustness_table.py:55``), and it is asymptotic
    rather than horizon-optimal: the sin/log arm's rate grows exponentially and so
    beats the ramp for *every* ``log_scale``, but only after
    :func:`visits_to_overtake` visits, which for the slower two thirds of the swept
    range is far beyond anything a run simulates.

    That gap **is** the environment. Inside the window the ramp looks better — at
    ``log_scale=1e-3`` it earns a rate of 6.25 against the sin/log arm's 1.39 — so an
    agent (or an evaluator) that judges only what it can see picks the ramp. Whether
    an agent resists depends on its rho: under R-learning the sin/log arm is
    preferred exactly when ``rho < (r1 - r0) / (tau1 - tau0)``, and SMART's
    whole-history ``rho = sum(r)/sum(tau)`` is inflated by the ramp's own ripening
    rewards, which pushes it over that threshold and onto the short-duration arm. The
    harmonic estimator's rho stays 3-4x lower and keeps the long arm.

    Note also that ``s1`` is **not Markov**: the ramp's reward is a global decision
    counter that is not part of the state, and the mutual reward hook advances it on
    every visit regardless of which action was taken. A tabular agent has no way to
    represent "action 0 is worth 25 now, not 0.05".
    """
    return SINCOSLOG_SOURCE_CONSTANT


#: The ``log_scale`` above which the sin/log arm out-earns the ramp *within* one
#: 1,000-step episode. Below it the arm is still asymptotically correct but its payoff
#: lies beyond the horizon, which is what makes those configurations traps. It falls
#: between the grid points 0.004175 and 0.005736 either way, so which sweep points are
#: traps is unchanged (20 of 30), but the exact boundary moves with the ``s2 -> s1``
#: leg's reward: 0.0042845 when that leg pays nothing, as the source had it, and
#: 0.0044206 at this repo's ``return_reward=1``.
SINCOSLOG_CROSSOVER = 0.0044206
#: The same boundary with the source's unpaid return leg, for comparison.
SINCOSLOG_CROSSOVER_UNPAID_RETURN = 0.0042845

#: ``PythonProject3/robustness_table.py:55``: ``CORRECT_ACTION = 1``, the sin/log arm,
#: for every environment in the sweep. Deliberate and consistently maintained — the
#: commit that swapped which index carries the sin/log processes changed this constant
#: in the same breath — and adopted here as the criterion.
#:
#: Worth knowing what it costs, though: within the simulated horizon it labels as
#: correct an arm earning up to 6.9x less, and in the source's own recorded sweep the
#: rows scored ``correct`` average a rate of 1.746 while the few scored wrong average
#: 5.833. The label rewards an agent whose value estimates *lag* the ripening ramp.
#: :func:`visits_to_overtake` is what reconciles the two: report it beside the horizon
#: and the trap is explicit rather than hidden in a binary.
SINCOSLOG_SOURCE_CONSTANT = 1

def _sincoslog_detail(env) -> str:
    """How far past the simulated horizon the correct arm's payoff lies."""
    log_scale = sincoslog_log_scale(env)
    if log_scale is None:  # pragma: no cover
        return ""
    visits = visits_to_overtake(log_scale,
                                return_reward=sincoslog_return_reward(env))
    horizon = env.max_steps // 2 if env.max_steps else SINCOSLOG_VISITS_PER_EPISODE
    if visits is None:
        return "the sin/log arm never overtakes within a reasonable horizon"
    if visits <= horizon:
        return (f"at log_scale={log_scale:g} the sin/log arm overtakes after "
                f"{visits:,} visits, inside the {horizon:,}-visit episode — so this "
                f"configuration is not actually a trap")
    return (f"at log_scale={log_scale:g} the sin/log arm needs **{visits:,} visits** "
            f"to overtake the ramp, but an episode provides only {horizon:,} "
            f"({visits / horizon:,.0f}x short). The correct answer is unobservable "
            f"within the run, which is the point")


#: environment name -> the correct choice at its decision state.
CORRECT: Dict[str, CorrectChoice] = {
    # --- criterion counterexamples -------------------------------------------
    "gemini": CorrectChoice(
        state="s1", action=1, criterion=RATIO_OF_EXPECTATIONS, bait=0,
        note="the sure 4 beats the gamble's 1 under a ratio of expectations, but "
             "the gamble pays 20 half the time and wins 10 to 4 under a "
             "time-average — the criteria pick different actions, so this entry is "
             "only meaningful with its criterion attached",
        source="config docstring; measured 4.0 against 0.749 long-run"),
    "ratio_vs_step_rate": CorrectChoice(
        state="s0", action=1, criterion=TIME_AVERAGE, bait=0,
        note="at the source's p=0.1 the jackpot action's mean per-transition rate "
             "is 10.0 but its time-average is 0.012, against a steady 1.0",
        source="measured 1.0 against 0.0122 long-run"),
    "high_time_variance": CorrectChoice(
        state="s0", action=0, criterion=LONG_RUN_RATE, bait=None,
        note="not a trap: the bursty option runs at 20/11 = 1.818 against the "
             "steady action's 1.0 and also pays 10 on the spot against 1, so it "
             "wins on both readings. It is a *rho* trap instead. Under R-learning "
             "the option survives only while rho < (20-1)/(11-1) = 1.9, which is "
             "4.5% above its own true rate, and the cycle spends 10 of its 11 time "
             "units in 1 of its 11 transitions — so an estimator that forgets per "
             "transition rather than per unit time is pulled toward the burst's "
             "rate of 10, clears 1.9, and drops the option. Over 60 seeds SMART "
             "and CumulativeWeightedHarmonic keep it 60 times, SmoothedSMART 39, "
             "RelaxedSMART and WeightedHarmonic 31, Harmonic 29 and "
             "CumulativeHarmonic 19. Two effects, not one: among the estimators "
             "that forget, the size of the overshoot orders the outcome — but "
             "CumulativeHarmonic overshoots *less* than Harmonic and still does "
             "worst, because it cannot forget the burst once the agent has "
             "switched away. The pairs that tie do so by identity: with every "
             "reward positive, a reward-weighted harmonic mean *is* the ratio of "
             "the same two averages",
        source="measured 1.818 against 1.0 long-run; the config docstring derives "
               "the 1.9 threshold and tabulates the measured rates"),
    "harmonic_criterion": CorrectChoice(
        state="s0", action=1, criterion=MEAN_TIME_PER_REWARD, bait=None,
        note="b costs 2 time units per unit of reward against a's 5.05, so it wins "
             "under 1/E[tau/r] — the one criterion only the unweighted harmonic "
             "estimator computes. It loses under both others (rate 0.5 against "
             "1.0 by ratio-of-means and 5.05 by mean-of-rates), so this entry is "
             "meaningless without its criterion attached, exactly as gemini's is. "
             "No agent actually takes it: a harmonic rho is not the gain of any "
             "policy, so R-learning's relative values diverge rather than ranking "
             "the arms — see the config docstring",
        source="exact from the transition table; the divergence is measured"),
    # --- risk -----------------------------------------------------------------
    "risk": CorrectChoice(
        state="s1", action=1, criterion=RISK_NEUTRAL, bait=None,
        note="not a trap in the immediate-reward sense: neutral has mean 17 against "
             "16 for both seek and averse, so it pays best straight away as well as "
             "in the long run (rate 8.33 against 8.0). What can still pull an agent "
             "away is averse's *zero variance* — it pays exactly 16 every single "
             "time — which is a risk attitude rather than a reward ordering, and the "
             "correct-choice measurement here does not detect it",
        source="measured over 4,000 samples: 17.03 against averse's 16.08"),
    # --- horizon --------------------------------------------------------------
    "hell_or_heaven": CorrectChoice(
        state="s1", action=0, criterion=LONG_RUN_RATE, bait=1,
        note="the jackpot pays 100 once and then -1 forever; refusing is worth "
             "0.998 against -0.798",
        source="measured; the config docstring derives the episode-length threshold"),
    "bonus": CorrectChoice(
        state="s1", action=1, criterion=LONG_RUN_RATE, bait=None,
        note="not a trap: the bonus pays more immediately and is also correct",
        source="measured 11.8 against 9.8 long-run"),
    "schwartz": CorrectChoice(
        state="s0", action=1, criterion=LONG_RUN_RATE, bait=None,
        note="advancing is correct (1.024 against 0.022) and both actions look "
             "identical immediately, so there is no bait — only noise to survive",
        source="measured"),
    "two_path": CorrectChoice(
        state=0, action=0, criterion=LONG_RUN_RATE, bait=1,
        note="path A totals 101 against B's 100, but B pays 50 of it on the first "
             "step",
        source="the environment's own secret()"),
    # --- drift ----------------------------------------------------------------
    "sincoslog": CorrectChoice(
        state="s1", action=_sincoslog_correct, criterion=ASYMPTOTIC_RATE, bait=0,
        bait_by=WINDOW_RATE, detail=lambda env: _sincoslog_detail(env),
        note="the sin/log arm is correct because its rate grows exponentially and "
             "the ramp's only linearly — but it needs 4,005 visits to overtake at "
             "the registered log_scale=1e-3, and 867,218 at 1e-5, against the 500 an "
             "episode provides. So the *bait is the ramp*: inside the window it "
             "earns 6.25 against 1.39, and an agent that trusts what it can measure "
             "takes it. Under R-learning the sin/log arm survives only while "
             "rho < (r1-r0)/(tau1-tau0), which is why a whole-history rho — inflated "
             "by the ramp's own ripening rewards — abandons it and a harmonic rho "
             "does not",
        source="PythonProject3/robustness_table.py:55 (CORRECT_ACTION = 1); the "
               "overtake counts are computed by visits_to_overtake()"),
    "non_stationary": CorrectChoice(
        state="s1", action=1, criterion=LONG_RUN_RATE, bait=None,
        note="not a trap: the steady action pays more immediately and is also "
             "correct, because the compounding action's holding time outruns its "
             "reward",
        source="measured 5.0 against 0.083 long-run"),
    # --- rates ----------------------------------------------------------------
    "uneven_cycling": CorrectChoice(
        state=0, action=1, criterion=BEST_STATIONARY, bait=0,
        note="action 0 pays 6.8x more on the first decision but the phase flips "
             "every 5 decisions and is not observable, so no state-conditioned "
             "answer exists — action 1 is the better of the two *stationary* "
             "choices over a full cycle (0.458 against 0.351). An agent with "
             "privileged access to the phase can and should beat both, which is why "
             "the Oracle is excluded from this measurement",
        source="measured"),
    "two_states_uneven": CorrectChoice(
        state=0, action=0, criterion=LONG_RUN_RATE, bait=None,
        note="no bait at state 0; the trade-off in this environment is at state 1, "
             "where the worse-paying action escapes faster",
        source="the environment's own secret()"),
    "stateless": CorrectChoice(
        state=0, action=0, criterion=LONG_RUN_RATE, bait=None,
        note="not a trap: rate 0.6 against 0.55, buried in noise of 100",
        source="the environment's own secret()"),
}

#: The state at which ``two_states_uneven``'s real trade-off lives — the poor state,
#: where the *worse-paying* action is the one that escapes. Probed in addition to the
#: start state, since that is where the environment's point is.
EXTRA_PROBES: Dict[str, CorrectChoice] = {
    "two_states_uneven": CorrectChoice(
        state=1, action=1, criterion=LONG_RUN_RATE, bait=0,
        note="in the poor state, action 0 pays twice what action 1 does but leaves "
             "four times more slowly; taking the loss is correct (0.693 against "
             "0.667 for always-0)",
        source="measured; the environment's own secret()"),
}


def choices(env_name: str) -> Dict[Hashable, CorrectChoice]:
    """Every probe for ``env_name``, keyed by the state to probe."""
    found: Dict[Hashable, CorrectChoice] = {}
    for table in (CORRECT, EXTRA_PROBES):
        entry = table.get(env_name)
        if entry is not None:
            found[entry.state] = entry
    return found


def traps() -> Dict[str, CorrectChoice]:
    """The environments where the correct action is not the attractive one."""
    return {name: choice for name, choice in CORRECT.items() if choice.is_trap}
