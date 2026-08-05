"""The catalogue of finite SMDPs, as :class:`~.tabular.SMDPConfig` graphs.

Each builder returns a config whose ``note`` says what the environment is *for*.
They fall into three groups:

**Criterion counterexamples** — ``gemini_three_state``, ``feinberg_three_state``
and ``ratio_vs_step_rate``. Small graphs where the candidate definitions of
"average reward per unit time" disagree, with the competing values worked out in
the docstring. These are the environments that tell you *which* criterion an
agent family actually optimises, so their numbers are asserted in
``tests/test_examples.py``.

**Risk** — ``risk_three_actions``. Three actions with the same mean rate and
increasing spread, so only a risk-sensitive agent has a preference.

**Non-stationary** — ``sincoslog`` and ``non_stationary_unichain``, built from
``envs/distributions.py``. A drifting world where the best action changes over
time, which is what separates an estimator that tracks a rate from one that
averages the whole history.

Ported from ``PythonProject3/{smdp_env,more_smdp_envs}.py`` and, for the risk
graph, from the ``SimpleRiskSMDP`` fixture on the ``risk`` branch.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .distributions import make_duration, make_reward
from .tabular import Action, SMDPConfig, State, Transition

Table = Dict[Tuple[State, Action], List[Transition]]

# Action names, for configs that use the shared two-action alphabet.
A, B = 0, 1


# --------------------------------------------------------- criterion examples
def gemini_three_state() -> SMDPConfig:
    """Time-average and ratio-of-expectations disagree, and pick *different* actions.

    ``s1`` offers a gamble and a sure thing::

        s1 --a--> s2   p=0.5  r=20  tau=1      s2 --a--> s2  r=20  tau=1
        s1 --a--> s3   p=0.5  r=0   tau=19     s3 --a--> s3  r=0   tau=19
        s1 --b--> s1   p=1.0  r=4   tau=1

    Action *b* is worth 4 under every criterion. Action *a* commits you forever
    to one of the two loops, so:

    * **time-average** (the long-run rate of the realised trajectory, then
      averaged over trajectories): ``0.5 * 20 + 0.5 * 0 = 10`` — *a* beats *b*.
    * **ratio of expectations** (``E[r] / E[tau]`` across the transition
      distribution): ``(0.5*20 + 0.5*0) / (0.5*1 + 0.5*19) = 10 / 10 = 1`` — *b*
      beats *a*.

    The chain is multichain, not unichain: ``s2`` and ``s3`` are separate
    absorbing loops, which is exactly what lets the two criteria part ways.
    """
    transitions: Table = {
        ("s1", A): [Transition("s2", 0.5, 20.0, 1.0),
                    Transition("s3", 0.5, 0.0, 19.0)],
        ("s1", B): [Transition("s1", 1.0, 4.0, 1.0)],
        ("s2", A): [Transition("s2", 1.0, 20.0, 1.0)],
        ("s3", A): [Transition("s3", 1.0, 0.0, 19.0)],
    }
    return SMDPConfig(
        states=["s1", "s2", "s3"],
        actions=[A, B],
        transitions=transitions,
        start_state="s1",
        note="a@s1: 10 by time-average but 1 by ratio-of-expectations; b@s1: 4 by both",
    )


def feinberg_three_state() -> SMDPConfig:
    """The same disagreement without a choice to make: 7.5 versus 6.667.

    ::

        s1 --a--> s2   p=0.5  r=0  tau=1       s2 --a--> s2  r=10  tau=1
        s1 --a--> s3   p=0.5  r=0  tau=1       s3 --a--> s3  r=10  tau=2

    One action throughout, so every criterion evaluates the same policy and the
    only question is what number it assigns:

    * **time-average**: the ``s2`` loop runs at 10 and the ``s3`` loop at 5, so
      ``0.5 * 10 + 0.5 * 5 = 7.5``.
    * **ratio of expectations**: ``10 / (0.5*1 + 0.5*2) = 6.667``.

    Useful as a pure measurement: whatever an agent's rho converges to here
    names its criterion, with no policy improvement in the way.
    """
    transitions: Table = {
        ("s1", A): [Transition("s2", 0.5, 0.0, 1.0),
                    Transition("s3", 0.5, 0.0, 1.0)],
        ("s2", A): [Transition("s2", 1.0, 10.0, 1.0)],
        ("s3", A): [Transition("s3", 1.0, 10.0, 2.0)],
    }
    return SMDPConfig(
        states=["s1", "s2", "s3"],
        actions=[A],
        transitions=transitions,
        start_state="s1",
        note="single policy worth 7.5 by time-average, 6.667 by ratio-of-expectations",
    )


def ratio_vs_step_rate(p: float = 0.01, r_hi: float = 100.0, t_hi: float = 1.0,
                       r_lo: float = 1.0, t_lo: float = 1000.0,
                       r_b: float = 1.0, t_b: float = 1.0) -> SMDPConfig:
    """Ratio of averages versus average of ratios, in one unichain state.

    ::

        s0 --a--> s0   p=p      r=r_hi  tau=t_hi
        s0 --a--> s0   p=1-p    r=r_lo  tau=t_lo
        s0 --b--> s0   p=1      r=r_b   tau=t_b

    At the defaults, action *a* is a rare jackpot (1% of the time, 100 in one
    time unit) buried in a long grind (99% of the time, 1 over 1000 time units),
    and action *b* pays a steady 1 per unit time:

    * **ratio of expectations** for *a*:
      ``(0.01*100 + 0.99*1) / (0.01*1 + 0.99*1000) = 1.99 / 990.01 = 0.00201``
      — far worse than *b*'s 1.
    * **mean of the per-transition rates** for *a*:
      ``0.01*(100/1) + 0.99*(1/1000) = 1.00099`` — better than *b*'s 1.

    So ``CumulativeTimeRate`` (which backs SMART) and ``CumulativeStepRate``
    rank the two actions in opposite orders here. Unlike the two graphs above
    this one is unichain, so the disagreement is not an artefact of separate
    absorbing loops: it is Jensen's inequality on ``r / tau``.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be a probability")
    transitions: Table = {
        ("s0", A): [Transition("s0", p, r_hi, t_hi),
                    Transition("s0", 1.0 - p, r_lo, t_lo)],
        ("s0", B): [Transition("s0", 1.0, r_b, t_b)],
    }
    return SMDPConfig(
        states=["s0"],
        actions=[A, B],
        transitions=transitions,
        start_state="s0",
        note=(f"a is a p={p} jackpot: worse than b under ratio-of-expectations, "
              f"better under the mean of per-transition rates"),
    )


# ------------------------------------------------------------ risk attitudes
#: Reward pairs for the three ``risk_three_actions`` gambles, each equiprobable.
#: Means are 16, 17 and 16 respectively — ``neutral`` is deliberately the best
#: gamble on expectation, so a risk-neutral agent has something to be right about.
RISK_OUTCOMES = {
    "seek": (2.0, 30.0),
    "neutral": (12.0, 22.0),
    "averse": (16.0, 16.0),
}
#: Action indices for :func:`risk_three_actions`.
SEEK, NEUTRAL, AVERSE, BACK = 0, 1, 2, 3


def risk_three_actions() -> SMDPConfig:
    """Three gambles that a risk-neutral, a risk-averse and a risk-seeking agent
    rank differently.

    ::

        s1 --seek----> s2   r = 2 or 30   tau=2      (equiprobable)
        s1 --neutral-> s2   r = 12 or 22  tau=2      (equiprobable)
        s1 --averse--> s2   r = 16        tau=2
        s2 --back----> s1   r = 8         tau=1

    Each round is a gamble in ``s1`` plus the fixed return leg, so 3 time units
    per round, and the three gambles are arranged so that each attitude has its
    own strict optimum:

    * ``neutral`` has the highest mean, 17 — rate ``(17 + 8) / 3 = 8.333``. It is
      what maximising expected reward alone picks.
    * ``seek`` and ``averse`` both have mean 16 — rate ``8`` — so they are tied on
      expectation and separated only by spread: ``seek`` is ±14 around it and
      ``averse`` is exactly 16 every time.

    A risk-neutral agent therefore takes ``neutral``, an averse one gives up the
    extra third of a unit for the certainty of ``averse``, and a seeking one gives
    it up for ``seek``'s upside of 30. Which of the three an agent settles on
    names its risk attitude.

    From the ``SimpleRiskSMDP`` fixture on the ``risk`` branch, with the string
    actions turned into indices; :data:`RISK_ACTION_NAMES` maps them back.
    """
    transitions: Table = {
        ("s1", SEEK): [Transition("s2", 0.5, RISK_OUTCOMES["seek"][0], 2.0),
                       Transition("s2", 0.5, RISK_OUTCOMES["seek"][1], 2.0)],
        ("s1", NEUTRAL): [Transition("s2", 0.5, RISK_OUTCOMES["neutral"][0], 2.0),
                          Transition("s2", 0.5, RISK_OUTCOMES["neutral"][1], 2.0)],
        ("s1", AVERSE): [Transition("s2", 1.0, RISK_OUTCOMES["averse"][0], 2.0)],
        ("s2", BACK): [Transition("s1", 1.0, 8.0, 1.0)],
    }
    return SMDPConfig(
        states=["s1", "s2"],
        actions=[SEEK, NEUTRAL, AVERSE, BACK],
        transitions=transitions,
        start_state="s1",
        note=("three gambles: neutral is best on expectation (rate 8.33), seek "
              "and averse tie at rate 8 and differ only in spread"),
    )


#: Human-readable names for :func:`risk_three_actions`'s action indices.
RISK_ACTION_NAMES = {SEEK: "seek", NEUTRAL: "neutral", AVERSE: "averse", BACK: "back"}


# ------------------------------------------------------- greed and postponement
def hell_or_heaven(jackpot: float = 100.0) -> SMDPConfig:
    """One irreversible jackpot against a small perpetual income.

    ::

        s1 --a--> s2   r=0        tau=1     s2 --a--> s2  r=+1  tau=1
        s1 --b--> s3   r=jackpot  tau=1     s3 --a--> s3  r=-1  tau=1

    A discounted agent with a short horizon takes the jackpot and then bleeds -1
    forever; an average-reward agent should refuse it, because the only thing that
    survives averaging is the loop it lands in.

    Both loops are absorbing, so ``s1`` is visited once and an agent needs
    *restarts* to learn anything — and the restart interval is part of the
    problem. Over episodes of ``L`` decisions, taking the jackpot is worth
    ``(jackpot - L + 1) / L`` and refusing is worth ``(L - 1) / L``, so refusing
    is only correct while ``L > jackpot / 2 + 1``. The registry pairs the default
    ``jackpot=100`` with ``L=500``, where refusing wins 0.998 to -0.798; shorten
    the episode past the threshold and the environment inverts.
    """
    transitions: Table = {
        ("s1", A): [Transition("s2", 1.0, 0.0, 1.0)],
        ("s1", B): [Transition("s3", 1.0, jackpot, 1.0)],
        ("s2", A): [Transition("s2", 1.0, 1.0, 1.0)],
        ("s3", A): [Transition("s3", 1.0, -1.0, 1.0)],
    }
    return SMDPConfig(
        states=["s1", "s2", "s3"],
        actions=[A, B],
        transitions=transitions,
        start_state="s1",
        note=(f"a@s1 -> rate +1 forever; b@s1 -> {jackpot:g} once then rate -1 "
              f"forever (refusing is correct only for episodes longer than "
              f"{jackpot / 2 + 1:g} decisions)"),
    )


def bonus_unichain(bonus: float = 100.0, income: float = 10.0) -> SMDPConfig:
    """A one-off bonus on the way into a loop that pays regardless.

    ::

        s1 --a--> s2   r=0      tau=1      s2 --a--> s2  r=income  tau=1
        s1 --b--> s2   r=bonus  tau=1

    Both actions land in the same loop, so over an infinite horizon both policies
    have exactly the same rate and the bonus is invisible to the criterion even
    though it is free money — capturing it is what the bias, or relative-value,
    term is for.

    Over episodes of ``L`` decisions it stops being invisible: the bonus adds
    ``bonus / L`` to the rate, so at the registry's ``L=50`` taking it is worth
    11.8 against 9.8, a 20% gap that an agent has no excuse to miss. What can
    still miss it is a rho estimator that washes a single large reward out
    against a long run of small ones.
    """
    transitions: Table = {
        ("s1", A): [Transition("s2", 1.0, 0.0, 1.0)],
        ("s1", B): [Transition("s2", 1.0, bonus, 1.0)],
        ("s2", A): [Transition("s2", 1.0, income, 1.0)],
    }
    return SMDPConfig(
        states=["s1", "s2"],
        actions=[A, B],
        transitions=transitions,
        start_state="s1",
        note=f"both policies run at rate {income:g}; b@s1 adds a one-off {bonus:g}",
    )


def schwartz_loop(length: int = 49, payout: float = 50.0) -> SMDPConfig:
    """A long corridor to a payout, with zero-mean noise all the way along.

    Every state ``s0..sN`` offers *b* (advance) and *a* (stay put); both pay
    ±1 with equal probability, and only the last *b* pays ``payout`` and returns
    to ``s0``. The noise has mean zero, so the whole signal is the payout spread
    over ``length + 1`` transitions — a rate of ``payout / (length + 1)``.

    The point of Schwartz's example: exploration has to survive a long stretch
    of pure noise before the payout is ever observed, so an agent that settles
    early sees no reason to advance.
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    coin = make_reward("choice", values=(-1.0, 1.0), seed=17)

    states = [f"s{i}" for i in range(length + 1)]
    transitions: Table = {}
    for i in range(length):
        transitions[(f"s{i}", B)] = [Transition(f"s{i + 1}", 1.0, coin, 1.0)]
        transitions[(f"s{i}", A)] = [Transition(f"s{i}", 1.0, coin, 1.0)]
    transitions[(f"s{length}", B)] = [Transition("s0", 1.0, payout, 1.0)]
    transitions[(f"s{length}", A)] = [Transition(f"s{length}", 1.0, coin, 1.0)]

    return SMDPConfig(
        states=states,
        actions=[A, B],
        transitions=transitions,
        start_state="s0",
        note=(f"advance {length + 1} times through zero-mean noise for {payout:g}: "
              f"optimal rate {payout / (length + 1):.4g}"),
    )


# --------------------------------------------------------------- drifting worlds
#: The ``log_scale`` sweep the sincoslog family was originally studied over
#: (``np.logspace(-5, -1, 30)``), slowest drift first.
SINCOSLOG_LOG_SCALES = tuple(round(10.0 ** (-5.0 + 4.0 * i / 29.0), 6) for i in range(30))


def sincoslog(log_scale: float = 1e-3, frequency: float = 1.0,
              slope: float = 1.0 / 20.0) -> SMDPConfig:
    """An oscillating, exponentially growing action against a steady ramp.

    ::

        s1 --a--> s2   r = slope * n                       tau ~ N(1, 0.1)
        s1 --b--> s2   r = (sin(n) + 10) * 10^(n*ls)       tau = (cos(n) + 10) * 10^(n*ls/2)
        s2 --a--> s1   r = 0                               tau = 1

    Action *a* is a linear ramp on a unit clock, so its rate climbs steadily and
    without bound. Action *b* oscillates *and* scales exponentially in both
    reward and holding time; the reward's exponent grows twice as fast as the
    holding time's, so its rate grows exponentially too — just not monotonically.

    Which action is better therefore changes at least twice. Measured as
    reward/time for the always-*a* and always-*b* policies at the default
    ``log_scale=1e-3``: *b* leads at the start (≈0.9 against *a*'s ≈0.05, since
    the ramp starts near zero), *a* overtakes within the first hundred decisions,
    and by 4000 decisions *a* is at 25 against *b*'s 5.4 — while at
    ``log_scale=1e-2`` the same 4000 decisions put *b* at 5e9. So the small end of
    the sweep is a world that eventually rewards the steady action and the large
    end one that eventually rewards the volatile one, with the crossover between
    ``1e-3`` and ``1e-2``.

    The two reward processes are hooked to each other, so both advance on every
    decision whichever action is taken: the world drifts on its own rather than
    only when sampled. The holding times are deliberately *not* hooked, so *b*'s
    clock advances only when *b* is chosen.

    ``log_scale`` spans :data:`SINCOSLOG_LOG_SCALES` in the original study. Mind
    the top of that range: at ``log_scale=1e-1`` *b*'s reward passes 1e100 within
    a few thousand decisions, which says more about floating point than about any
    agent, so treat the upper decade as a stress test rather than a benchmark.
    """
    reward_a = make_reward("linear", start=0.0, step=slope)
    duration_a = make_duration("normal", mean=1.0, stddev=0.1)
    reward_b = make_reward("sin_log", amplitude=1.0, frequency=frequency, offset=10.0,
                           log_base=10.0, start_exp=0.0, log_scale=log_scale)
    duration_b = make_duration("cos_log", amplitude=1.0, frequency=frequency, offset=10.0,
                               log_base=10.0, start_exp=0.0, log_scale=log_scale * 0.5)

    # Both rewards drift on every decision, whichever action produced it.
    reward_a.register_hook(reward_b)
    reward_b.register_hook(reward_a)

    transitions: Table = {
        ("s1", A): [Transition("s2", 1.0, reward_a, duration_a)],
        ("s1", B): [Transition("s2", 1.0, reward_b, duration_b)],
        ("s2", A): [Transition("s1", 1.0, 0.0, 1.0)],
    }
    return SMDPConfig(
        states=["s1", "s2"],
        actions=[A, B],
        transitions=transitions,
        start_state="s1",
        note=(f"linear ramp (slope {slope:g}) versus sin/cos-log drift at "
              f"log_scale={log_scale:g}; the better action flips over time"),
    )


def non_stationary_unichain(reward_growth: float = 1.5,
                            duration_growth: float = 2.0,
                            steady_reward: float = 10.0,
                            ramp_visits: int = 12) -> SMDPConfig:
    """A geometrically growing reward whose holding time grows faster.

    ::

        s1 --a--> s2   r *= reward_growth   tau *= duration_growth   (per visit)
        s1 --b--> s2   r = steady_reward    tau = 1
        s2 --a--> s1   r = 0                tau = 1

    Action *a*'s reward explodes, which is irresistible to anything that looks
    at reward alone — but its holding time explodes faster, so its *rate* decays
    to zero. The cheapest possible check that an agent is dividing by time.

    Both compounding processes stop after ``ramp_visits`` visits, after which *a*
    pays a fixed ``reward_growth ** n`` over a fixed ``duration_growth ** n``.
    Without that cap the holding time doubles without bound and a handful of
    exploratory choices consume any time budget you set — at the default the full
    ramp costs about ``2 ** 13`` time units.
    """
    reward = make_reward("linear", start=1.0, step=reward_growth,
                         drift_op="mul", max_drifts=ramp_visits)
    duration = make_duration("linear", start=1.0, step=duration_growth,
                             drift_op="mul", max_drifts=ramp_visits)
    transitions: Table = {
        ("s1", A): [Transition("s2", 1.0, reward, duration)],
        ("s1", B): [Transition("s2", 1.0, steady_reward, 1.0)],
        ("s2", A): [Transition("s1", 1.0, 0.0, 1.0)],
    }
    return SMDPConfig(
        states=["s1", "s2"],
        actions=[A, B],
        transitions=transitions,
        start_state="s1",
        note=(f"a@s1 grows reward x{reward_growth:g} but time x{duration_growth:g} "
              f"for {ramp_visits} visits, so its rate decays; b@s1 holds rate "
              f"{steady_reward:g}"),
    )
