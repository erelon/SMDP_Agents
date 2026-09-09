"""The catalogue of finite SMDPs, as :class:`~.tabular.SMDPConfig` graphs.

Each builder returns a config whose ``note`` says what the environment is *for*.
They fall into three groups:

**Criterion counterexamples** — ``gemini_three_state``, ``feinberg_three_state``,
``ratio_vs_step_rate``, ``high_time_variance`` and ``harmonic_criterion``. Small graphs where the
candidate definitions of "average reward per unit time" disagree, with the
competing values worked out in the docstring. These are the environments that
tell you *which* criterion an agent family actually optimises, so their numbers
are asserted in ``tests/test_examples.py``. The first three separate criteria
that differ in the limit; ``high_time_variance`` separates *estimators* of the
same criterion, by making the split of a trajectory into transitions carry no
information about how the time is spent.

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

import math
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


def high_time_variance(long_reward: float = 10.0, long_time: float = 10.0,
                       short_reward: float = 1.0, short_time: float = 0.1,
                       burst_length: int = 10, steady_reward: float = 1.0,
                       steady_time: float = 1.0) -> SMDPConfig:
    """One long slow transition, then a burst of short fast ones: a holding-time
    variance counterexample.

    ::

        s0 --a--> b1   r=10  tau=10        (rate 1, and 10 of the 11 time units)
        b1 --a--> b2   r=1   tau=0.1       (rate 10)
        ...                                (burst_length of these)
        b10 --a--> s0  r=1   tau=0.1
        s0 --b--> s0   r=1   tau=1         (rate 1, no variance at all)

    Action *a* is a macro-action: committing to it buys one long, slow
    transition and then ``burst_length`` short, fast ones before the next
    decision. Over the cycle it earns ``(10 + 10*1) / (10 + 10*0.1) = 20/11 =
    1.818``, so it beats *b*'s steady 1.0 by 82% — and it also pays 10 on the
    spot against *b*'s 1, so nothing about it is a trap in the usual sense.

    What the cycle *is* built for is holding-time variance. Its 11 transitions
    are split evenly in **time** — 10 units in the slow step, 1 unit across the
    burst — but 1-to-10 in **events**. Every estimator that forgets per
    transition therefore weights the burst ten times too heavily:

    * the pathwise time average, ``sum(r) / sum(tau)``, is 1.818 — the answer;
    * the mean of the per-transition rates is ``(1 + 10*10) / 11 = 9.18``;
    * a fixed-gain ratio of EMAs sits between them and depends on beta. With the
      md's ``beta = 0.5``, ten transitions after the slow step its weight has
      decayed by ``2**-10`` even though only one time unit has passed, and it
      reads 9.58 against a truth of 1.818;
    * a fixed-gain filter in *elapsed time* at the matching ``lambda = -log(1 -
      beta)`` reads 5.50 there instead, because the slow step is forgotten
      according to the 10 units it occupied rather than the single event it was.

    The disagreement is a fixed-gain effect, not an asymptotic bias: as
    ``beta -> 0`` both smoothers converge on 1.818. It is at the gains the agents
    actually run that they part, and by how much names the estimator.

    That the burst is delivered as ``burst_length`` transitions rather than one
    is also the point. Its physical trajectory — one time unit at rate 10 — is
    the same however it is chopped, so an estimator worth the name must return
    the same rho for any ``burst_length`` holding ``burst_length * short_time``
    fixed. The time-domain filter does; a per-transition one drifts toward the
    burst's rate of 10 as the chopping gets finer.

    Finally, rho is not a passive readout here. Under R-learning the two actions
    are compared by ``r - rho*tau`` over the whole option, so *a* is preferred
    exactly while

        rho < (R_cycle - steady_reward) / (T_cycle - steady_time) = 19/10 = 1.9

    which is only 4.5% above the cycle's own true rate. An overshooting rho
    therefore costs rate rather than merely mis-reporting one. The middle column
    below is the rho each estimator averages over the transitions of the always-*a*
    cycle at ``beta = 0.3``, which is deterministic and is what the updates
    actually see; the last two are measured over 60 seeds at the registered
    20,000-decision budget:

    ================================  ==========  ============  ===========
    estimator / agent                   mean rho  keeps *a* in  greedy rate
    ================================  ==========  ============  ===========
    the truth                              1.818             —        1.818
    SMART (cumulative)                     1.817       60 / 60        1.817
    CumulativeWeightedHarmonic (w = r)     1.817       60 / 60        1.817
    SmoothedSMART (time)                   2.488       39 / 60        1.531
    RelaxedSMART (ratio)                   2.868       31 / 60        1.422
    WeightedHarmonic (w = r)               2.868       31 / 60        1.422
    Harmonic (w = 1)                       6.449       29 / 60        1.395
    CumulativeHarmonic (w = 1)             5.496       19 / 60        1.259
    ================================  ==========  ============  ===========

    Two separate things decide whether the option survives, and the table is
    ordered by the first:

    **How far rho overshoots.** Every fixed-gain estimate is an overshoot, which
    is what a fixed gain buys, and among the estimators that forget at the same
    beta the size of it orders the outcome: 2.488 keeps the option 39 times, 2.868
    keeps it 31, 6.449 keeps it 29. The time-domain smoother lands between the
    cumulative rate and the per-transition ratio, exactly where forgetting in
    seconds rather than in events should put it, and that ordering holds at every
    beta.

    **Whether the estimator can come back down.** ``CumulativeHarmonic`` isolates
    this: it overshoots *less* than ``Harmonic`` (5.496, and rock steady against a
    swing from 2.66 to 9.28) and yet keeps the option least often of anything
    here. Once the agent has switched to *b* the stream becomes a flat rate of 1,
    and ``Harmonic``'s EMA is back under the 1.9 threshold within five
    transitions, so the option becomes attractive again; the cumulative version
    still holds every burst sample ever seen and is at 3.25 four hundred
    transitions later. A steady overshoot you cannot forget is worse than a larger
    one you can.

    ``WeightedHarmonic`` and ``RelaxedSMART`` land on the same row because on this
    environment they are the same agent: identical rho by the collapse below, and
    the same R-learning target. (``agents/experemental_harmonic_r.py`` breaks that
    tie deliberately, by dividing the reward-weighted agents' advantage by
    ``|rho|``; it is a scaling of the TD error, not a reordering, so it moves the
    learning dynamics rather than the threshold.)

    The collapse holds on any domain whose rewards are all strictly positive --
    as these are. With the weight set to the reward, the positive branch averages
    ``(tau/r) * r = tau`` against ``r``, so the harmonic mean *is* the ratio of
    those two averages and the sign mix reduces to that one branch:
    ``WeightedHarmonic``'s rho becomes ``RelaxedSMART``'s and
    ``CumulativeWeightedHarmonic``'s becomes SMART's, in both cases to floating
    point. Only the unit-weight pair stays a harmonic mean of the rates here, and
    it is pulled hardest of all toward the burst's rate of 10 -- the cumulative one
    to exactly ``11 / 2 = 5.5``.
    """
    if burst_length < 1:
        raise ValueError("burst_length must be >= 1")
    for name, value in (("long_time", long_time), ("short_time", short_time),
                        ("steady_time", steady_time)):
        if value <= 0:
            raise ValueError(f"{name} must be > 0")

    burst = [f"b{i}" for i in range(1, burst_length + 1)]
    transitions: Table = {
        ("s0", A): [Transition(burst[0], 1.0, long_reward, long_time)],
        ("s0", B): [Transition("s0", 1.0, steady_reward, steady_time)],
    }
    for i, state in enumerate(burst):
        target = burst[i + 1] if i + 1 < burst_length else "s0"
        transitions[(state, A)] = [Transition(target, 1.0, short_reward, short_time)]

    cycle_reward = long_reward + burst_length * short_reward
    cycle_time = long_time + burst_length * short_time
    return SMDPConfig(
        states=["s0"] + burst,
        actions=[A, B],
        transitions=transitions,
        start_state="s0",
        note=(f"a@s0: one tau={long_time:g} step at rate "
              f"{long_reward / long_time:g} then {burst_length} tau="
              f"{short_time:g} steps at rate {short_reward / short_time:g}, "
              f"worth {cycle_reward / cycle_time:.4g} overall; b@s0 holds "
              f"{steady_reward / steady_time:g} with no holding-time variance"),
    )


def harmonic_criterion(spread: float = 10.0, steady_rate: float = 0.5,
                       steady_time: float = 100.0) -> SMDPConfig:
    """The third way to average a rate, and the only environment that isolates it.

    ::

        s0 --a--> s0   p=0.5  r=spread  tau=1        (rate 10)
        s0 --a--> s0   p=0.5  r=1       tau=spread   (rate 0.1)
        s0 --b--> s0   p=1.0  r=50      tau=100      (rate 0.5, no variance)

    There are three ways to average reward against time over a set of
    transitions, and they are three different numbers:

    ==========================  =================  =========  ==============
    criterion                   what it is         arm *a*    estimator
    ==========================  =================  =========  ==============
    ratio of means              ``E[r] / E[tau]``     1.000    CumulativeTimeRate
    mean of the rates           ``E[r / tau]``        5.050    CumulativeStepRate
    reciprocal mean of inverse  ``1 / E[tau / r]``    0.198    CumulativeHarmonic
    ==========================  =================  =========  ==============

    ``ratio_vs_step_rate`` separates the first two. This one exists for the third,
    which nothing else in the suite measures: ``1 / E[tau/r]`` is the reciprocal of
    the **mean time spent per unit of reward**, and the unweighted harmonic
    estimator is the only one in ``average_rates`` that computes it.

    Arm *a* is deliberately symmetric -- ``(spread, 1)`` and ``(1, spread)`` with
    equal probability -- which makes the three values a tidy geometric family:
    ``E[r/tau] * 1/E[tau/r] = 1`` exactly, so the time-average is the geometric
    mean of the other two, whatever ``spread`` is. Arm *b* is a constant rate of
    0.5, chosen to sit *between* the harmonic value and the other two.

    So the criteria disagree about which arm to take:

    * by ratio of means, *a* wins 1.000 to 0.500;
    * by mean of the rates, *a* wins 5.050 to 0.500;
    * by mean time per unit reward, **b** wins 0.500 to 0.198 -- it costs 2 time
      units per unit of reward against *a*'s 5.05.

    ``b`` is also the *long* arm, at ``tau=100`` against *a*'s mean of 5.5, and
    that is not decoration. Under R-learning ``b`` is preferred exactly while
    ``rho < (50 - 5.5) / (100 - 5.5) = 0.471``, so only an estimator whose rho
    lands below 0.471 takes it. Both fixed points are stable at the library's
    ``exploration_rate=0.1``: a whole-history ratio-of-means rho reads 0.666 while
    greedy on *a* and 0.503 while greedy on *b*, staying above the line either
    way; the harmonic rho reads 0.211 and 0.434, staying below it either way.

    Note the harmonic arm is the *worse* one by the standard time-average
    criterion, so an agent that takes it scores 0.5 against 1.0 on
    ``lifetime_rate``. That is the point rather than a defect, exactly as in
    ``gemini``: the correct action here is correct **under a stated criterion**,
    and reporting it without that label would be meaningless.

    **Measured: no agent takes it, and the reason is structural.** Over 20 seeds
    ``Harmonic`` picks *b* 0/20 and ``CumulativeHarmonic`` 2/20, despite their rho
    landing at 0.329 and 0.229 -- well below the 0.471 threshold that should
    select it. Instrumenting the run shows why: ``Harmonic``'s Q values *diverge*,
    running 38 -> 389 -> 1904 -> 7640 over 20,000 decisions, while SMART's stay
    bounded.

    R-learning's relative-value update ``Q <- Q + alpha[(r - rho*tau) + max Q' -
    Q]`` has a fixed point only when ``rho`` equals the **gain of the policy being
    run**, because that is what makes ``r - rho*tau`` average to zero along it.
    Pinning rho and measuring the drift of ``Q`` per 1,000 decisions on this
    environment, against the predicted ``alpha * (r - rho*tau)``:

    ==========  ============  ==========  ===========
    pinned rho  r - rho*tau   Q drift     predicted
    ==========  ============  ==========  ===========
    0.198          4.411       +2893          --
    0.500          2.750        +254        +248
    1.000          0.000          -10           0
    1.500         -2.750         -272        -248
    ==========  ============  ==========  ===========

    Only ``rho = 1.0``, arm *a*'s time-average gain, is stationary. A harmonic rho
    is a different quantity, so the iteration has no fixed point, ``Q`` grows
    without bound, and the greedy choice ends up reflecting which arm was updated
    more often against a drifting baseline rather than any advantage.

    So the honest reading of this environment: ``1/E[tau/r]`` is a real criterion
    that only the unweighted harmonic estimator computes, and this graph pins its
    value exactly -- but it cannot be *optimised* by feeding it to ``r - rho*tau``.
    Doing so is a type error, not a better gain estimate. Note the corollary: the
    reward-weighted harmonic **is** the time average on any positive domain (see
    :func:`high_time_variance`), which is precisely why that one behaves and this
    one does not.
    """
    if spread <= 1.0:
        raise ValueError("spread must be > 1, or the two outcomes coincide")
    if steady_time <= 0 or steady_rate <= 0:
        raise ValueError("steady_rate and steady_time must be > 0")
    transitions: Table = {
        ("s0", A): [Transition("s0", 0.5, spread, 1.0),
                    Transition("s0", 0.5, 1.0, spread)],
        ("s0", B): [Transition("s0", 1.0, steady_rate * steady_time, steady_time)],
    }
    mean_rates = 0.5 * (spread + 1.0 / spread)
    return SMDPConfig(
        states=["s0"],
        actions=[A, B],
        transitions=transitions,
        start_state="s0",
        note=(f"a is worth 1 by ratio-of-means, {mean_rates:.3g} by mean-of-rates "
              f"and {1 / mean_rates:.3g} by mean-time-per-reward; b holds "
              f"{steady_rate:g} under all three"),
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


class _PlusConstant:
    """A duration process shifted by a fixed amount, delegating reset and reseed.

    Used to *fold* a constant return leg into the transition that precedes it: one
    decision per visit carrying the same reward over the same total time, instead
    of a second transition that pays nothing. Delegation matters -- the engine
    resets and reseeds whatever ``SMDPConfig.processes`` collects, and a bare
    lambda would silently freeze the wrapped process's noise.
    """

    def __init__(self, inner, offset: float):
        self.inner = inner
        self.offset = float(offset)

    def __call__(self, *args, **kwargs) -> float:
        return float(self.inner(*args, **kwargs)) + self.offset

    def reset(self) -> None:
        self.inner.reset()

    def reseed(self, seed: int) -> None:
        self.inner.reseed(seed)

    def __repr__(self) -> str:
        return f"_PlusConstant({self.inner!r}, {self.offset})"


# --------------------------------------------------------------- drifting worlds
#: The ``log_scale`` sweep the sincoslog family was originally studied over
#: (``np.logspace(-5, -1, 30)``), slowest drift first.
SINCOSLOG_LOG_SCALES = tuple(round(10.0 ** (-5.0 + 4.0 * i / 29.0), 6) for i in range(30))


def sincoslog(log_scale: float = 1e-3, frequency: float = 1.0,
              slope: float = 1.0 / 20.0, return_reward: float = 1.0,
              fold_return: bool = False) -> SMDPConfig:
    """An oscillating, exponentially growing action against a steady ramp.

    ::

        s1 --a--> s2   r = slope * n                       tau ~ N(1, 0.1)
        s1 --b--> s2   r = (sin(n) + 10) * 10^(n*ls)       tau = (cos(n) + 10) * 10^(n*ls/2)
        s2 --a--> s1   r = return_reward                   tau = 1

    Action *a* is a linear ramp on a unit clock, so its rate climbs steadily and
    without bound. Action *b* oscillates *and* scales exponentially in both
    reward and holding time; the reward's exponent grows twice as fast as the
    holding time's, so its rate grows exponentially too — just not monotonically.

    Which action is better therefore changes at least twice. Measured as
    reward/time for the always-*a* and always-*b* policies over one uncapped
    trajectory at the default ``log_scale=1e-3``: *b* leads at the start (1.03
    against *a*'s 0.64 over 20 decisions, since the ramp starts near zero), *a*
    overtakes at around 84 decisions, and by 4000 decisions *a* is at 25.5 against
    *b*'s 5.4 — while at ``log_scale=1e-2`` the same 4000 decisions put *b* at
    5e9. So the small end of the sweep is a world that eventually rewards the
    steady action and the large end one that eventually rewards the volatile one,
    with the crossover between ``1e-3`` and ``1e-2``.

    The two reward processes are hooked to each other, so both advance on every
    decision whichever action is taken: the world drifts on its own rather than
    only when sampled. The holding times are deliberately *not* hooked, so *b*'s
    clock advances only when *b* is chosen.

    **``return_reward`` deviates from the source, which paid 0.** ``s2`` exists
    only to hand control back to ``s1`` -- it offers one action and carries no
    decision -- yet the alternation makes that leg *half of every trajectory's
    transitions*, against 6.9% of its elapsed time. Paying nothing for it made
    sincoslog a worst case for any estimator that treats a reward of zero as an
    occurrence without a rate: the harmonic family books those 50% of steps into
    its sign mix but never charges their duration to a branch, which deflated its
    rho by the ratio of count share to time share -- a factor of 0.54 here, enough
    to be a large part of why that family alone keeps the correct arm. Paying 1
    removes every zero reward from the environment, and with it an artefact of how
    a two-armed choice was encoded as a two-state cycle. Set it back to 0.0 to
    recover the source's version; :data:`SINCOSLOG_CROSSOVER_UNPAID_RETURN` and
    ``visits_to_overtake(..., return_reward=0.0)`` describe that one. The change
    moves the overtake counts by under 0.3% and leaves which sweep points are
    traps untouched.

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

    if fold_return:
        # One decision per visit: the return leg's time is folded into the arm that
        # precedes it and its reward dropped, so the physical process is unchanged
        # (same reward, same total time per visit) but there is no transition that
        # pays nothing. s2 offered no choice, so no control is lost.
        transitions: Table = {
            ("s1", A): [Transition("s1", 1.0, reward_a, _PlusConstant(duration_a, 1.0))],
            ("s1", B): [Transition("s1", 1.0, reward_b, _PlusConstant(duration_b, 1.0))],
        }
        return SMDPConfig(
            states=["s1"], actions=[A, B], transitions=transitions,
            start_state="s1",
            note=(f"sincoslog with the empty return leg folded in: linear ramp "
                  f"(slope {slope:g}) versus sin/cos-log drift at "
                  f"log_scale={log_scale:g}, one decision per visit"))

    transitions: Table = {
        ("s1", A): [Transition("s2", 1.0, reward_a, duration_a)],
        ("s1", B): [Transition("s2", 1.0, reward_b, duration_b)],
        ("s2", A): [Transition("s1", 1.0, return_reward, 1.0)],
    }
    return SMDPConfig(
        states=["s1", "s2"],
        actions=[A, B],
        transitions=transitions,
        start_state="s1",
        note=(f"linear ramp (slope {slope:g}) versus sin/cos-log drift at "
              f"log_scale={log_scale:g}; the better action flips over time"),
    )


def ripening_bait(build_reward: float = 19.0, build_slope: float = 1.0 / 10_000.0,
                  build_time: float = 10.0, bait_offset: float = 6.0,
                  bait_amplitude: float = 5.5, bait_period: float = 2000.0,
                  bait_time: float = 5.0) -> SMDPConfig:
    """A slowly ripening arm against one that ripens fast and rots, in one state.

    ::

        s0 --build--> s0   r = 19 + n/10000                     tau = 10
        s0 --bait---> s0   r = 6 + 5.5*sin(2*pi*n / 2000)       tau = 5

    Both are self-loops on the only state, so there is no bookkeeping transition,
    no reward is ever zero, and the whole environment is the choice between two
    drifting arms. Both reward processes are hooked together, so the world drifts
    per decision whichever arm is pulled.

    *build* climbs slowly and without bound: rate ``1.9`` at the start, ``2.1`` by
    the end of a 20,000-decision run, ``2.0`` on average. *bait* is stationary in
    the long run -- mean rate ``6/5 = 1.2`` -- but swings between ``0.1`` and
    ``2.3`` on a 2,000-decision cycle. So *build* is the correct arm over the
    horizon by 67%, while *bait* genuinely out-rates it at every peak, since 2.3
    stays above *build*'s ceiling of 2.1 for the whole run.

    The shallow ramp is forced, not chosen. A sinusoidal bait peaks at most at
    twice its own mean, so "the bait leads at peaks" and "build leads on average"
    together require ``build_max < 2 * build_mean`` -- a build arm that rises
    steeply outruns the bait's ceiling early and the trap stops existing.

    Under R-learning the arms are compared by ``r - rho*tau``, so *build* survives
    exactly while

        rho < (r_build - r_bait) / (tau_build - tau_bait)

    At a bait trough that threshold is ``(20 - 0.5) / 5 = 3.9``, far above any
    rho these estimators reach, so every agent holds *build*. At a bait peak it
    is ``(20 - 11.5) / 5 = 1.7``, and that is where the environment bites: an
    agent whose rho exceeds 1.7 drops *build* for the ripe bait, and one whose
    rho stays below it does not. *build*'s own true rate is 2.0, so a correctly
    estimated rho sits *above* the threshold and switches -- which is locally
    right, the bait really is the better arm at a peak -- while a harmonic rho,
    pinned near 1.3, holds on through the peak and never pays the cost of
    switching back.

    The environment therefore asks one question: **after a ripe phase ends, does
    the agent come back?** That depends entirely on whether its rho came back, and
    the estimators split on exactly that. A rho that averages the *rates*
    arithmetically -- in time (``CumulativeTimeRate``, ``SmoothedSMART``) or per
    transition (``ExponentialMovingRatioRate``) -- is pulled up by the ripe phase's
    large rewards and stays above 1.8. A rho that averages them *harmonically* is
    pinned near the smallest rate it has seen and cannot be inflated by large
    values at all. Measured directly on a stream that is 90% build and 10% ripe
    bait, mean rho over the settled tail:

    ==========================  ========
    estimator                      rho
    ==========================  ========
    SMART / CumulativeTimeRate     1.155
    RelaxedSMART                   1.136
    SmoothedSMART                  1.136
    Harmonic (w = 1)               0.667
    CumulativeHarmonic (w = 1)     0.661
    ==========================  ========

    Two things this needs that are easy to get wrong, both established by
    measurement rather than assumed. The bait's excursion is in **reward** at a
    comparable duration: with equal rewards the unweighted harmonic mean of rates
    is *identically* the time average (``n*R / sum(tau)``), and a short
    high-rate bait separates nothing. And the durations differ by only 2x: they
    have to differ at all or rho drops out of the comparison, but a large gap
    would put the threshold back on a knife edge.

    Note this separates only the **unweighted** harmonics. Their reward-weighted
    counterparts collapse onto ``RelaxedSMART`` and SMART on any strictly positive
    domain, and this one is strictly positive by construction.
    """
    if bait_amplitude >= bait_offset:
        raise ValueError("bait_amplitude must be < bait_offset, or the bait pays "
                         "zero or negative rewards")
    for name, value in (("build_time", build_time), ("bait_time", bait_time)):
        if value <= 0:
            raise ValueError(f"{name} must be > 0")
    if build_time == bait_time:
        raise ValueError("build_time must differ from bait_time, or rho drops out "
                         "of the greedy comparison entirely")

    reward_build = make_reward("linear", start=build_reward, step=build_slope)
    reward_bait = make_reward("sin", amplitude=bait_amplitude,
                              frequency=2.0 * math.pi / bait_period,
                              offset=bait_offset)
    # Hooked both ways, as sincoslog does: the world ripens per decision, not per
    # visit to a particular arm, so an agent cannot freeze the bait by ignoring it.
    reward_build.register_hook(reward_bait)
    reward_bait.register_hook(reward_build)

    transitions: Table = {
        ("s0", A): [Transition("s0", 1.0, reward_build, build_time)],
        ("s0", B): [Transition("s0", 1.0, reward_bait, bait_time)],
    }
    return SMDPConfig(
        states=["s0"],
        actions=[A, B],
        transitions=transitions,
        start_state="s0",
        note=(f"a climbs from rate {build_reward / build_time:g} without bound; b "
              f"cycles {(bait_offset - bait_amplitude) / bait_time:g}-"
              f"{(bait_offset + bait_amplitude) / bait_time:g} every "
              f"{bait_period:g} decisions and is only ever briefly better"),
    )


def rotting_bait(build_reward: float = 20.0, build_time: float = 14.0,
                 bait_reward: float = 10.0, bait_decay: float = 0.998,
                 bait_floor_after: int = 2000, bait_time: float = 1.0) -> SMDPConfig:
    """A steady long arm against a short one that starts far better and rots.

    ::

        s0 --build--> s0   r = 20                       tau = 14   rate 1.43
        s0 --rot----> s0   r = 10 * 0.998**n            tau = 1    rate 10 -> 0.18

    The companion to :func:`ripening_bait`, built to the one shape the harmonic
    estimator's advantage actually needs: a **10x spread in the per-transition
    rates** (1.43 against 10 early and 0.18 late), where a harmonic mean departs
    furthest from an arithmetic one. ``ripening_bait``'s sinusoid could not express
    that -- a sine peaks at twice its mean, which pins both arms into one narrow
    band -- so the drift here is an exponential decay instead.

    *build* is the correct arm over the horizon, 1.43 against *rot*'s 0.41
    averaged over 20,000 decisions, and *rot* genuinely leads for its first few
    hundred. Under R-learning *build* survives while

        rho < (20 - r_rot) / (14 - 1)

    which is 1.525 once the bait has rotted -- just above *build*'s own rate of
    1.43, so a correctly estimated rho holds it and a rho still inflated by the
    early ripe phase does not.

    Requires ``with_rho_trick=False`` to be meaningful: the trick updates rho only
    on on-policy transitions, and the exploratory samples are the entire mechanism.

    **Measured: the rho separation appears and still does not convert.** Over 20
    seeds with the trick off, mean rho is 1.088 for ``CumulativeHarmonic`` and
    1.162 for ``Harmonic``, against 1.423-1.429 for ``SMART``, ``RelaxedSMART``,
    ``SmoothedSMART`` and ``ContinuousRLearning`` -- a real 1.3x split in the
    predicted direction, and it vanishes entirely with the trick on (everything
    lands on 1.429). But every one of those values is *below* the 1.525 threshold,
    so all sixteen agents hold *build* on all 20 seeds and the lifetime rates sit
    between 1.426 and 1.434. Being lower does not help when nobody is above the
    line.

    That is the structural obstruction, and it is not specific to this shape. For
    a low rho to rescue an arm, some rival estimator has to be *above* the
    threshold -- but the threshold sits just above the optimal arm's own rate by
    construction (that is what makes the arm optimal), and a correctly estimated
    rho converges to exactly that rate. So a correct estimator never fails in
    steady state, and an estimator that is merely *lower* than correct has nothing
    to rescue. Pushing a rival above the line needs persistent inflation, and the
    whole-history estimators wash it out over a long run while the fixed-gain ones
    forget it by design.

    The harmonic pair does edge SMART on lifetime here -- 1.4326 against 1.4258 --
    but that is the learning transient, not a policy difference.
    """
    if not 0 < bait_decay < 1:
        raise ValueError("bait_decay must be in (0, 1)")
    if build_time == bait_time:
        raise ValueError("build_time must differ from bait_time")

    reward_build = make_reward("constant", value=build_reward)
    reward_rot = make_reward("linear", start=bait_reward, step=bait_decay,
                             drift_op="mul", max_drifts=bait_floor_after)
    reward_build.register_hook(reward_rot)
    reward_rot.register_hook(reward_build)

    transitions: Table = {
        ("s0", A): [Transition("s0", 1.0, reward_build, build_time)],
        ("s0", B): [Transition("s0", 1.0, reward_rot, bait_time)],
    }
    floor = bait_reward * bait_decay ** bait_floor_after
    return SMDPConfig(
        states=["s0"],
        actions=[A, B],
        transitions=transitions,
        start_state="s0",
        note=(f"a holds rate {build_reward / build_time:.3g}; b starts at "
              f"{bait_reward / bait_time:g} and rots to {floor / bait_time:.3g}"),
    )


def sincoslog_self_similar(
        s: float = 0.01, *, return_reward: float = 0.0, frequency: float = 1.0,
        dur_factor: float = 0.5, o_r: float = 200.0, amp_r: float = 20.0,
        o_d: float = 3.0, amp_d: float = 2.8, a_r: float = 40.0,
        a_tau: float = 0.75, tau_ret: float = 1.0,
        log_base: float = 10.0) -> SMDPConfig:
    """Sin/cos-log with **both** arms on one exogenous envelope.

    ::

        G(k) = 10^(k*s)                 H(k) = 10^(k*s*dur_factor)     k = decisions

        s1 --A--> s2   r = a_r * G(k)                        tau = a_tau * H(k)
        s1 --B--> s2   r = (amp_r*sin(k) + o_r) * G(k)       tau = (amp_d*cos(k) + o_d) * H(k)
        s2 --A--> s1   r = return_reward                     tau = tau_ret

    Ported from ``PythonProject3/sincoslog_env.py``. The legacy :func:`sincoslog`
    pits an exploding arm against a *stationary-scale* one, which makes
    ``tau_B >> tau_A, tau_ret`` and collapses the decision to a knife edge: an
    agent prefers the long arm while ``rho < rho_crit = (r_B - r_A)/(tau_B -
    tau_A)``, and as ``tau_B`` runs away ``rho_crit -> r_B/tau_B -> rho*``, so the
    relative margin goes to zero and the benchmark measures only whether an
    estimator is biased low.

    Scaling *every* reward by ``G`` and *every* duration by ``H`` fixes that. The
    margin is envelope-independent and therefore constant over an episode:

        M = (a_tau / (o_d - a_tau)) * (1 - (a_r/a_tau) / (o_r/o_d))

    which is ``0.0667`` at these defaults -- arm B is optimal, and an agent needs
    ``rho`` below ``1.0667 * rho*`` to see it. Geometry now sets the difficulty and
    ``s`` independently sets how many decades the estimator has to chase inside one
    episode; in the legacy env those two were the same confounded knob.

    Both reward processes are cross-hooked and both duration processes are too, so
    the envelope advances once per decision whichever arm is pulled. That matters:
    it makes the drift exogenous, so the two fixed policies are comparable. The
    legacy config deliberately leaves the *durations* unhooked, which is part of
    why its two arms are not on a common clock.

    See :func:`self_similar_margin`. ``return_reward`` is the ``s2 -> s1`` leg the
    legacy env pinned at 0; here the arm rewards dominate it as soon as the
    envelope grows, so the ranking is invariant to it over a wide range.
    """
    if o_d <= a_tau:
        raise ValueError("o_d must exceed a_tau, or arm B is not the long arm")
    reward_b = make_reward("sin_log", amplitude=amp_r, frequency=frequency,
                           offset=o_r, log_base=log_base, start_exp=0.0, log_scale=s)
    reward_a = make_reward("sin_log", amplitude=0.0, frequency=frequency,
                           offset=a_r, log_base=log_base, start_exp=0.0, log_scale=s)
    duration_b = make_duration("cos_log", amplitude=amp_d, frequency=frequency,
                               offset=o_d, log_base=log_base, start_exp=0.0,
                               log_scale=s * dur_factor)
    duration_a = make_duration("cos_log", amplitude=0.0, frequency=frequency,
                               offset=a_tau, log_base=log_base, start_exp=0.0,
                               log_scale=s * dur_factor)
    # One clock for the rewards and one for the durations; each advances exactly
    # once per decision regardless of which arm was chosen.
    reward_a.register_hook(reward_b)
    reward_b.register_hook(reward_a)
    duration_a.register_hook(duration_b)
    duration_b.register_hook(duration_a)

    transitions: Table = {
        ("s1", A): [Transition("s2", 1.0, reward_a, duration_a)],
        ("s1", B): [Transition("s2", 1.0, reward_b, duration_b)],
        ("s2", A): [Transition("s1", 1.0, return_reward, tau_ret)],
    }
    return SMDPConfig(
        states=["s1", "s2"], actions=[A, B], transitions=transitions,
        start_state="s1",
        note=(f"both arms on one 10^(k*{s:g}) envelope; arm B optimal with a "
              f"constant relative margin of "
              f"{self_similar_margin(o_r, o_d, a_r, a_tau):.4f}"))


def self_similar_margin(o_r: float = 200.0, o_d: float = 3.0, a_r: float = 40.0,
                        a_tau: float = 0.75) -> float:
    """``M = (rho_crit - rho*)/rho*`` for :func:`sincoslog_self_similar`.

    Envelope-independent, so it does not decay over an episode. ``M > 0`` means arm
    B is optimal and an agent sees that exactly while ``rho < (1 + M) * rho*``.
    """
    if o_d <= a_tau:
        return math.inf
    return (a_tau / (o_d - a_tau)) * (1.0 - (a_r / a_tau) / (o_r / o_d))


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
