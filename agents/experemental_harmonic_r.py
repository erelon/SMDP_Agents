"""Experimental harmonic agents whose TD target is divided by ``|rho|``.

The shipped agents in :mod:`agents.harmonic_r` all use R-learning's target,
``r - rho*tau + max_a' Q(s', a')``.  The two here keep everything else about
:class:`~agents.harmonic_r.WeightedHarmonic` and
:class:`~agents.harmonic_r.CumulativeWeightedHarmonic` -- the same rho, the same
reward weighting -- and change only that::

    (r - rho*tau) / |rho| + max_a' Q(s', a')

Read at ``rho > 0`` the advantage is ``r / rho - tau``: the reward re-expressed
as the *time* it would take to earn at the current average rate, less the time it
actually took.  An advantage measured in time saved rather than in reward, which
is the natural pairing for a rho that is itself a reward-weighted average.

**Why the absolute value.**  Dividing by a signed rho reverses the comparison
whenever rho is negative, which inverts the policy on any domain whose long-run
rate is negative.  Measured on ``hell_or_heaven``, where the trap loop pays -1
forever, the signed form took rho to -0.991 and the agent from *100% correct on
every seed* to *0% on every seed*, with the lifetime rate falling from 0.854 to
-0.529 -- and self-reinforcing, since taking the bait is what drives rho negative
in the first place.  ``|rho|`` keeps the magnitude scaling and drops the sign, so
the ranking of two actions always matches the plain target's.

**What the scaling does, then.**  Since ``|rho| > 0`` is a positive constant
within one decision, it cannot reorder two actions: the greedy policy at a fixed
rho is exactly R-learning's.  What it changes is the *size* of every TD error, and
so the effective learning rate, which now moves inversely with rho over a run --
small when the rate estimate is inflated, large when it collapses toward zero.
That is the whole of the effect, and it is a real one: on ``non_stationary``,
whose compounding action has a rate decaying to zero, the signed version lifted
``WeightedHarmonic`` from 0.093 to 3.242 and its correct-choice rate from 0% to
70%.

**Still experimental.**  Across the 29-environment sweep the signed version was
worse on 19 of 28 comparable environments and better on 9, the losses
concentrated in whack-a-mole (all positive rewards, so no sign flip -- the ``1/rho``
scaling simply fights learning rates that were grid-searched against the plain
target).  ``|rho|`` removes the catastrophic failure but not that interaction, so
these agents want their own ``learning_rate`` before they are judged.
"""

from .harmonic_r import CumulativeWeightedHarmonic, WeightedHarmonic


class AbsRhoScaledTarget:
    """Mixin: divide the R-learning advantage by ``|rho|``.

    Mixed in *before* an agent class so ``set_target`` resolves here and falls
    through to the plain target when there is no rate to divide by.

    Only ever mix this into a reward-weighted agent.  The unweighted
    ``Harmonic`` subclasses ``WeightedHarmonic`` and ``CumulativeHarmonic``
    subclasses ``CumulativeWeightedHarmonic``, so an experimental *unweighted*
    variant must not be built by subclassing an experimental weighted one -- it
    would inherit this target.
    """

    def set_target(self, reward, time, next_q):
        """``(r - rho*tau) / |rho| + next_q``, or the plain target at ``rho == 0``.

        ``rho`` is zero on the first update of every run -- ``learn`` builds the
        target before ``calc_new_rho`` gets to set a rate -- and again whenever
        the averaged rewards cancel.  The fallback is the target the agent would
        have used anyway before any rate was known.
        """
        if self.rho == 0:
            return super().set_target(reward, time, next_q)
        return (reward - self.rho * time) / abs(self.rho) + next_q


class ExperimentalWeightedHarmonic(AbsRhoScaledTarget, WeightedHarmonic):
    """:class:`~agents.harmonic_r.WeightedHarmonic` with the ``|rho|``-scaled target."""


class ExperimentalCumulativeWeightedHarmonic(AbsRhoScaledTarget,
                                             CumulativeWeightedHarmonic):
    """:class:`~agents.harmonic_r.CumulativeWeightedHarmonic`, same scaled target."""
