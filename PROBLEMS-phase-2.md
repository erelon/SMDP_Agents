Phase 2 feasibility report
==========================

Status: STOP. The required equivalences use incompatible weighting and domain
semantics. Implementing them in one parameterized power-mean rate family would
require power-specific special cases rather than a single coherent estimator.

Conflicting definitions
-----------------------

Let x_i = r_i / t_i. The SMDP-native duration-weighted power rate proposed in
the plan is

    rho_p = (sum_i a_i t_i x_i^p / sum_i a_i t_i)^(1/p),

where a_i is 1 for the cumulative estimator and is the exponential recency
coefficient for the smoothed estimator. At p=0, the corresponding expression
uses the weighted average of log(x_i).

At p=1 this gives

    rho_1 = sum_i a_i r_i / sum_i a_i t_i.

For cumulative coefficients, this is CumulativeTimeRate. For exponential
coefficients, it is the reward-EMA/duration-EMA ratio used by RelaxedSMART.

For positive rewards and unit update weights, WeightedHarmonicRate instead
computes the event-weighted harmonic mean

    H_event = sum_i a_i / sum_i a_i x_i^(-1).

The p=-1 member of the duration-weighted family is

    H_time = sum_i a_i t_i / sum_i a_i t_i x_i^(-1)
           = sum_i a_i t_i / sum_i a_i t_i^2 / r_i.

H_event and H_time are unequal whenever holding times vary in a relevant way.
Consequently, one weighting convention cannot simultaneously provide the
required p=1 SMART/RelaxedSMART equivalences and the p=-1
WeightedHarmonicRate equivalence.

Minimal counterexample
----------------------

Use two transitions:

    (reward, duration) = (1, 1), (9, 3)
    local rates         = 1, 3

For cumulative estimates:

    event arithmetic p=1 = (1 + 3) / 2 = 2
    CumulativeTimeRate    = (1 + 9) / (1 + 3) = 2.5

Thus feeding local rates directly to CumulativePowerMean cannot reproduce
CumulativeTimeRate.

For normalized exponential estimates with beta=0.5, the two unnormalized
recency coefficients after the second update are a_1=0.25 and a_2=0.5.
Normalization cancels in ratios. The results are:

    RelaxedSMART p=1
      = (0.25*1 + 0.5*9) / (0.25*1 + 0.5*3)
      = 19/7
      = 2.714285714285714

    event-weighted p=1
      = (0.25*1 + 0.5*3) / (0.25 + 0.5)
      = 7/3
      = 2.333333333333333

    duration-weighted p=-1
      = (0.25*1 + 0.5*3) / (0.25*1 + 0.5*1)
      = 7/3
      = 2.333333333333333

    WeightedHarmonicRate p=-1
      = (0.25 + 0.5) / (0.25*1 + 0.5*(1/3))
      = 9/5
      = 1.8

These values were also checked against the current CumulativeTimeRate,
ExponentialMovingRatioRate, and WeightedHarmonicRate implementations.

Reuse and domain problem
------------------------

The Phase 1 classes accept update(value) only. A duration-weighted mean needs
both a transformed local rate and a duration weight. It can be assembled from
two arithmetic accumulators (one for t*x^p and one for t), but that does not
use a CumulativePowerMean(p) or NormalizedExponentialPowerMean(p, beta) as the
power-mean operation. Alternatively, adding a weight argument to Phase 1 would
violate the frozen Phase 1 interface and the instruction that each update
accept one value.

There is also a domain mismatch. Phase 1 implements the standard CRRA/power
mean for strictly positive values. WeightedHarmonicRate separately handles
negative and zero rewards using positive, negative, and zero partitions. That
operator is not the standard p=-1 power mean and has no direct CRRA certainty-
equivalent interpretation. Reproducing its full behavior would require a
p=-1-specific signed extension outside the Phase 1 classes.

Options
-------

1. Use one SMDP-native, duration-weighted CRRA family for positive local rates.
   Keep the p=1 equivalences with CumulativeTimeRate and RelaxedSMART, but drop
   the exact p=-1 equivalence with Harmonic/WeightedHarmonicRate. Treat those
   existing agents as event-weighted, signed heuristic estimators.

2. Use one event-weighted power-mean family. This naturally reuses the Phase 1
   classes and matches WeightedHarmonicRate at p=-1 only for positive rewards
   and compatible update weights. It does not match CumulativeTimeRate at p=1,
   does not match RelaxedSMART at smoothed p=1, and still does not support the
   signed/zero harmonic extension.

3. Explicitly support two objectives, for example `weighting="event"` and
   `weighting="duration"`, and add weighted observations to the Phase 1 API.
   Equivalence tests would then target different modes. This is coherent but
   requires revising Phase 1 and the Phase 3 requirements before continuing.

4. Special-case p=1 by delegating to SMART-style estimators and p=-1 by
   delegating to WeightedHarmonicRate. This would make the requested tests pass
   but would not be a single power-mean family, would not elegantly reuse
   power_means.py, and would be discontinuous in implementation semantics as p
   changes. This option is not recommended.

Recommendation
--------------

Choose option 1: use the positive-domain, duration-weighted CRRA objective from
the article because it is the SMDP-native definition and has the exact p=1
risk-neutral reduction. Revise the plan so p=-1 is tested against the direct
duration-weighted harmonic formula rather than WeightedHarmonicRate, and remove
the Phase 3 requirement that RiskSmoothedR(p=-1) equal Harmonic. If preserving
the existing harmonic behavior is more important, choose option 2 and remove
the SMART and RelaxedSMART equivalence requirements instead.

No power_rates.py, Phase 3 agent, or Phase 4 agent was created because the plan
requires execution to stop at this gate.


# Fixes Proposed

## Codex wrote (see also above)

Execution stopped at the Phase 2 feasibility gate, as required.
The required equivalences conflict for unequal durations:
- Duration-weighted power means match SMART and Relaxed SMART at p=1.
- Event-weighted power means match WeightedHarmonicRate at p=-1.
- One consistent power-mean family cannot provide both.
- Phase 1’s positive-only domain also cannot reproduce the harmonic estimator’s signed/zero behavior.

The equations, counterexample, alternatives, and recommendation are documented in [PROBLEMS-phase-2](/home/galk/Unisoned/Students/ErelShtusel/zGits/SMDP_Agents/PROBLEMS-phase-2).
No power_rates.py or Phase 3/4 agent files were created. The recommendation is to retain the duration-weighted CRRA definition and remove the exact WeightedHarmonicRate equivalence requirement.

## I replied

ah. you have a different notion of a weighted power ratio than I had in mind. Let's ignore exponential smoothing for now and focus on What if we consider the formula for a *cumulative* , **non-weighted** power ratio as 
$\rho_p =  \left(\frac{1}{K}\cdot \sum_i (r_i / t_i)^p\right)^(1/p)$, where $r_i$ is the reward, $t_i$ the time.  This seems like the event-weighted power mean of the ration r/t. Then the weighted version becomes 
$ \left(\frac{1}{K}\cdot \sum_i w_i\cdot (r_i / t_i)^p\right)^(1/p)$ where $w_i$ is the weight.  Now:
- setting w_i = r_i when p=-1 results in the harmonic cumulative rate being equal to the time-weighted arithmetic average.
- setting w_i = t_i when p=1 gives the time-weighted arithmetic SMDP path-wise time rate (total reward / total time).

## codex confirmed

Yes—the intended object is an event-weighted power mean of the local rates \(x_i=r_i/t_i\). One correction is needed: the weighted version must normalize by the total weight, not by \(K\):

\[
\rho_p =
\left(
\frac{\sum_i w_i(r_i/t_i)^p}{\sum_i w_i}
\right)^{1/p}.
\]

Otherwise, using \(1/K\) causes the scale of the weights to change the result.

With normalized weights, both identities hold:

For \(p=-1\) and \(w_i=r_i\):

\[
\rho_{-1}
=
\left(
\frac{\sum_i r_i(t_i/r_i)}{\sum_i r_i}
\right)^{-1}
=
\frac{\sum_i r_i}{\sum_i t_i}.
\]

For \(p=1\) and \(w_i=t_i\):

\[
\rho_1
=
\frac{\sum_i t_i(r_i/t_i)}{\sum_i t_i}
=
\frac{\sum_i r_i}{\sum_i t_i}.
\]

So the cumulative design can support:

- `weight=1`: event-weighted power mean
- `weight=reward`, `p=-1`: reward-weighted harmonic rate
- `weight=duration`, `p=1`: standard path-wise SMDP rate

This resolves the cumulative weighting conflict identified in `PROBLEMS-phase-2`. The remaining caveat is that ordinary power means require positive rates and positive weights; negative or zero rewards still need separate semantics.

## I replied with a question about whether w_i can be found for any other power mean

If we wanted to set the the weight for p=0 and p=2 so that they match time-weighted arithmetic average, what weights should we select?

## Codex says:  "NO"

Let

\[
x_i=\frac{r_i}{t_i},
\qquad
A=\frac{\sum_i r_i}{\sum_i t_i}
=\frac{\sum_i t_i x_i}{\sum_i t_i}.
\]

There is no fixed local choice such as \(w_i=r_i\) or \(w_i=t_i\) that makes the \(p=0\) or \(p=2\) mean equal \(A\) for every sequence. The weights must depend on the overall target \(A\).

For \(p=2\), one valid choice is

\[
\boxed{w_i=\frac{t_i}{x_i+A}}.
\]

Then

\[
\frac{\sum_i w_i x_i^2}{\sum_i w_i}=A^2,
\]

so the weighted quadratic mean is \(A\).

For \(p=0\), using the weighted geometric mean

\[
\rho_0=\exp\left(\frac{\sum_i w_i\log x_i}{\sum_i w_i}\right),
\]

choose

\[
\boxed{
w_i=t_i\frac{x_i-A}{\log x_i-\log A}
}.
\]

When \(x_i=A\), use the continuous-limit value

\[
w_i=t_iA.
\]

This produces \(\rho_0=A\).

The drawback is circularity: both weighting formulas require knowing \(A\) already. Thus they prove such weights exist, but they are not useful for estimating the time-average rate. The particularly useful identities at \(p=-1\) and \(p=1\) are special because their weights can be computed locally from each transition.




