# Phase 1 — Power means

## Scope and files

- Add `agents/power_means.py` at the repository root. Do not modify any existing file in this phase.
- Add `tests/test_power_means.py`. Tests should load the new module directly if importing it through a package would pull in optional agent dependencies.
- Provide two public stateful estimators, with final names settled before implementation:
  - `CumulativePowerMean(p)`
  - `NormalizedExponentialPowerMean(p, beta)`
- Give both estimators the common interface `update(value, weight=1.0) -> float`, `reset()`, and a public `value`. Constructors validate that `p` and, where applicable, `beta` are finite; require `0 < beta <= 1`. Updates require a finite positive weight and normalize by the accumulated or exponentially smoothed total weight.

## Mathematical contract

- For observations \(x_1,\ldots,x_n\), implement

  \[
  M_p(x)=\left(\frac{1}{n}\sum_i x_i^p\right)^{1/p},\quad p\ne0,
  \]

  and the continuous limit

  \[
  M_0(x)=\exp\left(\frac{1}{n}\sum_i\log x_i\right).
  \]

- The exponentially smoothed estimator replaces the ordinary average of the transformed values by a normalized EMA, so its first result is the first observation rather than a zero-biased value. Reuse `NormalizedEMA` from `agents/average_rates.py` where doing so preserves this contract. If direct package import is problematic, import the module without executing `agents/__init__.py`; do not copy the EMA implementation.
- Implement `p == 0` as a dedicated log/exp path. Do not approximate it with a very small nonzero `p`. Use numerically stable expressions where useful and document expected overflow/underflow behavior.
- Begin with the mathematically standard positive domain. Require `value > 0` for `p <= 0` and for non-integer powers. Decide and document whether positive integral powers accept negative values; never return a complex number implicitly. Zero and signed inputs outside the selected real-valued contract must raise a clear `ValueError`.
- At `p == 1`, delegate to or compose the existing normalized arithmetic EMA as directly as possible. The cumulative implementation may keep a running sum and count. `reset()` must restore fresh-instance behavior.

## Tests and completion criteria

- Test cumulative and smoothed results item by item for at least `p=-1`, `p=0`, `p=1`, and `p=2`, using hand-computed positive sequences of at least ten values.
- For cumulative means, compare against direct formulas for harmonic, geometric, arithmetic, and quadratic means after every observation.
- For smoothed means, independently compute normalized exponentially decayed transformed moments and compare after every observation for both a small and a large `beta`.
- For `p=1`, compare every smoothed result exactly or to tight floating-point tolerance with an existing `NormalizedEMA(beta)` fed the same values with unit weight.
- Test single-item identity, reset/replay behavior, invalid `p`/`beta`, nonfinite inputs, and the documented zero/negative-domain rules.
- Phase 1 is complete only when its new focused test file passes and no tracked pre-existing file has changed.

# Phase 2 — Power-mean rate estimators

## Scope and feasibility gate

- Add `agents/power_rates.py` at the repository root and `tests/test_power_rates.py` without changing Phase 1 or existing files.
- the weighting convention for the power_rates formulas all begin with the step average.
The general formula for the cumulative weighted power ratio is:
$$
\rho_p = \left(\frac{1}{\sum_i w_i}\cdot \sum_i w_i\cdot (r_i / t_i)^p\right)^(1/p)$ where $r_i$ is the reward given to the update (see the update() API for the WeightedHaronicRate), $t_i$ is the time, and  $w_i$ is the weight
- Confirm $\rho_{-1} = \rho_{1} IFF $w_i = r_i$ for $p=-1$, and $w_i = t_i$ for p=1.  
- Phase 1 estimators accept an optional observation weight, so rate estimators should pass `reward / duration` as the value and delegate weighting directly to the selected power mean.
- If reuse of `power_means.py` is not straightforward and elegant for any required `p`, or if the `p=1` and `p=-1` equivalences require incompatible weighting/domain semantics, create `PROBLEMS-phase-2`. Record the conflicting equations, a minimal counterexample, options for resolving the API or objective, and a recommendation. Then stop implementation and report; do not proceed to Phases 3 or 4.
- Create NormalizedExponentialPowerMeanRate(p,\beta) using the Cumulative version above as the basis.


## Proposed interface and implementation, conditional on passing the gate

- Provide `CumulativePowerMeanRate(p)` and `NormalizedExponentialPowerMeanRate(p, beta)`. Each exposes `update(reward, duration, weight=1.0) -> float`, `reset()`, `value`, and `rho`.
- Validate finite reward and weight and require finite, strictly positive duration. Apply the same local-rate domain rules chosen in Phase 1.
- Make `power_rates.py` import and compose the public classes in `power_means.py`; it must not reimplement the transform/inverse-transform cases for powers and logarithms. If weighted observations require a small generalization of the Phase 1 API, that need must be discovered at the gate and resolved before Phase 1 is declared frozen.
- Handle `p=0` through the geometric/log limit and avoid division by `p`. Preserve normalization of the smoothed estimator from its first update.

## Tests and completion criteria

- Compare both estimators item by item with independent direct calculations on varying reward/duration sequences for `p=-1`, `p=0`, `p=1`, and `p=2`, at multiple `beta` values.  Do this for w_i = 1.0.
- Verify the cumulative `p=1`, time-weighted estimator matches `CumulativeTimeRate` on every update.
- verify the cumulative `p=1` with weight = 1.0 is same as CumulativeStepRate. If the test fails, flag it but continue to other tests.
- Verify the smoothed `p=1` estimator against the selected existing p=1 reference with time-weighted normalized counterpart, or the reward-EMA/duration-EMA ratio. 
- Verify smoothed `p=-1` with weight = 1.0 matches `WeightedHarmonicRate` after every update on the full supported input domain, including the agreed weight behavior. Include unequal durations so an accidental event-weighted/time-weighted match cannot pass. If the intended equivalence is limited to positive rewards, say so in the class and test names; otherwise include positive, negative, zero, and mixed-sign sequences.
- Verify smoothed `p=-1` with weight = reward matches analogous WeightedHarmonicRate with same weight.
- Verify smoothed `p=-1` with weight = reward matches smoothed `p=1` with weight=time.
- Test reset, invalid durations, nonfinite values, domain errors, single observations, and non-unit weights.
- Phase 2 is complete only if the reuse requirement and all declared equivalence tests pass without a `PROBLEMS-phase-2` report.

# Phase 3 — Risk-sensitive rate-based agents

## Scope and design

- Proceed only after Phase 2 completes without a stop report.
- Add `agents/risk_total_r.py` and `agents/risk_smoothed_r.py`. Add exactly corresponding test files `tests/test_risk_total_r.py` and `tests/test_risk_smoothed_r.py`, following the repository convention of one test file per agent module.
- `RiskTotalR` should inherit `ContinuousRLearning`, own a `CumulativePowerMeanRate(p)`, reset it with the agent, and set `self.rho` from its returned estimate in `calc_new_rho`. Model its constructor, error translation, and observable cumulative properties on `SMART` where meaningful.
- `RiskSmoothedR` should inherit `ContinuousRLearning`, own a `NormalizedExponentialPowerMeanRate(p, rho_learning_rate)`, reset it with the agent, and set `self.rho` in `calc_new_rho`. Model lifecycle and constructor forwarding on `Harmonic`/`WeightedHarmonic` and `RelaxedSMART`.
- Expose `p` directly. Document the CRRA mapping `theta = 1 - p`: `p>1` is risk-seeking, `p=1` risk-neutral, `0<p<1` risk-averse, `p=0` logarithmic, and `p<0` increasingly sensitive to low rates. Carry forward the positive/signed-domain limitations from Phases 1 and 2.
- Do not change `ContinuousRLearning.set_target`: these agents alter the learned baseline rate, not the TD target’s reward representation. Avoid adding exports to `agents/__init__.py` unless a later integration task explicitly requests them.

## Tests and equivalence criteria

- Test construction, parameter forwarding, reset, `calc_new_rho`, and item-by-item estimator ownership for several powers including `-1`, `0`, `1`, and `2`.
- Use deterministic sequences with unequal rewards and durations, and test at least two smoothing factors for the smoothed agent.
- Verify `RiskTotalR(p=1)` and `SMART` produce equal `rho` after every transition and equal TD targets/Q-table updates when initialized identically.
- Verify `RiskSmoothedR(p=-1)` and the intended harmonic reference (`Harmonic`, not reward-weighted `WeightedHarmonic`, unless Phase 2 explicitly selects otherwise) produce equal `rho` after every transition and equivalent learning updates.
- Verify `RiskSmoothedR(p=1)` and `RelaxedSMART` produce equal `rho` after every transition and equivalent learning updates.
- These three equivalences must use nontrivial, unequal durations. If any fails because Phase 2 selected incompatible weighting semantics, treat that as a Phase 2 design failure: write/update `PROBLEMS-phase-2`, stop, and report rather than special-casing the agents.
- Run the two new focused test modules, then the existing SMART, harmonic, relaxed-SMART, rate-estimator, and full test suites. Preserve existing behavior outside the new files.

# Phase 4 — Utility-transformed risk-sensitive R-learning

## Scope and interpretation

- Add only `agents/risky_reward_r.py`; do not modify files from earlier phases or any other file. The filename must include the `.py` suffix even though the requested shorthand omitted it.
- Implement a `ContinuousRLearning` subclass based on the article’s CRRA certainty-equivalent proposal for positive local rates \(x=r/t\), with `p = 1 - theta`:

  \[
  u_\theta(x)=
  \begin{cases}
  (x^{1-\theta}-1)/(1-\theta), & \theta\ne1,\\
  \log x, & \theta=1.
  \end{cases}
  \]

- Unlike Phase 3, this agent should make risk sensitivity part of the reward/TD criterion, not merely substitute a different estimator for `rho`. The initial design to validate is a time-additive utility target

  \[
  t\,[u_\theta(r/t)-u_\theta(\rho)] + \max_{a'}Q(s',a'),
  \]

  while maintaining the corresponding time-average utility baseline and exposing its inverse-utility certainty equivalent as `rho`. Derive this carefully against the article before coding; do not assume that inserting a power mean into the ordinary target is equivalent.
- Keep all new helpers required for utility, inverse utility, validation, and baseline state inside `agents/risky_reward_r.py`. Reuse imports from existing modules where possible, but do not alter their APIs or source.

## Domain, checks, and stop rule

- Require `duration > 0` and local rate `reward / duration > 0`, matching the direct CRRA interpretation. Handle `theta == 1` with the logarithmic limit and `theta == 0` as the linear/risk-neutral special case. Document that shifting rates, treating costs, or introducing a sign-partitioned extension changes the objective and is out of scope unless explicitly approved.
- Within the same file, include small pure helper functions or executable self-checks sufficient to validate utility/inverse round trips, the `theta=0` risk-neutral reduction, and selected certainty-equivalent calculations without changing a test file. Normal repository tests can be added only in a later task because this phase forbids modifying any other file.
- Confirm whether the additive transformed-reward TD target and the estimator’s normalization produce a coherent average-reward SMDP algorithm for every supported `theta`. Pay particular attention to units, baseline subtraction, positivity of `rho`, initialization before the first transition, and whether policy ordering is invariant to positive rescaling of rates.
- If the formulation is not straightforward or elegant for any requested power/risk parameter, create only `PROBLEMS-phase-4` in addition to the agent file. Explain the mathematical or architectural conflict, give a minimal example, list viable alternatives, then stop and report. Do not modify Phases 1–3 or any existing file to force compatibility.
- Completion means the new module imports independently, its self-checks pass, `theta=0` reduces to the ordinary risk-neutral target under the documented normalization, and the implementation clearly states that `theta>0` is risk-averse and `theta<0` risk-seeking for positive rates.
