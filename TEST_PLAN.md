# Baseline Characterization Test Plan

## Goal

Lock in the behavior of the current `SMDP_Agents` implementation before the
average-rate and learning-hook refactors described in the workspace `PLAN.md`.
These are characterization tests: they intentionally describe current behavior,
including unusual edge-case behavior, rather than silently correcting it.

## Test groups

1. Core `Agent`
   - action-space validation;
   - environment-specific actions;
   - deterministic RNG/reset behavior;
   - convergence/policy-change detection.
2. Tabular Q-learning
   - table initialization and greedy/exploratory action selection;
   - discrete and duration-discounted TD updates;
   - fractional and zero durations;
   - reset behavior.
3. Average-reward agents
   - R-learning target and rho trick;
   - SMART cumulative reward/time rate;
   - Relaxed SMART EMA ratio;
   - weighted and unweighted harmonic estimators;
   - positive, negative, zero, and mixed rewards;
   - zero-time and zero-denominator behavior.
4. Bandits
   - MAB sample means;
   - continuous MAB reward/time rates;
   - UCB initialization and updates;
   - continuous UCB initialization, rates, and current zero-time exception.
5. Stateless agents and package API
   - oracle validation/delegation;
   - seeded random behavior;
   - public exports when optional dependencies are present;
   - current import failure when PyTorch is absent.
6. Deep Q and PPO, when PyTorch is installed
   - tensor conversion, action selection, TD target, gradient update;
   - replay-buffer threshold, target-network synchronization, reset;
   - Gaussian log probability/entropy;
   - rollout stacking;
   - PPO GAE and one update;
   - SMART, Relaxed SMART, and harmonic PPO rho reduction.

## Execution

Run all dependency-free tests:

```bash
python -m tests
```

The screen runner displays successful tests in green, skips and warnings in
yellow, and failures/errors in bright red. The suite remains compatible with plain
`python -m unittest discover -s tests -v` for tools that require non-colored
standard unittest output.

Tests are executed in contiguous algorithm groups. The final summary reports
only nonzero ok, warning, skip, and failure counts for every group, as well as
unittest's overall totals.

Run one algorithm group by supplying its fully qualified test class:

```bash
python -m tests tests.test_q_and_r_learning.RLearningTests
```

Run one test by supplying its fully qualified test method:

```bash
python -m tests tests.test_q_and_r_learning.RLearningTests.test_rho_trick_skips_non_greedy_action
```

The same names work with plain unittest when non-colored output is required:

```bash
python -m unittest -v tests.test_q_and_r_learning.RLearningTests
python -m unittest -v tests.test_q_and_r_learning.RLearningTests.test_rho_trick_skips_non_greedy_action
```

Deep tests are skipped automatically when PyTorch is unavailable. In a complete
environment, install the repository requirements and rerun the same command.

For line coverage, if `coverage.py` is available:

```bash
coverage run --branch -m unittest discover -s tests
coverage report -m
```

## Parity policy

After refactoring, all characterization tests should continue to pass unless a
behavior change is intentional. An intentional change must update the relevant
test in the same commit and explain the old and new behavior.
