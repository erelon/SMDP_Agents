Created branch test/baseline-characterization and added the baseline test suite.

  Key files:

  - SMDP_Agents/TEST_PLAN.md
  - SMDP_Agents/tests

  Coverage includes:

  - Core agent behavior, RNG, reset, convergence, and restricted actions.
  - Q-learning and continuous-time discounting.
  - R-learning and rho-trick behavior.
  - SMART, Relaxed SMART, Harmonic, and Weighted Harmonic.
  - Positive, negative, zero, mixed rewards, fractional/zero durations.
  - MAB and UCB variants.
  - Oracle and random agents.
  - Conditional Deep Q, Gaussian MLP, rollout buffer, and PPO tests.
  - Package import behavior with missing optional dependencies.

  Results:

  Ran 43 tests
  OK (skipped=7)

  The seven skipped tests require PyTorch, which is not installed in the current environment. They will run automatically where
  PyTorch is available. Standard-library tracing exercised every executable line in the tabular agent modules.

  The tests also characterize existing quirks, including:

  - Several resets do not reseed RNG state.
  - SMART variants raise on zero total duration.
  - Continuous UCB carries a tiny pseudotime offset.
  - Continuous UCB does not increment its action count during learning.
  - Importing the top-level package fails when PyTorch is unavailable.

  No production algorithm files were changed.

