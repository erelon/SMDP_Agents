
I’ll verify the active branch and reread Phase 4 directly from RISK-PLAN.md. Then I’ll derive the utility-baseline update and target carefully before creating only agents/risky_reward_r.py, as the phase requires.

The reread supports interpretation 1 cleanly: maintain the duration-weighted average of CRRA utility, expose its inverse as rho, and use duration × (u(local_rate) − u(rho)) in the TD target. At theta=0, this simplifies exactly to reward − rho × duration. Initializing the utility baseline at zero gives the natural positive certainty equivalent rho=1, avoiding an invalid zero inside CRRA utility.
Interpretation 2—changing only rho while leaving the linear TD reward unchanged—is not supported as a risk-sensitive objective: a common baseline shift cannot distinguish equal-mean actions with different risk. I’ll implement interpretation 1 and document that counterexample in the specifically requested interpretation-2 problem file.

Interpretation 1 is implemented with numerically stable CRRA transforms and a cumulative duration-weighted utility baseline; interpretation 2 is documented as unsupported by a concrete equal-mean/same-duration counterexample. I’ll now run the embedded self-checks plus import, lifecycle, domain, risk-neutral-reduction, and scale-invariance checks without adding a test file.

Phase 4 is implemented on branch `risk_gpt-version`. No Phase 1–3 or existing files were modified.

Created:

- [risky_reward_r.py](/home/galk/Unisoned/Students/ErelShtusel/zGits/SMDP_Agents/agents/risky_reward_r.py)
  - `RiskyRewardR`, derived from `ContinuousRLearning`
  - CRRA-transformed TD reward
  - Duration-weighted utility baseline
  - Certainty-equivalent `rho`
  - Exact risk-neutral reduction at `theta=0`
  - Strictly positive reward-rate domain

- [risky_reward_tools.py](/home/galk/Unisoned/Students/ErelShtusel/zGits/SMDP_Agents/agents/risky_reward_tools.py)
  - Numerically stable CRRA utility and inverse
  - Baseline estimator
  - Validation helpers
  - Embedded checks for round trips, power-mean equivalence, risk-neutral reduction, and scaling invariance

- [PROBLEMS-Phase-4_intrep2.md](/home/galk/Unisoned/Students/ErelShtusel/zGits/SMDP_Agents/PROBLEMS-Phase-4_intrep2.md)
  - Explains why modifying only `rho` while retaining a linear TD reward does not generally optimize the CRRA objective.

All embedded and manual Phase 4 checks pass. No separate test file was added, as required by the plan.