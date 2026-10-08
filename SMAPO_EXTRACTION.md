# SMAPO extraction — what was kept, what was dropped, what is reproducible

This records the decisions behind `agents/smapo.py`. The source implementation
accumulated a large number of experimental flags while the algorithm was being
developed; almost all of them were tested and not adopted. This repo carries the
**final algorithm only**.

## 1. How "final" was decided

Not from memory or from the development notes, which disagree with each other in
places. From the configuration of the runs that produced the paper's final
figure, read back from the run metadata:

```
longrun                true          discount                1.0
eta_debias             true          gae_lambda              0.9
centre_advantage_only  true          lr_eta                  0.3
entropy_pressure       0.005         entropy_loss_coeff      0.0
entropy_pressure_track true          entropy_track_memory_steps  2_000_000
entropy_pressure_boost true          entropy_boost_efold_steps     400_000
entropy_warmup_iters   20            pg_divisor              false
normalize_advantage    false         rm_vbias_coeff          0.0
```

Two of these are worth flagging because the development notes say otherwise.

* **`entropy_pressure_track` and `entropy_pressure_boost` are ON.** The source
  repo's notes still describe both as experimental, default-off and not shipped.
  The paper's final runs use both, and the paper's text describes them directly
  ("a bounded multiplicative boost raises `c_H` while the starvation signal is
  active", "the calibrated form delivered the searched `p` throughout training,
  where a frozen coefficient drifted away from it"). The runs and the paper
  agree, so they are part of the algorithm and are **on by default here**.
* **`pg_divisor` is false**, i.e. the `1/mean(tau)` factor on the SMDP advantage
  is NOT applied. It was tested and found to be criterion-invariant — a positive
  per-batch constant cannot move maximisers, gradient direction or fixed points.
  This repo's PPO never had it, so there was nothing to remove and no flag
  exists.

## 2. Kept

| mechanism | where |
|---|---|
| undiscounted objective, `r - rho*tau` in the TD residual | `SMAPO.rate_residual` |
| the rate estimator, by inheritance | `APO` / `RsmartSMAPO` / `SmartSMAPO` / `SmoothedSmartSMAPO` |
| A-centering | `SMAPO.shape_advantage` |
| calibrated entropy pressure, with tracking and the starvation boost | `SMAPO._update_entropy_coeff`, `SMAPO._boost_step` |
| debiased rate estimators | already in `agents/average_rates.py` (`NormalizedEMA`) |
| value-bias correction `rm_vbias_coeff` | inherited from `PPO`; a searched hyperparameter, kept as a parameter |

Controller rates are specified in **environment steps**, not iterations, because
the number of iterations per unit of simulated time depends on the dwell regime.
`SMAPO._batch_steps` does the conversion from the batch's own summed holding time.

## 3. Dropped, with the reason

Every one of these was an arm that was run and not adopted. None appears in the
final configuration, and none is referenced by the paper.

| dropped | what it was | why it is not here |
|---|---|---|
| `entropy_time_pressure` | `c_H` scaled by mean dwell instead of advantage scale | tested across three dwell regimes; the predicted ordering did not hold |
| `adv_nstep` | a hard n-step advantage window replacing GAE's exponential blend | indistinguishable from GAE at matched horizon — an alternative, not an improvement |
| `eta_consistency_weight` | per-sample model-consistency weighting of the rate estimator | no effect in the shipped configuration; the one regime where it helped is unexplained and was not adopted |
| `auto_entropy` | entropy as a constraint solved by dual ascent | the dual rediscovered the right scale but the schedule never reached it; superseded by specifying the ratio directly |
| `residual_eta_beta` | closed-loop rate estimator | halved the rate error and changed no outcome |
| `profiled_centring` | dwell-profiled rather than mean centring | the dwell-correlated component of the offset is not distinguishable from zero |
| `exact_value_centring` | an exact form of the value-constraint correction | acts on the critic target; the offset that matters lives in the advantage |
| `normalize_advantage` | z-scoring the advantage | mutually exclusive with A-centering and not the published choice. It remains a `PPO` option because it predates this work; `SMAPO` overrides the hook and does not use it |
| `pg_divisor` | the `1/mean(tau)` advantage factor | criterion-invariant; see above |

## 4. What this repo can and cannot reproduce

**The algorithm is here in full. Most of the paper's environments are not.** The
MuJoCo SMDP environments are forked control XMLs plus hold-until-target wrappers
that live in the experiment repo, not here.

| paper figure | environment | reproducible here |
|---|---|---|
| episodic vs continuing training (Swimmer) | `examples/envs/mujoco_smdp.py` | **algorithm and env yes, numbers no** |
| A-centering (Swimmer, Ant) | `examples/envs/mujoco_smdp.py` | **algorithm and env yes, numbers no** |
| A-centering + pressure (Swimmer, Ant) | `examples/envs/mujoco_smdp.py` | **algorithm and env yes, numbers no** |
| PPO with/without calibrated pressure (Swimmer) | `examples/envs/mujoco_smdp.py` | **algorithm and env yes, numbers no** |
| MDP vs SMDP, sample efficiency (Swimmer) | `examples/envs/mujoco_smdp.py` | **algorithm and env yes, numbers no** |
| Whack-a-Mole champions | `examples/envs/kinova_wam.py` | **algorithm yes, numbers no** — the env is here; the hyperparameter sweep that selected the champions is not |
| Whack-a-Mole deep vs tabular | `examples/envs/kinova_wam.py` | **yes in principle** — both agent families and the env are here |

"Numbers no" everywhere above means the same thing and it is worth being precise
about it: every figure plots the median over 30 seeds of a CHAMPION
configuration, and those champions were chosen by a hyperparameter sweep that is
not in this repo. The algorithm, the environments and the training loop are all
here, so the experiments can be run — but reproducing a specific curve means
re-running the search, and `entropy_pressure` in particular is environment-
specific and was searched per environment.

The two locomotion environments are a faithful port, not a reimplementation: the
hold wrapper, the effort cost and both forked XMLs are the ones used for the
published runs. Measured after porting, under random actions: Ant's mean holding
time is 15.2 frames against the 14-16 of the original runs, and Swimmer's rises
monotonically with the commanded distance, which is the property the SMDP
formulation turns on.

## 5. Resolved while extracting

1. **The pre-SMAPO deep variants are gone.** `RsmartPPO`, `SmartPPO`,
   `HarmonicPPO`, `SmoothedSmartPPO` and `ExperimentalWeightedHarmonicPPO` were
   the same rate estimators without A-centering or pressure; they are superseded
   by the SMAPO variants and were removed. `agents/ppo.py` is now the discounted
   baseline and the rollout buffer, nothing else. Their test coverage was
   retargeted onto the SMAPO variants rather than deleted.
2. **Harmonic survives as `HarmonicSMAPO`.** The tabular harmonic estimators are
   untouched; only the deep wrapper changed.
3. **Both additions can be switched off** — `a_centering=True` and
   `calibrated_pressure=True` are the defaults, and either can be set `False` to
   measure what it contributes. With both off this is average-reward PPO with a
   fixed entropy coefficient.

   `entropy_pressure=0` is **not** the way to disable the bonus and is rejected
   with an error. It would leave the mechanism running and overwrite
   `entropy_loss_coeff` with 0 on every iteration, silently discarding a fixed
   coefficient the caller had set.
4. **The default `entropy_pressure = 0.02` is a placeholder.** The value is
   environment-specific and was searched per environment; the Swimmer champion
   used 0.005. There is no universal setting and the paper claims none.
