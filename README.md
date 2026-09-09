# RL Agents

A modular reinforcement learning library supporting tabular, bandit, deep Q-learning, and PPO algorithms, with first-class support for **Semi-Markov Decision Processes (SMDPs)** — environments where actions have variable durations.

---

## Installation

```bash
pip install -r requirements.txt   # numpy, torch + gymnasium, pandas, matplotlib
```

`import agents` pulls in the deep agents, so **torch is required even to use the tabular ones**, and to run the test suite. The one exception is `rate_comparison.py`, a standalone CLI that loads `agents/average_rates.py` by path so it stays dependency-free.

`gymnasium`, `pandas` and `matplotlib` are needed only by `examples/` — the
environments are Gymnasium environments, `btc_market` reads a CSV, and the plots
are matplotlib. Nothing in `agents/` imports any of them.

---

## How to Use

### 1. Instantiate an agent

Every agent requires a `name` and an `action_space` (a list of valid actions). Optional hyperparameters vary by algorithm.

```python
from agents import QLearning, RLearning, SMART, UCB, DeepQWrapper

agent = QLearning(
    name="my_agent",
    action_space=[0, 1, 2],
    learning_rate=0.1,
    discount_factor=0.9,
    exploration_rate=0.1,
)
```

To allow an agent to query the environment for state-specific available actions (rather than always using the full action space), pass the environment via `env=`:

```python
agent = QLearning(name="agent", action_space=[0, 1, 2], env=my_env)
# env must implement: get_available_actions(state) -> list
```

### 2. The training loop

All agents share the same four-method interface:

```python
agent.reset()  # call once before each episode

for episode in range(n_episodes):
    state = env.reset()

    while not done:
        action = agent.act(state)            # ε-greedy (exploration + exploitation)
        next_state, reward, done, time = env.step(action)
        agent.learn(state, action, reward, next_state, time)
        state = next_state
```

The `time` argument to `learn()` represents the **holding time** of the action — how long it took. For standard MDPs, pass `time=1`. For SMDPs, pass the actual elapsed duration (see the SMDP section below).

### 3. Evaluation (greedy policy)

```python
action = agent.eval(state)  # no exploration, pure greedy
```

### 4. Seeding and reset semantics

```python
agent.set_seed(123)
agent.reset()
```

`reset()` restores the agent to its constructed state: it clears the Q-table and the reward-rate accumulators, and **re-seeds the RNG from `self.seed`**. Consequently an agent reset before every episode draws the *same* exploration sequence in each one. If you want exploration to keep advancing across episodes, either reset once before training rather than per-episode, or call `set_seed()` with a per-episode seed before each reset.

### 5. Deep Q-Learning wrapper

Wrap any `ContinuousQLearning`-derived agent with a neural network. The wrapped agent's reward-rate logic (R-Learning, SMART, Harmonic, etc.) is fully preserved.

```python
import torch.nn as nn
from agents import SMART, DeepQWrapper

base = SMART(name="smart", action_space=[0, 1, 2], learning_rate=0.001)

net = nn.Sequential(
    nn.Linear(state_dim, 64), nn.ReLU(),
    nn.Linear(64, 64),        nn.ReLU(),
    nn.Linear(64, 3),
)

agent = DeepQWrapper(
    agent=base,
    network=net,
    replay_buffer_size=10_000,
    batch_size=32,
    target_update_freq=500,
)
```

### 6. PPO and its average-reward variants

The PPO agents are env-agnostic (torch + numpy); you provide the rollout loop. Each average-reward variant reuses a tabular agent's `calc_new_rho` through multiple inheritance, so the rate logic lives in exactly one place.

```python
from agents import RsmartPPO, RolloutBuffer

agent = RsmartPPO(obs_dim, act_dim)
buf = RolloutBuffer()
obs = envs.reset()                                   # [B, obs_dim]
for itr in range(n_itr):
    buf.clear()
    for t in range(agent.batch_T):
        action, value, logp = agent.act(obs)
        next_obs, reward, terminated, truncated, _ = envs.step(action.numpy())
        buf.add(obs, action, reward, terminated, truncated, value, logp, time=tau)
        obs = next_obs
    stats = agent.update(buf, agent.value(obs))      # bootstrap from the final obs
```

`buf.add(..., time=)` is the per-step dwell (`1.0` for an MDP, the macro-step duration for an SMDP). Adding a variant is one line — inherit the deep core plus a tabular rate agent, and pick how a batch feeds the rate updater via `rho_reduce` (`"mean"`, `"sum"`, or `"none"` for per-transition):

```python
class WeightedHarmonicPPO(PPO, WeightedHarmonic):
    longrun = True
    rho_reduce = "none"      # the pos/neg split needs each reward's sign
```

A variant that changes the *correction* rather than the rate overrides `rate_residual`, which is where `r - \rho\tau` enters the GAE recursion. That hook matters: PPO never calls the tabular `set_target`, so overriding that instead would compile, run, and silently do nothing.

### 7. Policy change tracking

Agents track whether the last `learn()` call changed the greedy policy for the updated state:

```python
changed = agent.get_policy_changed()        # bool
last_at = agent.last_policy_changed_at      # episode index (set by the caller)
steps   = agent.step_count                  # learn() calls since construction/reset
```

---

## Algorithms

| Class | Variant | Paper |
|---|---|---|
| `QLearning` | Tabular Q-Learning (discrete time) | Watkins & Dayan, [*Q-Learning*](https://link.springer.com/article/10.1007/BF00992698), Machine Learning 1992 |
| `ContinuousQLearning` | Q-Learning for SMDPs (continuous / variable time) | Bradtke & Duff, [*Reinforcement Learning Methods for Continuous-Time Markov Decision Problems*](https://proceedings.neurips.cc/paper_files/paper/1994/file/07871915a8107172b3b5dc15a6574ad3-Paper.pdf), NeurIPS 1994 |
| `RLearning` | Average-reward R-Learning (discrete time) | Schwartz, [*A Reinforcement Learning Method for Maximizing Undiscounted Rewards*](https://www.sciencedirect.com/science/chapter/monograph/abs/pii/B9781558603073500459?via%3Dihub), ICML 1993 |
| `ContinuousRLearning` | R-Learning for SMDPs | Mahadevan, [*Average Reward Reinforcement Learning: Foundations, Algorithms, and Empirical Results*](https://doi.org/10.1007/BF00114727), Machine Learning 1996 |
| `SMART` | Sample Mean Average Reward Technique | Das et al., [*Solving Semi-Markov Decision Problems Using Average Reward Reinforcement Learning*](https://doi.org/10.1287/mnsc.45.4.560), Management Science 1999 |
| `RelaxedSMART` | Exponentially-smoothed variant of SMART | Gosavi, [*Reinforcement Learning for Long-Run Average Cost*](https://doi.org/10.1016/S0377-2217(02)00874-3), EJOR 2004 |
| `SmoothedSMART` | SMART smoothed in *elapsed time* rather than per transition | — |
| `Harmonic` | Harmonic Moving Average rho estimator | Shtossel et al., [*A Harmonic Mean Formulation of Average Reward RL in SMDPs*](https://arxiv.org/abs/2605.04880), ALA 2026 *(official implementation)* |
| `WeightedHarmonic` | Reward-weighted Harmonic Moving Average | Shtossel et al., [*A Harmonic Mean Formulation of Average Reward RL in SMDPs*](https://arxiv.org/abs/2605.04880), ALA 2026 *(official implementation)* |
| `CumulativeHarmonic` | The same harmonic ρ over the whole history, with no forgetting | — |
| `CumulativeWeightedHarmonic` | The reward-weighted variant of the above | — |
| `ExperimentalWeightedHarmonic` | `WeightedHarmonic` with the TD advantage divided by $\|\rho\|$ | — |
| `ExperimentalCumulativeWeightedHarmonic` | The cumulative counterpart of the above | — |
| `EpsilonGreedyMAB` | ε-greedy Multi-Armed Bandit (sample mean, discrete) | Robbins, [*Some Aspects of the Sequential Design of Experiments*](https://www.jstor.org/stable/2236094), 1952 |
| `ContinuousEpsilonGreedyMAB` | ε-greedy MAB with time-averaged rewards (SMDP) | — |
| `UCB` | Upper Confidence Bound bandit (discrete) | Auer et al., [*Finite-time Analysis of the Multiarmed Bandit Problem*](https://link.springer.com/article/10.1023/A:1013689704352), Machine Learning 2002 |
| `ContinuosUCB` | UCB with time-averaged rewards (SMDP) | — |
| `DeepQWrapper` | Neural network Q-function around any of the above | Mnih et al., [*Human-level control through deep reinforcement learning*](https://www.nature.com/articles/nature14236), Nature 2015 |
| `PPO` | Clipped-surrogate PPO with GAE (discounted, no rate correction) | Schulman et al., [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347), 2017 |
| `SmartPPO` | PPO with the SMART cumulative rate correction | Das et al. 1999 (rate) + Schulman et al. 2017 |
| `RsmartPPO` | PPO with the Relaxed SMART smoothed rate correction (APO) | Gosavi 2004 (rate) + Schulman et al. 2017 |
| `HarmonicPPO` | PPO with the Harmonic Moving Average rate correction | Shtossel et al. 2026 (rate) + Schulman et al. 2017 |
| `SmoothedSmartPPO` | PPO with the elapsed-time smoothed rate correction | — |
| `ExperimentalWeightedHarmonicPPO` | PPO with the reward-weighted harmonic rate, residual divided by $\|\rho\|$ | — |
| `RandomAgent` | Uniformly random baseline | — |
| `Oracle` | Optimal-action oracle (requires environment secret) | — |

### Reward-rate estimators

`agents/average_rates.py` holds the averaging primitives the rate-based agents are built from. It has no dependencies beyond the standard library, so it can be imported on its own.

| Class | Estimate |
|---|---|
| `CumulativeTimeRate` | $\sum r_i / \sum \tau_i$ — backs `SMART` |
| `CumulativeStepRate` | mean of the per-transition rates $r_i / \tau_i$ |
| `ExponentialMovingRatioRate` | $\mathrm{EMA}(r) / \mathrm{EMA}(\tau)$ — backs `RelaxedSMART` |
| `WeightedHarmonicRate` | signed harmonic moving average — backs `Harmonic` / `WeightedHarmonic` |
| `CumulativeWeightedHarmonicRate` | the same, over the whole history — backs `CumulativeHarmonic` / `CumulativeWeightedHarmonic` |
| `ExponentialMovingTimeRate` | rate smoothed by elapsed time rather than by step count |
| `NormalizedExponentialMovingTimeRate` | the same, debiased for the zero initialization — backs `SmoothedSMART` |
| `ExponentialMovingAverage`, `NormalizedEMA`, `TimeDecayedEMA`, `CumulativeAverage` | the underlying averaging blocks |

Every estimator takes `update(reward, duration, weight=1.0)` and returns the new `rho`. Durations must be finite and non-negative; all but `WeightedHarmonicRate` additionally require them to be strictly positive, since they divide by the duration.

To compare the estimators on a recorded run, `rate_comparison.py` writes one CSV column per estimator from a whitespace-delimited `reward duration` log:

```bash
python rate_comparison.py tests/test-data/sincoslog.data --beta 0.3
```

---

## Semi-Markov Decision Processes (SMDPs)

### What is an SMDP?

A standard Markov Decision Process (MDP) assumes that every action takes exactly one time step. In many real-world problems this is too restrictive: a robot maneuver might take 50 ms or 2 s, a network packet might be transmitted in 1 ms or 10 ms, and a high-level "option" in hierarchical RL might span dozens of primitive steps.

A **Semi-Markov Decision Process** generalises the MDP by allowing actions (or *options*) to have variable, stochastic holding times. The key quantities are:

- **State** $s$: as in an MDP.
- **Action** $a$: chosen at each decision epoch.
- **Reward** $r$: the total (or average) reward accumulated during the action.
- **Holding time** $\tau$: how long the action took before the next decision epoch.

The Markov property still holds at decision epochs — it is only between epochs that the process runs in continuous time.

### How this library handles it

The `time` parameter passed to `learn()` is the holding time $\tau$. Each algorithm scales its updates accordingly:

**Discounted Q-Learning** (`ContinuousQLearning`): the discount is applied over the full holding time, not just one step:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma^{\tau} \max_{a'} Q(s',a') - Q(s,a) \right]$$

**Average-reward R-Learning** (`ContinuousRLearning`): the system average reward-rate $\rho$ (reward per unit time) is subtracted proportionally to the holding time:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r - \rho\cdot\tau + \max_{a'} Q(s',a') - Q(s,a) \right]$$

**SMART**: $\rho$ is the ratio of accumulated reward to accumulated time, making it naturally unit-consistent across variable-duration actions:

$$\rho = \frac{\sum_i r_i}{\sum_i \tau_i}$$

**RelaxedSMART**: the same ratio, but over exponentially smoothed reward and time, so the estimate tracks a drifting rate instead of averaging the whole history:

$$\rho = \frac{\mathrm{EMA}_\beta(r)}{\mathrm{EMA}_\beta(\tau)}$$

**SmoothedSMART**: also a tracking estimate, but smoothed in *elapsed time* instead of per transition. One filter rather than a ratio of two, driven by the realised rate $r_k/\tau_k$ and forgetting at $\lambda = -\log(1-\beta)$ per unit time:

$$\rho_k = e^{-\lambda \tau_k}\,\rho_{k-1} + \left(1 - e^{-\lambda \tau_k}\right)\frac{r_k}{\tau_k}$$

which is the continuous-time filter $\dot\rho = \lambda\,(q(t) - \rho)$ integrated across the transition. A $\tau = 1$ transition has gain exactly $\beta$, so the two smoothers are comparable at the same `rho_learning_rate`. Two things follow when the holding times vary: there is no $\mathbb{E}[\mathrm{EMA}(r)/\mathrm{EMA}(\tau)] \ne \mathbb{E}[r]/\mathbb{E}[\tau]$ ratio bias to acquire, and the estimate is *segmentation-invariant* — splitting a transition into pieces covering the same time at the same rate leaves it unchanged, since $e^{-\lambda\tau_1}e^{-\lambda\tau_2} = e^{-\lambda(\tau_1+\tau_2)}$. The `high_time_variance` example environment is built to separate the two.

**Harmonic / WeightedHarmonic**: $\rho$ is a harmonic moving average of the per-transition rates. Each sign of the reward is averaged in its own branch — a harmonic mean across a sign change is meaningless — and the branches are mixed by how often each sign occurs:

$$\rho = \frac{H_+ p_+ + H_- p_-}{p_+ + p_- + p_0}, \qquad H_\pm = \frac{\mathrm{EMA}_\beta(w\,\mathbb{1}_\pm)}{\mathrm{EMA}_\beta\left(w\,\mathbb{1}_\pm \tau_i / r_i\right)}$$

where $p_+, p_-, p_0$ are exponential averages of the sign indicators and the weight is $w = 1$ for `Harmonic` and $w = r_i$ for `WeightedHarmonic`. Because $\tau$ appears only in the numerator of $\tau_i / r_i$, these two accept $\tau = 0$ (an instantaneous transition contributes nothing to the branch and merely decays it); the ratio-based estimators above require $\tau > 0$.

`CumulativeHarmonic` and `CumulativeWeightedHarmonic` are the same two with running means in place of the EMAs — no forgetting, the whole history. With unit weight and strictly positive rewards the first is the harmonic mean of the rates, $n / \sum_i \tau_i / r_i$; with the reward as the weight the $r$ in $(\tau/r)\cdot r$ cancels and the second's $\rho$ becomes $\sum r / \sum \tau$, i.e. SMART's.

All four use R-learning's plain $r - \rho\tau$ target; only $\rho$ differs between them.

**Experimental: the $|\rho|$-scaled target.** `agents/experemental_harmonic_r.py` holds `ExperimentalWeightedHarmonic` and `ExperimentalCumulativeWeightedHarmonic`, identical to the two reward-weighted agents except that the advantage is divided by $|\rho|$:

$$Q(s,a) \leftarrow Q(s,a) + \alpha\left[\frac{r - \rho\,\tau}{|\rho|} + \max_{a'} Q(s',a') - Q(s,a)\right]$$

At $\rho > 0$ that advantage is $r/\rho - \tau$: the reward re-expressed as the *time* it would take to earn at the current average rate, less the time it actually took — an advantage in time saved rather than in reward, which is the natural pairing for a $\rho$ that is itself reward-weighted. Because $|\rho|$ is a positive constant within one decision it can never reorder two actions, so the greedy policy at a fixed $\rho$ is exactly R-learning's; what moves is the scale of every TD error, and hence the effective learning rate, which now runs inversely with $\rho$. The absolute value is load-bearing — dividing by a *signed* $\rho$ inverts the ordering wherever $\rho < 0$, which on `hell_or_heaven` took `WeightedHarmonic` from 100% correct to 0% and its rate from 0.854 to −0.529. $\rho$ is 0 on the first update of every run (the target is built before `calc_new_rho` sets a rate), where the plain target is used instead.

Still experimental: across the 29-environment sweep the scaling helps sharply where the rate collapses toward zero (`non_stationary`, 0.093 → 3.242) and hurts across whack-a-mole, whose learning rates were grid-searched against the plain target. Give these agents their own `learning_rate` before judging them.

**Bandit variants** (`ContinuousEpsilonGreedyMAB`, `ContinuosUCB`): action values are estimated as total reward divided by total holding time, yielding a *reward-rate* estimate per action rather than a per-step average.

### Discrete-time variants

`QLearning` and `RLearning` are the discrete-time counterparts of `ContinuousQLearning` and `ContinuousRLearning`: they ignore the `time` you pass and charge one unit per transition. That clamp is the `holding_time(time)` hook rather than a `learn()` override, so it survives wrappers that replace `learn` — a `QLearning` inside a `DeepQWrapper` still discounts by $\gamma^1$, not $\gamma^\tau$. Override it to define a different clock:

```python
class HalfStep(ContinuousQLearning):
    def holding_time(self, time):
        return time / 2
```

### When to use SMDP agents

Use the `Continuous*` variants whenever:

- Actions correspond to macro-actions or options that span multiple primitive steps.
- The environment reports elapsed wall-clock or simulation time rather than a step count.
- You are doing hierarchical RL and the lower-level controller runs for a variable number of steps.

For standard gym-style environments where every step has the same duration, pass `time=1.0` to any agent and the SMDP formulas reduce to their standard MDP equivalents.

---

## Example environments

`examples/` holds 28 SMDP environments behind one interface, a runner that measures
every tabular agent on them, and the report and figures generated from the results.
See [examples/README.md](examples/README.md) for the catalogue; the short version:

```bash
python -m examples.envs                     # list every environment
python -m examples.run --all --seeds 8      # the full sweep -> examples/results/*.json
python -m examples.make_report              # -> examples/results/REPORT.md
python -m examples.make_plots               # -> examples/results/plots/*.png
```

An environment there is a Gymnasium environment that reports the holding time of
each action in `info["tau"]`, plus `state_of(obs)` for a hashable state,
`get_available_actions(state)` for legality, and `secret()` where the optimal policy
is known analytically. `examples.envs.check_smdp_env` audits all of that and runs
against every registered environment in the test suite.

They are grouped by the question each one asks. Several are small enough to have an
analytic answer, and where they do the number is asserted in the tests rather than
merely asserted in prose — `gemini`, for instance, is worth 10 under a time-average
but 1 under a ratio of expectations, so which action an agent takes there names the
criterion it is really optimising.

| Family | Question |
|---|---|
| `criterion` | Which definition of "reward per unit time" does this agent optimise? |
| `risk` | Same rate, different spread — what is the agent's risk attitude? |
| `horizon` | Does it look far enough ahead to refuse a poisoned jackpot? |
| `drift` | Can it follow a world where the best action changes? |
| `rates` | Reward earned at a rate over a random duration, so reward and time are coupled. |
| `whack_a_mole` | The same grid world with τ ≡ 1 and with variable τ — does time-awareness buy anything? |
| `market` | Real hourly Bitcoin bars, holding times derived from the returns. |

---

## Running the tests

```bash
python -m tests
```

The runner in `tests/__main__.py` executes the suite in contiguous algorithm groups and colors successes green, skips and warnings yellow, and failures and errors bright red. The final summary reports the nonzero ok/warning/skip/failure counts per group alongside unittest's overall totals.

Run one group by naming its test class, or one case by naming its method:

```bash
python -m tests tests.test_r_learning.RLearningTests
python -m tests tests.test_r_learning.RLearningTests.test_rho_trick_skips_non_greedy_action
```

The same names work with plain unittest when uncolored output is needed:

```bash
python -m unittest -v tests.test_r_learning.RLearningTests
python -m unittest discover -s tests          # full suite, standard output
```

Each `tests/test_*.py` module covers the `agents/` module it is named after, and imports it directly — the suite requires torch, so nothing is skipped or conditionally loaded.

For line coverage, if `coverage.py` is installed:

```bash
coverage run --branch -m unittest discover -s tests
coverage report -m
```

### Parity policy

The suite pins current behavior, including edge cases that are quirks rather than design. A change that alters observable behavior must update the affected test **in the same commit** and state the old and new behavior in the message — a test renamed to describe a bug is how a bug becomes a feature.
