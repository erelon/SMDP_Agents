# RL Agents

A modular reinforcement learning library supporting tabular, bandit, and deep Q-learning algorithms, with first-class support for **Semi-Markov Decision Processes (SMDPs)** — environments where actions have variable durations.

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

### 4. Seeding for reproducibility

```python
agent.set_seed(123)
agent.reset()
```

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

### 6. Policy change tracking

Agents track whether the last `learn()` call changed the greedy policy for the updated state:

```python
changed = agent.get_policy_changed()        # bool
last_at = agent.last_policy_changed_at      # episode index (set by the caller)
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
| `Harmonic` | Harmonic Moving Average rho estimator | Shtossel et al., [*A Harmonic Mean Formulation of Average Reward RL in SMDPs*](https://arxiv.org/abs/2605.04880), ALA 2026 *(official implementation)* |
| `WeightedHarmonic` | Reward-weighted Harmonic Moving Average | Shtossel et al., [*A Harmonic Mean Formulation of Average Reward RL in SMDPs*](https://arxiv.org/abs/2605.04880), ALA 2026 *(official implementation)* |
| `MAB` | ε-greedy Multi-Armed Bandit (sample mean, discrete) | Robbins, [*Some Aspects of the Sequential Design of Experiments*](https://www.jstor.org/stable/2236094), 1952 |
| `ContinuesMAB` | ε-greedy MAB with time-averaged rewards (SMDP) | — |
| `UCB` | Upper Confidence Bound bandit (discrete) | Auer et al., [*Finite-time Analysis of the Multiarmed Bandit Problem*](https://link.springer.com/article/10.1023/A:1013689704352), Machine Learning 2002 |
| `ContinuosUCB` | UCB with time-averaged rewards (SMDP) | — |
| `DeepQWrapper` | Neural network Q-function around any of the above | Mnih et al., [*Human-level control through deep reinforcement learning*](https://www.nature.com/articles/nature14236), Nature 2015 |
| `RandomAgent` | Uniformly random baseline | — |
| `Oracle` | Optimal-action oracle (requires environment secret) | — |

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

**SMART / RelaxedSMART**: $\rho$ is estimated as the ratio of accumulated reward to accumulated time, making it naturally unit-consistent across variable-duration actions:

$$\rho = \frac{\sum r_i}{\sum \tau_i}$$

**Bandit variants** (`ContinuesMAB`, `ContinuosUCB`): action values are estimated as total reward divided by total holding time, yielding a *reward-rate* estimate per action rather than a per-step average.

### When to use SMDP agents

Use the `Continuous*` variants whenever:

- Actions correspond to macro-actions or options that span multiple primitive steps.
- The environment reports elapsed wall-clock or simulation time rather than a step count.
- You are doing hierarchical RL and the lower-level controller runs for a variable number of steps.

For standard gym-style environments where every step has the same duration, pass `time=1.0` to any agent and the SMDP formulas reduce to their standard MDP equivalents.
