# Branch · Harmonic Mean RL Insights

**User:** Anonymous  
**Created:** 2026/7/20 22:40:43  
**Updated:** 2026/7/20 23:04:57  
**Exported:** 2026/7/20 23:10:51  
**Link:** [https://chatgpt.com/c/6a5e7958-e980-83eb-8133-fc86ed8c3eda](https://chatgpt.com/c/6a5e7958-e980-83eb-8133-fc86ed8c3eda)  

## Prompt:
09/07/2026, 22:14:00

Here is a paper proposing a method for doing SMDP reinforcement learning. In a toy example with two states but non-stationary reward/holding-time ratio, the harmonic mean RL algorithm beats all other algorithms.  My suspicion is that while it optimizes R/t like all others, it manages to beat the others because it is treats the rewards differently.  But I cannot figure what is going on.
 SMART and Relaxed-SMART assume additive rewards, and that's pretty straightforward.  The Harmonic mean is supposedly computing the same entity (rho, the average reward rate), but it is not converging to the same numbers, and the example shows that in fact it is working much better.

What is it actually doing?  Why, mathematically, is it able to establish that action B is so much better than action A, even though its reward rates are smaller?

## Response:
09/07/2026, 22:14:02

I don't think I can answer this yet, because I don't actually have the paper you're referring to.

From your description, there is something very interesting going on:

- SMART and Relaxed-SMART estimate
  $$
  \rho \approx \frac{\sum_i r_i}{\sum_i t_i},
  $$
  i.e., an arithmetic estimate of the reward rate.

- Your harmonic algorithm also claims to estimate the same long-run reward rate, yet it converges to a different numerical value and chooses a much better policy in a non-stationary example.

That can only happen if **one of the following is true**:

1. **It is not actually estimating the same quantity.**
   The update may look like a reward-rate estimator, but it is converging to
   $$
   E\!\left[\frac{r}{t}\right]
   $$
   or to a weighted version of it, rather than
   $$
   \frac{E[r]}{E[t]}.
   $$

2. **It uses a different weighting of samples.**
   Harmonic means give much larger influence to small values. If the algorithm averages reciprocals or rates, then transitions with short holding times receive much larger weight than SMART gives them.

3. **The estimate is biased but the bias helps control.**
   Remember that in RL the purpose of $\rho$ is not statistical estimation—it appears inside the TD error
   $$
   \delta=r-\rho t+\max Q(s')-Q(s).
   $$
   If the bias causes the TD errors of one action to be consistently larger than another, the policy can improve even if $\rho$ itself is not an unbiased reward-rate estimate.

4. **The harmonic update acts as an adaptive step size.**
   Harmonic averaging often responds much more strongly to rare poor observations than arithmetic averaging. The improvement may come from learning dynamics rather than from the limiting value of $\rho$.

---

### Your remark about action B is especially revealing

You wrote:

> it is able to establish that action B is so much better than action A, even though its reward rates are smaller.

That immediately suggests that the algorithm is **not** ranking actions by their average reward rate.

Instead it is probably ranking them by something like

$$
E[f(r,t)]
$$

for some nonlinear function $f$.

Examples include

- harmonic mean of the per-transition rates $r_i/t_i$,
- reciprocal holding times,
- inverse TD errors,
- weighted rates.

These quantities need not preserve the ordering of

$$
\frac{E[r]}{E[t]}.
$$

---

### Why harmonic means can reverse rankings

Suppose

Action A has rates

$$
1,\;9
$$

and Action B has rates

$$
4,\;5.
$$

Both have arithmetic mean 5.

But

$$
H_A=\frac{2}{1+1/9}=1.8,
$$

while

$$
H_B=\frac{2}{1/4+1/5}=4.44.
$$

The harmonic mean strongly penalizes variability.

So if B has **more consistent reward/time ratios**, a harmonic estimator can strongly prefer it even when the arithmetic reward rate is lower.

This is exactly the kind of phenomenon I'd expect in a non-stationary two-state example.

---

### My suspicion

Given what we've discussed over the last few weeks about your harmonic estimator, my current guess is that it is **not really estimating the average reward rate at all**.

Instead it is implicitly optimizing something closer to

$$
\left(E\!\left[\frac{t}{r}\right]\right)^{-1},
$$

or

$$
H\!\left(\frac{r_i}{t_i}\right),
$$

which is a **risk-sensitive objective**. It rewards policies with stable reward rates and penalizes policies having occasional poor reward/time ratios.

That objective can produce dramatically better behavior in environments where the reward/time ratio is highly non-stationary.

---

**Could you upload the paper?**

Especially the sections containing

- the definition of the harmonic estimate of $\rho$,
- the update equations,
- and the two-state example.

Once I see the exact equations, I can derive mathematically **what quantity the algorithm is actually converging to**, and explain precisely why it separates actions A and B so effectively. I suspect the answer is quite subtle, and it is probably not "just another estimator of the same $\rho$."

## Prompt:
09/07/2026, 22:18:51

here is the paper.

[Attachment: main(3).pdf]

## Response:
09/07/2026, 22:18:53

> **Thinking longer for a better answer**
>
> 
>
> Thought for 9s

It is **not optimizing the same $\rho$**.

SMART estimates

$$
\rho_{\text{SMART}}=\frac{\sum r_i}{\sum \tau_i}
$$

or, in Relaxed-SMART,

$$
\rho_{\text{Relaxed}}=\frac{\operatorname{EMA}(r)}{\operatorname{EMA}(\tau)}.
$$

The harmonic algorithm instead estimates something closer to

$$
\rho_H =
\left(\operatorname{EMA}\left(\frac{\tau_i}{r_i}\right)\right)^{-1}
$$

i.e. a harmonic mean of the **per-transition rates** $r_i/\tau_i$. The paper explicitly says Harmonic $R$-Learning keeps the SMART $Q$-update, but changes the $\rho$ estimator to an incremental modified harmonic mean of sampled reward rates. `main.pdf`

The key mathematical distinction is:

$$
\frac{A(r)}{A(\tau)}
\neq
\frac{1}{A(\tau/r)}
$$

unless

$$
\operatorname{Cov}\!\left(r,\frac{\tau}{r}\right)=0.
$$

The paper states exactly this condition in Theorem 1. So when reward and holding time are coupled, the harmonic estimator is not merely a better estimator of SMART’s quantity; it is estimating a different functional. `main.pdf`

Why does this help action B? Because action B has a **structured coupling** between reward and duration. Its raw visible rate $r_t/\tau_t$ may look smaller early on, but the harmonic update averages reciprocals:

$$
\frac{\tau_t}{r_t}.
$$

That means it is sensitive to the **shape of the reward-time relationship**, not just accumulated reward divided by accumulated time. In the toy example, B’s reward grows with log-scale $v$, while its duration grows with log-scale $v/2$. Therefore its true ratio has an exponential growth trend, even if the first 1000 samples do not yet show the crossover. SMART sees mostly “A has higher observed rate so far.” Harmonic $R$-Learning sees “B’s reciprocal rate structure is improving in a way A’s is not.”

So the honest interpretation is:

$$
\textbf{Harmonic RL is not just optimizing average reward rate.}
$$

It is optimizing a **harmonic/risk-shaped/transformed average of local reward rates**, and in this constructed example that transformation happens to reveal the latent trend of B earlier than arithmetic ratio methods.

That is why it can say B is better even when B’s observed rates are smaller: it is not ranking by the same observed arithmetic rate.

## Prompt:
09/07/2026, 22:23:40

Under what conditions will harmonic do better?  in what "shape of the reward-time relationship" will it work better.

## Response:
09/07/2026, 22:23:40

I think this is actually the central scientific question raised by the paper. After reading it carefully, I don't think the answer is exactly the one the paper gives.

The paper argues that the harmonic mean is preferable because rewards and durations are dependent. I don't think **dependence alone** is the real criterion. Plenty of dependent reward/time processes are still correctly optimized by the arithmetic ratio
$$
\frac{E[R]}{E[T]}.
$$

The harmonic estimator is optimizing a different functional, so it is better only when **that functional is more predictive of future return than the arithmetic ratio**.

Let me make this more precise.

---

## Case 1: Stationary environment

Suppose

$$
(r_t,\tau_t)
$$

are IID.

Then

$$
\rho^*=\frac{E[R]}{E[T]}
$$

is the correct objective.

In this setting I would actually expect SMART to be asymptotically optimal.

The harmonic estimator converges to

$$
H=\frac1{E[T/R]},
$$

which is generally different.

So there is **no reason** harmonic should outperform.

Indeed, your Bitcoin experiments where reward and duration are independent show essentially no advantage. `main.pdf`

---

## Case 2: Positive correlation

Suppose

$$
R=aT.
$$

Then every sample has identical rate

$$
\frac RT=a.
$$

Arithmetic:

$$
\frac{E[R]}{E[T]}=a.
$$

Harmonic:

$$
\frac1{E[T/R]}
=
\frac1{E[1/a]}
=a.
$$

Again, they are identical.

So dependence itself changes nothing.

---

## Case 3: Variable local rates

Suppose instead

$$
R=f(T)
$$

is nonlinear.

Now every transition has its own instantaneous rate

$$
x_t=\frac{R_t}{T_t}.
$$

SMART estimates

$$
\frac{\sum R}{\sum T},
$$

which weights each transition proportionally to its duration.

Harmonic estimates roughly

$$
H(x_1,\ldots,x_n).
$$

This gives much greater influence to **small rates**.

So harmonic is pessimistic.

---

## This immediately tells us what environments favor it

The harmonic mean always emphasizes the worst rates.

Suppose action A produces

|Rate|
|----:|
|100|
|100|
|100|
|1|

Arithmetic mean

$$
75.25.
$$

Harmonic mean

$$
3.88.
$$

Now suppose B produces

|Rate|
|----:|
|25|
|25|
|25|
|25|

Arithmetic

25.

Harmonic

25.

SMART prefers A.

Harmonic prefers B.

So harmonic rewards **consistency**.

---

# But your paper's toy example is subtler

Look carefully at what they constructed.

Action A:

- linear reward
- almost constant duration

Therefore

$$
\frac RT
$$

is almost linear.

Action B:

Reward

$$
10^{vt}\sin(t)
$$

Duration

$$
10^{v t/2}\cos(t).
$$

Therefore

$$
\frac RT
\approx
10^{vt/2}
\frac{\sin(t)+10}
{\cos(t)+10}.
$$

Notice something remarkable:

Both numerator and denominator explode.

But because the denominator grows only with exponent $v/2$, the ratio has an exponential trend.

SMART averages

$$
\frac{\sum R}{\sum T}.
$$

Those huge late rewards are accompanied by huge late durations.

They partially cancel.

The ratio grows only slowly.

The harmonic estimator instead averages

$$
\frac TR.
$$

Late samples have

$$
\frac TR
\approx
10^{-vt/2},
$$

which becomes exponentially tiny.

Tiny reciprocals dominate a harmonic mean.

So the harmonic estimate reacts much earlier to the latent exponential improvement.

I think this is actually why Figure 5 looks the way it does. `main.pdf`

---

# Geometrically

SMART asks

> What is the average slope from the origin to all observations?

The harmonic estimator asks

> What is the reciprocal average slope?

These are fundamentally different statistics.

Whenever the local slopes become progressively steeper,

the reciprocal collapses toward zero much faster than the arithmetic ratio grows.

That makes Harmonic RL "see" accelerating trends earlier.

---

# My conjecture

After reading the paper, I would summarize the regimes where Harmonic RL should outperform as follows.

It should outperform when **all** of these hold:

1. **The reward rate is non-stationary**, rather than merely noisy.

2. **Future rates systematically improve** (or deteriorate) over time, not just fluctuate.

3. **Reward and duration change at different growth rates**, so
   $$
   \frac{R_t}{T_t}
   $$
   has a persistent trend.

4. **Recent transitions are more informative than earlier ones**, so emphasizing the newest (highest-rate or lowest-rate) samples is beneficial.

---

## One thing I would challenge in the paper

The paper repeatedly claims that the harmonic mean is "the mathematically correct" average for rates. I don't think that statement is justified in the context of SMDPs.

The harmonic mean is the correct average **only for a very specific physical situation**: averaging rates over **equal amounts of work** (equal distances, equal volumes, equal file sizes, etc.). In an SMDP, each transition is not necessarily an equal unit of work. A transition is just one sample from a stochastic process. The average reward criterion in SMDPs is derived from renewal theory and is
$$
\frac{E[R]}{E[T]},
$$
not the harmonic mean of per-transition rates.

So I think the empirical success of Harmonic RL comes not because it estimates the same quantity more accurately, but because it **changes the optimization objective** in a way that favors certain classes of non-stationary environments-particularly those with accelerating reward-rate trends and where robustness to occasional poor rates or early misleading observations is beneficial. That distinction is important theoretically: the algorithm appears to be solving a different optimization problem, and characterizing exactly which objective it converges to would make for a strong follow-up analysis.

## Prompt:
09/07/2026, 22:38:40

Interesting.  Would using the harmonic RL have an advantage is some specific applications of finance?  or how about using harmonic RL in specific robotics RL applications?

## Response:
09/07/2026, 22:38:40

I actually think robotics is a much more promising application than finance, but for a very specific reason.

The harmonic estimator seems to prefer **policies that avoid locally poor reward rates**, rather than policies with the largest average reward rate. That makes it look much more like a **risk-sensitive** or **bottleneck-sensitive** objective than a standard average-reward objective.

Here are some examples.

---

# 1. Robot motion planning with unpredictable execution times (Excellent fit)

Imagine a mobile robot choosing between two corridors.

Corridor A:

- usually 1 second
- occasionally 20 seconds because people block it

Corridor B:

- always 4 seconds

Average reward rate might favor A.

But in practice, every once in a while the robot becomes completely stuck.

The harmonic estimator heavily penalizes those occasional terrible rates.

This is exactly what you often want in robotics:

> avoid catastrophic delays rather than maximize average speed.

---

# 2. Dual-arm manipulation (Excellent fit)

You've worked on this exact problem.

Suppose each action consists of

(left target, right target)

The reward depends on

$$
\text{makespan}
=
\max(T_L,T_R).
$$

Arithmetic averaging treats one very slow arm as "washed out" by many good motions.

The harmonic estimator is much less forgiving.

It naturally prefers actions where

both arms finish together.

That's actually closer to minimizing makespan than maximizing average velocity.

---

# 3. Robot navigation under congestion (Excellent fit)

Suppose travel times fluctuate because of humans.

Some paths

```
1 1 1 20 1 1
```

Others

```
4 4 4 4 4
```

The harmonic estimator strongly prefers the second path.

Many warehouse robots would too.

---

# 4. Multi-robot task allocation (Excellent fit)

Suppose one robot repeatedly receives tasks requiring

```
1
1
1
40
```

seconds.

Another receives

```
10
10
10
10
```

Average reward might not care.

Harmonic RL naturally encourages balancing.

Interestingly, this resembles minimizing bottlenecks.

---

# Finance

Here I'm much more skeptical.

---

## Trend following

Your paper's Bitcoin experiment is really exploiting

> accelerating reward rates.

In trend-following strategies,

early profits are small,

later profits become huge.

The harmonic estimator reacts earlier because

$$
\tau/r
$$

shrinks rapidly.

So I can imagine benefits for

- trend following
- momentum strategies
- breakout detection

where reward grows superlinearly.

---

## Market making

Probably not.

Market making seeks

small,

stationary,

approximately IID profits.

Arithmetic averages are exactly the right statistic there.

---

## Portfolio optimization

Possibly.

Imagine two assets.

Asset A

```
+2%
+2%
+2%
-20%
```

Asset B

```
+0.8%
+0.9%
+1.1%
+1.0%
```

The harmonic estimator would strongly prefer B.

That resembles minimizing downside risk.

But finance already has better-established objectives:

- Sharpe ratio
- Sortino ratio
- CVaR
- Entropic utility

So harmonic RL would need to demonstrate that it captures something these do not.

---

# Where I think it could really shine

After reading your paper, I think the strongest applications are actually **robotics tasks with synchronization or bottlenecks**.

For example:

### Swarm robotics

Suppose a swarm repeatedly transports objects.

One robot getting delayed delays the whole swarm.

The harmonic estimator naturally dislikes these bottlenecks.

---

### Multi-robot exploration

Suppose exploration cycles take

```
5
5
5
40
```

seconds because one robot repeatedly gets trapped.

Average reward still looks good.

Mission throughput is terrible.

The harmonic estimator would penalize exactly that.

---

### Manipulation

Suppose grasp success is

```
99%
```

but occasionally

the grasp takes

20 seconds instead of 2.

Human operators usually hate that.

The harmonic estimator probably would too.

---

# My biggest insight

After reading the paper, I don't think Harmonic RL is fundamentally a "better reward-rate estimator."

I think it is much closer to an algorithm for optimizing something like

> **the weakest local throughput**

or

> **the consistency of throughput**

rather than average throughput.

That immediately suggests applications wherever **the slowest stage dominates overall performance**:

- dual-arm robots (makespan),
- robot teams,
- assembly lines,
- warehouse fleets,
- network routing,
- cloud scheduling,
- manufacturing,
- logistics.

These are all domains where average performance is often less important than avoiding occasional severe slowdowns.

In fact, this leads to a research direction that I find even more interesting than the paper's current framing: characterize Harmonic RL as optimizing a **robust average-reward criterion** rather than the classical average reward. If you can identify the exact objective function it converges to, you may be able to connect it to the broader literature on risk-sensitive RL, bottleneck optimization, and robust control. That would explain *why* it succeeds in the domains above and delineate the class of problems where it should outperform conventional average-reward methods.

## Prompt:
09/07/2026, 22:51:23

Please provide references to the broader literature on risk-sensitive RL, and bottleneck optimization.  In addition to important references, also find papers on risk-sensitive average-rewards in SMDP settings

## Response:
09/07/2026, 22:52:14

This is actually a very interesting direction. I think there are **three separate literatures** that are relevant to your work, and each suggests a different interpretation of Harmonic RL.

---

# 1. Risk-sensitive reinforcement learning (most closely related)

The central idea is that the objective is **not** the expected reward, but a risk-adjusted functional.

## Seminal paper

**Howard & Matheson (1972)**

> *Risk-Sensitive Markov Decision Processes*.
> Management Science 18(7):356-369.

This is the classic paper introducing exponential utility into MDPs. Instead of maximizing

$$
E[R],
$$

they maximize

$$
\frac1\theta
\log E[e^{\theta R}],
$$

where $\theta$ controls risk aversion/risk seeking. ([PubsOnline](https://pubsonline.informs.org/doi/10.1287/mnsc.18.7.356?utm_source=chatgpt.com))

---

## Risk-sensitive Q-learning

**Borkar (2002)**

> *Q-Learning for Risk-Sensitive Control*.
> Mathematics of Operations Research.

One of the first RL algorithms with convergence guarantees for risk-sensitive control. It develops a Q-learning analogue for exponential-utility objectives rather than expected reward. ([PubsOnline](https://pubsonline.informs.org/doi/abs/10.1287/moor.27.2.294.324?journalCode=moor&utm_source=chatgpt.com))

---

## Modern survey

**Bäuerle & Jaśkiewicz (2024)**

> *Markov decision processes with risk-sensitive criteria: an overview.*

This is probably the best modern survey.

It covers

- exponential utility,
- certainty equivalents,
- CVaR,
- coherent risk measures,
- average-cost and undiscounted settings,
- infinite-horizon theory. ([DOI](https://doi.org/10.1007/s00186-024-00857-0?utm_source=chatgpt.com))

---

## RL survey

**Prashanth & Fu (2018)**

> *Risk-Sensitive Reinforcement Learning via Policy Gradient Search.*

Very useful because it surveys many risk criteria:

- variance
- CVaR
- chance constraints
- exponential utility
- prospect theory

and discusses average-reward formulations as well. ([arXiv](https://arxiv.org/abs/1810.09126?utm_source=chatgpt.com))

---

## Ergodic (average-reward) risk-sensitive control

**Biswas & Borkar (2023)**

> *Ergodic Risk-sensitive Control - A Survey.*

This survey is especially relevant because your work is about **continuing tasks**.

It focuses on long-run average criteria instead of discounted returns. ([arXiv](https://arxiv.org/abs/2301.00224?utm_source=chatgpt.com))

---

# 2. Bottleneck optimization

This literature may actually be even closer conceptually.

Here the objective is not

$$
\sum r
$$

or

$$
E[r],
$$

but something like

$$
\min_i r_i,
$$

or minimizing the largest delay.

Examples include

- bottleneck shortest path
- min-max routing
- minimax scheduling
- makespan minimization
- max-min fairness

These are not usually framed as RL problems, but the objectives resemble what your harmonic estimator seems to emphasize.

---

## Makespan scheduling

Classic references include

**Pinedo**

> *Scheduling: Theory, Algorithms, and Systems.*

Makespan optimization is

$$
\min \max_i C_i.
$$

This is immediately relevant to your dual-arm robotics work.

---

## Bottleneck shortest path

Examples:

- Burkard et al.
- Ahuja, Magnanti & Orlin

These optimize the largest edge cost rather than the sum.

Again, conceptually close to "avoid the worst local rate."

---

## Robust routing

Many papers in

- network routing
- logistics
- warehouse planning

replace average travel time by

worst-case delay

or

maximum latency.

The philosophy is similar.

---

# 3. Semi-Markov average-reward literature

This is surprisingly sparse.

---

## SMART

Das et al. (1999)

> *Solving Semi-Markov Decision Problems Using Average Reward Reinforcement Learning.*

The foundational RL algorithm for average-reward SMDPs. ([PubsOnline](https://pubsonline.informs.org/doi/pdf/10.1287/mnsc.45.4.560?utm_source=chatgpt.com))

---

## Relaxed SMART

Gosavi (2004)

> *Reinforcement Learning for Long-Run Average Cost.*

Introduces Relaxed SMART and discusses long-run average RL in SMDPs. ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0377221702008743?utm_source=chatgpt.com))

---

## New convergence theory

Yu, Wan & Sutton (2025)

> *Average-reward reinforcement learning in semi-Markov decision processes via relative value iteration.*

This develops RVI Q-learning for average-reward SMDPs with convergence guarantees. ([arXiv](https://arxiv.org/abs/2512.06218?utm_source=chatgpt.com))

---

# Risk-sensitive SMDPs

Here is where things become interesting.

There is **very little** directly combining

- SMDPs
- average reward
- risk-sensitive RL.

Most papers are one of

- risk-sensitive MDPs,

or

- average-reward SMDPs.

The overlap is remarkably small.

One paper that comes reasonably close is

**Mankowitz, Tamar & Mannor (2016)**

> *Situational Awareness by Risk-Conscious Skills.*

This considers **risk-aware options** in a **Probabilistic Goal Semi-Markov Decision Process (PG-SMDP)**. It is hierarchical RL rather than average reward, but it explicitly introduces risk-aware temporally extended actions (options), making it one of the closest existing links between risk sensitivity and semi-Markov decision processes. ([arXiv](https://arxiv.org/abs/1610.02847?utm_source=chatgpt.com))

From the control-theory side, there is also an active body of work on **risk-sensitive average-cost semi-Markov decision processes** (not RL). For example, papers on "risk-sensitive average semi-Markov decision chains" develop average-cost optimality equations and existence results for SMDPs under exponential-utility-type criteria, but they assume a known model rather than learning from interaction. The 2024 survey by Bäuerle & Jaśkiewicz points to this line of work and its relationship to average-cost risk-sensitive control. ([DOI](https://doi.org/10.1007/s00186-024-00857-0?utm_source=chatgpt.com))

---

# Why I think Harmonic RL is different

After reading your paper, I don't think Harmonic RL fits naturally into any of these categories.

It is **not** minimizing variance.

It is **not** optimizing exponential utility.

It is **not** minimizing CVaR.

It is **not** optimizing worst-case reward.

Instead, it seems to optimize something like

> **the harmonic mean of local throughput.**

That is a completely different objective.

---

# A research direction I find particularly exciting

I would actually position Harmonic RL as introducing a **new class of risk-sensitive objectives**.

Today, risk-sensitive RL has essentially three major families:

- **Utility-based:** exponential utility, certainty equivalents.
- **Tail-risk:** CVaR, VaR, coherent risk measures.
- **Variance-based:** mean-variance optimization.

Harmonic RL suggests a fourth family:

> **Rate-sensitive risk measures**, which penalize low instantaneous throughput disproportionately by aggregating local reward rates through a harmonic operator.

This framing would immediately connect your work to the risk-sensitive RL literature while making clear that it is *not another exponential-utility method*. It would also naturally explain why the algorithm appears particularly attractive for robotics domains dominated by throughput and bottlenecks-multi-arm manipulation, swarm coordination, warehouse robots, and task allocation-where occasional severe slowdowns can dominate overall system performance even when the arithmetic average reward rate looks favorable. I think that positioning is both more accurate and potentially more impactful than arguing that the harmonic mean is the universally "correct" estimator for reward rates.

---

**Sources:**

- [Risk-Sensitive Markov Decision Processes | Management Science](https://pubsonline.informs.org/doi/10.1287/mnsc.18.7.356?utm_source=chatgpt.com)
- [Markov decision processes with risk-sensitive criteria: an overview | Mathematical Methods of Operations Research | Springer Nature Link](https://doi.org/10.1007/s00186-024-00857-0?utm_source=chatgpt.com)
- [Risk-Sensitive Reinforcement Learning via Policy Gradient Search](https://arxiv.org/abs/1810.09126?utm_source=chatgpt.com)
- [Reinforcement learning for long-run average cost - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0377221702008743?utm_source=chatgpt.com)

## Prompt:
09/07/2026, 23:22:41

If I replace the harmonic mean in the harmonic RL with the geometric mean, wouldn't (1) I get the same phenomenon of preference to smaller values, and (2) a more natural connection to exponential utility with risk parameter $\theta$?

## Response:
09/07/2026, 23:22:42

> **Considering citation and paper relevance**
>
> User's asking for a conceptual answer about geometric replacing harmonic but didn't specifically ask for current references. No need to use web unless citations are needed. Since a paper’s been uploaded, I should cite it with the correct file reference marker from earlier, ensuring I follow the guidelines about line numbers if appropriate.
>
> **Searching for modified harmonic mean**
>
> It looks like I need to search for the modified harmonic mean, specifically as it appears in Equation 17. It's important to ensure any information aligns with that context, so I’ll focus my search on finding relevant details for this particular equation and its modifications.
>
> **Explaining generalized means and exponential utility**
>
> The generalized mean is defined as $M_p = (E[x^p])^(1/p)$. When p = -1, it's harmonic; p → 0, it's geometric; and p = 1, it's arithmetic. While harmonic and geometric means prioritize low values, the harmonic mean is more sensitive to near-zero values. The geometric mean maximizes expected log-rate, while exponential utility handles risk aversion and certainty equivalence, with a focus on multiplicative growth. Geometric and exponential utilities are closely connected with logarithmic behavior.
>
> **Exploring geometric mean, reward properties, and finance**
>
> In reinforcement learning, rewards can be negative, and the paper’s modified harmonic mean handles these cases. Geometric mean can't easily handle mixed signs. For finance, returns can be negative, but gross return is positive, while rates may transform to wealth factors. A natural connection exists between exponential utility and log-rates. This is clearer with an example showing how different means—arithmetic, geometric, and harmonic—compare with sensitivity to small values using their respective derivatives.
>
> **Derivative properties and use in various fields**
>
> The harmonic mean’s derivative shows higher leverage for small values, making it sensitive to bottlenecks. The geometric mean
>
> **Exploring power means and update inconsistencies**
>
> The power mean formula defines different risk attitudes depending on p. The EMA update formula adjusts the mean using a parameterized exponent. However, integrating this into a SMART Q update with reward regularization ($r-\rho\tau$) may cause inconsistencies unless the reward rate matches units. If it doesn't, the Q update becomes biased. Additionally, exponential utility connects to power means when using log-transformations, relating them to entropic risk over log-rates for varying risk preferences.
>
> **Clarifying geometric and harmonic mean properties**
>
> For fixed arithmetic mean, both geometric and harmonic means decrease with increasing variance. The geometric mean punishes variance in the log space, approximating as $G ≈ μ (1 - CV^2/2)$ and $H ≈ μ(1 - CV^2)$. Geometric mean is natural for multiplicative rates, such as investment growth or survival probabilities, while harmonic mean penalizes variance more strongly. The key conditions involve positive rates, multiplicative events, and log-normal distributions, making geometric mean well-suited for estimating multiplicative growth factors.
>
> Thought for 1m 17s

Yes - replacing harmonic by geometric is a very natural idea, but it would change the interpretation.

Let

$$
x_t=\frac{r_t}{\tau_t}
$$

be the sampled reward rate. For positive rates,

$$
A(x)=E[x],
\qquad
G(x)=\exp(E[\log x]),
\qquad
H(x)=\frac{1}{E[1/x]}.
$$

They satisfy

$$
H(x)\leq G(x)\leq A(x).
$$

So yes, the geometric mean also penalizes variability and gives more importance to small values than the arithmetic mean.

But it does so **much more weakly** than the harmonic mean.

For example:

$$
A=(1,100), \qquad B=(10,10).
$$

Then

$$
A_{\text{arith}}=50.5,\qquad B_{\text{arith}}=10,
$$

so arithmetic strongly prefers $A$.

$$
G_A=\sqrt{100}=10,\qquad G_B=10,
$$

so geometric is indifferent.

$$
H_A=\frac{2}{1+1/100}\approx 1.98,\qquad H_B=10,
$$

so harmonic strongly prefers $B$.

So the geometric mean has the same *direction* of effect, but not the same *strength*. Harmonic is much more bottleneck-sensitive.

---

The deeper point is that arithmetic, geometric, and harmonic are all members of the same power-mean family:

$$
M_p(x)=
\left(E[x^p]\right)^{1/p}.
$$

Special cases:

$$
M_1=A,
$$

$$
M_0=\lim_{p\to 0}M_p=G,
$$

$$
M_{-1}=H.
$$

So instead of saying “replace harmonic by geometric,” I would say: introduce a **risk parameter** $p$, and define a family of SMDP rate estimators:

$$
\rho_p =
\left(E\left[\left(\frac{r_t}{\tau_t}\right)^p\right]\right)^{1/p}.
$$

Then:

- $p=1$: arithmetic rate averaging,
- $p=0$: geometric rate averaging,
- $p=-1$: harmonic rate averaging,
- $p\to -\infty$: worst local rate,
- $p\to +\infty$: best local rate.

That gives you a continuum between average-seeking and bottleneck-seeking behavior.

---

On your second point: yes, but with one important correction.

The geometric mean is not exactly standard exponential utility over $x_t=r_t/\tau_t$. Standard exponential utility would be

$$
\operatorname{CE}_\theta(x)
=
\frac{1}{\theta}\log E[e^{\theta x}].
$$

The geometric mean instead corresponds to expected **log rate**:

$$
G(x)=\exp(E[\log x]).
$$

However, the full power-mean family has a very clean exponential-utility interpretation if you define

$$
z_t = \log x_t.
$$

Then

$$
\frac{1}{\theta}\log E[e^{\theta z_t}]
=
\frac{1}{\theta}\log E[x_t^\theta].
$$

Exponentiating both sides gives

$$
\exp\left(
\frac{1}{\theta}\log E[e^{\theta z_t}]
\right)
=
\left(E[x_t^\theta]\right)^{1/\theta}
=
M_\theta(x).
$$

So the power mean $M_\theta$ is exactly the exponential-certainty-equivalent of the **log reward rate**, mapped back to rate units.

That is beautiful, because it means:

$$
\theta=1
$$

gives arithmetic averaging,

$$
\theta\to 0
$$

gives geometric averaging,

$$
\theta=-1
$$

gives harmonic averaging.

So harmonic already has a natural risk-sensitive interpretation: it is the $\theta=-1$ member of an entropic-risk family over log-rates.

---

This also clarifies the difference between geometric and harmonic.

The geometric mean is risk-neutral in log-rate space:

$$
G(x)=\exp(E[\log x]).
$$

The harmonic mean is risk-averse in log-rate space:

$$
H(x)=\left(E[x^{-1}]\right)^{-1}.
$$

So if your goal is to connect to exponential utility with a tunable risk parameter, the best move is probably **not** to replace harmonic by geometric, but to generalize Harmonic RL to **Power-Mean $R$-Learning**:

$$
\rho_{p,t}
=
\left(
\operatorname{EMA}\left[
\left(\frac{r_t}{\tau_t}\right)^p
\right]
\right)^{1/p}.
$$

For $p=0$, use the log-limit:

$$
\rho_{0,t}
=
\exp\left(
\operatorname{EMA}\left[
\log\left(\frac{r_t}{\tau_t}\right)
\right]
\right).
$$

Then $p$ becomes a genuine risk/bottleneck sensitivity parameter.

---

There is one serious caveat: positivity.

The geometric and power means require

$$
\frac{r_t}{\tau_t}>0.
$$

But ordinary RL rewards can be zero or negative. The paper’s modified harmonic operator was introduced specifically because the ordinary harmonic mean is undefined for zeros and behaves badly with mixed signs. `main.pdf`

You could define a modified geometric operator analogously:

$$
G_{\pm}(X)
=
\frac{
|X^+|G(X^+)
-
|X^-|G(|X^-|)
}{
|X^+|+|X^-|+|X^0|
}.
$$

That would mirror the paper’s sign-partitioned modified harmonic operator. But then the clean exponential-utility interpretation becomes less direct, because logs of signed values are not real.

So the cleanest theory is for **positive reward rates**.

---

My suggested framing:

> Harmonic RL is one point in a broader family of risk-sensitive SMDP rate objectives based on power means of local reward rates.

Then the paper’s method becomes

$$
p=-1,
$$

geometric RL becomes

$$
p=0,
$$

SMART-like arithmetic rate averaging becomes

$$
p=1.
$$

This would let you experimentally test whether the success comes specifically from harmonic averaging, or from choosing a negative risk parameter that penalizes low local throughput.

## Prompt:
09/07/2026, 23:32:56

Interesting. This characterizes geometric mean as being risk-neutral for what kind of SMDP rewards?

## Response:
09/07/2026, 23:32:57

> **Clarifying file citation use**
>
> Okay, the user might be asking if I need to cite the uploaded file. Since some system tools require the citation to reference specific lines, I’m wondering if it's okay to cite the file without specifying lines. The system's guidance seems a bit unclear, but I’ll likely proceed with the full citation including the file if it’s requested.
>
> **Explaining risk-neutral geometric mean in SMDP**
>
> The user asks about the geometric mean and its relation to risk neutrality in SMDP rewards. It’s about log-rate space, where rewards are expressed as log-rewards rather than additive ones. Geometric means maximize the average log of reward rates, which is similar to optimizing using arithmetic means in transformed reward terms.
>
> **Clarifying SMDP reward structure and risk-neutral behavior**
>
> Time durations in SMDPs are crucial when considering risk-neutral averages of log-rates, which aren’t directly per unit time unless weighted by duration. There’s also the concept of multiplicative or log-throughput rewards where risk-neutral behavior maximizes the geometric mean. This idea links to Kelly’s utility in finance and relative improvements in robotics.
>
> Thought for 22s

It characterizes the geometric mean as **risk-neutral for logarithmic reward-rate rewards**.

Let the ordinary SMDP transition produce a positive reward rate

$$
x_t=\frac{r_t}{\tau_t}>0.
$$

If we replace the raw rate $x_t$ by the transformed reward

$$
\tilde r_t = \log x_t
=
\log\left(\frac{r_t}{\tau_t}\right),
$$

then maximizing the geometric mean

$$
G(x)=\exp\left(E[\log x]\right)
$$

is equivalent to maximizing the ordinary arithmetic expectation

$$
E[\tilde r_t].
$$

So geometric-mean RL is risk-neutral **not with respect to ordinary additive rewards $r_t$**, but with respect to **log reward rates**.

---

For SMDPs, there are two subtly different interpretations.

### 1. Event-average log-rate objective

If each decision step counts equally, then geometric mean optimizes

$$
\lim_{n\to\infty}
\frac{1}{n}
\sum_{t=1}^n
\log\left(\frac{r_t}{\tau_t}\right).
$$

Exponentiating gives

$$
\exp\left(
\lim_{n\to\infty}
\frac{1}{n}
\sum_{t=1}^n
\log\left(\frac{r_t}{\tau_t}\right)
\right),
$$

which is the long-run geometric mean of per-transition reward rates.

This is risk-neutral if the SMDP reward is defined as

$$
\tilde r_t=\log(r_t/\tau_t).
$$

But it is **event-time neutral**, not physical-time neutral.

---

### 2. Time-average log-rate objective

If you want a true SMDP time-average criterion, then each transition should be weighted by its duration:

$$
\rho_{\log}
=
\lim_{n\to\infty}
\frac{\sum_{t=1}^n \tau_t
\log(r_t/\tau_t)}
{\sum_{t=1}^n \tau_t}.
$$

Then

$$
\exp(\rho_{\log})
$$

is the **time-weighted geometric mean reward rate**.

In this case, the transformed SMDP lump reward would be

$$
\tilde R_t
=
\tau_t\log\left(\frac{r_t}{\tau_t}\right).
$$

Then ordinary average-reward SMDP learning on $\tilde R_t$ is risk-neutral in the transformed reward.

---

So the clean answer is:

> The geometric mean is risk-neutral for SMDPs whose primitive reward is the **logarithm of the positive reward rate**, or equivalently for tasks where utility accumulates additively in log-throughput rather than in raw reward.

That is exactly the same logic as log utility in finance: maximizing expected log return is risk-neutral in log wealth growth, but risk-averse relative to arithmetic wealth growth.

---

This matters for your harmonic/geometric comparison.

Arithmetic rate objective:

$$
E[x]
$$

is risk-neutral for raw local reward rate.

Geometric rate objective:

$$
\exp(E[\log x])
$$

is risk-neutral for log local reward rate.

Harmonic rate objective:

$$
\left(E[x^{-1}]\right)^{-1}
$$

is risk-averse in log-rate space; it is the $\theta=-1$ entropic certainty equivalent over

$$
z=\log x.
$$

So geometric is the neutral midpoint only after changing the primitive reward from

$$
r_t/\tau_t
$$

to

$$
\log(r_t/\tau_t).
$$

That makes it natural for multiplicative-growth domains such as finance, but less obviously natural for robotics unless the task utility is genuinely relative/percentage-based rather than additive in distance, energy, time, or task completions.

## Prompt:
09/07/2026, 23:51:29

What I'm aiming for is to relate the \theta parameter of exponential utility functions to the power mean.  Let's assume regular additive rewards R in an MDP?  What value of \theta should be used for risk-neutral optimization?  for risk-averse or risk-seeking?

## Response:
09/07/2026, 23:51:32

> **Explaining exponential utility and power mean**
>
> The user is asking about the relationship between exponential utility and the power mean in the context of MDPs. For risk-neutral parameters, theta approaches 0, implying no risk preference. For negative theta, risks are penalized (risk-averse), and positive theta emphasizes higher outcomes (risk-seeking). The power mean connects to these risk preferences, showing the impact of different parameters on expectations.
>
> Thought for 9s

For additive MDP rewards, the clean answer is:

$$
\theta=0
$$

is risk-neutral.

Using the common reward-maximization convention,

$$
\operatorname{CE}_{\theta}(G)
=
\frac{1}{\theta}\log E[e^{\theta G}],
$$

where $G$ is the return, then:

$$
\theta \to 0
\quad\Rightarrow\quad
\operatorname{CE}_{0}(G)=E[G].
$$

So:

$$
\boxed{\theta=0 \text{ is risk-neutral}}
$$

$$
\boxed{\theta<0 \text{ is risk-averse}}
$$

$$
\boxed{\theta>0 \text{ is risk-seeking}}
$$

for reward maximization.

The local Taylor expansion makes this clear:

$$
\operatorname{CE}_{\theta}(G)
=
E[G]
+
\frac{\theta}{2}\operatorname{Var}(G)
+
O(\theta^2).
$$

So positive $\theta$ rewards variance, while negative $\theta$ penalizes variance.

---

Now the relation to power means is subtle.

The power mean of positive values $x$ is

$$
M_p(x)
=
\left(E[x^p]\right)^{1/p}.
$$

Rewrite it as

$$
M_p(x)
=
\exp\left(
\frac{1}{p}\log E[e^{p\log x}]
\right).
$$

So:

$$
\boxed{
M_p(x)
=
\exp\left(\operatorname{CE}_{p}(\log x)\right)
}
$$

This means the power-mean parameter $p$ is exactly the exponential-utility parameter $\theta$, but **not over $x$**. It is the exponential-utility parameter over

$$
z=\log x.
$$

Therefore:

$$
p=0
$$

is risk-neutral **in log-reward space**, because

$$
M_0(x)=\exp(E[\log x]),
$$

the geometric mean.

But for ordinary additive rewards $R$, risk neutrality is

$$
E[R],
$$

which corresponds to the arithmetic mean:

$$
M_1(R)=E[R].
$$

So there are two different notions of neutrality:

| Objective | Risk-neutral with respect to | Parameter |
|---|---:|---:|
| Exponential utility $\operatorname{CE}_\theta(R)$ | additive reward $R$ | $\theta=0$ |
| Power mean $M_p(R)$ | log-reward $\log R$ | $p=0$ |
| Arithmetic mean $E[R]$ | additive reward $R$ | $p=1$ |

That is the key distinction.

---

So if you want to relate Harmonic RL to exponential utility, the correct statement is:

$$
H(x)=M_{-1}(x)
=
\exp\left(\operatorname{CE}_{-1}(\log x)\right).
$$

Thus harmonic averaging is equivalent to using exponential utility with

$$
\theta=-1
$$

on **log reward-rates**, not on ordinary additive rewards.

Similarly:

$$
G(x)=M_0(x)
=
\exp(E[\log x])
$$

is risk-neutral for log reward-rates.

And:

$$
A(x)=M_1(x)=E[x]
$$

is risk-neutral for raw reward-rates.

So for a regular additive-reward MDP, $\theta=0$ is risk-neutral. But for the power-mean family, the parameter $p$ corresponds to $\theta$ only after transforming rewards by a logarithm.

## Prompt:
09/07/2026, 23:56:58

perhaps i could just set p=\theta-1?

## Response:
09/07/2026, 23:57:01

> **Explaining power means and risk-neutral mappings**
>
> So, I’m trying to consider how these mappings work, especially with the relation between power means and entropic utility. They seem to be proposing an affine shift for p=theta-1, but that might not align perfectly with risk-neutral behavior. If we're aiming for risk-neutral, p=theta+1 looks more fitting for additive x. Got it.
>
> Thought for 19s

You *could*, but I would not. It would give the wrong neutral point.

For the power mean

$$
M_p(X)=\left(E[X^p]\right)^{1/p},
$$

the risk-neutral case for ordinary positive additive rewards $X$ is

$$
p=1,
$$

because

$$
M_1(X)=E[X].
$$

So if you want an exponential-utility-like parameter $\theta$ where

$$
\theta=0
$$

means risk-neutral, then the natural shift is

$$
\boxed{p = 1+\theta}
$$

not

$$
p=\theta-1.
$$

Then:

$$
\theta=0 \Rightarrow p=1
$$

risk-neutral arithmetic mean,

$$
\theta<0 \Rightarrow p<1
$$

risk-averse,

$$
\theta>0 \Rightarrow p>1
$$

risk-seeking.

So the mapping would be:

| $\theta$ | $p=1+\theta$ | mean | interpretation |
|---:|---:|---|---|
| $0$ | $1$ | arithmetic | risk-neutral |
| $-1$ | $0$ | geometric | log-utility-like risk aversion |
| $-2$ | $-1$ | harmonic | stronger bottleneck/risk aversion |
| $>0$ | $>1$ | super-arithmetic | risk-seeking |

With your proposed mapping,

$$
p=\theta-1,
$$

we get

$$
\theta=0 \Rightarrow p=-1,
$$

so the harmonic mean would become the risk-neutral point. That seems mathematically misleading unless you intentionally want to define a new convention where harmonic aggregation is “neutral.”

---

But there is an even cleaner interpretation:

The power mean is not really exponential utility over $X$. It is the certainty equivalent of **power utility**:

$$
u_p(x)=
\begin{cases}
\frac{x^p}{p}, & p\neq 0,\\[4pt]
\log x, & p=0.
\end{cases}
$$

The certainty equivalent is

$$
u_p^{-1}(E[u_p(X)])
=
\left(E[X^p]\right)^{1/p}
=
M_p(X).
$$

So the power-mean parameter $p$ is naturally a **power-utility / CRRA-style risk parameter**, not exactly the exponential-utility parameter.

In that framing:

$$
p=1
$$

is linear utility, hence risk-neutral;

$$
p<1
$$

is concave utility, hence risk-averse;

$$
p>1
$$

is convex utility, hence risk-seeking.

So I would define

$$
\boxed{\theta := p-1}
$$

or equivalently

$$
\boxed{p := 1+\theta}.
$$

Then $\theta$ behaves like a signed risk-attitude parameter:

$$
\theta=0
\quad\text{neutral},
$$

$$
\theta<0
\quad\text{risk-averse},
$$

$$
\theta>0
\quad\text{risk-seeking}.
$$

This gives you the clean story:

$$
\rho_\theta
=
M_{1+\theta}\!\left(\frac{R}{\tau}\right).
$$

Then:

$$
\theta=0:
\quad
\rho_0
=
E\left[\frac{R}{\tau}\right],
$$

$$
\theta=-1:
\quad
\rho_{-1}
=
\exp E\left[\log \frac{R}{\tau}\right],
$$

$$
\theta=-2:
\quad
\rho_{-2}
=
\left(E\left[\frac{\tau}{R}\right]\right)^{-1}.
$$

So harmonic RL corresponds to

$$
\boxed{\theta=-2}
$$

under this shifted power-mean convention.

## Prompt:
10/07/2026, 00:16:34

is there a different form of risk-sensitive reward functions accepting a risk parameter \theta, that is more easily tied to power means? especially for reward rates (r/t)?

## Response:
10/07/2026, 00:16:36

> **Clarifying risk-sensitive reward functions and power means**
>
> The user is asking about a form of risk-sensitive reward functions, which involves a parameter (theta) and relates to power means. I'll discuss concepts like CRRA (constant relative risk aversion) and power utility, as well as how power means can act as certainty equivalents under power utility. This extends to reward rates in an MDP!
>
> **Optimizing additive utility with transformed rewards**
>
> In a Semi-Markov Decision Process (SMDP), the transformed lumped reward is calculated with the utility function. The goal is to optimize the average transformed reward over time, using a weighted power mean. For updates, I’ll apply a transformation in reward and time. Bellman optimality adjusts for utility rate and uses this transformed data carefully.
>
> Thought for 47s

Yes. The much more natural bridge is **power utility**, also called **isoelastic utility** or **CRRA utility**, rather than exponential utility.

For positive reward rates

$$
x_t=\frac{r_t}{\tau_t}>0,
$$

define

$$
u_\theta(x)=
\begin{cases}
\dfrac{x^{1-\theta}-1}{1-\theta}, & \theta\neq 1,\\[6pt]
\log x, & \theta=1.
\end{cases}
$$

Then define the certainty-equivalent rate by

$$
\rho_\theta
=
u_\theta^{-1}\left(E[u_\theta(X)]\right).
$$

For $\theta\neq 1$, this becomes

$$
\rho_\theta
=
\left(E[X^{1-\theta}]\right)^{1/(1-\theta)}.
$$

That is exactly a **power mean** with

$$
p = 1-\theta.
$$

So the mapping is:

$$
\boxed{\rho_\theta = M_{1-\theta}(X)}
$$

where

$$
X=\frac{R}{\tau}.
$$

This gives a very clean interpretation:

| $\theta$ | $p=1-\theta$ | mean | interpretation |
|---:|---:|---|---|
| $0$ | $1$ | arithmetic | risk-neutral |
| $1$ | $0$ | geometric | log utility |
| $2$ | $-1$ | harmonic | stronger risk aversion |
| $\theta>0$ | $p<1$ | sub-arithmetic | risk-averse |
| $\theta<0$ | $p>1$ | super-arithmetic | risk-seeking |

So under this convention, **harmonic RL corresponds to $\theta=2$**, not $\theta=-1$ or $\theta=-2$.

---

For SMDPs, I would distinguish two versions.

The paper’s harmonic estimator is essentially an event-average over sampled local rates $r_t/\tau_t$, while the classical SMDP average-reward objective is a physical-time reward rate of the form $\sum r_t/\sum \tau_t$. The paper itself defines the SMDP objective as an average reward rate and introduces the harmonic method as a modified estimator over sampled reward rates. `main.pdf`

The **event-rate** power objective is

$$
\rho_\theta^{\text{event}}
=
\left(
E\left[
\left(\frac{R}{\tau}\right)^{1-\theta}
\right]
\right)^{1/(1-\theta)}.
$$

This is closest to the current Harmonic RL idea.

But the more SMDP-native version is the **time-weighted** power objective:

$$
\rho_\theta^{\text{time}}
=
u_\theta^{-1}
\left(
\frac{E[\tau\,u_\theta(R/\tau)]}{E[\tau]}
\right).
$$

For $\theta\neq 1$, this is

$$
\rho_\theta^{\text{time}}
=
\left(
\frac{
E\left[
\tau\left(\frac{R}{\tau}\right)^{1-\theta}
\right]
}{
E[\tau]
}
\right)^{1/(1-\theta)}.
$$

This is a **duration-weighted power mean of local reward rates**.

The nice thing is that for $\theta=0$,

$$
\rho_0^{\text{time}}
=
\frac{E[\tau(R/\tau)]}{E[\tau]}
=
\frac{E[R]}{E[\tau]},
$$

so it reduces exactly to the standard SMDP average reward rate.

For $\theta=1$,

$$
\rho_1^{\text{time}}
=
\exp
\left(
\frac{E[\tau\log(R/\tau)]}{E[\tau]}
\right),
$$

a time-weighted geometric mean.

For $\theta=2$,

$$
\rho_2^{\text{time}}
=
\left(
\frac{E[\tau(R/\tau)^{-1}]}{E[\tau]}
\right)^{-1}
=
\frac{E[\tau]}{E[\tau^2/R]},
$$

a time-weighted harmonic reward rate.

---

This is probably the cleanest theoretical framing:

$$
\boxed{
\text{Risk-sensitive SMDP rate learning via CRRA utility over local rates } R/\tau.
}
$$

It is cleaner than exponential utility because power utility is scale-invariant:

$$
M_p(cX)=cM_p(X).
$$

That is exactly what you want for rates. If the units change from meters/second to centimeters/second, the objective just rescales. Exponential utility does not have this property; its $\theta$ depends on the numerical units of reward.

One important caveat: this theory is clean only for

$$
R/\tau>0.
$$

For zero or negative rates, you would need one of three fixes: model positive costs instead of signed rewards, shift all rates by a positive constant, or define a sign-partitioned power-mean analogue. The last option is closest to the paper’s modified harmonic operator, but it weakens the direct utility-theoretic interpretation.

## Prompt:
20/07/2026, 22:40:41

What SMDP risk-sensitive RL algorithms should I look at?  group discounted SMDP RL risk-sensitive algorithms separately from average-reward SMDP risk-sensitive algorithms

## Prompt:
20/07/2026, 22:44:17

What SMDP risk-sensitive RL algorithms should I look at?  group discounted SMDP RL risk-sensitive algorithms separately from average-reward SMDP risk-sensitive algorithms.

## Response:
20/07/2026, 22:44:18

> **Searched multiple research websites and databases**
>
> 
>
> Worked for a few seconds

There is a surprisingly sharp divide between the discounted and average-reward literatures.

The discounted case has a mature risk-sensitive RL literature. The average-reward SMDP case is still very sparse.

I would organize it as follows.

---

# I. Discounted risk-sensitive RL

This is the "classical" literature.

## A. Exponential utility (entropic risk)

This is the dominant framework.

### Howard & Matheson (1972)

> *Risk-Sensitive Markov Decision Processes*

The foundational paper.

Objective:

$$
J_\theta(\pi)
=
\frac1\theta
\log
E\!\left[
e^{\theta G}
\right]
$$

where $G$ is the discounted return.

This paper already treats infinite-horizon problems and derives Bellman equations. ([pubsonline.informs.org](https://pubsonline.informs.org/doi/10.1287/mnsc.18.7.356?utm_source=chatgpt.com))

---

### Borkar (2002)

> *Q-Learning for Risk-Sensitive Control*

One of the earliest risk-sensitive RL algorithms.

Develops Q-learning under exponential utility.

Very important if you want algorithmic inspiration.

---

### Fei et al. (2020)

> *Risk-Sensitive Reinforcement Learning*

Modern regret analysis.

Risk-sensitive Value Iteration

Risk-sensitive Q-learning

Model-free

Provable regret bounds. ([arXiv](https://arxiv.org/abs/2006.13827?utm_source=chatgpt.com))

---

### General utility

Recent work studies arbitrary utility functions.

Wu & Xu (2023)

> *Risk-sensitive MDP and Learning under General Utility Functions.*

Allows

- exponential
- power
- arbitrary monotone utility

rather than only entropic risk. ([arXiv](https://arxiv.org/abs/2311.13589?utm_source=chatgpt.com))

---

# II. Discounted SMDPs

This literature is surprisingly thin.

Most work on SMDPs assumes either

- discounted expected return

or

- average reward,

but not risk-sensitive objectives.

The closest papers I know are

### Huang, Lian & Guo (2018)

> *Risk-sensitive Semi-Markov Decision Processes with General Utilities and Multiple Criteria*

This is **not RL**.

It studies

- SMDPs
- general utility functions
- occupation measures
- linear programming

Very relevant mathematically. ([Cambridge University Press & Assessment](https://www.cambridge.org/core/journals/advances-in-applied-probability/article/abs/risksensitive-semimarkov-decision-processes-with-general-utilities-and-multiple-criteria/EC574AF62541158A57864D84F2A1C281?utm_source=chatgpt.com))

---

Also useful:

Guo & Huang's survey on SMDPs

covers

- discounted
- average reward
- constrained
- mean-variance
- probability criteria

although not RL specifically. ([sciengine.com](https://www.sciengine.com/SSM/doi/10.1360/N012015-00041?utm_source=chatgpt.com))

---

# III. Average-reward SMDP RL

This is your home territory.

The literature is tiny.

Essentially:

### SMART

Das et al. (1999)

Average reward

Ratio objective

Long-term arithmetic average. ([pubsonline.informs.org](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.45.4.560?utm_source=chatgpt.com))

---

### Relaxed SMART

EMA version

ratio of EMAs

---

### RVI Q-learning

Yu, Wan & Sutton (2025)

Relative Value Iteration

Convergence proofs

Average-reward SMDPs. ([arXiv](https://arxiv.org/abs/2512.06218?utm_source=chatgpt.com))

---

### Deterministic Policy Gradient for SMDPs

Hosseinloo & Dahleh (2022)

Average reward

continuous actions

actor-critic. ([DOI](https://doi.org/10.1002/int.22709?utm_source=chatgpt.com))

---

### Harmonic R-learning

Your paper.

This introduces a new estimator of the average reward rate based on a modified harmonic mean rather than the arithmetic ratio. `main.pdf`

---

# IV. Risk-sensitive average-reward MDPs (not SMDPs)

This is where I think your strongest theoretical connection lies.

There is a substantial control-theory literature on

**ergodic risk-sensitive control**

using exponential utility.

For example

Howard & Matheson

already discuss infinite-horizon certainty-equivalent gain.

The recent survey by

Bäuerle & Jaśkiewicz

covers

- discounted
- average reward
- ergodic risk-sensitive control

and gives an excellent roadmap. ([DOI](https://doi.org/10.1007/s00186-024-00857-0?utm_source=chatgpt.com))

---

# V. Risk-sensitive average-reward SMDPs

This is the remarkable part.

I could not find a genuine **model-free RL algorithm** combining all three:

- SMDP
- average reward
- risk sensitivity.

There are

✔ risk-sensitive SMDPs (control)

✔ average-reward SMDP RL

✔ risk-sensitive MDP RL

but almost nobody seems to have combined all three.

That makes your work potentially quite novel.

---

# If I were writing a follow-up paper

I would position it as follows.

Rather than

> Harmonic RL

I would introduce

## Power-Mean R-learning

Parameter

$$
\theta
$$

controls risk attitude.

Estimator

$$
\rho_\theta
=
M_{1-\theta}
\!\left(
\frac{r}{\tau}
\right).
$$

Special cases

$$
\theta=0
$$

Arithmetic (SMART-like)

$$
\theta=1
$$

Geometric

$$
\theta=2
$$

Harmonic

This gives an entire continuum between

- expected reward rate

- log-growth rate

- bottleneck-sensitive throughput.

---

## One more literature I think you should investigate

I actually think the **strongest connection is not to exponential utility**.

It is to **Constant Relative Risk Aversion (CRRA)** from economics.

CRRA utility

$$
u(x)
=
\frac{x^{1-\theta}-1}{1-\theta}
$$

has certainty equivalent

$$
u^{-1}(E[u(x)])
=
\left(E[x^{1-\theta}]\right)^{1/(1-\theta)},
$$

which is **exactly** the power mean.

That means your proposed family

$$
M_{1-\theta}(r/\tau)
$$

is not merely analogous to a known risk-sensitive criterion-it **is** the certainty equivalent under CRRA utility applied to reward rates.

I think this is a much stronger theoretical foundation than trying to force a connection to exponential utility. Exponential utility naturally leads to entropic certainty equivalents, whereas the power mean emerges directly and exactly from CRRA utility, with harmonic, geometric, and arithmetic means appearing as special cases. That observation could provide a principled interpretation of a "Power-Mean $R$-Learning" algorithm as a family of **CRRA risk-sensitive average-reward SMDP algorithms**, which, to my knowledge, has not been explored in the model-free RL literature.

---

**Sources:**

- [Risk-Sensitive Markov Decision Processes | Management Science](https://pubsonline.informs.org/doi/10.1287/mnsc.18.7.356?utm_source=chatgpt.com)
- [Risk-Sensitive Reinforcement Learning: Near-Optimal Risk-Sample Tradeoff in Regret](https://arxiv.org/abs/2006.13827?utm_source=chatgpt.com)
- [Risk-sensitive semi-Markov decision processes with general utilities and multiple criteria | Advances in Applied Probability | Cambridge Core](https://www.cambridge.org/core/journals/advances-in-applied-probability/article/abs/risksensitive-semimarkov-decision-processes-with-general-utilities-and-multiple-criteria/EC574AF62541158A57864D84F2A1C281?utm_source=chatgpt.com)
- [半Markov决策过程的研究进展](https://www.sciengine.com/SSM/doi/10.1360/N012015-00041?utm_source=chatgpt.com)
- [Deterministic policy gradient algorithms for semi‐Markov decision processes - Hosseinloo - 2022 - International Journal of Intelligent Systems - Wiley Online Library](https://doi.org/10.1002/int.22709?utm_source=chatgpt.com)

## Prompt:
20/07/2026, 23:01:13

Excellent, I can adapt that. I want to better understand how different power means would choose different actions, in the same state, given the risk sensitivity parameter \theta.  Suppose I have three actions from a state s1. One is optimizes risk-seeking, one is risk-neutral, and one is appropriate for risk-averse selection.  Show me an SMDP with at least two states (and no more than 5) where, starting at s1, hypothetical different algorithms will choose different actions in s1.

## Response:
20/07/2026, 23:01:16

> **Considering the approach**
>
> I’m wondering if a web search is necessary for the example asked by the user. While I could construct an answer, technical guidelines seem to require at least one web search to support it.
>
> **Considered the approach**
>
> 
>
> **Searching for primary sources on CRRA utility**
>
> [Markov decision processes with risk-sensitive criteria: an overview | Mathematical Methods of Operations Research | Springer Nature Link](https://doi.org/10.1007/s00186-024-00857-0)
> [Risk-sensitive reinforcement learning using expectiles, shortfall risk and optimized certainty equivalent risk](https://arxiv.org/abs/2602.09300)
>
> **Searched primary sources on CRRA utility, DOI, and arXiv**
>
> 
>
> **Clarifying model structure**
>
> I’m deciding how to define states and actions in a way that aligns with dynamic consistency and avoids violating conditions. The goal is to maintain a clean, consistent system while exploring potential reward and duration functions.
>
> **Clarified model structure**
>
> 
>
> **Refining the model structure**
>
> I’m considering incorporating varying time durations for each state to emphasize the SMDP framework. This will adjust time weighting to ensure consistency in calculating rewards and outcomes over cycles.
>
> **Computing optimal action selection across parameters**
>
>
> **Computed optimal action selection across parameters**
>
> 
>
> **Formulating the example**
>
> I’m detailing this SMDP example with two states and time-weighted power means for action ranking, ensuring the calculation is correct. I’ll reference the uploaded paper where appropriate, per guidelines.
>
> **Formulated the example**
>
> 
>
> Worked for 1m 30s

Consider a two-state continuing SMDP designed so that each visit to $s_1$ starts a new regenerative cycle. This avoids complications from different future state distributions: all three actions lead to the same state $s_2$, which then returns to $s_1$.

For positive local reward rates

$$
X=\frac{R}{\tau},
$$

use the CRRA certainty-equivalent rate

$$
\rho_\theta
=
\left(
\frac{\mathbb E[\tau X^{1-\theta}]}
{\mathbb E[\tau]}
\right)^{1/(1-\theta)},
\qquad \theta\neq 1,
$$

with the geometric limit at $\theta=1$. This is the duration-weighted power mean with

$$
p=1-\theta.
$$

The certainty-equivalent construction $u^{-1}(\mathbb E[u(X)])$ is standard in risk-sensitive control. ([DOI](https://doi.org/10.1007/s00186-024-00857-0?utm_source=chatgpt.com)) The use of reward per holding time as the central SMDP quantity also matches the continuing average-rate formulation of the uploaded paper. `main.pdf`

## The SMDP

From $s_1$, choose one of three actions. Each action lasts exactly

$$
\tau=2
$$

time units and transitions to $s_2$.

| Action | Probability | Reward $R$ | Duration $\tau$ | Rate $R/\tau$ |
|---|---:|---:|---:|---:|
| $a_{\mathrm{seek}}$ | $1/2$ | $2$ | $2$ | $1$ |
|  | $1/2$ | $30$ | $2$ | $15$ |
| $a_{\mathrm{neutral}}$ | $1/2$ | $12$ | $2$ | $6$ |
|  | $1/2$ | $22$ | $2$ | $11$ |
| $a_{\mathrm{averse}}$ | $1$ | $16$ | $2$ | $8$ |

From $s_2$, there is only one action:

$$
s_2 \longrightarrow s_1,
\qquad
R=8,\quad \tau=1,\quad R/\tau=8.
$$

Thus every cycle lasts three time units.

The structure is:

$$
s_1
\overset{a_{\mathrm{seek}},a_{\mathrm{neutral}},a_{\mathrm{averse}}}{\longrightarrow}
s_2
\longrightarrow
s_1.
$$

Because the $s_1$ transition lasts two units and each random outcome occurs with probability $1/2$, each of the two possible $s_1$ rates contributes one expected unit of time. The fixed $s_2$ transition contributes one unit. Therefore, the time-weighted rate distributions used for comparison are

$$
a_{\mathrm{seek}}:\quad (1,15,8),
$$

$$
a_{\mathrm{neutral}}:\quad (6,11,8),
$$

$$
a_{\mathrm{averse}}:\quad (8,8,8),
$$

with equal weights $1/3$.

---

## 1. Risk-seeking algorithm: $\theta=-1$

Here

$$
p=1-\theta=2,
$$

so the criterion is the quadratic mean:

$$
\rho_{-1}=M_2(X)
=
\sqrt{\frac{x_1^2+x_2^2+x_3^2}{3}}.
$$

For the three actions:

$$
\rho_{-1}(a_{\mathrm{seek}})
=
\sqrt{\frac{1^2+15^2+8^2}{3}}
\approx 9.83,
$$

$$
\rho_{-1}(a_{\mathrm{neutral}})
=
\sqrt{\frac{6^2+11^2+8^2}{3}}
\approx 8.58,
$$

$$
\rho_{-1}(a_{\mathrm{averse}})=8.
$$

Therefore,

$$
\boxed{
\pi_{\theta=-1}(s_1)=a_{\mathrm{seek}}
}
$$

The large possible rate $15$ receives disproportionate weight. The bad outcome $1$ is tolerated because the convex power $x^2$ amplifies the upside.

---

## 2. Risk-neutral algorithm: $\theta=0$

Now

$$
p=1,
$$

so the criterion is the arithmetic, physical-time average rate.

For $a_{\mathrm{seek}}$,

$$
\rho_0
=
\frac{1+15+8}{3}
=8.
$$

For $a_{\mathrm{neutral}}$,

$$
\rho_0
=
\frac{6+11+8}{3}
=
\frac{25}{3}
\approx 8.33.
$$

For $a_{\mathrm{averse}}$,

$$
\rho_0=8.
$$

Therefore,

$$
\boxed{
\pi_{\theta=0}(s_1)=a_{\mathrm{neutral}}
}
$$

Equivalently, using expected cycle rewards:

$$
a_{\mathrm{seek}}:
\quad
\frac{E[R_{\text{cycle}}]}{E[\tau_{\text{cycle}}]}
=
\frac{16+8}{3}
=8,
$$

$$
a_{\mathrm{neutral}}:
\quad
\frac{17+8}{3}
=
\frac{25}{3},
$$

$$
a_{\mathrm{averse}}:
\quad
\frac{16+8}{3}
=8.
$$

The neutral policy chooses $a_{\mathrm{neutral}}$ because it has the largest expected reward per unit time.

---

## 3. Risk-averse algorithm: $\theta=2$

Here

$$
p=1-\theta=-1,
$$

so the criterion is the harmonic mean:

$$
\rho_2=M_{-1}(X)
=
\frac{3}{
1/x_1+1/x_2+1/x_3
}.
$$

For $a_{\mathrm{seek}}$,

$$
\rho_2
=
\frac{3}{
1+\frac1{15}+\frac18
}
\approx 2.52.
$$

For $a_{\mathrm{neutral}}$,

$$
\rho_2
=
\frac{3}{
\frac16+\frac1{11}+\frac18
}
\approx 7.84.
$$

For $a_{\mathrm{averse}}$,

$$
\rho_2=8.
$$

Therefore,

$$
\boxed{
\pi_{\theta=2}(s_1)=a_{\mathrm{averse}}
}
$$

The occasional rate $1$ destroys the harmonic value of $a_{\mathrm{seek}}$. Even $a_{\mathrm{neutral}}$, whose arithmetic mean is higher than $8$, loses to the perfectly reliable rate $8$.

---

## Summary

| Risk parameter | Power $p=1-\theta$ | Criterion | Selected action |
|---:|---:|---|---|
| $\theta=-1$ | $p=2$ | quadratic mean | $a_{\mathrm{seek}}$ |
| $\theta=0$ | $p=1$ | arithmetic mean | $a_{\mathrm{neutral}}$ |
| $\theta=1$ | $p=0$ | geometric mean | $a_{\mathrm{neutral}}$ |
| $\theta=2$ | $p=-1$ | harmonic mean | $a_{\mathrm{averse}}$ |

At $\theta=1$, for example,

$$
G(a_{\mathrm{seek}})
=
(1\cdot15\cdot8)^{1/3}
\approx4.93,
$$

$$
G(a_{\mathrm{neutral}})
=
(6\cdot11\cdot8)^{1/3}
\approx8.09,
$$

$$
G(a_{\mathrm{averse}})=8,
$$

so the geometric criterion still chooses $a_{\mathrm{neutral}}$, though only slightly.

The action regions for this particular example are approximately

$$
\theta<-0.16
\quad\Longrightarrow\quad
a_{\mathrm{seek}},
$$

$$
-0.16<\theta<1.34
\quad\Longrightarrow\quad
a_{\mathrm{neutral}},
$$

$$
\theta>1.34
\quad\Longrightarrow\quad
a_{\mathrm{averse}}.
$$

Thus, varying a single continuous parameter produces genuine policy changes:

$$
\boxed{
\text{high-upside action}
\;\longrightarrow\;
\text{highest-average action}
\;\longrightarrow\;
\text{most reliable action}.
}
$$

The regenerative two-state construction is important: because all three actions share the same next state and common continuation, the different choices arise purely from the risk-sensitive treatment of the reward-rate distribution, not from differences in future opportunities.

---

**Sources:**

- [Markov decision processes with risk-sensitive criteria: an overview | Mathematical Methods of Operations Research | Springer Nature Link](https://doi.org/10.1007/s00186-024-00857-0?utm_source=chatgpt.com)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)
