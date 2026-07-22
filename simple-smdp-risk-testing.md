Consider a two-state continuing SMDP designed so that each visit to (s_1) starts a new regenerative cycle. This avoids complications from different future state distributions: all three actions lead to the same state (s_2), which then returns to (s_1).

For positive local reward rates

[
X=\frac{R}{\tau},
]

use the CRRA certainty-equivalent rate

[
\rho_\theta
===========

\left(
\frac{\mathbb E[\tau X^{1-\theta}]}
{\mathbb E[\tau]}
\right)^{1/(1-\theta)},
\qquad \theta\neq 1,
]

with the geometric limit at (\theta=1). This is the duration-weighted power mean with

[
p=1-\theta.
]

The certainty-equivalent construction (u^{-1}(\mathbb E[u(X)])) is standard in risk-sensitive control. ([DOI][1]) The use of reward per holding time as the central SMDP quantity also matches the continuing average-rate formulation of the uploaded paper. 

## The SMDP

From (s_1), choose one of three actions. Each action lasts exactly

[
\tau=2
]

time units and transitions to (s_2).

| Action                 | Probability | Reward (R) | Duration (\tau) | Rate (R/\tau) |
| ---------------------- | ----------: | ---------: | --------------: | ------------: |
| (a_{\mathrm{seek}})    |       (1/2) |        (2) |             (2) |           (1) |
|                        |       (1/2) |       (30) |             (2) |          (15) |
| (a_{\mathrm{neutral}}) |       (1/2) |       (12) |             (2) |           (6) |
|                        |       (1/2) |       (22) |             (2) |          (11) |
| (a_{\mathrm{averse}})  |         (1) |       (16) |             (2) |           (8) |

From (s_2), there is only one action:

[
s_2 \longrightarrow s_1,
\qquad
R=8,\quad \tau=1,\quad R/\tau=8.
]

Thus every cycle lasts three time units.

The structure is:

[
s_1
\overset{a_{\mathrm{seek}},a_{\mathrm{neutral}},a_{\mathrm{averse}}}{\longrightarrow}
s_2
\longrightarrow
s_1.
]

Because the (s_1) transition lasts two units and each random outcome occurs with probability (1/2), each of the two possible (s_1) rates contributes one expected unit of time. The fixed (s_2) transition contributes one unit. Therefore, the time-weighted rate distributions used for comparison are

[
a_{\mathrm{seek}}:\quad (1,15,8),
]

[
a_{\mathrm{neutral}}:\quad (6,11,8),
]

[
a_{\mathrm{averse}}:\quad (8,8,8),
]

with equal weights (1/3).

---

## 1. Risk-seeking algorithm: (\theta=-1)

Here

[
p=1-\theta=2,
]

so the criterion is the quadratic mean:

[
\rho_{-1}=M_2(X)
================

\sqrt{\frac{x_1^2+x_2^2+x_3^2}{3}}.
]

For the three actions:

[
\rho_{-1}(a_{\mathrm{seek}})
============================

\sqrt{\frac{1^2+15^2+8^2}{3}}
\approx 9.83,
]

[
\rho_{-1}(a_{\mathrm{neutral}})
===============================

\sqrt{\frac{6^2+11^2+8^2}{3}}
\approx 8.58,
]

[
\rho_{-1}(a_{\mathrm{averse}})=8.
]

Therefore,

[
\boxed{
\pi_{\theta=-1}(s_1)=a_{\mathrm{seek}}
}
]

The large possible rate (15) receives disproportionate weight. The bad outcome (1) is tolerated because the convex power (x^2) amplifies the upside.

---

## 2. Risk-neutral algorithm: (\theta=0)

Now

[
p=1,
]

so the criterion is the arithmetic, physical-time average rate.

For (a_{\mathrm{seek}}),

[
\rho_0
======

\frac{1+15+8}{3}
=8.
]

For (a_{\mathrm{neutral}}),

[
\rho_0
======

# \frac{6+11+8}{3}

\frac{25}{3}
\approx 8.33.
]

For (a_{\mathrm{averse}}),

[
\rho_0=8.
]

Therefore,

[
\boxed{
\pi_{\theta=0}(s_1)=a_{\mathrm{neutral}}
}
]

Equivalently, using expected cycle rewards:

[
a_{\mathrm{seek}}:
\quad
\frac{E[R_{\text{cycle}}]}{E[\tau_{\text{cycle}}]}
==================================================

\frac{16+8}{3}
=8,
]

[
a_{\mathrm{neutral}}:
\quad
\frac{17+8}{3}
==============

\frac{25}{3},
]

[
a_{\mathrm{averse}}:
\quad
\frac{16+8}{3}
=8.
]

The neutral policy chooses (a_{\mathrm{neutral}}) because it has the largest expected reward per unit time.

---

## 3. Risk-averse algorithm: (\theta=2)

Here

[
p=1-\theta=-1,
]

so the criterion is the harmonic mean:

[
\rho_2=M_{-1}(X)
================

\frac{3}{
1/x_1+1/x_2+1/x_3
}.
]

For (a_{\mathrm{seek}}),

[
\rho_2
======

\frac{3}{
1+\frac1{15}+\frac18
}
\approx 2.52.
]

For (a_{\mathrm{neutral}}),

[
\rho_2
======

\frac{3}{
\frac16+\frac1{11}+\frac18
}
\approx 7.84.
]

For (a_{\mathrm{averse}}),

[
\rho_2=8.
]

Therefore,

[
\boxed{
\pi_{\theta=2}(s_1)=a_{\mathrm{averse}}
}
]

The occasional rate (1) destroys the harmonic value of (a_{\mathrm{seek}}). Even (a_{\mathrm{neutral}}), whose arithmetic mean is higher than (8), loses to the perfectly reliable rate (8).

---

## Summary

| Risk parameter | Power (p=1-\theta) | Criterion       | Selected action        |
| -------------: | -----------------: | --------------- | ---------------------- |
|    (\theta=-1) |              (p=2) | quadratic mean  | (a_{\mathrm{seek}})    |
|     (\theta=0) |              (p=1) | arithmetic mean | (a_{\mathrm{neutral}}) |
|     (\theta=1) |              (p=0) | geometric mean  | (a_{\mathrm{neutral}}) |
|     (\theta=2) |             (p=-1) | harmonic mean   | (a_{\mathrm{averse}})  |

At (\theta=1), for example,

[
G(a_{\mathrm{seek}})
====================

(1\cdot15\cdot8)^{1/3}
\approx4.93,
]

[
G(a_{\mathrm{neutral}})
=======================

(6\cdot11\cdot8)^{1/3}
\approx8.09,
]

[
G(a_{\mathrm{averse}})=8,
]

so the geometric criterion still chooses (a_{\mathrm{neutral}}), though only slightly.

The action regions for this particular example are approximately

[
\theta<-0.16
\quad\Longrightarrow\quad
a_{\mathrm{seek}},
]

[
-0.16<\theta<1.34
\quad\Longrightarrow\quad
a_{\mathrm{neutral}},
]

[
\theta>1.34
\quad\Longrightarrow\quad
a_{\mathrm{averse}}.
]

Thus, varying a single continuous parameter produces genuine policy changes:

[
\boxed{
\text{high-upside action}
;\longrightarrow;
\text{highest-average action}
;\longrightarrow;
\text{most reliable action}.
}
]

The regenerative two-state construction is important: because all three actions share the same next state and common continuation, the different choices arise purely from the risk-sensitive treatment of the reward-rate distribution, not from differences in future opportunities.

[1]: https://doi.org/10.1007/s00186-024-00857-0?utm_source=chatgpt.com "Markov decision processes with risk-sensitive criteria: an overview | Mathematical Methods of Operations Research | Springer Nature Link"
