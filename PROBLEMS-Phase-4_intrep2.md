# Interpretation 2: changing only rho is insufficient

The conversation supports using the CRRA certainty equivalent

$$
\rho_\theta=u_\theta^{-1}(E[u_\theta(R/T)])
$$

as the risk-sensitive objective. It does not establish that putting this
number into the otherwise linear R-learning target

$$
R-\rho_\theta T+\max Q
$$

makes the learned action values optimize that objective.

Consider equal-duration actions in one recurrent state. A safe action always
returns 5. A risky action returns 1 or 9 with equal probability. Both have
expected reward 5, so under the unchanged linear TD reward their expected
targets are

$$
5-\rho_\theta+\max Q.
$$

Changing the single global baseline shifts both actions by the same amount and
leaves them tied. But for `theta=2` (`p=-1`), their CRRA certainty equivalents
are different:

$$
M_{-1}(5,5)=5,
\qquad
M_{-1}(1,9)=\frac{2}{1+1/9}=1.8.
$$

A risk-averse criterion should prefer the safe action, while the unchanged
linear TD criterion remains indifferent. Therefore changing only `rho`, as in
Phase 3, is an estimator modification or learning heuristic rather than a
derived CRRA Bellman criterion.

The implemented interpretation instead transforms both the observed local
rate and its certainty-equivalent baseline:

$$
T\,[u_\theta(R/T)-u_\theta(\rho_\theta)]+\max Q.
$$

This distinguishes the two actions and reduces exactly to ordinary continuous
R-learning when `theta=0`.
