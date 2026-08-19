"""The contract every example environment in this package obeys.

An SMDP environment is an ordinary Gymnasium environment with one addition: the
action it just executed took a *holding time*, and that time is reported in
``info["tau"]``. Everything else — spaces, ``reset``/``step`` signatures,
seeding — is stock Gymnasium, so these environments drop into any gym-based
tooling and into the agents in ``agents/`` without an adapter:

    obs, info = env.reset(seed=1)
    s = env.state_of(obs)
    while True:
        a = agent.act(s)
        obs, r, terminated, truncated, info = env.step(a)
        ns = env.state_of(obs)
        agent.learn(s, a, r, ns, info["tau"])
        s = ns

Three hooks on top of ``gym.Env``:

``state_of(obs)``
    The hashable, Markov state the tabular agents key their Q-tables on. The
    default is the observation itself, which is right for the environments whose
    observation *is* a discrete state; environments with structured
    observations (whack-a-mole's ``Dict``) override it.
``get_available_actions(state)``
    The actions legal in ``state``. Agents call this through ``env=`` to avoid
    putting entries for impossible actions in their tables. The default is the
    full action space.
``secret()``
    The optimal policy, when it is known analytically. Feeds the ``Oracle``
    agent as an upper baseline; returns ``None`` when no closed form is known.

``check_smdp_env`` audits an environment against all of this and is applied to
every entry of the registry in ``tests/test_examples.py``.
"""

from __future__ import annotations

import math
from random import Random
from typing import Any, Callable, Dict, Hashable, List, Optional

import gymnasium as gym
from gymnasium import spaces

#: Key under which ``step`` reports the holding time of the action just taken.
TAU_KEY = "tau"
#: Key under which ``step`` reports the cumulative clock after the transition.
TIME_KEY = "time"


class SMDPEnv(gym.Env):
    """A Gymnasium environment whose actions have variable holding times.

    Subclasses implement ``reset``/``step`` as usual and must put a finite,
    non-negative ``info[TAU_KEY]`` and a cumulative ``info[TIME_KEY]`` in every
    ``step`` (and in ``reset``, where ``tau`` is 0).
    """

    #: Set by subclasses whose "time" is always 1, i.e. plain MDPs. Purely
    #: informational — the runner reports it so a τ≡1 result is not mistaken
    #: for an SMDP one.
    is_mdp: bool = False

    # ------------------------------------------------------------------ hooks
    def state_of(self, obs: Any) -> Hashable:
        """The hashable state the tabular agents key on. Default: ``obs``."""
        return obs

    def get_available_actions(self, state: Optional[Hashable] = None) -> List[int]:
        """Actions legal in ``state``. Default: the whole action space."""
        return self.action_list

    def secret(self) -> Optional[Callable[[Hashable], int]]:
        """The optimal policy if known analytically, else ``None``."""
        return None

    # -------------------------------------------------------------- utilities
    @property
    def action_list(self) -> List[int]:
        """The action space as the plain list the agents are constructed with."""
        space = self.action_space
        if not isinstance(space, spaces.Discrete):
            raise TypeError(
                f"{type(self).__name__}.action_list needs a Discrete action "
                f"space, got {space!r}; override action_list."
            )
        return list(range(int(space.start), int(space.start) + int(space.n)))

    def _info(self, tau: float, **extra: Any) -> Dict[str, Any]:
        """Build a step ``info`` with the two mandatory keys."""
        info = {TAU_KEY: float(tau), TIME_KEY: float(getattr(self, "time", 0.0))}
        info.update(extra)
        return info


# --------------------------------------------------------------------- audit
class EnvContractError(AssertionError):
    """An environment violated the SMDP contract."""


def check_smdp_env(
    env: SMDPEnv,
    steps: int = 300,
    seed: int = 0,
    check_determinism: bool = True,
) -> Dict[str, Any]:
    """Audit ``env`` against the contract; raise ``EnvContractError`` if it fails.

    Drives the environment for ``steps`` decisions under a seeded uniform-random
    policy over ``get_available_actions``, re-resetting whenever an episode ends,
    and checks the τ bookkeeping, the state hashability, the action space, and
    (unless disabled) that a second run with the same seed and the same action
    sequence reproduces the first exactly.

    Returns a small summary — step count, elapsed time, τ range, distinct states
    — which the tests use to assert the environment is actually doing something.
    """
    first = _audit_run(env, steps, seed)
    if check_determinism:
        second = _audit_run(env, steps, seed)
        if first["trace"] != second["trace"]:
            where = next(
                i for i, (a, b) in enumerate(zip(first["trace"], second["trace"]))
                if a != b
            )
            raise EnvContractError(
                f"{_name(env)} is not reproducible from seed {seed}: transition "
                f"{where} gave {first['trace'][where]} then {second['trace'][where]}"
            )
    summary = {k: v for k, v in first.items() if k != "trace"}
    return summary


def _audit_run(env: SMDPEnv, steps: int, seed: int) -> Dict[str, Any]:
    name = _name(env)
    rng = Random(seed)

    reset = env.reset(seed=seed)
    if not (isinstance(reset, tuple) and len(reset) == 2):
        raise EnvContractError(
            f"{name}.reset must return (obs, info), got {type(reset).__name__}"
        )
    obs, info = reset
    if not isinstance(info, dict):
        raise EnvContractError(f"{name}.reset info must be a dict, got {info!r}")
    state = _hashable_state(env, obs, name)

    trace: List[tuple] = []
    taus: List[float] = []
    states = {state}
    clock = 0.0
    episodes = 0

    for i in range(steps):
        available = env.get_available_actions(state)
        if not available:
            # A terminal state with no legal action: the episode must be over.
            obs, info = env.reset(seed=seed + 1000 + episodes)
            state = _hashable_state(env, obs, name)
            states.add(state)
            episodes += 1
            clock = 0.0
            continue
        for action in available:
            if not env.action_space.contains(action):
                raise EnvContractError(
                    f"{name}.get_available_actions({state!r}) offered {action!r}, "
                    f"which is not in {env.action_space!r}"
                )
        action = available[rng.randrange(len(available))]

        result = env.step(action)
        if not (isinstance(result, tuple) and len(result) == 5):
            raise EnvContractError(
                f"{name}.step must return the gymnasium 5-tuple "
                f"(obs, reward, terminated, truncated, info), got "
                f"{type(result).__name__} of length "
                f"{len(result) if isinstance(result, tuple) else 'n/a'}"
            )
        obs, reward, terminated, truncated, info = result

        if not isinstance(info, dict):
            raise EnvContractError(f"{name}.step info must be a dict, got {info!r}")
        for key in (TAU_KEY, TIME_KEY):
            if key not in info:
                raise EnvContractError(
                    f"{name}.step info is missing {key!r} at step {i} "
                    f"(keys: {sorted(info)})"
                )
        tau = float(info[TAU_KEY])
        if not math.isfinite(tau) or tau < 0.0:
            raise EnvContractError(
                f"{name} reported tau={tau!r} at step {i}; holding times must be "
                f"finite and non-negative"
            )
        reward = float(reward)
        if not math.isfinite(reward):
            raise EnvContractError(f"{name} reported reward={reward!r} at step {i}")
        if terminated not in (True, False) or truncated not in (True, False):
            raise EnvContractError(
                f"{name} reported terminated={terminated!r} truncated={truncated!r} "
                f"at step {i}; both must be booleans"
            )

        clock += tau
        reported = float(info[TIME_KEY])
        if not math.isclose(reported, clock, rel_tol=1e-9, abs_tol=1e-9):
            raise EnvContractError(
                f"{name} info['{TIME_KEY}']={reported} disagrees with the sum of "
                f"holding times {clock} at step {i}"
            )

        state = _hashable_state(env, obs, name)
        states.add(state)
        taus.append(tau)
        trace.append((round(reward, 12), round(tau, 12), repr(state)))

        if terminated or truncated:
            obs, info = env.reset(seed=seed + 1000 + episodes)
            state = _hashable_state(env, obs, name)
            states.add(state)
            episodes += 1
            clock = 0.0

    _check_secret(env, states, name)

    return {
        "steps": len(taus),
        "episodes": episodes,
        "elapsed": sum(taus),
        "tau_min": min(taus) if taus else None,
        "tau_max": max(taus) if taus else None,
        "distinct_states": len(states),
        "trace": trace,
    }


def _check_secret(env: SMDPEnv, states, name: str) -> None:
    secret = env.secret()
    if secret is None:
        return
    if not callable(secret):
        raise EnvContractError(f"{name}.secret() must return a callable or None")
    for state in states:
        available = env.get_available_actions(state)
        if not available:
            continue
        action = secret(state)
        if action not in available:
            raise EnvContractError(
                f"{name}.secret() chose {action!r} in state {state!r}, which is "
                f"not among the available actions {available!r}"
            )


def _hashable_state(env: SMDPEnv, obs: Any, name: str) -> Hashable:
    state = env.state_of(obs)
    try:
        hash(state)
    except TypeError as exc:
        raise EnvContractError(
            f"{name}.state_of returned the unhashable {type(state).__name__} "
            f"{state!r}; tabular agents key their Q-tables on it"
        ) from exc
    return state


def _name(env: SMDPEnv) -> str:
    return getattr(env, "name", None) or type(env).__name__
