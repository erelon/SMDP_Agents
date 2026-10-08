"""MuJoCo locomotion as an SMDP: the agent commands a joint-angle target and
waits for the joint to get there.

A stock MuJoCo env is an MDP with a fixed `frame_skip`: every decision costs the
same amount of simulated time. These envs replace that with a *macro-step*. The
action is a target joint angle; the simulation runs at `frame_skip=1` until the
joints have reached it (or `max_frames` elapses), and the number of frames that
took is the decision's HOLDING TIME. A small move finishes in a few frames, a
large one takes many, so the holding time varies with the action — which is what
makes the average-reward-RATE objective different from average reward per
decision.

Each step returns the holding time in ``info["tau"]``; feed it to the agent::

    from examples.envs.mujoco_smdp import make_swimmer
    from smdp_agents import RsmartSMAPO, RolloutBuffer

    env = make_swimmer()
    obs, _ = env.reset(seed=0)
    agent, buf = RsmartSMAPO(env.observation_space.shape[0],
                             env.action_space.shape[0]), RolloutBuffer()
    action, value, logp = agent.act(torch.as_tensor(obs).float()[None])
    obs, reward, terminated, truncated, info = env.step(action[0].numpy())
    buf.add(..., time=info["tau"])

Needs ``gymnasium[mujoco]``, which is NOT a requirement of this library; import
this module only if you want these environments.

The forked XMLs in ``assets/`` replace the stock ``<motor>`` actuators with
``<position>`` servos. They are copied verbatim from the ones used to produce the
published results and are deliberately NOT corrected: ``swimmer_position.xml``
sets ``kp=500`` with no ``forcerange``, so the servo's torque is unbounded, which
is not how a real position-controlled joint behaves. Changing it would change the
dynamics and so would not reproduce anything. Treat it as a reproduction artefact
rather than a model to copy.
"""
import os

import gymnasium as gym
import numpy as np

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


class EffortCost(gym.Wrapper):
    """Charge the actual normalised torque rather than the commanded angle.

    A stock locomotion env charges ``w * ||ctrl||^2``, where ``ctrl`` is the
    fraction of available torque: it penalises EFFORT, and coasting is free.
    Under position control ``ctrl`` is an ANGLE, so the same formula charges
    POSTURE — and where a joint's range excludes zero it acquires a cost floor
    the original never had. ``actuator_force / forcerange`` is the quantity the
    stock env actually squares, so charging that restores both properties using
    the original's own weight and formula.

    Applied INSIDE the hold wrapper, so it accrues per physics frame exactly as
    the stock per-step cost does. Set the env's own ``ctrl_cost_weight=0`` when
    using this, or posture is taxed twice.
    """

    def __init__(self, env, weight):
        super().__init__(env)
        self.weight = float(weight)
        model = env.unwrapped.model
        force = np.asarray(model.actuator_forcerange[:, 1], dtype=np.float64).copy()
        gear = np.asarray(model.actuator_gear[:, 0], dtype=np.float64)
        self._norm = np.where(force > 0, force,
                              np.where(gear != 0, np.abs(gear), 1.0))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        f = np.asarray(self.env.unwrapped.data.actuator_force,
                       dtype=np.float64) / self._norm
        cost = self.weight * float(np.dot(f, f))
        info = dict(info)
        info["reward_effort"] = -cost
        return obs, float(reward) - cost, terminated, truncated, info


class HoldUntilTargetReached(gym.Wrapper):
    """One macro-step: hold the commanded joint-angle target until it is reached.

    The step ends when ``max|qpos - target| < pos_tol`` AND ``max|qvel| < vel_tol``
    have EACH been satisfied at least once during the macro-step — not necessarily
    on the same frame. That is deliberately looser than requiring both at the same
    instant: under PD control the joint overshoots slightly on approach, so the
    two conditions alternate during the damped oscillation and the strict version
    waits for it to die out entirely, which gives back control late and produces a
    visibly floaty gait.

    The actuated joints are resolved through the model's actuator -> joint ->
    qpos/dof addressing rather than assumed to be the trailing entries, and the
    target is divided by ``actuator_gear`` so it is correct under any gear scaling.
    """

    def __init__(self, env, max_frames=100, min_frames=1, pos_tol=0.05,
                 vel_tol=10.0):
        super().__init__(env)
        self.max_frames = int(max_frames)
        self.min_frames = int(min_frames)
        self.pos_tol = float(pos_tol)
        self.vel_tol = float(vel_tol)
        n = env.action_space.shape[0]
        model = env.unwrapped.model
        self._gear = np.asarray(model.actuator_gear[:n, 0], dtype=np.float64)
        joints = [int(model.actuator_trnid[i, 0]) for i in range(n)]
        self._q_idx = np.asarray([int(model.jnt_qposadr[j]) for j in joints],
                                 dtype=np.intp)
        self._v_idx = np.asarray([int(model.jnt_dofadr[j]) for j in joints],
                                 dtype=np.intp)
        ctrlrange = np.asarray(model.actuator_ctrlrange, dtype=np.float64)
        self._lo = ctrlrange[:n, 0] / self._gear
        self._hi = ctrlrange[:n, 1] / self._gear

    def step(self, action):
        target = np.clip(np.asarray(action, dtype=np.float64) / self._gear,
                         self._lo, self._hi)
        data = self.env.unwrapped.data
        total, frames, vel_sum = 0.0, 0, 0.0
        pos_ok = vel_ok = False
        obs, terminated, truncated, info = None, False, False, {}

        for _ in range(self.max_frames):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total += float(reward)
            frames += 1
            if "x_velocity" in info:
                vel_sum += float(info["x_velocity"])
            if terminated or truncated or frames < self.min_frames:
                if terminated or truncated:
                    break
                continue
            if float(np.max(np.abs(np.asarray(data.qpos)[self._q_idx] - target))) < self.pos_tol:
                pos_ok = True
            if float(np.max(np.abs(np.asarray(data.qvel)[self._v_idx]))) < self.vel_tol:
                vel_ok = True
            if pos_ok and vel_ok:
                break

        info = dict(info)
        info["tau"] = float(frames)          # the holding time, in physics frames
        if frames and "x_velocity" in info:
            info["x_velocity"] = vel_sum / frames
        return obs, total, terminated, truncated, info


#: Per-environment settings used for the published results.
CONFIGS = {
    "swimmer": dict(gym_id="Swimmer-v5", xml="swimmer_position.xml",
                    max_frames=100, pos_tol=0.05, vel_tol=10.0,
                    effort_cost_weight=0.0, env_kwargs={}),
    "ant": dict(gym_id="Ant-v5", xml="ant_position.xml",
                max_frames=50, pos_tol=0.05, vel_tol=1e9,
                effort_cost_weight=0.5, env_kwargs=dict(ctrl_cost_weight=0.0)),
}


def make(name, max_episode_steps=None, **overrides):
    """Build one of :data:`CONFIGS` as an SMDP.

    ``max_episode_steps`` counts PHYSICS FRAMES, not decisions, because the time
    limit wraps the inner ``frame_skip=1`` environment and the hold wrapper sits
    outside it. ``None`` means the episode never truncates — a continuing task,
    which is the setting the average-reward objective is for.

    The action space is rescaled to ``[-1, 1]`` regardless of the XML's own
    ``ctrlrange``, so a policy with a tanh-squashed mean spans exactly the
    reachable targets.
    """
    if name not in CONFIGS:
        raise KeyError("unknown environment %r; expected one of %s"
                       % (name, sorted(CONFIGS)))
    cfg = dict(CONFIGS[name])
    cfg.update(overrides)

    env = gym.make(cfg["gym_id"], xml_file=os.path.join(ASSETS, cfg["xml"]),
                   frame_skip=1, max_episode_steps=max_episode_steps,
                   **cfg["env_kwargs"])
    if cfg["effort_cost_weight"]:
        env = EffortCost(env, cfg["effort_cost_weight"])
    env = HoldUntilTargetReached(env, max_frames=cfg["max_frames"],
                                 pos_tol=cfg["pos_tol"], vel_tol=cfg["vel_tol"])
    lo = np.full(env.action_space.shape, -1.0, dtype=env.action_space.dtype)
    return gym.wrappers.RescaleAction(env, lo, -lo)


def make_swimmer(**kwargs):
    """Swimmer as an SMDP. Cannot terminate, so episodes end only on a time limit."""
    return make("swimmer", **kwargs)


def make_ant(**kwargs):
    """Ant as an SMDP. CAN terminate, so compare it on a discount-free quantity."""
    return make("ant", **kwargs)
