import os
import tempfile
import unittest

import torch

import smdp_agents as A
from smdp_agents.checkpoint import load, save, state_of


def drive_tabular(agent, states=("a", "b", "c"), rounds=4):
    """A few real learning steps, so the estimator and table hold something."""
    for r in range(rounds):
        for i, s in enumerate(states):
            agent.tabulate(agent.q_table, s)
            agent.update_table(s, 0, 1.0 + i, 2.0, 1.0, 0.5, True)
    return agent


def rollout(agent, obs_dim=4, act_dim=2, steps=8, batch=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    buf = A.RolloutBuffer()
    for _ in range(steps):
        o = torch.randn(batch, obs_dim, generator=g)
        a, v, lp = agent.act(o)
        buf.add(o, a, torch.randn(batch, generator=g), torch.zeros(batch),
                torch.zeros(batch), v, lp, time=torch.full((batch,), 2.0))
    return buf, torch.zeros(batch)


class TabularCheckpointTests(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def path(self, n="a.pkl"):
        return os.path.join(self.dir, n)

    def test_every_tabular_agent_round_trips(self):
        for name in ("QLearning", "RLearning", "SMART", "RelaxedSMART",
                     "SmoothedSMART", "Harmonic", "WeightedHarmonic",
                     "CumulativeHarmonic", "ExperimentalWeightedHarmonic"):
            with self.subTest(agent=name):
                agent = drive_tabular(getattr(A, name)("x", action_space=[0, 1]))
                back = load(save(agent, self.path(name + ".pkl")))
                self.assertIs(type(back), type(agent))
                self.assertEqual(back.q_table, agent.q_table)
                self.assertAlmostEqual(back.rho, agent.rho)

    def test_the_rate_estimator_object_is_restored_not_reset(self):
        """A checkpoint that forgets the estimator still runs, and silently
        starts from the wrong rate. This is the test for that."""
        agent = drive_tabular(A.RelaxedSMART("x", action_space=[0, 1]))
        back = load(save(agent, self.path()))
        self.assertAlmostEqual(back.ratio_rate.mean_reward,
                               agent.ratio_rate.mean_reward)
        self.assertAlmostEqual(back.ratio_rate.mean_duration,
                               agent.ratio_rate.mean_duration)
        self.assertNotEqual(agent.ratio_rate.mean_reward, 0.0)   # it held something

    def test_a_resumed_agent_continues_identically(self):
        a = drive_tabular(A.RelaxedSMART("x", action_space=[0, 1]))
        b = load(save(a, self.path()))
        drive_tabular(a, rounds=3)
        drive_tabular(b, rounds=3)
        self.assertEqual(a.q_table, b.q_table)
        self.assertAlmostEqual(a.rho, b.rho)
        self.assertEqual(a.step_count, b.step_count)

    def test_the_rng_is_restored_so_exploration_replays(self):
        a = drive_tabular(A.QLearning("x", action_space=[0, 1, 2]))
        b = load(save(a, self.path()))
        self.assertEqual([a.act("a") for _ in range(50)],
                         [b.act("a") for _ in range(50)])

    def test_an_unpicklable_attribute_fails_loudly(self):
        agent = A.QLearning("x", action_space=[0, 1])
        agent.handle = lambda: None            # a lambda cannot be pickled
        with self.assertRaises(TypeError) as ctx:
            state_of(agent)
        self.assertIn("handle", str(ctx.exception))

    def test_the_environment_reference_is_not_state(self):
        agent = A.QLearning("x", action_space=[0, 1], env=object())
        self.assertNotIn("env", state_of(agent)["attrs"])


class DeepCheckpointTests(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def path(self, n="d.pkl"):
        return os.path.join(self.dir, n)

    def test_every_deep_agent_round_trips(self):
        for name in ("PPO", "APO", "RsmartSMAPO", "SmartSMAPO", "HarmonicSMAPO",
                     "SmoothedSmartSMAPO", "DiscreteRsmartSMAPO"):
            with self.subTest(agent=name):
                cls = getattr(A, name)
                agent = cls(4, 2, hidden=(8,), seed=0)
                agent.update(*rollout(agent))
                back = load(save(agent, self.path(name + ".pkl")))
                self.assertIs(type(back), type(agent))
                obs = torch.randn(5, 4)
                with torch.no_grad():
                    for x, y in zip(agent.net(obs), back.net(obs)):
                        self.assertTrue(torch.equal(x, y))

    def test_the_optimiser_state_survives(self):
        """Adam's moments are state; dropping them restarts the optimiser."""
        agent = A.RsmartSMAPO(4, 2, hidden=(8,), seed=0)
        agent.update(*rollout(agent))
        back = load(save(agent, self.path()))
        src = agent.optimizer.state_dict()["state"]
        dst = back.optimizer.state_dict()["state"]
        self.assertEqual(len(src), len(dst))
        self.assertTrue(len(src) > 0)
        for k in src:
            self.assertTrue(torch.allclose(src[k]["exp_avg"], dst[k]["exp_avg"]))
            self.assertEqual(int(src[k]["step"]), int(dst[k]["step"]))

    def test_the_pressure_controller_state_survives(self):
        """SMAPO's calibration is slow state; losing it silently re-runs warmup."""
        agent = A.RsmartSMAPO(4, 2, hidden=(8,), seed=0, entropy_warmup_iters=1)
        for _ in range(3):
            agent.update(*rollout(agent))
        back = load(save(agent, self.path()))
        for attr in ("_adv_std_frozen", "_scale_ema", "_scale_peak",
                     "_log_boost", "entropy_loss_coeff", "boost"):
            self.assertEqual(getattr(back, attr), getattr(agent, attr), attr)
        self.assertIsNotNone(agent._adv_std_frozen)       # warmup had completed

    def test_a_resumed_agent_continues_identically(self):
        a = A.RsmartSMAPO(4, 2, hidden=(8,), seed=0, entropy_warmup_iters=1)
        a.update(*rollout(a))
        path = save(a, self.path())
        # `a` keeps going first, advancing torch's GLOBAL rng; loading with
        # restore_torch_rng rewinds it to the save point, so `b` sees the same
        # draws `a` did. Loading before this would leave `b` downstream of them.
        sa = a.update(*rollout(a, seed=7))
        b = load(path, restore_torch_rng=True)
        sb = b.update(*rollout(b, seed=7))
        for k in sa:
            self.assertAlmostEqual(sa[k], sb[k], places=5, msg=k)

    def test_the_discrete_head_is_rebuilt_as_a_discrete_head(self):
        agent = A.DiscreteRsmartSMAPO(4, 5, hidden=(8,), seed=0)
        back = load(save(agent, self.path()))
        self.assertTrue(back.discrete)
        self.assertIsInstance(back.net, A.CategoricalMLP)
        self.assertEqual(tuple(back.eval_act(torch.zeros(3, 4)).shape), (3,))

    def test_without_restore_torch_rng_the_weights_still_match(self):
        """The agent is restored either way; only the global draw sequence differs."""
        a = A.RsmartSMAPO(4, 2, hidden=(8,), seed=0)
        a.update(*rollout(a))
        b = load(save(a, self.path()))                 # default: RNG untouched
        obs = torch.randn(5, 4)
        with torch.no_grad():
            self.assertTrue(torch.equal(a.net(obs)[0], b.net(obs)[0]))

    def test_no_torch_object_is_pickled(self):
        """Weights go out as numpy so a different torch version can read them."""
        agent = A.RsmartSMAPO(4, 2, hidden=(8,), seed=0)
        agent.update(*rollout(agent))
        blob = state_of(agent)
        for v in blob["net"].values():
            self.assertFalse(torch.is_tensor(v))
        self.assertNotIn("net", blob["attrs"])
        self.assertNotIn("optimizer", blob["attrs"])


if __name__ == "__main__":
    unittest.main()
