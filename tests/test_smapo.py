import math
import unittest

import torch

from agents.policy_heads import CategoricalHead, GaussianHead
from agents.ppo import RolloutBuffer
from agents.smapo import (APO, DiscreteRsmartSMAPO, RsmartSMAPO, SmartSMAPO,
                          SmoothedSmartSMAPO)

OBS, ACT, T, B = 4, 2, 12, 3


def rollout(agent, obs_dim=OBS, n=T, b=B, tau=2.0, seed=0):
    """A synthetic [T, B] rollout; returns the buffer and the bootstrap value."""
    g = torch.Generator().manual_seed(seed)
    buf = RolloutBuffer()
    for _ in range(n):
        o = torch.randn(b, obs_dim, generator=g)
        a, v, lp = agent.act(o)
        buf.add(o, a, torch.randn(b, generator=g), torch.zeros(b),
                torch.zeros(b), v, lp, time=torch.full((b,), float(tau)))
    return buf, torch.zeros(b)


class SMAPOTests(unittest.TestCase):

    def test_every_variant_updates(self):
        for cls in (APO, RsmartSMAPO, SmartSMAPO, SmoothedSmartSMAPO):
            agent = cls(OBS, ACT, seed=0)
            buf, bv = rollout(agent)
            stats = agent.update(buf, bv)
            self.assertTrue(math.isfinite(stats["loss"]), cls.__name__)
            self.assertIn("realised_pressure", stats)

    def test_a_centering_removes_the_batch_mean(self):
        agent = RsmartSMAPO(OBS, ACT, seed=0)
        adv = torch.randn(T, B) + 7.0           # a large shared offset
        valid = torch.ones(T, B)
        batch = {"obs": torch.randn(T, B, OBS), "rew": torch.zeros(T, B),
                 "time": torch.full((T, B), 2.0)}
        out = agent.shape_advantage(adv, valid, batch)
        self.assertAlmostEqual(float(out.mean()), 0.0, places=5)
        # the ranking the policy gradient uses is untouched
        self.assertTrue(torch.allclose(out - out.mean(), adv - adv.mean(), atol=1e-6))

    def test_pressure_is_off_during_warmup_then_calibrated(self):
        agent = RsmartSMAPO(OBS, ACT, seed=0, entropy_pressure=0.05,
                            entropy_warmup_iters=3, pressure_track=False,
                            pressure_boost=False)
        valid = torch.ones(T, B)
        batch = {"obs": torch.randn(T, B, OBS), "rew": torch.zeros(T, B),
                 "time": torch.full((T, B), 2.0)}
        scales = [1.0, 3.0, 2.0]                 # median is 2.0
        for s in scales[:-1]:
            agent.shape_advantage(torch.randn(T, B, generator=torch.Generator().manual_seed(1)) * s, valid, batch)
            self.assertEqual(agent.entropy_loss_coeff, 0.0, "bonus is off during warmup")
        agent.shape_advantage(torch.randn(T, B) * scales[-1], valid, batch)
        self.assertIsNotNone(agent._adv_std_frozen)
        self.assertAlmostEqual(agent.entropy_loss_coeff,
                               0.05 * agent._adv_std_frozen, places=9)

    def test_tracking_holds_the_ratio_as_the_scale_moves(self):
        agent = RsmartSMAPO(OBS, ACT, seed=0, entropy_pressure=0.05,
                            entropy_warmup_iters=1, pressure_track=True,
                            pressure_track_steps=1.0,   # one iteration = full memory
                            pressure_boost=False)
        valid = torch.ones(T, B)
        batch = {"obs": torch.randn(T, B, OBS), "rew": torch.zeros(T, B),
                 "time": torch.full((T, B), 2.0)}
        base = torch.randn(T, B, generator=torch.Generator().manual_seed(2))
        agent.shape_advantage(base * 10.0, valid, batch)     # freeze at a large scale
        frozen = agent._adv_std_frozen
        agent.shape_advantage(base * 40.0, valid, batch)     # scale grows 4x
        self.assertAlmostEqual(agent.realised_pressure, 0.05, places=6)
        self.assertGreater(agent.entropy_loss_coeff, 0.05 * frozen)

    def test_tracked_coefficient_never_falls_below_the_frozen_floor(self):
        agent = RsmartSMAPO(OBS, ACT, seed=0, entropy_pressure=0.05,
                            entropy_warmup_iters=1, pressure_track=True,
                            pressure_track_steps=1.0, pressure_boost=False)
        valid = torch.ones(T, B)
        batch = {"obs": torch.randn(T, B, OBS), "rew": torch.zeros(T, B),
                 "time": torch.full((T, B), 2.0)}
        base = torch.randn(T, B, generator=torch.Generator().manual_seed(3))
        agent.shape_advantage(base * 10.0, valid, batch)
        floor = 0.05 * agent._adv_std_frozen
        for _ in range(5):                                   # scale collapses
            agent.shape_advantage(base * 0.01, valid, batch)
        self.assertGreaterEqual(agent.entropy_loss_coeff, floor - 1e-12)

    def test_boost_engages_only_when_the_scale_starves(self):
        # One iteration consumes T*B*tau = 72 environment steps. The time
        # constant must span SEVERAL iterations, or the tracked scale jumps
        # instantly and its running peak never records the healthy level --
        # starvation is relative to that peak, so it could never trigger.
        kw = dict(entropy_pressure=0.05, entropy_warmup_iters=1,
                  pressure_track=True, pressure_track_steps=720.0,
                  pressure_boost=True, boost_efold_steps=720.0)
        valid = torch.ones(T, B)
        batch = {"obs": torch.randn(T, B, OBS), "rew": torch.zeros(T, B),
                 "time": torch.full((T, B), 2.0)}
        base = torch.randn(T, B, generator=torch.Generator().manual_seed(4))

        healthy = RsmartSMAPO(OBS, ACT, seed=0, **kw)
        for _ in range(40):
            healthy.shape_advantage(base * 10.0, valid, batch)
        self.assertAlmostEqual(healthy.boost, 1.0, places=9)

        starved = RsmartSMAPO(OBS, ACT, seed=0, **kw)
        for _ in range(20):                       # establish a healthy peak
            starved.shape_advantage(base * 10.0, valid, batch)
        self.assertAlmostEqual(starved.boost, 1.0, places=9)
        for _ in range(40):                       # then the scale collapses
            starved.shape_advantage(base * 0.01, valid, batch)
        self.assertGreater(starved.boost, 1.0)
        self.assertLessEqual(starved.boost, starved.boost_max + 1e-9)

    def test_apo_ignores_the_holding_time(self):
        apo, rsmart = APO(OBS, ACT, seed=0), RsmartSMAPO(OBS, ACT, seed=0)
        r, t = torch.full((T, B), 1.0), torch.full((T, B), 5.0)
        apo.rho = rsmart.rho = 0.3
        self.assertTrue(torch.allclose(apo.rate_residual(r, t), r - 0.3 * 1.0))
        self.assertTrue(torch.allclose(rsmart.rate_residual(r, t), r - 0.3 * 5.0))

    def test_discrete_variant_runs_and_evaluates_by_argmax(self):
        n_actions = 5
        agent = DiscreteRsmartSMAPO(OBS, n_actions, seed=0)
        self.assertIsInstance(agent, CategoricalHead)
        buf, bv = rollout(agent)
        stats = agent.update(buf, bv)
        self.assertTrue(math.isfinite(stats["loss"]))
        obs = torch.randn(7, OBS)
        a = agent.eval_act(obs)
        self.assertEqual(tuple(a.shape), (7,))
        self.assertTrue(((a >= 0) & (a < n_actions)).all())
        params, _ = agent.forward_net(obs)
        self.assertTrue(torch.equal(a.long(), params.argmax(-1)))

    def test_continuous_default_is_gaussian(self):
        self.assertIsInstance(RsmartSMAPO(OBS, ACT, seed=0), GaussianHead)


if __name__ == "__main__":
    unittest.main()
