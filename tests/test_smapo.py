import math
import unittest

import torch

from agents.policy_heads import CategoricalHead, GaussianHead
from agents.ppo import RolloutBuffer
from agents.average_rates import (NormalizedExponentialMovingTimeRate,
                                  WeightedHarmonicRate)
from agents.smapo import (APO, CumulativeHarmonicSMAPO, WeightedHarmonicSMAPO,
                          CumulativeWeightedHarmonicSMAPO,
                          DiscreteRsmartSMAPO,
                          ExperimentalCumulativeWeightedHarmonicSMAPO,
                          ExperimentalWeightedHarmonicSMAPO, HarmonicSMAPO,
                          RsmartSMAPO, SmartSMAPO, SmoothedSmartSMAPO)

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
        for cls in (APO, RsmartSMAPO, SmartSMAPO, HarmonicSMAPO,
                    WeightedHarmonicSMAPO, CumulativeHarmonicSMAPO,
                    CumulativeWeightedHarmonicSMAPO,
                    ExperimentalWeightedHarmonicSMAPO,
                    ExperimentalCumulativeWeightedHarmonicSMAPO,
                    SmoothedSmartSMAPO):
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


class RateEstimatorTests(unittest.TestCase):
    """Each variant feeds the batch to its estimator the way that estimator needs."""

    def test_rho_reduction_per_variant(self):
        reward = torch.tensor([[2.0, -1.0], [4.0, 1.0]])
        duration = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
        value = torch.tensor([[3.0, 1.0], [2.0, 2.0]])

        smart = SmartSMAPO(1, 1, hidden=(2,), rho_lr=0.5)
        smart.update_rho(reward, value, duration)
        self.assertAlmostEqual(smart.rho, 6.0 / 6.0)      # sum r / sum tau
        self.assertEqual(smart.value_bias, 2.0)

        rsmart = RsmartSMAPO(1, 1, hidden=(2,), rho_lr=0.5)
        rsmart.update_rho(reward, value, duration)
        self.assertAlmostEqual(rsmart.rho,
                               reward.mean().item() / duration.mean().item())

        harmonic = HarmonicSMAPO(1, 1, hidden=(2,), rho_lr=0.5)
        harmonic.update_rho(reward, value, duration)
        self.assertTrue(math.isfinite(harmonic.rho))

    def test_apo_rate_ignores_duration(self):
        reward = torch.tensor([[2.0, -1.0], [4.0, 1.0]])
        duration = torch.tensor([[7.0, 7.0], [7.0, 7.0]])
        apo = APO(1, 1, hidden=(2,), rho_lr=0.5)
        apo.update_rho(reward, torch.zeros(2, 2), duration)
        self.assertAlmostEqual(apo.rho, reward.mean().item())   # tau == 1


class SmoothedSmartSMAPOTests(unittest.TestCase):
    """The deep SmoothedSMART reuses the tabular time-decayed estimator."""

    def build(self):
        return SmoothedSmartSMAPO(2, 1, hidden=(4,), seed=0, rho_lr=0.2)

    def test_it_is_an_average_reward_variant_fed_per_transition(self):
        agent = self.build()
        self.assertTrue(agent.longrun)
        self.assertEqual(agent.discount, 1.0)
        self.assertEqual(agent.rho_reduce, "none")

    def test_rho_is_the_tabular_time_decayed_estimator(self):
        agent = self.build()
        self.assertIsInstance(agent.time_rate, NormalizedExponentialMovingTimeRate)
        self.assertAlmostEqual(agent.lambda_, -math.log(1 - 0.2))
        reference = NormalizedExponentialMovingTimeRate(0.2)
        for reward, duration in ((4.0, 2.0), (1.0, 0.5), (9.0, 3.0)):
            agent.calc_new_rho(reward, duration, None, None)
            self.assertAlmostEqual(agent.rho, reference.update(reward, duration))

    def test_update_rho_walks_the_batch_transition_by_transition(self):
        agent = self.build()
        reward = torch.tensor([[2.0, 6.0], [4.0, 1.0]])
        duration = torch.tensor([[1.0, 3.0], [2.0, 0.5]])
        agent.update_rho(reward, torch.zeros(2, 2), duration)
        reference = NormalizedExponentialMovingTimeRate(0.2)
        for r, t in zip(reward.reshape(-1).tolist(), duration.reshape(-1).tolist()):
            reference.update(r, t)
        self.assertAlmostEqual(agent.rho, reference.rho)

    def test_it_keeps_the_plain_rate_residual(self):
        agent = self.build()
        agent.rho = 2.0
        residual = agent.rate_residual(torch.tensor([6.0]), torch.tensor([2.0]))
        self.assertAlmostEqual(residual.item(), 2.0)       # 6 - 2*2, unscaled


class ToggleTests(unittest.TestCase):
    """Both changes SMAPO adds can be switched off independently."""

    def setUp(self):
        self.adv = torch.randn(T, B, generator=torch.Generator().manual_seed(9)) + 4.0
        self.valid = torch.ones(T, B)
        self.batch = {"obs": torch.randn(T, B, OBS), "rew": torch.zeros(T, B),
                      "time": torch.full((T, B), 2.0)}

    def test_a_centering_off_leaves_the_offset(self):
        agent = RsmartSMAPO(OBS, ACT, seed=0, a_centering=False)
        out = agent.shape_advantage(self.adv.clone(), self.valid, self.batch)
        self.assertGreater(abs(float(out.mean())), 1.0)

    def test_calibrated_pressure_off_keeps_a_fixed_coefficient(self):
        agent = RsmartSMAPO(OBS, ACT, seed=0, calibrated_pressure=False,
                            entropy_loss_coeff=0.01, entropy_warmup_iters=1)
        for _ in range(3):
            agent.shape_advantage(self.adv.clone(), self.valid, self.batch)
        self.assertEqual(agent.entropy_loss_coeff, 0.01)

    def test_zero_pressure_is_rejected_rather_than_silently_disabling(self):
        with self.assertRaises(ValueError):
            RsmartSMAPO(OBS, ACT, entropy_pressure=0.0)


class AbsRhoScaledSMAPOTests(unittest.TestCase):
    """The |rho| scaling has to reach the GAE residual, not set_target."""

    def build(self):
        return ExperimentalWeightedHarmonicSMAPO(2, 1, hidden=(4,), seed=0,
                                                 rho_lr=0.3)

    def test_the_residual_is_divided_by_the_magnitude_of_rho(self):
        agent, plain = self.build(), HarmonicSMAPO(2, 1, hidden=(4,), seed=0)
        reward, duration = torch.tensor([6.0]), torch.tensor([2.0])
        for rho in (2.0, -2.0):
            with self.subTest(rho=rho):
                agent.rho = plain.rho = rho
                self.assertAlmostEqual(agent.rate_residual(reward, duration).item(),
                                       (6.0 - rho * 2.0) / abs(rho))
                self.assertAlmostEqual(plain.rate_residual(reward, duration).item(),
                                       6.0 - rho * 2.0)

    def test_a_zero_rho_falls_back_instead_of_dividing(self):
        agent = self.build()
        self.assertEqual(agent.rho, 0.0)
        self.assertAlmostEqual(
            agent.rate_residual(torch.tensor([6.0]), torch.tensor([2.0])).item(), 6.0)

    def test_the_scaling_reaches_gae_rather_than_being_ignored(self):
        """SMAPO defines rate_residual and precedes AbsRhoScaledTarget in the MRO,
        so inheriting the mixin alone would silently apply no scaling. This is the
        test that catches that."""
        agent, plain = self.build(), HarmonicSMAPO(2, 1, hidden=(4,), seed=0)
        args = (torch.tensor([[4.0]]), torch.zeros(1, 1), torch.ones(1, 1),
                torch.tensor([0.0]), torch.tensor([[2.0]]))
        for a in (agent, plain):
            a.discount, a.gae_lambda, a.rho = 1.0, 1.0, 2.0
        self.assertAlmostEqual(agent._gae(*args)[0].item(), 0.0)    # (4 - 4)/2
        self.assertAlmostEqual(plain._gae(*args)[0].item(), 0.0)
        for a in (agent, plain):
            a.rho = 0.5
        self.assertAlmostEqual(agent._gae(*args)[0].item(), 6.0)    # (4 - 1)/0.5
        self.assertAlmostEqual(plain._gae(*args)[0].item(), 3.0)    # 4 - 1

    def test_rho_is_the_reward_weighted_harmonic_estimator(self):
        agent = self.build()
        self.assertIsInstance(agent.hma, WeightedHarmonicRate)
        self.assertEqual(agent.rho_reduce, "none")
        reference = WeightedHarmonicRate(0.3)
        for reward, duration in ((4.0, 2.0), (-1.0, 1.0), (3.0, 2.0)):
            agent.calc_new_rho(reward, duration, None, None)
            self.assertAlmostEqual(agent.rho,
                                   reference.update(reward, duration, reward))


if __name__ == "__main__":
    unittest.main()
