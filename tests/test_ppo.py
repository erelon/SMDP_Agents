import math
import unittest

import torch

from agents.average_rates import (NormalizedExponentialMovingTimeRate,
                                  WeightedHarmonicRate)
from agents.ppo import (PPO, ExperimentalWeightedHarmonicPPO, HarmonicPPO,
                        RolloutBuffer, RsmartPPO, SmartPPO, SmoothedSmartPPO)


class PPOTests(unittest.TestCase):
    def test_rollout_buffer_and_ppo_gae(self):
        buffer = RolloutBuffer()
        for reward, duration in ((1.0, 2.0), (2.0, 3.0)):
            buffer.add([[0.0]], [[0.0]], [reward], [False], [False], [0.0], [0.0],
                       time=duration)
        stacked = buffer.stacked("cpu")
        self.assertEqual(tuple(stacked["time"].shape), (2, 1))
        ppo = PPO(1, 1, hidden=(2,), discount=1.0, gae_lambda=1.0, epochs=1,
                  minibatches=1)
        advantage, returns = ppo._gae(
            torch.tensor([[1.0], [2.0]]), torch.zeros(2, 1), torch.ones(2, 1),
            torch.tensor([0.0]), torch.tensor([[2.0], [3.0]])
        )
        self.assertTrue(torch.equal(advantage, torch.tensor([[3.0], [2.0]])))
        self.assertTrue(torch.equal(returns, advantage))

    def test_ppo_rho_reduction_variants(self):
        reward = torch.tensor([[2.0, -1.0], [4.0, 1.0]])
        duration = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
        value = torch.tensor([[3.0, 1.0], [2.0, 2.0]])

        smart = SmartPPO(1, 1, hidden=(2,), rho_lr=0.5)
        smart.update_rho(reward, value, duration)
        self.assertAlmostEqual(smart.rho, 6.0 / 6.0)
        self.assertEqual(smart.value_bias, 2.0)

        relaxed = RsmartPPO(1, 1, hidden=(2,), rho_lr=0.5)
        relaxed.update_rho(reward, value, duration)
        self.assertAlmostEqual(relaxed.rho, reward.mean().item() / duration.mean().item())

        harmonic = HarmonicPPO(1, 1, hidden=(2,), rho_lr=0.5)
        harmonic.update_rho(reward, value, duration)
        self.assertTrue(math.isfinite(harmonic.rho))

    def test_small_ppo_update_returns_finite_stats(self):
        torch.manual_seed(4)
        ppo = PPO(1, 1, hidden=(4,), epochs=1, minibatches=1, batch_T=2)
        buffer = RolloutBuffer()
        obs = torch.zeros(1, 1)
        for reward in (1.0, -0.5):
            action, value, logp = ppo.act(obs)
            buffer.add(obs, action, torch.tensor([reward]), torch.tensor([False]),
                       torch.tensor([False]), value, logp)
        stats = ppo.update(buffer, ppo.value(obs))
        for value in stats.values():
            self.assertTrue(math.isfinite(value))


class SmoothedSmartPPOTests(unittest.TestCase):
    """The deep SmoothedSMART reuses the tabular time-decayed estimator."""

    def build(self):
        return SmoothedSmartPPO(2, 1, hidden=(4,), seed=0, rho_lr=0.2)

    def test_it_is_an_average_reward_variant_fed_per_transition(self):
        agent = self.build()
        self.assertTrue(agent.longrun)
        self.assertEqual(agent.discount, 1.0)
        # Per-transition, not aggregated: the estimator decays by exp(-lambda*tau),
        # so collapsing a batch to one (sum r, sum tau) is exact only when the rate
        # is constant across it.
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
        self.assertAlmostEqual(residual.item(), 2.0)   # 6 - 2*2, unscaled


class ExperimentalWeightedHarmonicPPOTests(unittest.TestCase):
    """The |rho| scaling has to reach the GAE residual, not set_target."""

    def build(self):
        return ExperimentalWeightedHarmonicPPO(2, 1, hidden=(4,), seed=0, rho_lr=0.3)

    def test_the_residual_is_divided_by_the_magnitude_of_rho(self):
        agent, plain = self.build(), HarmonicPPO(2, 1, hidden=(4,), seed=0)
        reward, duration = torch.tensor([6.0]), torch.tensor([2.0])
        for rho in (2.0, -2.0):
            with self.subTest(rho=rho):
                agent.rho = plain.rho = rho
                expected = (6.0 - rho * 2.0) / abs(rho)
                self.assertAlmostEqual(agent.rate_residual(reward, duration).item(),
                                       expected)
                self.assertAlmostEqual(plain.rate_residual(reward, duration).item(),
                                       6.0 - rho * 2.0)

    def test_a_zero_rho_falls_back_instead_of_dividing(self):
        agent = self.build()
        self.assertEqual(agent.rho, 0.0)
        self.assertAlmostEqual(
            agent.rate_residual(torch.tensor([6.0]), torch.tensor([2.0])).item(), 6.0)

    def test_the_scaling_reaches_gae_rather_than_being_ignored(self):
        # PPO never calls set_target, so an override there would be silent. This is
        # the test that would catch that regression.
        agent, plain = self.build(), HarmonicPPO(2, 1, hidden=(4,), seed=0)
        args = (torch.tensor([[4.0]]), torch.zeros(1, 1), torch.ones(1, 1),
                torch.tensor([0.0]), torch.tensor([[2.0]]))
        for a in (agent, plain):
            a.discount, a.gae_lambda, a.rho = 1.0, 1.0, 2.0
        scaled, _ = agent._gae(*args)
        unscaled, _ = plain._gae(*args)
        self.assertAlmostEqual(scaled.item(), 0.0)      # (4 - 4)/2
        self.assertAlmostEqual(unscaled.item(), 0.0)
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
            self.assertAlmostEqual(agent.rho, reference.update(reward, duration, reward))


if __name__ == "__main__":
    unittest.main()
