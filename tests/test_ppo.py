import math
import unittest

import torch
from torch.distributions import Categorical

from smdp_agents.average_rates import (NormalizedExponentialMovingTimeRate,
                                       WeightedHarmonicRate)
from smdp_agents.gaussian_mlp import CategoricalMLP, GaussianMLP
from smdp_agents.ppo import PPO, RolloutBuffer
from smdp_agents.smapo import (ExperimentalWeightedHarmonicSMAPO,
                               RsmartSMAPO)


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


class SMDPDiscountTests(unittest.TestCase):
    """The GAE discount is ``gamma^tau``, not a flat ``gamma`` per transition.

    A macro-step of duration tau stands in for tau primitive steps, so it must be
    discounted as such. The reference numbers below are hand-computed from the
    recursion rather than captured from a run.
    """

    def gae(self, agent, tau, bootstrap=1.0, steps=2):
        """``_gae`` on an all-zero [steps, 1] batch, bootstrapping ``bootstrap``."""
        zeros = torch.zeros(steps, 1)
        return agent._gae(zeros, zeros, torch.ones(steps, 1),
                          torch.tensor([bootstrap]),
                          torch.full((steps, 1), float(tau)))[0]

    def build(self, discount):
        return PPO(1, 1, hidden=(2,), discount=discount, gae_lambda=1.0, seed=0)

    def test_a_unit_holding_time_is_the_plain_mdp_discount(self):
        # gamma^1 == gamma, so nothing changes for a conventional MDP rollout.
        advantage = self.gae(self.build(0.5), tau=1.0)
        self.assertAlmostEqual(advantage[1].item(), 0.5)     # 0.5 * bootstrap
        self.assertAlmostEqual(advantage[0].item(), 0.25)    # 0.5 * lambda * adv[1]

    def test_a_longer_holding_time_compounds_the_discount(self):
        # tau=2 -> 0.5^2 = 0.25 in both the residual and the lambda recursion.
        advantage = self.gae(self.build(0.5), tau=2.0)
        self.assertAlmostEqual(advantage[1].item(), 0.25)
        self.assertAlmostEqual(advantage[0].item(), 0.0625)

    def test_it_is_gamma_to_the_tau_and_not_gamma_times_tau(self):
        # The plausible wrong implementation, gamma*tau, agrees at tau=1 and
        # diverges immediately after; pin the exponent.
        for tau in (2.0, 3.0, 7.5):
            with self.subTest(tau=tau):
                advantage = self.gae(self.build(0.9), tau=tau, steps=1)
                self.assertAlmostEqual(advantage[0].item(), 0.9 ** tau, places=5)

    def test_the_average_reward_variants_are_untouched_by_it(self):
        # discount == 1.0 makes gamma^tau == 1 for every tau, so holding time
        # reaches the objective only through rate_residual's -rho*tau.
        agent = RsmartSMAPO(1, 1, hidden=(2,), gae_lambda=1.0, seed=0)
        self.assertEqual(agent.discount, 1.0)
        agent.rho = 0.0
        reference = self.gae(agent, tau=1.0)
        for tau in (2.0, 9.0):
            with self.subTest(tau=tau):
                self.assertTrue(torch.allclose(self.gae(agent, tau=tau), reference))

    def test_holding_time_still_reaches_rho_when_the_discount_is_inert(self):
        # The other half of the above: with gamma^tau == 1, tau must still bite.
        agent = RsmartSMAPO(1, 1, hidden=(2,), gae_lambda=1.0, seed=0)
        agent.rho = 1.0
        one, two = self.gae(agent, tau=1.0, steps=1), self.gae(agent, tau=2.0, steps=1)
        self.assertAlmostEqual(one.item(), 0.0)    # 0 - 1*1 + 1*1
        self.assertAlmostEqual(two.item(), -1.0)   # 0 - 1*2 + 1*1

    def test_it_composes_with_a_scaled_rate_residual(self):
        # gamma^tau must multiply the bootstrap only — the residual keeps whatever
        # rate_residual returns, including the |rho| scaling.
        agent = ExperimentalWeightedHarmonicSMAPO(1, 1, hidden=(2,), seed=0)
        agent.discount, agent.gae_lambda, agent.rho = 0.5, 1.0, 2.0
        advantage, _ = agent._gae(
            torch.tensor([[6.0]]), torch.zeros(1, 1), torch.ones(1, 1),
            torch.tensor([1.0]), torch.tensor([[2.0]]))
        # (6 - 2*2)/|2| + 0.5^2 * 1
        self.assertAlmostEqual(advantage.item(), 1.0 + 0.25)


class DiscreteActorTests(unittest.TestCase):
    """``discrete=True`` swaps the Gaussian head for a categorical one.

    Everything downstream of the head — rho, GAE, the clipped surrogate — is
    shared, so these tests pin the head swap and the action round-trip.
    """

    N_ACTIONS = 5

    def build(self, **kwargs):
        return PPO(3, self.N_ACTIONS, hidden=(4,), seed=0, discrete=True, **kwargs)

    def test_the_flag_picks_the_head(self):
        self.assertIsInstance(self.build().net, CategoricalMLP)
        self.assertIsInstance(PPO(3, 2, hidden=(4,), seed=0).net, GaussianMLP)

    def test_act_returns_integer_option_indices(self):
        action, value, logp = self.build().act(torch.zeros(6, 3))
        self.assertEqual(action.dtype, torch.int64)
        self.assertEqual(action.shape, (6,))
        self.assertTrue(bool(((action >= 0) & (action < self.N_ACTIONS)).all()))
        self.assertEqual(value.shape, (6,))
        self.assertEqual(logp.shape, (6,))

    def test_the_logp_act_returns_is_the_categorical_one(self):
        agent = self.build()
        obs = torch.randn(6, 3)
        action, _, logp = agent.act(obs)
        expected = Categorical(logits=agent.net(obs)[0]).log_prob(action)
        self.assertTrue(torch.allclose(logp, expected))

    def test_eval_act_is_the_argmax_not_the_logits(self):
        agent = self.build()
        obs = torch.randn(6, 3)
        self.assertTrue(torch.equal(agent.eval_act(obs),
                                    agent.net(obs)[0].argmax(-1)))

    def test_value_reads_the_last_tuple_element_for_either_head(self):
        # CategoricalMLP returns (logits, value) and GaussianMLP (mu, log_std,
        # value); PPO.value indexes [-1] so it must not care which.
        obs = torch.randn(6, 3)
        for agent in (self.build(), PPO(3, 2, hidden=(4,), seed=0)):
            with self.subTest(discrete=agent.discrete):
                self.assertTrue(torch.equal(agent.value(obs), agent.net(obs)[-1]))

    def test_an_unbatched_observation_still_works(self):
        action, value, logp = self.build().act(torch.zeros(3))
        self.assertEqual(action.shape, ())
        self.assertEqual(value.shape, ())
        self.assertEqual(logp.shape, ())

    def test_the_buffer_round_trips_indices_through_its_float_store(self):
        # RolloutBuffer stores everything as float32; update() casts back with
        # .long(), so the indices have to survive the trip exactly.
        buffer = RolloutBuffer()
        buffer.add(torch.zeros(2, 3), torch.tensor([3, 0]), torch.zeros(2),
                   torch.zeros(2), torch.zeros(2), torch.zeros(2), torch.zeros(2))
        stored = buffer.stacked("cpu")["act"]
        self.assertEqual(stored.dtype, torch.float32)
        self.assertTrue(torch.equal(stored.reshape(-1).long(), torch.tensor([3, 0])))

    def rollout(self, agent, steps=3, envs=2):
        buffer, obs = RolloutBuffer(), torch.randn(envs, 3)
        for _ in range(steps):
            action, value, logp = agent.act(obs)
            buffer.add(obs, action, torch.randn(envs), torch.zeros(envs),
                       torch.zeros(envs), value, logp)
            obs = torch.randn(envs, 3)
        return buffer, obs

    def test_an_update_runs_and_reports_finite_stats(self):
        torch.manual_seed(4)
        agent = self.build(epochs=2, minibatches=2)
        stats = agent.update(*self.rollout(agent)[:1], agent.value(torch.zeros(2, 3)))
        for key, value in stats.items():
            with self.subTest(stat=key):
                self.assertTrue(math.isfinite(value))
        # Categorical entropy is bounded by log(n) — a Gaussian's is not, so this
        # also catches the entropy term coming from the wrong distribution.
        self.assertLessEqual(stats["entropy"], math.log(self.N_ACTIONS) + 1e-6)

    def test_the_sampled_indices_are_the_ones_scored_in_the_surrogate(self):
        # On the very first minibatch the policy is unchanged, so the ratio is
        # exactly 1 and the loss is fully predictable. Any mix-up between an
        # action's index and its slot would move new_logp off old_logp and break
        # this equality — which a finite-stats smoke test would not catch.
        torch.manual_seed(4)
        agent = self.build(epochs=1, minibatches=1)
        buffer, final_obs = self.rollout(agent)
        batch = buffer.stacked("cpu")
        advantage, _ = agent._gae(batch["rew"], batch["val"],
                                  torch.ones_like(batch["rew"]),
                                  agent.value(final_obs), batch["time"])
        entropy = Categorical(
            logits=agent.net(batch["obs"].reshape(-1, 3))[0]).entropy().mean()
        expected = (-advantage.mean() + 0.5 * (advantage ** 2).mean()
                    - agent.entropy_loss_coeff * entropy)
        stats = agent.update(buffer, agent.value(final_obs))
        self.assertAlmostEqual(stats["loss"], expected.item(), places=5)


if __name__ == "__main__":
    unittest.main()
