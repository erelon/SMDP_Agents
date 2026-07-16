import importlib.util
import math
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class DeepAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global torch, nn, QLearning, RLearning, DeepQWrapper
        global GaussianMLP, gaussian_entropy, gaussian_logp
        global PPO, SmartPPO, RsmartPPO, HarmonicPPO, RolloutBuffer
        import torch
        import torch.nn as nn
        from agents.deep_q_wrapper import DeepQWrapper
        from agents.gaussian_mlp import GaussianMLP, gaussian_entropy, gaussian_logp
        from agents.ppo import PPO, SmartPPO, RsmartPPO, HarmonicPPO, RolloutBuffer
        from agents.q_learning import QLearning
        from agents.r_learning import RLearning

    def test_gaussian_helpers_and_network_shapes(self):
        net = GaussianMLP(3, 2, hidden=(4,), init_log_std=math.log(2))
        mu, log_std, value = net(torch.zeros(5, 3))
        self.assertEqual(mu.shape, (5, 2))
        self.assertEqual(log_std.shape, (5, 2))
        self.assertEqual(value.shape, (5,))
        action = mu.clone()
        expected = -2 * math.log(2) - math.log(2 * math.pi)
        self.assertTrue(torch.allclose(gaussian_logp(action, mu, log_std),
                                       torch.full((5,), expected)))
        self.assertEqual(gaussian_entropy(log_std).shape, (5,))

    def test_deep_q_target_gradient_and_action(self):
        torch.manual_seed(1)
        network = nn.Linear(1, 2)
        with torch.no_grad():
            network.weight.zero_()
            network.bias.copy_(torch.tensor([1.0, 2.0]))
        wrapped = DeepQWrapper(
            QLearning("q", [0, 1], learning_rate=0.1, discount_factor=0.5,
                      exploration_rate=0),
            network,
        )
        self.assertEqual(wrapped.eval([3.0]), 1)
        self.assertAlmostEqual(wrapped._compute_td_target([0], 0, 1, [1], 7), 2.0)
        before = wrapped.network.bias.detach().clone()
        wrapped.learn([0], 0, 1, [1], 7)
        self.assertFalse(torch.equal(before, wrapped.network.bias.detach()))

    def test_deep_r_learning_updates_rho_even_for_non_greedy_action(self):
        network = nn.Linear(1, 2)
        with torch.no_grad():
            network.weight.zero_()
            network.bias.copy_(torch.tensor([0.0, 1.0]))
        wrapped = DeepQWrapper(
            RLearning("r", [0, 1], learning_rate=0.1, rho_learning_rate=0.5,
                      exploration_rate=0, with_rho_trick=True),
            network,
        )
        wrapped.learn([0], 0, 2.0, [1], 1)
        self.assertNotEqual(wrapped.rho, 0.0)

    def test_replay_threshold_target_sync_and_reset(self):
        network = nn.Linear(1, 1)
        wrapped = DeepQWrapper(
            QLearning("q", [0], learning_rate=0.1, exploration_rate=0),
            network, replay_buffer_size=4, batch_size=2, target_update_freq=2
        )
        initial = wrapped.network.weight.detach().clone()
        wrapped.learn([0], 0, 1, [1], 1)
        self.assertTrue(torch.equal(initial, wrapped.network.weight.detach()))
        wrapped.learn([0], 0, 1, [1], 1)
        self.assertEqual(wrapped._learn_step, 2)
        for a, b in zip(wrapped.network.parameters(), wrapped.target_network.parameters()):
            self.assertTrue(torch.equal(a, b))
        wrapped.reset()
        self.assertEqual(len(wrapped.replay_buffer), 0)
        self.assertEqual(wrapped._learn_step, 0)

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


if __name__ == "__main__":
    unittest.main()
