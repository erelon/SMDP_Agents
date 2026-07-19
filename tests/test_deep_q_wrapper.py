import importlib.util
import math
import unittest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")


class DeepQWrapperTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
