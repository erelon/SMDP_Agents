import unittest

import torch
import torch.nn as nn

from agents.deep_q_wrapper import DeepQWrapper
from agents.q_learning import ContinuousQLearning, QLearning
from agents.r_learning import RLearning


class DeepQWrapperTests(unittest.TestCase):
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
        # Discrete Q-learning charges one step whatever the elapsed time, so
        # the target discounts by gamma**1: 1 + 0.5 * 2.0.
        self.assertEqual(wrapped.holding_time(7), 1.0)
        self.assertAlmostEqual(
            wrapped._compute_td_target([0], 0, 1, [1], wrapped.holding_time(7)), 2.0)
        before = wrapped.network.bias.detach().clone()
        wrapped.learn([0], 0, 1, [1], 7)
        self.assertFalse(torch.equal(before, wrapped.network.bias.detach()))

    def test_holding_time_is_applied_exactly_once(self):
        # A non-idempotent clock catches double application; the discrete
        # agents' constant 1.0 would hide it.
        seen = []

        class HalfStep(ContinuousQLearning):
            def holding_time(self, time):
                return time / 2

            def set_target(self, reward, time, next_q):
                seen.append(time)
                return super().set_target(reward, time, next_q)

        wrapped = DeepQWrapper(
            HalfStep("half", [0, 1], exploration_rate=0), nn.Linear(1, 2)
        )
        wrapped.learn([0], 0, 1.0, [1], 4.0)
        self.assertEqual(seen, [2.0])

    def test_deep_r_learning_applies_the_rho_trick_like_the_tabular_agent(self):
        def wrap(with_rho_trick):
            network = nn.Linear(1, 2)
            with torch.no_grad():
                network.weight.zero_()
                network.bias.copy_(torch.tensor([0.0, 1.0]))
            return DeepQWrapper(
                RLearning("r", [0, 1], learning_rate=0.1, rho_learning_rate=0.5,
                          exploration_rate=0, with_rho_trick=with_rho_trick),
                network,
            )

        # Action 1 is greedy under the bias [0.0, 1.0], so action 0 is off-policy.
        gated = wrap(True)
        gated.learn([0], 0, 2.0, [1], 1)
        self.assertEqual(gated.rho, 0.0)

        ungated = wrap(False)
        ungated.learn([0], 0, 2.0, [1], 1)
        self.assertNotEqual(ungated.rho, 0.0)

        greedy = wrap(True)
        greedy.learn([0], 1, 2.0, [1], 1)
        self.assertNotEqual(greedy.rho, 0.0)

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
