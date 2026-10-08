import math
import unittest

import torch

from agents.ppo import PPO, RolloutBuffer


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


if __name__ == "__main__":
    unittest.main()
