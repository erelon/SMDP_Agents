import math
import unittest

import torch

from agents.gaussian_mlp import GaussianMLP, gaussian_entropy, gaussian_logp


class GaussianMLPTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
