import math
import unittest

import torch

from smdp_agents.gaussian_mlp import (CategoricalMLP, GaussianMLP,
                                      gaussian_entropy, gaussian_logp)


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


class CategoricalMLPTests(unittest.TestCase):
    def test_network_shapes(self):
        logits, value = CategoricalMLP(3, 4, hidden=(8,))(torch.zeros(5, 3))
        self.assertEqual(logits.shape, (5, 4))
        self.assertEqual(value.shape, (5,))

    def test_the_value_is_the_last_element_as_it_is_for_the_gaussian_head(self):
        # PPO.value indexes [-1], so the two heads have to agree on where the
        # value sits even though they return tuples of different lengths.
        obs = torch.randn(5, 3)
        self.assertEqual(len(CategoricalMLP(3, 4, hidden=(8,))(obs)), 2)
        self.assertEqual(len(GaussianMLP(3, 4, hidden=(8,))(obs)), 3)
        for net in (CategoricalMLP(3, 4, hidden=(8,)), GaussianMLP(3, 4, hidden=(8,))):
            with self.subTest(net=type(net).__name__):
                self.assertTrue(torch.equal(net(obs)[-1], net.v(obs).squeeze(-1)))

    def test_the_logits_are_unsquashed(self):
        # GaussianMLP tanh-squashes its mean into the action range; logits are not
        # an action, so a tanh there would silently cap the achievable sharpness.
        net = CategoricalMLP(3, 4, hidden=(8,))
        with torch.no_grad():
            for layer in reversed(net.logits):
                if isinstance(layer, torch.nn.Linear):
                    layer.bias.fill_(5.0)
                    break
        self.assertGreater(net(torch.zeros(1, 3))[0].abs().max().item(), 1.0)

    def test_it_has_no_log_std_parameter(self):
        names = dict(CategoricalMLP(3, 4, hidden=(8,)).named_parameters())
        self.assertNotIn("log_std", names)
        self.assertIn("log_std", dict(GaussianMLP(3, 4, hidden=(8,)).named_parameters()))


if __name__ == "__main__":
    unittest.main()
