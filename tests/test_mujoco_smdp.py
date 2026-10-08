import unittest

import numpy as np

try:
    from examples.envs.mujoco_smdp import CONFIGS, make_ant, make_swimmer
    HAVE_MUJOCO = True
except Exception:                                  # gymnasium[mujoco] not installed
    HAVE_MUJOCO = False


@unittest.skipUnless(HAVE_MUJOCO, "needs gymnasium[mujoco]")
class MujocoSMDPTests(unittest.TestCase):

    def test_both_environments_build_and_report_a_holding_time(self):
        for make, act_dim in ((make_swimmer, 2), (make_ant, 8)):
            env = make()
            obs, _ = env.reset(seed=0)
            self.assertEqual(env.action_space.shape, (act_dim,))
            self.assertEqual(tuple(env.action_space.low), (-1.0,) * act_dim)
            _, _, _, _, info = env.step(np.zeros(act_dim))
            self.assertIn("tau", info)
            self.assertGreaterEqual(info["tau"], 1.0)

    def test_the_holding_time_varies_with_the_commanded_distance(self):
        """The defining SMDP property: a bigger move costs more simulated time."""
        env = make_swimmer()
        dwell = []
        for magnitude in (0.02, 0.2, 1.0):
            env.reset(seed=1)
            dwell.append(np.mean([env.step(np.full(2, magnitude))[4]["tau"]
                                  for _ in range(20)]))
        self.assertLess(dwell[0], dwell[1])
        self.assertLess(dwell[1], dwell[2])

    def test_the_holding_time_is_capped(self):
        env = make_swimmer()
        env.reset(seed=2)
        rng = np.random.default_rng(0)
        taus = [env.step(rng.uniform(-1, 1, 2))[4]["tau"] for _ in range(60)]
        self.assertLessEqual(max(taus), CONFIGS["swimmer"]["max_frames"])
        self.assertGreater(np.std(taus), 0.0)      # and it is not constant

    def test_ant_charges_effort_rather_than_posture(self):
        """Ant's cost is on actuator force, so a still pose is not taxed."""
        env = make_ant()
        env.reset(seed=3)
        _, _, _, _, info = env.step(np.zeros(8))
        self.assertIn("reward_effort", info)
        self.assertLessEqual(info["reward_effort"], 0.0)

    def test_swimmer_never_terminates(self):
        env = make_swimmer()
        env.reset(seed=4)
        rng = np.random.default_rng(1)
        for _ in range(50):
            _, _, terminated, _, _ = env.step(rng.uniform(-1, 1, 2))
            self.assertFalse(terminated)


if __name__ == "__main__":
    unittest.main()
