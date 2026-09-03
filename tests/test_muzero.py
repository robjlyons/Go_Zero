import unittest
from unittest import mock

import gymnasium as gym
import numpy as np

from MuZero_Simple import (
    Config,
    Experience,
    MuZero,
    Network,
    ReplayBuffer,
    _prepare_observation,
    reset_env,
    step_env,
)


class MuZeroTest(unittest.TestCase):
    def setUp(self):
        self.env = gym.make("CartPole-v1", max_episode_steps=3)
        self.config = Config(
            hidden_units=8,
            num_simulations=3,
            max_steps_per_rollout=3,
            batch_size=2,
            training_steps_per_rollout=1,
        )
        self.agent = MuZero(self.env, Network(self.env.action_space.n, 8), self.config)

    def tearDown(self):
        self.env.close()

    def test_gymnasium_reset_and_step(self):
        observation = reset_env(self.env, seed=7)
        next_observation, reward, done, info = step_env(self.env, 0)

        self.assertEqual(observation.shape, next_observation.shape)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

    def test_mcts_distributes_all_visits(self):
        observation = reset_env(self.env, seed=7)
        root = self.agent.mcts(observation)

        self.assertEqual(sum(child.visit_count for child in root.children.values()), 3)
        self.assertTrue(np.isclose(sum(child.prior for child in root.children.values()), 1.0))

    def test_training_populates_replay_and_has_finite_loss(self):
        self.agent.train(1, seed=7)

        self.assertGreater(len(self.agent.replay), 0)
        self.assertTrue(np.isfinite(self.agent._train_batch()))

    def test_training_uses_a_hard_environment_step_budget(self):
        with mock.patch("MuZero_Simple.step_env", wraps=step_env) as counted_step:
            self.agent.train(5, seed=7)

        self.assertEqual(counted_step.call_count, 5)

    def test_training_rejects_non_positive_step_budget(self):
        with self.assertRaisesRegex(ValueError, "max_steps must be positive"):
            self.agent.train(0, seed=7)


class PerformanceHelpersTest(unittest.TestCase):
    @staticmethod
    def _experience(action):
        observation = np.zeros(1, dtype=np.float32)
        return Experience(observation, action, 0.0, observation, observation, 0.0)

    def test_replay_buffer_overwrites_oldest_slot(self):
        replay = ReplayBuffer(2)
        replay.append(self._experience(0))
        replay.append(self._experience(1))
        replay.append(self._experience(2))

        self.assertEqual(len(replay), 2)
        self.assertEqual({replay[0].action, replay[1].action}, {1, 2})

    def test_byte_observations_are_scaled_and_flattened(self):
        observation = np.array([[0, 255]], dtype=np.uint8)

        prepared = _prepare_observation(observation)

        np.testing.assert_array_equal(prepared, np.array([0.0, 1.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
