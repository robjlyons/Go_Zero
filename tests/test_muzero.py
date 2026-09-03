import unittest

import gymnasium as gym
import numpy as np

from MuZero_Simple import Config, MuZero, Network, reset_env, step_env


class MuZeroTest(unittest.TestCase):
    def setUp(self):
        self.env = gym.make("CartPole-v1", max_episode_steps=3)
        self.config = Config(
            hidden_units=8,
            num_simulations=3,
            max_steps_per_episode=3,
            batch_size=2,
            training_steps_per_episode=1,
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


if __name__ == "__main__":
    unittest.main()
