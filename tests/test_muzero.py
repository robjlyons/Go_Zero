import unittest

import gymnasium as gym
import numpy as np

from MuZero_Simple import (
    ArchiveEntry,
    CellEncoder,
    Config,
    GoExploreArchive,
    MuZero,
    Network,
    make_env,
    reset_env,
    step_env,
)


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

    def test_environment_factory_keeps_classic_control_unwrapped(self):
        env = make_env("CartPole-v1")
        try:
            self.assertEqual(env.spec.id, "CartPole-v1")
            self.assertEqual(reset_env(env, seed=7).shape, (4,))
        finally:
            env.close()

    def test_mcts_distributes_all_visits(self):
        observation = reset_env(self.env, seed=7)
        root = self.agent.mcts(observation)

        self.assertEqual(sum(child.visit_count for child in root.children.values()), 3)
        self.assertTrue(np.isclose(sum(child.prior for child in root.children.values()), 1.0))

    def test_training_populates_replay_and_has_finite_loss(self):
        self.agent.train(1, seed=7)

        self.assertGreater(len(self.agent.replay), 0)
        self.assertGreater(len(self.agent.archive.cells), 1)
        self.assertTrue(np.isfinite(self.agent._train_batch()))

    def test_archive_keeps_best_path_and_does_not_select_terminal_cells(self):
        archive = GoExploreArchive(CellEncoder(2, 8, 7), 10, np.random.default_rng(7))
        observation = np.array([1.0, 2.0])
        archive.add(observation, ArchiveEntry(7, (1,), 2.0, terminal=True))
        archive.add(observation, ArchiveEntry(7, (0, 1), 1.0))
        archive.add(observation, ArchiveEntry(7, (0,), 2.0))
        archive.add(np.array([-1.0, -2.0]), ArchiveEntry(7, (), 3.0, terminal=True))

        retained = archive.cells[archive.encoder.encode(observation)]
        self.assertFalse(retained.terminal)
        self.assertEqual(retained.actions, (0,))
        self.assertEqual(archive.select().actions, (0,))

    def test_small_archive_does_not_evict_its_only_return_point(self):
        archive = GoExploreArchive(CellEncoder(2, 8, 11), 1, np.random.default_rng(11))
        archive.add(np.array([0.0, 0.0]), ArchiveEntry(11, (), 0.0))
        archive.add(np.array([10.0, 10.0]), ArchiveEntry(11, (0,), 1.0, terminal=True))

        self.assertFalse(archive.select().terminal)

    def test_training_empty_replay_has_clear_error(self):
        with self.assertRaisesRegex(RuntimeError, "before collecting"):
            self.agent._train_batch()


if __name__ == "__main__":
    unittest.main()
