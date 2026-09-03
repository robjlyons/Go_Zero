"""A small, self-contained MuZero-style training example.

This is intentionally an educational implementation: it uses episode returns
and replay rather than MuZero's full unrolled training procedure.
Importing this module is safe; training only starts through :func:`main`.
"""

from __future__ import annotations

import argparse
import os
from collections import OrderedDict, deque
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

# Avoid TensorFlow probing CUDA on CPU-only hosts. Set GO_ZERO_USE_GPU=1 to opt
# in; this must happen before TensorFlow is imported.
if os.environ.get("GO_ZERO_USE_GPU") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf


@dataclass(frozen=True)
class Config:
    gamma: float = 0.99
    learning_rate: float = 1e-3
    num_simulations: int = 50
    hidden_units: int = 128
    max_steps_per_episode: int = 500
    exploration_constant: float = 1.25
    batch_size: int = 32
    replay_capacity: int = 10_000
    training_steps_per_episode: int = 20
    gradient_clip_norm: float = 5.0
    archive_capacity: int = 10_000
    cell_bits: int = 16

    def __post_init__(self) -> None:
        positive_fields = (
            "learning_rate",
            "num_simulations",
            "hidden_units",
            "max_steps_per_episode",
            "batch_size",
            "replay_capacity",
            "training_steps_per_episode",
            "gradient_clip_norm",
            "archive_capacity",
            "cell_bits",
        )
        for name in positive_fields:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be between zero and one")


def reset_env(env: gym.Env, seed: int | None = None) -> np.ndarray:
    """Reset a Gymnasium environment and return a flattened observation."""
    observation, _ = env.reset(seed=seed)
    return np.asarray(gym.spaces.flatten(env.observation_space, observation), dtype=np.float32)


def step_env(env: gym.Env, action: int) -> tuple[np.ndarray, float, bool, dict]:
    """Step a Gymnasium environment and combine its two completion flags."""
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    flattened = np.asarray(gym.spaces.flatten(env.observation_space, observation), dtype=np.float32)
    return flattened, float(reward), bool(done), info


class Network(tf.keras.Model):
    """Representation, dynamics, and prediction functions used by MuZero."""

    def __init__(self, num_actions: int, hidden_units: int):
        super().__init__()
        self.num_actions = num_actions
        self.representation = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(hidden_units, activation="relu"),
                tf.keras.layers.Dense(hidden_units, activation="relu"),
            ]
        )
        self.dynamics = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(hidden_units, activation="relu"),
                tf.keras.layers.Dense(hidden_units + 1),
            ]
        )
        self.prediction = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(hidden_units, activation="relu"),
                tf.keras.layers.Dense(num_actions + 1),
            ]
        )

    def initial_inference(self, observation: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        hidden_state = self.representation(observation)
        value, policy_logits = self._predict(hidden_state)
        return hidden_state, value, policy_logits

    def recurrent_inference(
        self, hidden_state: tf.Tensor, action: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        action = tf.one_hot(tf.cast(action, tf.int32), self.num_actions)
        dynamics_output = self.dynamics(tf.concat([hidden_state, action], axis=-1))
        next_hidden = tf.nn.relu(dynamics_output[:, :-1])
        reward = dynamics_output[:, -1]
        value, policy_logits = self._predict(next_hidden)
        return next_hidden, reward, value, policy_logits

    def _predict(self, hidden_state: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        output = self.prediction(hidden_state)
        return output[:, 0], output[:, 1:]


@dataclass
class Node:
    prior: float
    hidden_state: np.ndarray | None = None
    reward: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "Node"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


@dataclass
class MinMaxStats:
    """Normalize search values so reward scale does not suppress exploration."""

    minimum: float = float("inf")
    maximum: float = float("-inf")

    def update(self, value: float) -> None:
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def normalize(self, value: float) -> float:
        value_range = self.maximum - self.minimum
        return (value - self.minimum) / value_range if value_range > 1e-8 else value


@dataclass(frozen=True)
class Experience:
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    policy: np.ndarray
    value: float


@dataclass
class ArchiveEntry:
    """A reproducible Go-Explore cell and the best trajectory reaching it."""

    seed: int
    actions: tuple[int, ...]
    score: float
    terminal: bool = False
    visits: int = 0


class CellEncoder:
    """Map arbitrary flat observations to stable cells without game features."""

    def __init__(self, observation_size: int, bits: int, seed: int):
        rng = np.random.default_rng(seed)
        # An appended constant makes these random hyperplanes affine, allowing
        # observations that differ mostly in magnitude to occupy different cells.
        self.projection = rng.normal(size=(observation_size + 1, bits)).astype(np.float32)

    def encode(self, observation: np.ndarray) -> bytes:
        vector = np.append(np.asarray(observation, dtype=np.float32), 1.0)
        bits = np.matmul(vector, self.projection) >= 0
        return np.packbits(bits).tobytes()


class GoExploreArchive:
    """Bounded archive with count-based cell selection and trajectory replay."""

    def __init__(self, encoder: CellEncoder, capacity: int, rng: np.random.Generator):
        self.encoder = encoder
        self.capacity = capacity
        self.rng = rng
        self.cells: OrderedDict[bytes, ArchiveEntry] = OrderedDict()

    def add(self, observation: np.ndarray, entry: ArchiveEntry) -> bool:
        key = self.encoder.encode(observation)
        previous = self.cells.get(key)
        if previous is None:
            improved = True
        elif previous.terminal != entry.terminal:
            # A hash collision must not make an explorable cell unreachable.
            improved = previous.terminal and not entry.terminal
        else:
            improved = entry.score > previous.score or (
                entry.score == previous.score and len(entry.actions) < len(previous.actions)
            )
        if not improved:
            return False
        if previous is not None:
            entry.visits = previous.visits
            del self.cells[key]
        self.cells[key] = entry
        while len(self.cells) > self.capacity:
            # Discard terminal cells before useful return points. In particular,
            # an archive of capacity one must retain its non-terminal root.
            terminal_key = next(
                (candidate for candidate, value in self.cells.items() if value.terminal),
                None,
            )
            if terminal_key is not None:
                del self.cells[terminal_key]
            else:
                self.cells.popitem(last=False)
        return True

    def select(self) -> ArchiveEntry:
        entries = [entry for entry in self.cells.values() if not entry.terminal]
        if not entries:
            raise RuntimeError("archive contains no explorable cells")
        weights = np.asarray([1.0 / np.sqrt(entry.visits + 1) for entry in entries])
        selected = entries[int(self.rng.choice(len(entries), p=weights / weights.sum()))]
        selected.visits += 1
        return selected


class MuZero:
    def __init__(self, env: gym.Env, network: Network, config: Config, seed: int = 0):
        if not isinstance(env, gym.Env):
            raise TypeError(
                "env must be a Gymnasium environment; legacy Gym environments "
                "are incompatible with NumPy 2"
            )
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("This example requires a discrete action space")
        self.env = env
        self.network = network
        self.config = config
        self.num_actions = env.action_space.n
        self.optimizer = tf.keras.optimizers.Adam(config.learning_rate)
        self.huber = tf.keras.losses.Huber()
        self.replay: deque[Experience] = deque(maxlen=config.replay_capacity)
        observation_size = gym.spaces.flatdim(env.observation_space)
        self.rng = np.random.default_rng(seed)
        self.archive = GoExploreArchive(
            CellEncoder(observation_size, config.cell_bits, seed),
            config.archive_capacity,
            self.rng,
        )

    @staticmethod
    def _expand(node: Node, logits: np.ndarray) -> None:
        priors = tf.nn.softmax(logits).numpy()
        node.children = {action: Node(float(prior)) for action, prior in enumerate(priors)}

    def _select_child(self, node: Node, min_max_stats: MinMaxStats) -> tuple[int, Node]:
        scale = np.sqrt(node.visit_count + 1)

        def score(child: Node) -> float:
            exploration = (
                self.config.exploration_constant * child.prior * scale / (child.visit_count + 1)
            )
            exploitation = min_max_stats.normalize(child.reward + self.config.gamma * child.value)
            return exploitation + exploration

        return max(node.children.items(), key=lambda item: score(item[1]))

    def mcts(self, observation: np.ndarray) -> Node:
        hidden, value, logits = self.network.initial_inference(
            tf.convert_to_tensor(observation[None, :], dtype=tf.float32)
        )
        root = Node(1.0, hidden.numpy()[0])
        self._expand(root, logits.numpy()[0])
        min_max_stats = MinMaxStats()

        for _ in range(self.config.num_simulations):
            node = root
            path = [root]
            while node.children:
                action, node = self._select_child(node, min_max_stats)
                path.append(node)
                if node.hidden_state is None:
                    parent_hidden = path[-2].hidden_state
                    next_hidden, reward, leaf_value, leaf_logits = self.network.recurrent_inference(
                        tf.convert_to_tensor(parent_hidden[None, :], dtype=tf.float32),
                        tf.convert_to_tensor([action]),
                    )
                    node.hidden_state = next_hidden.numpy()[0]
                    node.reward = float(reward.numpy()[0])
                    self._expand(node, leaf_logits.numpy()[0])
                    value_to_back_up = float(leaf_value.numpy()[0])
                    break
            else:
                value_to_back_up = float(value.numpy()[0])

            for visited in reversed(path):
                visited.visit_count += 1
                visited.value_sum += value_to_back_up
                min_max_stats.update(visited.reward + self.config.gamma * visited.value)
                value_to_back_up = visited.reward + self.config.gamma * value_to_back_up
        return root

    def train(self, num_episodes: int, seed: int | None = None) -> None:
        initial_seed = 0 if seed is None else seed
        initial_observation = reset_env(self.env, initial_seed)
        self.archive.add(initial_observation, ArchiveEntry(initial_seed, (), 0.0))
        for episode in range(num_episodes):
            start = self.archive.select()
            observation, total_reward, restored_actions, done = self._return_to_cell(start)
            trajectory: list[tuple[np.ndarray, int, float, np.ndarray, np.ndarray]] = []
            actions = list(restored_actions)
            steps_remaining = self.config.max_steps_per_episode - len(actions)
            for _ in range(max(0, steps_remaining)):
                if done:
                    break
                root = self.mcts(observation)
                visits = np.array([root.children[a].visit_count for a in range(self.num_actions)])
                policy_target = visits / max(visits.sum(), 1)
                action = int(self.rng.choice(self.num_actions, p=policy_target))
                next_observation, reward, done, _ = step_env(self.env, action)
                actions.append(action)
                trajectory.append(
                    (
                        observation.copy(),
                        action,
                        reward,
                        next_observation.copy(),
                        policy_target,
                    )
                )
                observation = next_observation
                total_reward += reward
                self.archive.add(
                    next_observation,
                    ArchiveEntry(start.seed, tuple(actions), total_reward, terminal=done),
                )
                if done:
                    break

            discounted_return = 0.0
            for observation, action, reward, next_observation, policy in reversed(trajectory):
                discounted_return = reward + self.config.gamma * discounted_return
                self.replay.append(
                    Experience(
                        observation,
                        action,
                        reward,
                        next_observation,
                        policy,
                        discounted_return,
                    )
                )

            losses = [self._train_batch() for _ in range(self.config.training_steps_per_episode)]
            print(
                f"Episode {episode + 1}: reward={total_reward:.2f}, "
                f"loss={np.mean(losses):.4f}, replay={len(self.replay)}, "
                f"archive={len(self.archive.cells)}"
            )

    def _return_to_cell(
        self, entry: ArchiveEntry
    ) -> tuple[np.ndarray, float, tuple[int, ...], bool]:
        """Restore a cell by deterministically replaying its action trajectory."""
        observation = reset_env(self.env, entry.seed)
        score = 0.0
        completed_actions: list[int] = []
        done = False
        for action in entry.actions:
            observation, reward, done, _ = step_env(self.env, action)
            score += reward
            completed_actions.append(action)
            if done:
                break
        return observation, score, tuple(completed_actions), done

    def _train_batch(self) -> float:
        batch_size = min(self.config.batch_size, len(self.replay))
        if batch_size == 0:
            raise RuntimeError("cannot train before collecting an experience")
        indices = self.rng.choice(len(self.replay), batch_size, replace=False)
        batch = [self.replay[index] for index in indices]
        observations = np.stack([sample.observation for sample in batch])
        actions = np.asarray([sample.action for sample in batch], dtype=np.int32)
        rewards = np.asarray([sample.reward for sample in batch], dtype=np.float32)
        next_observations = np.stack([sample.next_observation for sample in batch])
        policies = np.stack([sample.policy for sample in batch])
        values = np.asarray([sample.value for sample in batch], dtype=np.float32)

        with tf.GradientTape() as tape:
            hidden, predicted_values, logits = self.network.initial_inference(observations)
            next_hidden, predicted_rewards, _, _ = self.network.recurrent_inference(hidden, actions)
            value_loss = self.huber(values, predicted_values)
            reward_loss = self.huber(rewards, predicted_rewards)
            policy_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=policies, logits=logits)
            )
            # Encourage the learned dynamics state to match the next represented state.
            target_hidden = self.network.representation(next_observations)
            consistency_loss = tf.reduce_mean(
                tf.square(next_hidden - tf.stop_gradient(target_hidden))
            )
            loss = value_loss + reward_loss + policy_loss + 0.1 * consistency_loss
        gradients = tape.gradient(loss, self.network.trainable_variables)
        gradient_pairs = [
            (gradient, variable)
            for gradient, variable in zip(gradients, self.network.trainable_variables)
            if gradient is not None
        ]
        clipped, _ = tf.clip_by_global_norm(
            [gradient for gradient, _ in gradient_pairs], self.config.gradient_clip_norm
        )
        self.optimizer.apply_gradients(zip(clipped, [variable for _, variable in gradient_pairs]))
        return float(loss.numpy())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--training-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--archive-capacity", type=int, default=10_000)
    parser.add_argument("--cell-bits", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    env = gym.make(args.env)
    env.action_space.seed(args.seed)
    config = Config(
        num_simulations=args.simulations,
        batch_size=args.batch_size,
        training_steps_per_episode=args.training_steps,
        learning_rate=args.learning_rate,
        archive_capacity=args.archive_capacity,
        cell_bits=args.cell_bits,
    )
    try:
        network = Network(env.action_space.n, config.hidden_units)
        MuZero(env, network, config, args.seed).train(args.episodes, args.seed)
    finally:
        env.close()


if __name__ == "__main__":
    main()
