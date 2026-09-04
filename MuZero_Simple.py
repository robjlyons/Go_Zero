"""A small, self-contained MuZero-style training example.

This is intentionally an educational implementation: it uses one-step online
targets rather than MuZero's replay buffer and unrolled training procedure.
Importing this module is safe; training only starts through :func:`main`.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np

np.bool8 = np.bool

# TensorFlow is allowed to see CUDA devices.  Device selection is handled
# explicitly in main() so Colab/Kaggle GPUs are used automatically when present.
import tensorflow as tf  # noqa: E402


def configure_tensorflow_device(device: str = "auto") -> str:
    """Configure TensorFlow for CPU/GPU execution before model construction.

    ``auto`` uses the first visible GPU when one is available and otherwise
    falls back to CPU.  ``gpu`` fails loudly if TensorFlow cannot see a GPU.
    ``cpu`` hides all GPUs from TensorFlow for reproducible CPU comparisons.
    """
    device = device.lower()
    if device not in {"auto", "gpu", "cpu"}:
        raise ValueError("device must be one of: auto, gpu, cpu")

    physical_gpus = tf.config.list_physical_devices("GPU")

    if device == "cpu":
        tf.config.set_visible_devices([], "GPU")
        print("TensorFlow device: CPU (GPU disabled by --device cpu)")
        return "cpu"

    if not physical_gpus:
        if device == "gpu":
            raise RuntimeError(
                "--device gpu was requested, but TensorFlow cannot see a GPU. "
                "In Colab, enable a GPU runtime and make sure a GPU-enabled "
                "TensorFlow build is installed."
            )
        print("TensorFlow device: CPU (no GPU detected)")
        return "cpu"

    # Prevent TensorFlow from reserving all accelerator memory up front.
    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except (RuntimeError, ValueError):
            # Memory growth may already have been configured by the notebook.
            pass

    logical_gpus = tf.config.list_logical_devices("GPU")
    gpu_name = physical_gpus[0].name
    print(
        f"TensorFlow device: GPU ({gpu_name}); "
        f"logical GPUs={len(logical_gpus)}"
    )
    return "gpu"


@dataclass(frozen=True)
class Config:
    gamma: float = 0.99
    learning_rate: float = 1e-3
    num_simulations: int = 50
    hidden_units: int = 128
    max_steps_per_rollout: int = 500
    exploration_constant: float = 1.25

    # MuZero root exploration noise.
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25

    # Explicit post-restore exploration.  Most actions still come from MCTS,
    # while a minority deliberately probe under-tried or random actions.
    novelty_exploration_fraction: float = 0.25
    random_exploration_fraction: float = 0.05

    # Replay/training settings.
    replay_capacity: int = 50_000
    batch_size: int = 32
    training_steps_per_rollout: int = 20
    gradient_clip_norm: float = 5.0

    # Go-Explore archive settings.
    archive_capacity: int = 10_000
    cell_bits: int = 16

    # Standard Atari visual pipeline: four 84x84 grayscale frames.  Atari
    # observations are kept flattened in replay/MCTS storage, then reshaped by
    # the CNN representation network immediately before convolution.
    atari_screen_size: int = 84
    atari_frame_skip: int = 4
    atari_frame_stack: int = 4


def make_env(env_id: str, config: Config | None = None) -> gym.Env:
    """Create either a normal Gymnasium env or an ALE Atari env.

    Go-Explore restores archived cells by replaying their action trajectories.
    ALE/v5 normally uses sticky actions, so Atari is created with
    repeat_action_probability=0.0 to make replay deterministic.
    """
    if env_id.startswith("ALE/"):
        # Registration is deliberately kept beside Atari construction so that
        # importing the educational module does not initialize ALE needlessly.
        gym.register_envs(ale_py)
        config = config or Config()
        env = gym.make(
            env_id,
            repeat_action_probability=0.0,
            # AtariPreprocessing performs frame skipping and max-pooling. ALE
            # itself must therefore advance exactly one frame per wrapper step.
            frameskip=1,
        )
        env = gym.wrappers.AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=config.atari_frame_skip,
            screen_size=config.atari_screen_size,
            terminal_on_life_loss=False,
            grayscale_obs=True,
            scale_obs=False,
        )
        return gym.wrappers.FrameStackObservation(
            env,
            stack_size=config.atari_frame_stack,
        )
    return gym.make(env_id)


def _prepare_observation(observation: object) -> np.ndarray:
    """Flatten observations, scaling byte images without an intermediate copy."""
    array = np.asarray(observation)
    if array.dtype == np.uint8:
        return array.reshape(-1).astype(np.float32) / 255.0
    return array.astype(np.float32, copy=False).reshape(-1)


def reset_env(env: gym.Env, seed: int | None = None) -> np.ndarray:
    """Reset old or new Gym environments and return only the observation."""
    if seed is None:
        result = env.reset()
    else:
        try:
            result = env.reset(seed=seed)
        except TypeError:  # Gym before 0.21 used a separate seed method.
            env.seed(seed)
            result = env.reset()
    observation = result[0] if isinstance(result, tuple) else result
    return _prepare_observation(observation)


def step_env(env: gym.Env, action: int) -> tuple[np.ndarray, float, bool, dict]:
    """Step old or new Gym environments using one consistent API."""
    result = env.step(action)
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        done = terminated or truncated
    else:
        observation, reward, done, info = result
    return _prepare_observation(observation), float(reward), bool(done), info


class Network(tf.keras.Model):
    """MuZero representation, dynamics, and prediction functions.

    Atari/image observations use a DQN-style convolutional representation.
    Vector observations such as CartPole retain the small dense representation.
    Observations remain flattened outside the network so the existing replay,
    Go-Explore archive, and MCTS code do not need to change.
    """

    def __init__(
        self,
        num_actions: int,
        hidden_units: int,
        observation_shape: tuple[int, ...],
    ):
        super().__init__()

        # Gymnasium/ALE may expose Discrete.n and shape dimensions as NumPy ints.
        self.num_actions = int(num_actions)
        self.hidden_units = int(hidden_units)
        self.observation_shape = tuple(int(dim) for dim in observation_shape)
        self.flat_observation_size = int(np.prod(self.observation_shape))

        self.uses_cnn = self._looks_like_image_observation(self.observation_shape)
        self.representation = self._build_representation()

        # MuZero latent dynamics remain compact. The expensive spatial processing
        # happens once in the representation model; recurrent MCTS inference then
        # operates on the learned latent vector.
        self.dynamics = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.hidden_units, activation="relu"),
                tf.keras.layers.Dense(self.hidden_units + 1),
            ],
            name="dynamics",
        )
        self.prediction = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.hidden_units, activation="relu"),
                tf.keras.layers.Dense(self.num_actions + 1),
            ],
            name="prediction",
        )

    @staticmethod
    def _looks_like_image_observation(shape: tuple[int, ...]) -> bool:
        if len(shape) == 2:
            return min(shape) >= 32
        if len(shape) != 3:
            return False

        # Covers FrameStackObservation's (stack, H, W) as well as (H, W, C).
        return (
            (shape[0] <= 8 and shape[1] >= 32 and shape[2] >= 32)
            or (shape[-1] <= 8 and shape[0] >= 32 and shape[1] >= 32)
        )

    def _build_representation(self) -> tf.keras.Model:
        inputs = tf.keras.Input(
            shape=(self.flat_observation_size,),
            dtype=tf.float32,
            name="flat_observation",
        )

        if not self.uses_cnn:
            x = tf.keras.layers.Dense(
                self.hidden_units,
                activation="relu",
                name="representation_dense_1",
            )(inputs)
            hidden = tf.keras.layers.Dense(
                self.hidden_units,
                activation="relu",
                name="representation_dense_2",
            )(x)
            return tf.keras.Model(inputs, hidden, name="representation_mlp")

        x = tf.keras.layers.Reshape(
            self.observation_shape,
            name="restore_image_shape",
        )(inputs)

        if len(self.observation_shape) == 2:
            # Single grayscale image: (H, W) -> (H, W, 1).
            x = tf.keras.layers.Reshape(
                (*self.observation_shape, 1),
                name="add_channel_dimension",
            )(x)
        elif self.observation_shape[0] <= 8:
            # Gymnasium FrameStackObservation gives Atari as (stack, H, W).
            # Conv2D expects channels-last: (H, W, stack).
            x = tf.keras.layers.Permute(
                (2, 3, 1),
                name="channels_first_to_last",
            )(x)

        # DQN-style visual encoder. This preserves spatial structure instead of
        # projecting all 28,224 Atari pixels directly through a dense layer.
        x = tf.keras.layers.Conv2D(
            32,
            kernel_size=8,
            strides=4,
            activation="relu",
            padding="valid",
            name="conv1",
        )(x)
        x = tf.keras.layers.Conv2D(
            64,
            kernel_size=4,
            strides=2,
            activation="relu",
            padding="valid",
            name="conv2",
        )(x)
        x = tf.keras.layers.Conv2D(
            64,
            kernel_size=3,
            strides=1,
            activation="relu",
            padding="valid",
            name="conv3",
        )(x)
        x = tf.keras.layers.Flatten(name="conv_flatten")(x)
        hidden = tf.keras.layers.Dense(
            self.hidden_units,
            activation="relu",
            name="representation_latent",
        )(x)

        return tf.keras.Model(inputs, hidden, name="representation_cnn")

    @tf.function(reduce_retracing=True)
    def initial_inference(
        self,
        observation: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        hidden_state = self.representation(observation, training=False)
        value, policy_logits = self._predict(hidden_state)
        return hidden_state, value, policy_logits

    @tf.function(reduce_retracing=True)
    def recurrent_inference(
        self,
        hidden_state: tf.Tensor,
        action: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        action = tf.one_hot(tf.cast(action, tf.int32), self.num_actions)
        dynamics_output = self.dynamics(
            tf.concat([hidden_state, action], axis=-1),
            training=False,
        )
        next_hidden = tf.nn.relu(dynamics_output[:, :-1])
        reward = dynamics_output[:, -1]
        value, policy_logits = self._predict(next_hidden)
        return next_hidden, reward, value, policy_logits

    def _predict(self, hidden_state: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        output = self.prediction(hidden_state, training=False)
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


@dataclass(frozen=True)
class Experience:
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    policy: np.ndarray
    value: float


class ReplayBuffer:
    """Fixed-size ring buffer with constant-time random access.

    ``collections.deque`` indexing walks from an end and becomes a major
    training bottleneck once an Atari replay buffer contains many transitions.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("replay capacity must be positive")
        self.capacity = capacity
        self._items: list[Experience] = []
        self._next = 0

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Experience:
        return self._items[index]

    def append(self, experience: Experience) -> None:
        if len(self._items) < self.capacity:
            self._items.append(experience)
        else:
            self._items[self._next] = experience
        self._next = (self._next + 1) % self.capacity


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
        affine = rng.normal(size=(observation_size + 1, bits)).astype(np.float32)
        self.projection = affine[:-1]
        self.bias = affine[-1]

    def encode(self, observation: np.ndarray) -> bytes:
        vector = np.asarray(observation, dtype=np.float32) - 0.5
        bits = np.matmul(vector, self.projection) + self.bias >= 0
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
        improved = (
            previous is None
            or entry.score > previous.score
            or (entry.score == previous.score and len(entry.actions) < len(previous.actions))
        )
        if not improved:
            return False
        if previous is not None:
            entry.visits = previous.visits
            del self.cells[key]
        self.cells[key] = entry
        while len(self.cells) > self.capacity:
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
    def __init__(
        self,
        env: gym.Env,
        network: Network,
        config: Config,
        seed: int = 0,
    ):
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("This example requires a discrete action space")

        self.env = env
        self.network = network
        self.config = config
        self.num_actions = int(env.action_space.n)
        self.optimizer = tf.keras.optimizers.Adam(config.learning_rate)

        self.replay = ReplayBuffer(config.replay_capacity)

        observation_size = gym.spaces.flatdim(env.observation_space)
        self.rng = np.random.default_rng(seed)
        self.archive = GoExploreArchive(
            CellEncoder(observation_size, config.cell_bits, seed),
            config.archive_capacity,
            self.rng,
        )

        # Per-cell action counts used for count-based novelty exploration.
        # The key comes from the same cell representation used by Go-Explore.
        self.cell_action_counts: dict[bytes, np.ndarray] = {}

        exploration_total = (
            config.novelty_exploration_fraction
            + config.random_exploration_fraction
        )
        if config.novelty_exploration_fraction < 0.0:
            raise ValueError("novelty_exploration_fraction must be >= 0")
        if config.random_exploration_fraction < 0.0:
            raise ValueError("random_exploration_fraction must be >= 0")
        if exploration_total > 1.0:
            raise ValueError(
                "novelty_exploration_fraction + random_exploration_fraction "
                "must be <= 1"
            )

    @staticmethod
    def _expand(node: Node, logits: np.ndarray) -> None:
        # Avoid dispatching a tiny TensorFlow op (and synchronizing the device)
        # for every expanded MCTS node.
        shifted = logits - np.max(logits)
        priors = np.exp(shifted)
        priors /= priors.sum()
        node.children = {action: Node(float(prior)) for action, prior in enumerate(priors)}

    def _add_root_exploration_noise(self, root: Node) -> None:
        """Mix Dirichlet noise into root priors for training-time exploration.

        Noise is applied only to the MCTS root; deeper nodes keep the
        network priors unchanged.
        """
        if not root.children:
            return

        fraction = self.config.root_exploration_fraction
        alpha = self.config.root_dirichlet_alpha

        if fraction <= 0.0:
            return
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("root_exploration_fraction must be in [0, 1]")
        if alpha <= 0.0:
            raise ValueError("root_dirichlet_alpha must be > 0")

        actions = list(root.children)
        noise = self.rng.dirichlet(
            np.full(len(actions), alpha, dtype=np.float64)
        )

        keep = 1.0 - fraction
        for action, noise_value in zip(actions, noise):
            child = root.children[action]
            child.prior = float(
                keep * child.prior + fraction * noise_value
            )

    def _select_training_action(
        self,
        observation: np.ndarray,
        policy_target: np.ndarray,
    ) -> tuple[int, str]:
        """Choose MCTS, novelty-seeking, or random action during collection.

        Novelty is count-based: for the current Go-Explore cell, choose
        uniformly among the least-tried actions.  Counts include actions taken
        by every mode, so novelty continuously favours actions that have been
        neglected in that state.

        The returned ``policy_target`` remains the MCTS visit distribution;
        exploratory actions therefore improve dynamics/reward coverage without
        teaching the policy head that random actions are intrinsically optimal.
        """
        cell_key = self.archive.encoder.encode(observation)
        counts = self.cell_action_counts.get(cell_key)
        if counts is None:
            counts = np.zeros(self.num_actions, dtype=np.int64)
            self.cell_action_counts[cell_key] = counts

        draw = float(self.rng.random())
        random_fraction = self.config.random_exploration_fraction
        novelty_fraction = self.config.novelty_exploration_fraction

        if draw < random_fraction:
            action = int(self.rng.integers(self.num_actions))
            mode = "random"
        elif draw < random_fraction + novelty_fraction:
            minimum = counts.min()
            candidates = np.flatnonzero(counts == minimum)
            action = int(self.rng.choice(candidates))
            mode = "novelty"
        else:
            action = int(
                self.rng.choice(
                    self.num_actions,
                    p=policy_target,
                )
            )
            mode = "mcts"

        counts[action] += 1
        return action, mode

    def _select_child(self, node: Node) -> tuple[int, Node]:
        scale = np.sqrt(node.visit_count + 1)

        def score(child: Node) -> float:
            exploration = (
                self.config.exploration_constant * child.prior * scale / (child.visit_count + 1)
            )
            return child.reward + self.config.gamma * child.value + exploration

        return max(node.children.items(), key=lambda item: score(item[1]))

    def mcts(self, observation: np.ndarray) -> Node:
        hidden, value, logits = self.network.initial_inference(
            tf.convert_to_tensor(observation[None, :], dtype=tf.float32)
        )

        root = Node(1.0, hidden.numpy()[0])
        self._expand(root, logits.numpy()[0])
        self._add_root_exploration_noise(root)

        for _ in range(self.config.num_simulations):
            node = root
            path = [root]
            value_to_back_up = float(value.numpy()[0])

            while node.children:
                action, node = self._select_child(node)
                path.append(node)

                if node.hidden_state is None:
                    parent_hidden = path[-2].hidden_state
                    next_hidden, reward, leaf_value, leaf_logits = self.network.recurrent_inference(
                        tf.convert_to_tensor(
                            parent_hidden[None, :],
                            dtype=tf.float32,
                        ),
                        tf.convert_to_tensor([action]),
                    )
                    node.hidden_state = next_hidden.numpy()[0]
                    node.reward = float(reward.numpy()[0])
                    self._expand(node, leaf_logits.numpy()[0])
                    value_to_back_up = float(leaf_value.numpy()[0])
                    break

            for visited in reversed(path):
                visited.visit_count += 1
                visited.value_sum += value_to_back_up
                value_to_back_up = visited.reward + self.config.gamma * value_to_back_up

        return root

    def train(
        self,
        max_steps: int,
        seed: int | None = None,
        log_path: str | os.PathLike[str] | None = "training_metrics.jsonl",
    ) -> None:
        """Train until ``max_steps`` environment transitions have been executed.

        Returning to a Go-Explore cell also advances the environment, so those
        replayed actions count toward the limit. This makes the budget a hard
        upper bound on emulator work rather than merely on collected samples.
        """
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")

        metrics_path = Path(log_path) if log_path is not None else None
        if metrics_path is not None:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text("", encoding="utf-8")

        training_started = time.perf_counter()
        total_steps = 0
        rollout = 0
        while total_steps < max_steps:
            rollout_started = time.perf_counter()
            rollout_seed = None if seed is None else seed + rollout

            # Always create a reproducible starting cell for this rollout.
            initial_observation = reset_env(self.env, rollout_seed)
            archive_seed = rollout if rollout_seed is None else rollout_seed
            self.archive.add(
                initial_observation,
                ArchiveEntry(
                    seed=archive_seed,
                    actions=(),
                    score=0.0,
                    terminal=False,
                ),
            )

            # Go back to a previously discovered, non-terminal cell.
            start = self.archive.select()
            (
                observation,
                total_reward,
                completed_actions,
                done,
            ) = self._return_to_cell(start, max_steps - total_steps)
            replayed_steps = len(completed_actions)
            total_steps += replayed_steps

            actions = list(completed_actions)
            trajectory: list[tuple[np.ndarray, int, float, np.ndarray, np.ndarray]] = []
            mcts_actions = 0
            novelty_actions = 0
            random_actions = 0

            if not done and total_steps < max_steps:
                for _ in range(self.config.max_steps_per_rollout):
                    root = self.mcts(observation)

                    visits = np.asarray(
                        [root.children[action].visit_count for action in range(self.num_actions)],
                        dtype=np.float64,
                    )

                    if visits.sum() <= 0:
                        policy_target = np.full(
                            self.num_actions,
                            1.0 / self.num_actions,
                            dtype=np.float32,
                        )
                    else:
                        policy_target = (visits / visits.sum()).astype(np.float32)

                    action, action_mode = self._select_training_action(
                        observation,
                        policy_target,
                    )
                    if action_mode == "mcts":
                        mcts_actions += 1
                    elif action_mode == "novelty":
                        novelty_actions += 1
                    else:
                        random_actions += 1

                    (
                        next_observation,
                        reward,
                        done,
                        _,
                    ) = step_env(self.env, action)
                    total_steps += 1

                    actions.append(action)
                    trajectory.append(
                        (
                            observation.copy(),
                            action,
                            reward,
                            next_observation.copy(),
                            policy_target.copy(),
                        )
                    )

                    observation = next_observation
                    total_reward += reward

                    self.archive.add(
                        next_observation,
                        ArchiveEntry(
                            seed=start.seed,
                            actions=tuple(actions),
                            score=total_reward,
                            terminal=done,
                        ),
                    )

                    if done or total_steps >= max_steps:
                        break

            # Monte-Carlo return targets for each transition.
            discounted_return = 0.0
            for (
                observation_t,
                action_t,
                reward_t,
                next_observation_t,
                policy_t,
            ) in reversed(trajectory):
                discounted_return = reward_t + self.config.gamma * discounted_return
                self.replay.append(
                    Experience(
                        observation=observation_t,
                        action=action_t,
                        reward=reward_t,
                        next_observation=next_observation_t,
                        policy=policy_t,
                        value=discounted_return,
                    )
                )

            losses = []
            if self.replay:
                losses = [
                    self._train_batch() for _ in range(self.config.training_steps_per_rollout)
                ]

            mean_loss = float(np.mean(losses)) if losses else float("nan")
            elapsed_seconds = time.perf_counter() - training_started
            rollout_seconds = time.perf_counter() - rollout_started
            collected_steps = len(trajectory)
            steps_per_second = total_steps / max(elapsed_seconds, np.finfo(float).eps)

            metrics = {
                "rollout": rollout + 1,
                "environment_steps": total_steps,
                "max_steps": max_steps,
                "collected_steps": collected_steps,
                "replayed_steps": replayed_steps,
                "mcts_actions": mcts_actions,
                "novelty_actions": novelty_actions,
                "random_actions": random_actions,
                "reward": total_reward,
                "loss": mean_loss if np.isfinite(mean_loss) else None,
                "replay_size": len(self.replay),
                "archive_size": len(self.archive.cells),
                "terminal": done,
                "rollout_seconds": rollout_seconds,
                "elapsed_seconds": elapsed_seconds,
                "steps_per_second": steps_per_second,
            }
            if metrics_path is not None:
                with metrics_path.open("a", encoding="utf-8") as metrics_file:
                    metrics_file.write(json.dumps(metrics, allow_nan=False) + "\n")

            print(
                f"Rollout {rollout + 1}: "
                f"steps={total_steps}/{max_steps}, "
                f"reward={total_reward:.2f}, "
                f"loss={mean_loss:.4f}, "
                f"steps/s={steps_per_second:.2f}, "
                f"replay={len(self.replay)}, "
                f"archive={len(self.archive.cells)}, "
                f"actions[mcts/novel/random]="
                f"{mcts_actions}/{novelty_actions}/{random_actions}"
            )
            rollout += 1

    def _return_to_cell(
        self,
        entry: ArchiveEntry,
        max_steps: int | None = None,
    ) -> tuple[np.ndarray, float, tuple[int, ...], bool]:
        """Restore a cell by deterministically replaying its action trajectory."""
        observation = reset_env(self.env, entry.seed)
        score = 0.0
        completed_actions: list[int] = []
        done = False

        for action in entry.actions:
            if max_steps is not None and len(completed_actions) >= max_steps:
                break
            observation, reward, done, _ = step_env(
                self.env,
                action,
            )
            score += reward
            completed_actions.append(action)

            if done:
                break

        return (
            observation,
            score,
            tuple(completed_actions),
            done,
        )

    def _train_batch(self) -> float:
        if not self.replay:
            return 0.0

        batch_size = min(
            self.config.batch_size,
            len(self.replay),
        )
        indices = self.rng.choice(
            len(self.replay),
            size=batch_size,
            replace=False,
        )
        batch = [self.replay[int(index)] for index in indices]

        observations = np.stack([sample.observation for sample in batch]).astype(np.float32)
        actions = np.asarray(
            [sample.action for sample in batch],
            dtype=np.int32,
        )
        rewards = np.asarray(
            [sample.reward for sample in batch],
            dtype=np.float32,
        )
        next_observations = np.stack([sample.next_observation for sample in batch]).astype(
            np.float32
        )
        policies = np.stack([sample.policy for sample in batch]).astype(np.float32)
        values = np.asarray(
            [sample.value for sample in batch],
            dtype=np.float32,
        )

        with tf.GradientTape() as tape:
            observation_tensor = tf.convert_to_tensor(
                observations,
                dtype=tf.float32,
            )
            action_tensor = tf.convert_to_tensor(actions)

            hidden = self.network.representation(
                observation_tensor,
                training=True,
            )
            prediction_output = self.network.prediction(
                hidden,
                training=True,
            )
            predicted_values = prediction_output[:, 0]
            logits = prediction_output[:, 1:]

            one_hot_actions = tf.one_hot(
                tf.cast(action_tensor, tf.int32),
                self.num_actions,
            )
            dynamics_output = self.network.dynamics(
                tf.concat([hidden, one_hot_actions], axis=-1),
                training=True,
            )
            next_hidden = tf.nn.relu(dynamics_output[:, :-1])
            predicted_rewards = dynamics_output[:, -1]

            value_loss = tf.reduce_mean(
                tf.keras.losses.huber(
                    values,
                    predicted_values,
                )
            )
            reward_loss = tf.reduce_mean(
                tf.keras.losses.huber(
                    rewards,
                    predicted_rewards,
                )
            )
            policy_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=policies,
                    logits=logits,
                )
            )

            target_hidden = self.network.representation(
                tf.convert_to_tensor(
                    next_observations,
                    dtype=tf.float32,
                ),
                training=False,
            )
            consistency_loss = tf.reduce_mean(
                tf.square(next_hidden - tf.stop_gradient(target_hidden))
            )

            loss = value_loss + reward_loss + policy_loss + 0.1 * consistency_loss

        gradients = tape.gradient(
            loss,
            self.network.trainable_variables,
        )
        gradient_variable_pairs = [
            (gradient, variable)
            for gradient, variable in zip(
                gradients,
                self.network.trainable_variables,
            )
            if gradient is not None
        ]

        if gradient_variable_pairs:
            grads, variables = zip(*gradient_variable_pairs)
            clipped_grads, _ = tf.clip_by_global_norm(
                grads,
                self.config.gradient_clip_norm,
            )
            self.optimizer.apply_gradients(zip(clipped_grads, variables))

        return float(loss.numpy())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument(
        "--device",
        choices=("auto", "gpu", "cpu"),
        default="auto",
        help="TensorFlow execution device; auto uses a GPU when available",
    )
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument(
        "--log-file",
        default="training_metrics.jsonl",
        help="JSON Lines destination for rollout and throughput metrics",
    )
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=0.3,
        help="Dirichlet alpha used for MCTS root exploration noise",
    )
    parser.add_argument(
        "--root-noise-fraction",
        type=float,
        default=0.25,
        help="Fraction of each MCTS root prior replaced by Dirichlet noise",
    )
    parser.add_argument(
        "--novelty-exploration-fraction",
        type=float,
        default=0.25,
        help=(
            "Fraction of collected actions chosen from the least-tried actions "
            "for the current Go-Explore cell"
        ),
    )
    parser.add_argument(
        "--random-exploration-fraction",
        type=float,
        default=0.05,
        help="Fraction of collected actions chosen uniformly at random",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--training-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--archive-capacity", type=int, default=10_000)
    parser.add_argument("--cell-bits", type=int, default=16)
    parser.add_argument("--atari-screen-size", type=int, default=84)
    parser.add_argument("--atari-frame-skip", type=int, default=4)
    parser.add_argument("--atari-frame-stack", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    selected_device = configure_tensorflow_device(args.device)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    config = Config(
        num_simulations=args.simulations,
        root_dirichlet_alpha=args.dirichlet_alpha,
        root_exploration_fraction=args.root_noise_fraction,
        novelty_exploration_fraction=args.novelty_exploration_fraction,
        random_exploration_fraction=args.random_exploration_fraction,
        batch_size=args.batch_size,
        training_steps_per_rollout=args.training_steps,
        learning_rate=args.learning_rate,
        archive_capacity=args.archive_capacity,
        cell_bits=args.cell_bits,
        atari_screen_size=args.atari_screen_size,
        atari_frame_skip=args.atari_frame_skip,
        atari_frame_stack=args.atari_frame_stack,
    )
    env = make_env(args.env, config)
    env.action_space.seed(args.seed)
    try:
        observation_shape = tuple(int(dim) for dim in env.observation_space.shape)
        network = Network(
            int(env.action_space.n),
            config.hidden_units,
            observation_shape,
        )

        # Build once so startup clearly reports which representation is active.
        dummy_observation = tf.zeros(
            (1, int(np.prod(observation_shape))),
            dtype=tf.float32,
        )
        dummy_hidden, _, _ = network.initial_inference(dummy_observation)
        network.recurrent_inference(
            dummy_hidden,
            tf.zeros((1,), dtype=tf.int32),
        )

        representation_name = "CNN" if network.uses_cnn else "MLP"
        print(
            f"Representation: {representation_name}; "
            f"observation_shape={observation_shape}; "
            f"latent={config.hidden_units}; "
            f"parameters={network.count_params():,}; "
            f"device={selected_device}"
        )

        MuZero(env, network, config, args.seed).train(
            args.max_steps,
            args.seed,
            args.log_file,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
