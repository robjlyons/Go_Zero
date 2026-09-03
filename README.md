# Go Zero

This repository contains a compact, educational MuZero-style agent using the
maintained [Gymnasium](https://gymnasium.farama.org/) environment API.

## Installation

Create a fresh virtual environment so that an older `gym` installation cannot
be imported accidentally, then install the declared dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the repository's entry point (rather than an older copied notebook or
`Go_Zero.py` file):

```bash
python MuZero_Simple.py --env CartPole-v1 --episodes 10
```

## Atari performance

Atari environments automatically use the input pipeline popularized by
Stable-Baselines3: deterministic ALE actions, 4-frame action repeat with
max-pooling, 84x84 grayscale frames, and a stack of four observations. Byte
pixels are normalized to `[0, 1]` before inference. This reduces each network
input from 100,800 raw RGB values to 28,224 values and avoids processing every
emulator frame independently.

```bash
python MuZero_Simple.py --env ALE/Pong-v5 --episodes 10
```

The defaults can be tuned with `--atari-screen-size`, `--atari-frame-skip`, and
`--atari-frame-stack`. The standard `84`, `4`, and `4` values are recommended
when comparing throughput with Stable-Baselines3. MuZero's MCTS still performs
many more model calls per action than model-free algorithms such as PPO or DQN;
lower `--simulations` when environment steps per second matter more than search
quality.

The replay store is a fixed-size, constant-time ring buffer, so random batch
sampling does not become progressively slower as a long Atari run fills it.

## Gymnasium migration

The implementation imports `gymnasium`, not the unmaintained `gym` package.
It follows Gymnasium's current API:

* `reset()` returns `(observation, info)`.
* `step()` returns `(observation, reward, terminated, truncated, info)`.
* An episode ends when either `terminated` or `truncated` is true.

If a traceback contains `site-packages/gym/` or code such as
`result = env.step(action)`, it is running an older copy of the program. The
current entry point calls the Gymnasium API directly. Warnings originating in
Matplotlib or a CUDA initialization message are dependency/runtime diagnostics
and are unrelated to the environment API migration.
