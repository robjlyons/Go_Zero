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

The example defaults to CPU execution, which avoids TensorFlow CUDA
initialization errors on machines without a working CUDA device. To opt into a
GPU explicitly, set its device before launching Python:

```bash
GO_ZERO_USE_GPU=1 CUDA_VISIBLE_DEVICES=0 python MuZero_Simple.py --env CartPole-v1 --episodes 10
```

## Gymnasium migration

The implementation imports `gymnasium`, not the unmaintained `gym` package.
It follows Gymnasium's current API:

* `reset()` returns `(observation, info)`.
* `step()` returns `(observation, reward, terminated, truncated, info)`.
* An episode ends when either `terminated` or `truncated` is true.

If a traceback contains `site-packages/gym/` or code such as
`result = env.step(action)`, it is running an older copy of the program. The
current entry point calls the Gymnasium API directly. Warnings originating in
Matplotlib are dependency diagnostics and are unrelated to the environment API
migration. The `MuZero` constructor also rejects legacy Gym environments early,
before they can fail inside Gym's NumPy-incompatible environment checker.
In an existing environment, remove the obsolete package with
`python -m pip uninstall gym`; installing Gymnasium does not automatically
uninstall it. Running `python -c "import gymnasium; print(gymnasium.__file__)"`
shows which Gymnasium installation Python will use.

## Training behavior

Training data is retained in a bounded replay buffer and sampled in batches.
Episode-level discounted returns provide fixed value targets, rather than
bootstrapping immediately from the same network. Huber losses and global
gradient clipping limit the loss growth visible in earlier runs. The output
includes replay-buffer size so it is easier to distinguish data collection from
optimization:

```text
Episode 10: reward=42.00, loss=1.2345, replay=287
```

This remains a compact teaching implementation, not a reproduction of the full
MuZero paper. Increase `--episodes` when assessing learning; ten CartPole
episodes are generally too few to judge performance. The optimization workload
can be adjusted with `--batch-size`, `--training-steps`, and `--learning-rate`.
