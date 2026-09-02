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
CUDA_VISIBLE_DEVICES=0 python MuZero_Simple.py --env CartPole-v1 --episodes 10
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
