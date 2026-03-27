# differential TD Investigation Notes

Date: 2026-03-25

Update: 2026-03-27

## What this repository implements

This repository implements the paper's differential temporal-difference method (dTD) in a PPO pipeline for Brax continuous-control tasks.

The key idea from the paper is implemented as three critic variants:

1. Baseline TD (standard PPO-style critic target)
2. naive-dTD (TD-like decomposition of dTD terms)
3. dTD (the paper's preferred parametrization)

The paper's stabilization strategy (beta-dTD) is implemented as a linear mixture between baseline TD loss and dTD loss via `mix_ratio` (beta).

## Paper-to-code mapping

### Core training entry and variant switch

- [dtd/ppo/main.py](dtd/ppo/main.py)
	- Hydra entrypoint and run configuration
	- Chooses variant by `cfg.algorithm.TD`:
		- `baseline` -> `train_baseline`
		- `naive` -> `train_naive_dtd`
		- `dtd` -> `train_dtd`

### Core algorithm implementations

- [dtd/ppo/train.py](dtd/ppo/train.py)
	- `train_baseline`: standard PPO critic target flow
	- `train_naive_dtd`: naive-dTD variant
	- `train_dtd`: dTD variant

### Differential terms in the dTD loss

- [dtd/common/train.py](dtd/common/train.py)
	- `dsV_s_fn`: first-order directional derivative via JAX JVP
	- `dsV_ssds_fn`: second-order term via gradient of the first-order term

### Stochastic dynamics/noise setup used in experiments

- [dtd/common/env_wrappers.py](dtd/common/env_wrappers.py)
	- `NoiseWrapper` perturbs observations with noise scaled by `abs(obs) * noise_lvl`
	- `create_env` composes wrappers: noise, episode handling, auto-reset, logging, vectorization

### Config and tuning plumbing

- [dtd/configs/config.yaml](dtd/configs/config.yaml)
	- Default environment and sweeper setup
	- Uses the DEHB sweeper by default
- [dtd/configs/algorithm/ppo_baseline.yaml](dtd/configs/algorithm/ppo_baseline.yaml)
- [dtd/configs/algorithm/ppo_naive.yaml](dtd/configs/algorithm/ppo_naive.yaml)
- [dtd/configs/algorithm/ppo_dtd.yaml](dtd/configs/algorithm/ppo_dtd.yaml)
	- Defines default algorithm settings including `mix_ratio`
- [dtd/configs/search_space/ppo_baseline.yaml](dtd/configs/search_space/ppo_baseline.yaml)
- [dtd/configs/search_space/ppo_naive.yaml](dtd/configs/search_space/ppo_naive.yaml)
- [dtd/configs/search_space/ppo_dtd.yaml](dtd/configs/search_space/ppo_dtd.yaml)
	- Hyperparameter search spaces (including `mix_ratio` for naive and dtd)

### Experiment scripts and incumbent parameters

- [dtd/runs/main.sh](dtd/runs/main.sh): single training run with incumbent hyperparameters
- [dtd/runs/tune.sh](dtd/runs/tune.sh): DEHB tuning run
- [dtd/runs/incumbent](dtd/runs/incumbent): tuned hyperparameter presets by env/TD/noise

## What to read first to understand the method quickly

1. Paper method section on dTD and beta-dTD in [Paper.pdf](Paper.pdf)
2. [dtd/common/train.py](dtd/common/train.py) (how derivative terms are computed)
3. [dtd/ppo/train.py](dtd/ppo/train.py) (baseline vs naive-dTD vs dTD differences)
4. [dtd/ppo/main.py](dtd/ppo/main.py) (how variants are selected and wired)
5. [dtd/common/env_wrappers.py](dtd/common/env_wrappers.py) (how process noise is modeled)

## Practical observations from the code

1. The implementation clearly follows the paper's practical framing: PPO + mixed critic loss.
2. `mix_ratio` is the central stability/performance knob corresponding to beta in beta-dTD.
3. The experiment pipeline is built around Brax vectorized envs and Hydra configs.

## Possible point to double-check against the paper

In `train_dtd`, the dTD target uses `transition.value` in:

```python
target = reward + jnp.log(gamma) * value
```

while the paper's discrete compatibility section discusses a target involving `V(s_{t+dt})` in its displayed formula. This may still be a valid rearrangement in the code's chosen prediction-target decomposition, but it is the first line to inspect if reproduction deviates.

## Running on macOS (native, no NVIDIA Docker)

The provided Docker path is CUDA/NVIDIA-specific and not suitable for a standard Mac workflow:

- [docker/Dockerfile](docker/Dockerfile) starts from an NVIDIA CUDA base image
- [docker/requirements.txt](docker/requirements.txt) includes `jax[cuda12]`
- [launch.sh](launch.sh) uses `--gpus=all`

Use a native Python environment on macOS instead.

### 1. Create environment and install dependencies

From repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools

# validated package set from this investigation
pip install -r requirements/macos-cpu.txt

# optional: same install through uv (pip-compatible mode)
# uv pip install -r requirements/macos-cpu.txt
```

Dependency file added during this investigation:

- [requirements/macos-cpu.txt](requirements/macos-cpu.txt)
  - Includes packages needed to run training on macOS CPU.
  - Includes warning-related optional packages installed during debugging (`wandb`, `warp-lang`).

Notes:

1. On macOS, start with CPU JAX for reliability.
2. The repository [setup.sh](setup.sh) is Linux-path oriented and includes site-package patching that may not match your Mac environment.
3. Running from [dtd](dtd) with script-style invocation can fail imports (`ModuleNotFoundError: No module named dtd`).

### 2. Run from repo root (validated launch mode)

Use module invocation from repository root:

```bash
cd /path/to/differential_TD
```

### 3. First small validation run (recommended)

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m dtd.ppo.main \
	env.name=hopper \
	env.noise_lvl=0.01 \
	algorithm=ppo_dtd \
	algorithm.total_timesteps=5000 \
	env.num_envs=2 \
	algorithm.num_env_steps_per_update=4 \
	algorithm.num_epochs_per_update=1 \
	algorithm.minibatch_size=8 \
	run_time=$(date +%s)
```

Why the tiny config above uses `minibatch_size=8`:

- The code enforces divisibility of batch size by minibatch count.
- With `num_envs=2` and `num_env_steps_per_update=4`, batch size is 8.
- A larger minibatch (for example 32) caused an assertion failure during this investigation.

### 4. Baseline comparison run

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m dtd.ppo.main \
	env.name=hopper \
	env.noise_lvl=0.01 \
	algorithm=ppo_baseline \
	algorithm.total_timesteps=200000 \
	env.num_envs=8 \
	algorithm.num_env_steps_per_update=8 \
	algorithm.num_epochs_per_update=5 \
	algorithm.minibatch_size=128 \
	run_time=$(date +%s)
```

### 5. Script updates made during this investigation

The run scripts were updated to be macOS-friendly and import-safe:

- [dtd/runs/main.sh](dtd/runs/main.sh)
- [dtd/runs/tune.sh](dtd/runs/tune.sh)

These now:

1. Auto-use `.venv/bin/python` if present.
2. Run via module mode from repo root.
3. Set `JAX_PLATFORMS=cpu` automatically on Darwin.

### 6. Warning status and interpretation

After fixes in this investigation:

1. Missing-module warnings for `wandb` were resolved by installing `wandb`.
2. Missing-module warnings for `warp`/`mujoco_warp` were resolved by installing `warp-lang`.
3. TPU backend probe warnings were removed from the validated command path by setting `JAX_PLATFORMS=cpu`.

Still expected and mostly informational:

1. Orbax checkpoint INFO logs are verbose but not failures.
2. Upstream Brax maintenance warning was filtered in [dtd/ppo/main.py](dtd/ppo/main.py).

### 7. Where outputs are written

- Checkpoints: `models/<agent>/<env>/<TD>/noise_lvlXXX/<run_time>/`
- Metrics/config dump: `results/metrics/<agent>/<env>/<TD>/noise_lvlXXX/<run_time>/`
- Hydra logs: under the run directory configured in command/config

## Suggested next steps

1. Re-run dTD and baseline with the same seed and compare return curves.
2. Increase `mix_ratio` sweep density around incumbent values for your target env.
3. If runtime is high on CPU, reduce `env.num_envs` and `total_timesteps`, then scale up after confirming stability.
4. If reproducibility matters, freeze the environment from [requirements/macos-cpu.txt](requirements/macos-cpu.txt) with a lockfile and keep it committed.
