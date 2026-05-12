from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("results/metrics/ppo")
OUT = Path("plots/all_envs_average.png")

ENVS = ["hopper", "ant", "humanoid", "halfcheetah"]

ALGS = {
    "baseline": ("TD", "#4C72B0"),
    "dtd": (r"$\beta$-dTD", "#55A868"),
    "dtd_full": (r"dTD ($\beta$=1)", "#C44E52"),
}

NOISES = [
    ("0.00", "000"),
    ("0.01", "001"),
    ("0.05", "005"),
]


def load_runs(env, alg, noise_suffix, total_timesteps=2_500_000):
    run_root = ROOT / env / alg / f"noise_lvl{noise_suffix}"
    if not run_root.exists():
        return []

    runs = []
    for run_dir in sorted(run_root.iterdir()):
        if not run_dir.is_dir():
            continue

        try:
            cfg = json.loads((run_dir / "configs.json").read_text())
            means = np.array(json.loads((run_dir / "means.json").read_text()), dtype=float)
        except FileNotFoundError:
            continue

        if int(cfg["algorithm"]["total_timesteps"]) != total_timesteps:
            continue

        steps_per_update = (
            int(cfg["env"]["num_envs"])
            * int(cfg["algorithm"]["num_env_steps_per_update"])
        )
        x = np.arange(1, len(means) + 1) * steps_per_update
        runs.append((x, means, run_dir.name))

    return runs


def aggregate(runs):
    min_len = min(len(y) for _, y, _ in runs)
    ys = np.stack([y[:min_len] for _, y, _ in runs], axis=0)
    x = runs[0][0][:min_len]
    return x, ys.mean(axis=0), ys.std(axis=0)


OUT.parent.mkdir(exist_ok=True)

fig, axes = plt.subplots(len(ENVS), len(NOISES), figsize=(11, 3.3 * len(ENVS)), sharey="row")

for row, env in enumerate(ENVS):
    for col, (noise_label, noise_suffix) in enumerate(NOISES):
        ax = axes[row][col]

        for alg, (label, color) in ALGS.items():
            runs = load_runs(env, alg, noise_suffix)

            if not runs:
                print(f"missing {env}/{alg} noise={noise_label}")
                continue

            x, mean, std = aggregate(runs)

            print(
                f"{env} {alg} noise={noise_label}: {len(runs)} runs, "
                f"final={mean[-1]:.2f} ± {std[-1]:.2f}"
            )

            ax.plot(x, mean, label=label, color=color, linewidth=2)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)

        if col == 0:
            ax.set_ylabel(f"{env.capitalize()}\nEpisodic return")
        if row == 0:
            ax.set_title(f"noise: {noise_label}")
        elif row == len(ENVS) - 1:
            ax.set_xlabel("Total episode step")

        ax.set_xlim(0, 2_500_000)
        ax.grid(alpha=0.25)

axes[0][0].legend(frameon=False, loc="upper left")

fig.tight_layout()
fig.savefig(OUT, dpi=200)
print(f"saved {OUT}")
