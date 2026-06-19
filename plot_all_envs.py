from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("results/metrics/ppo")
INCLUDE_PURE_DTD = False
PLOTS = [
    (
        Path("plots/all_envs_average.png"),
        {
            "baseline": ("TD", "#808080"),
            "dtd": (r"$\beta$-dTD", "#C44E52"),
            **(
                {"dtd_full": (r"dTD ($\beta$=1)", "#8B1E3F")}
                if INCLUDE_PURE_DTD
                else {}
            ),
        },
        -1000,
    ),
]

ENVS = ["hopper", "halfcheetah", "ant", "humanoid"]

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
        beta = cfg.get("algorithm", {}).get("model_kwargs", {}).get("mix_ratio")
        x = np.arange(1, len(means) + 1) * steps_per_update
        runs.append((x, means, run_dir.name, beta))

    return runs


def aggregate(runs):
    min_len = min(len(y) for _, y, _, _ in runs)
    ys = np.stack([y[:min_len] for _, y, _, _ in runs], axis=0)
    x = runs[0][0][:min_len]
    return x, ys.mean(axis=0), ys.std(axis=0)


for out, algs, y_floor in PLOTS:
    out.parent.mkdir(exist_ok=True)
    fig, axes = plt.subplots(len(ENVS), len(NOISES), figsize=(11, 3.3 * len(ENVS)), sharey="row")

    for row, env in enumerate(ENVS):
        row_mean_min = np.inf
        row_mean_max = -np.inf
        for col, (noise_label, noise_suffix) in enumerate(NOISES):
            ax = axes[row][col]

            for alg, (label, color) in algs.items():
                runs = load_runs(env, alg, noise_suffix)

                if not runs:
                    print(f"missing {env}/{alg} noise={noise_label}")
                    continue

                x, mean, std = aggregate(runs)

                print(
                    f"{out.name} | {env} {alg} noise={noise_label}: {len(runs)} runs, "
                    f"final={mean[-1]:.2f} ± {std[-1]:.2f}"
                )

                label_for_plot = label
                if alg == "dtd":
                    beta_values = sorted(
                        {
                            float(beta)
                            for _, _, _, beta in runs
                            if beta is not None
                        }
                    )
                    if beta_values:
                        label_for_plot = rf"$\beta$-dTD ($\beta$={beta_values[0]:.3f})"

                if alg == "baseline":
                    line_width = 1.8
                    line_alpha = 0.85
                    line_style = "--"
                else:
                    line_width = 2.4 if alg == "dtd" else 2.0
                    line_alpha = 1.0 if alg == "dtd" else 0.9
                    line_style = "-"
                ax.plot(
                    x,
                    mean,
                    label=label_for_plot,
                    color=color,
                    linewidth=line_width,
                    alpha=line_alpha,
                    linestyle=line_style,
                )
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12)
                row_mean_min = min(row_mean_min, float(mean.min()))
                row_mean_max = max(row_mean_max, float(mean.max()))

            if col == 0:
                ax.set_ylabel("Episodic return", fontsize=13)
            ax.set_title(f"{env.capitalize()} (Noise: {noise_label})")
            if row == len(ENVS) - 1:
                ax.set_xlabel("Total episode step")

            ax.set_xlim(0, 2_500_000)
            ax.grid(alpha=0.15, linestyle="--", linewidth=0.7)
            ax.set_xticks(np.arange(0, 2_500_001, 1_000_000))
            y_tick_step = 250 if env == "humanoid" else 1000
            ax.set_yticks(np.arange(-1000, 8001, y_tick_step))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(3, 3), useMathText=True)
            ax.tick_params(axis="both", which="major", direction="out", length=4, width=0.8)
            if col == 0:
                handles, labels = ax.get_legend_handles_labels()
                order = sorted(range(len(labels)), key=lambda i: labels[i] == "TD")
                ax.legend(
                    [handles[i] for i in order],
                    [labels[i] for i in order],
                    frameon=False,
                    loc="upper left",
                )

        if np.isfinite(row_mean_min) and np.isfinite(row_mean_max):
            if row_mean_max == row_mean_min:
                pad = max(1.0, abs(row_mean_max) * 0.05)
            else:
                pad = (row_mean_max - row_mean_min) * 0.05
            for ax in axes[row]:
                ymin = max(y_floor, row_mean_min - pad)
                ax.set_ylim(ymin, row_mean_max + pad)

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved {out}")
