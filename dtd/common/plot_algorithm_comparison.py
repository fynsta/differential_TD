import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def noise_to_dir_suffix(noise_lvl: str) -> str:
    return str(noise_lvl).replace(".", "").zfill(3)


def parse_run_times(items: List[str]) -> Dict[str, str]:
    run_times: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --run-times item '{item}'. Expected format: algorithm=run_time"
            )
        alg, run_time = item.split("=", 1)
        run_times[alg.strip()] = run_time.strip()
    return run_times


def load_run_payload(run_dir: Path) -> Optional[Dict]:
    cfg_path = run_dir / "configs.json"
    means_path = run_dir / "means.json"
    stds_path = run_dir / "stds.json"

    if not (cfg_path.exists() and means_path.exists() and stds_path.exists()):
        return None

    cfg = json.loads(cfg_path.read_text())
    means = json.loads(means_path.read_text())
    stds = json.loads(stds_path.read_text())

    if not isinstance(means, list) or not isinstance(stds, list):
        return None
    if len(means) == 0 or len(stds) == 0:
        return None

    return {
        "run_time": run_dir.name,
        "run_dir": run_dir,
        "cfg": cfg,
        "means": means,
        "stds": stds,
    }


def matches_setup(
    payload: Dict,
    seed: Optional[int],
    total_timesteps: Optional[int],
    num_envs: Optional[int],
    steps_per_update: Optional[int],
) -> bool:
    cfg = payload["cfg"]
    algorithm_cfg = cfg.get("algorithm", {})
    env_cfg = cfg.get("env", {})

    if seed is not None and algorithm_cfg.get("seed") != seed:
        return False
    if total_timesteps is not None and algorithm_cfg.get("total_timesteps") != total_timesteps:
        return False
    if num_envs is not None and env_cfg.get("num_envs") != num_envs:
        return False
    if (
        steps_per_update is not None
        and algorithm_cfg.get("num_env_steps_per_update") != steps_per_update
    ):
        return False

    return True


def select_run(candidates: List[Dict], strategy: str) -> Dict:
    if strategy == "latest":
        return max(candidates, key=lambda p: int(p["run_time"]))

    if strategy == "best-final":
        return max(candidates, key=lambda p: float(p["means"][-1]))

    raise ValueError(f"Unknown selection strategy: {strategy}")


def find_run_for_algorithm(
    results_root: Path,
    agent: str,
    env: str,
    algorithm: str,
    noise_lvl: str,
    explicit_run_time: Optional[str],
    selection: str,
    seed: Optional[int],
    total_timesteps: Optional[int],
    num_envs: Optional[int],
    steps_per_update: Optional[int],
) -> Optional[Dict]:
    noise_suffix = noise_to_dir_suffix(noise_lvl)
    alg_dir = results_root / agent / env / algorithm / f"noise_lvl{noise_suffix}"

    if explicit_run_time is not None:
        payload = load_run_payload(alg_dir / explicit_run_time)
        if payload is None:
            return None
        if not matches_setup(payload, seed, total_timesteps, num_envs, steps_per_update):
            return None
        return payload

    if not alg_dir.exists():
        return None

    candidates: List[Dict] = []
    for run_dir in alg_dir.iterdir():
        if not run_dir.is_dir() or not run_dir.name.isdigit():
            continue
        payload = load_run_payload(run_dir)
        if payload is None:
            continue
        if not matches_setup(payload, seed, total_timesteps, num_envs, steps_per_update):
            continue
        candidates.append(payload)

    if not candidates:
        return None

    return select_run(candidates, selection)


def summarize(payload: Dict) -> Tuple[float, float, float]:
    means = np.asarray(payload["means"], dtype=float)
    final_mean = float(means[-1])
    best_mean = float(np.max(means))
    auc = float(np.trapezoid(means, dx=1.0))
    return final_mean, best_mean, auc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot and compare algorithm return curves for a given setup using saved results/metrics JSON files."
        )
    )
    parser.add_argument("--results-root", default="results/metrics", help="Root metrics directory")
    parser.add_argument("--agent", default="ppo", help="Agent class in the results path")
    parser.add_argument("--env", required=True, help="Environment name (e.g. hopper)")
    parser.add_argument("--noise-lvl", required=True, help="Noise level value used in the run (e.g. 0.01)")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["baseline", "naive", "dtd"],
        help="Algorithms to compare",
    )
    parser.add_argument(
        "--run-times",
        nargs="*",
        default=[],
        help="Optional explicit run times per algorithm: baseline=123 dtd=456",
    )
    parser.add_argument(
        "--selection",
        choices=["latest", "best-final"],
        default="latest",
        help="How to pick runs when --run-times is not provided",
    )

    # Optional setup filters to ensure apples-to-apples comparison.
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--steps-per-update", type=int, default=None)

    parser.add_argument(
        "--output",
        default=None,
        help="Output image path. Defaults to plots/compare_<agent>_<env>_noise_lvlXXX.png",
    )
    parser.add_argument("--show", action="store_true", help="Also show the plot window")
    parser.add_argument(
        "--no-std-band",
        action="store_true",
        help="Disable standard deviation shaded region",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    results_root = Path(args.results_root)
    run_times = parse_run_times(args.run_times)

    selected: Dict[str, Dict] = {}
    missing: List[str] = []

    for algorithm in args.algorithms:
        payload = find_run_for_algorithm(
            results_root=results_root,
            agent=args.agent,
            env=args.env,
            algorithm=algorithm,
            noise_lvl=args.noise_lvl,
            explicit_run_time=run_times.get(algorithm),
            selection=args.selection,
            seed=args.seed,
            total_timesteps=args.total_timesteps,
            num_envs=args.num_envs,
            steps_per_update=args.steps_per_update,
        )
        if payload is None:
            missing.append(algorithm)
            continue
        selected[algorithm] = payload

    if not selected:
        raise SystemExit(
            "No matching runs found for the requested setup. "
            "Try relaxing filters or specifying --run-times explicitly."
        )

    if missing:
        print("Warning: no matching run found for:", ", ".join(missing))

    noise_suffix = noise_to_dir_suffix(args.noise_lvl)
    if args.output is None:
        out_path = Path("plots") / f"compare_{args.agent}_{args.env}_noise_lvl{noise_suffix}.png"
    else:
        out_path = Path(args.output)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for algorithm, payload in selected.items():
        means = np.asarray(payload["means"], dtype=float)
        stds = np.asarray(payload["stds"], dtype=float)
        x = np.arange(1, len(means) + 1)

        label = f"{algorithm} (run={payload['run_time']})"
        plt.plot(x, means, label=label, linewidth=2)

        if not args.no_std_band and len(stds) == len(means):
            plt.fill_between(x, means - stds, means + stds, alpha=0.2)

    plt.title(
        f"Algorithm Comparison | agent={args.agent}, env={args.env}, noise={args.noise_lvl}"
    )
    plt.xlabel("Update")
    plt.ylabel("Episode Return")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    print(f"Saved plot: {out_path}")

    print("\nRun summary:")
    for algorithm, payload in selected.items():
        final_mean, best_mean, auc = summarize(payload)
        print(
            f"- {algorithm:8s} run={payload['run_time']} final_mean={final_mean:.3f} "
            f"best_mean={best_mean:.3f} auc={auc:.3f}"
        )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
