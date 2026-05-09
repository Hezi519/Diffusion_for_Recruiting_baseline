import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from core.utils import load_pickle


STD_NAMES = ["Gonorrhea", "Chlamydia", "Syphilis", "HIV", "Hepatitis"]
EXP_TYPES = {
    0: "lomax",
    1: "uniform",
    2: "lomax",
    3: "uniform",
    4: "ICPSR_sim",
    5: "ICPSR_real",
}
POLICY_LABELS = {
    ("OurPolicy", None): "Adaptive surrogate",
    ("ConstantPolicy", 2): "Fixed allocation k=2",
    ("ConstantPolicy", 3): "Fixed allocation k=3",
    ("ConstantPolicy", 5): "Fixed allocation k=5",
    ("ConstantPolicy", 10): "Fixed allocation k=10",
}
POLICY_COLORS = {
    ("OurPolicy", None): "tab:green",
    ("ConstantPolicy", 2): "tab:blue",
    ("ConstantPolicy", 3): "tab:purple",
    ("ConstantPolicy", 5): "tab:orange",
    ("ConstantPolicy", 10): "tab:red",
}


def result_path(args: argparse.Namespace) -> Path:
    std = STD_NAMES[args.std_idx] if args.test_mode in {4, 5} else None
    exp_type = EXP_TYPES[args.test_mode]
    surrogate_suffix = "" if args.surrogate_samples == 1000 else f"_surr{args.surrogate_samples}"
    param_string = (
        f"{args.num_times}_{args.max_budget}_{args.gamma}_"
        f"{args.initial_frontier_size}_{args.eps}_{std}_{args.rng_seed}{surrogate_suffix}"
    )
    return Path("results") / exp_type / f"full_{param_string}.pkl"


def collapse_duplicate_x(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = {}
    for xi, yi in zip(x, y):
        values[float(xi)] = float(yi)
    xs = np.array(sorted(values), dtype=float)
    ys = np.array([values[xi] for xi in xs], dtype=float)
    return xs, ys


def interpolate_trajectory(
    trajectory: dict,
    grid: np.ndarray,
    max_budget: int,
) -> np.ndarray:
    if not isinstance(trajectory, dict) or "budget_spent" not in trajectory:
        raise ValueError(
            "This result file does not contain budget_spent. "
            "Rerun run_experiments.py after the budget tracking change."
        )

    x = np.asarray(trajectory["budget_spent"], dtype=float)
    y = np.asarray(trajectory["system_size"], dtype=float)
    x, y = collapse_duplicate_x(x, y)
    if x[-1] < max_budget:
        x = np.append(x, max_budget)
        y = np.append(y, y[-1])
    return np.interp(grid, x, y)


def plot_budget_recruits(args: argparse.Namespace) -> Path:
    path = result_path(args)
    results = load_pickle(path)

    grid = np.linspace(0, args.max_budget, args.points)
    plt.figure(figsize=(11, 7))
    selected_policies = {("OurPolicy", None)} if args.adaptive_only else set(POLICY_LABELS)

    for policy_key, trajectories in results.items():
        if policy_key not in selected_policies:
            continue

        curves = np.vstack([
            interpolate_trajectory(trajectory, grid, args.max_budget)
            for trajectory in trajectories
        ])
        mean_curve = curves.mean(axis=0)
        std_curve = curves.std(axis=0)
        final_mean = mean_curve[-1]

        label = f"{POLICY_LABELS[policy_key]} ({final_mean:.1f} recruits)"
        color = POLICY_COLORS[policy_key]
        plt.plot(grid, mean_curve, label=label, color=color, linewidth=2.5)
        plt.fill_between(
            grid,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=color,
            alpha=0.15,
        )
        plt.scatter([grid[-1]], [mean_curve[-1]], color=color, s=45, zorder=3)
        plt.text(
            grid[-1],
            mean_curve[-1],
            f" {final_mean:.1f}",
            color=color,
            va="center",
            fontsize=10,
        )

    std = STD_NAMES[args.std_idx] if args.test_mode in {4, 5} else "Synthetic"
    plt.title(f"{std} recruiting performance comparison (B={args.max_budget}, gamma={args.gamma})")
    plt.xlabel("Cumulative budget spent")
    plt.ylabel("Cumulative recruits")
    plt.xlim(0, args.max_budget)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()

    output = args.output
    if output is None:
        exp_type = EXP_TYPES[args.test_mode]
        surrogate_suffix = "" if args.surrogate_samples == 1000 else f"_surr{args.surrogate_samples}"
        output = Path("figures") / exp_type / (
            f"budget_recruits_B{args.max_budget}_gamma{args.gamma}_"
            f"n{args.initial_frontier_size}_{std}{surrogate_suffix}.png"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot cumulative recruits against cumulative budget spent."
    )
    parser.add_argument("--test-mode", type=int, default=4)
    parser.add_argument("--num-times", type=int, default=30)
    parser.add_argument("--max-budget", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--initial-frontier-size", type=int, default=10)
    parser.add_argument("--eps", type=float, default=0.0)
    parser.add_argument("--std-idx", type=int, default=3)
    parser.add_argument("--rng-seed", type=int, default=42)
    parser.add_argument("--surrogate-samples", type=int, default=1000)
    parser.add_argument("--points", type=int, default=200)
    parser.add_argument("--adaptive-only", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    output_path = plot_budget_recruits(parse_args())
    print(f"Saved figure to {output_path}")
