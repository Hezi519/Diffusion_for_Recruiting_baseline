"""Plot DQN, structured RL, and adaptive surrogate on shared axes.

The individual drivers save trajectory CSV files through ``save_single_curve``.
This script reads those CSVs and produces three comparison figures:

1. Budget spent vs cumulative recruits.
2. Time spent vs cumulative recruits.
3. Budget spent vs accumulated discounted reward.
4. Time spent vs accumulated discounted reward.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_DIFFUSION_PROJECT_DIR = (
    REPO_DIR / "Diffusion_for_Recruiting_baseline-leakage-free-split"
)

METHODS = {
    "dqn": {
        "label": "Budget-DQN + GreedyAlloc",
        "color": "tab:blue",
    },
    "structured": {
        "label": "Three-Head Structured RL",
        "color": "tab:green",
    },
    "adaptive": {
        "label": "Adaptive Surrogate",
        "color": "tab:red",
    },
}


def resolve_path(path: str | None, default: Path) -> Path:
    return (Path(path).expanduser() if path else default).resolve()


def default_trajectory_paths(args) -> dict[str, Path]:
    diffusion_results_dir = resolve_path(
        args.diffusion_results_dir,
        DEFAULT_DIFFUSION_PROJECT_DIR / "results",
    )
    adaptive_results_dir = resolve_path(args.adaptive_results_dir, REPO_DIR / "results")

    dqn_tag = (
        f"{args.std_name}"
        f"_B{args.budget}"
        f"_F{args.initial_frontier_size}"
        f"_disc{args.discount}"
        f"_seed{args.seed}"
        f"_train{args.train_episodes}"
        f"_hid{args.hidden_dim}"
    )
    structured_tag = f"{dqn_tag}_structured"
    adaptive_tag = (
        f"adaptive_surrogate_{args.std_name}"
        f"_B{args.budget}"
        f"_F{args.initial_frontier_size}"
        f"_disc{args.discount}"
        f"_seed{args.seed}"
        f"_surr{args.surrogate_samples}"
    )

    return {
        "dqn": resolve_path(args.dqn_csv, diffusion_results_dir / f"trajectories_{dqn_tag}.csv"),
        "structured": resolve_path(
            args.structured_csv,
            diffusion_results_dir / f"trajectories_{structured_tag}.csv",
        ),
        "adaptive": resolve_path(
            args.adaptive_csv,
            adaptive_results_dir / f"trajectories_{adaptive_tag}.csv",
        ),
    }


def load_trajectory_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"trajectory CSV not found: {path}")
    df = pd.read_csv(path)
    required = {
        "episode",
        "round",
        "reward",
        "cumulative_budget_spent",
        "cumulative_recruits",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df


def build_curves(
    df: pd.DataFrame,
    horizon: int,
    gamma: float,
) -> dict[str, np.ndarray]:
    budget_mat = []
    recruits_mat = []
    discounted_mat = []

    for _, ep_df in df.groupby("episode"):
        ep_df = ep_df.sort_values("round")
        budgets = []
        recruits = []
        discounted = []
        cumulative_discounted = 0.0

        for _, row in ep_df.iterrows():
            round_idx = int(row["round"]) - 1
            cumulative_discounted += float(row["reward"]) * (gamma ** round_idx)
            budgets.append(float(row["cumulative_budget_spent"]))
            recruits.append(float(row["cumulative_recruits"]))
            discounted.append(cumulative_discounted)

        while len(budgets) < horizon:
            budgets.append(budgets[-1] if budgets else 0.0)
            recruits.append(recruits[-1] if recruits else 0.0)
            discounted.append(discounted[-1] if discounted else 0.0)

        budget_mat.append(np.asarray(budgets[:horizon], dtype=np.float32))
        recruits_mat.append(np.asarray(recruits[:horizon], dtype=np.float32))
        discounted_mat.append(np.asarray(discounted[:horizon], dtype=np.float32))

    budget_arr = np.asarray(budget_mat, dtype=np.float32)
    recruits_arr = np.asarray(recruits_mat, dtype=np.float32)
    discounted_arr = np.asarray(discounted_mat, dtype=np.float32)

    return {
        "budget_x": budget_arr.mean(axis=0),
        "time_x": np.arange(1, horizon + 1, dtype=np.float32),
        "recruits_y": recruits_arr.mean(axis=0),
        "recruits_y_std": recruits_arr.std(axis=0),
        "discounted_y": discounted_arr.mean(axis=0),
        "discounted_y_std": discounted_arr.std(axis=0),
        "final_recruits": recruits_arr[:, -1],
        "final_discounted": discounted_arr[:, -1],
    }


def prepend_origin(x: np.ndarray, y: np.ndarray, y_std: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.concatenate([[0.0], x]),
        np.concatenate([[0.0], y]),
        np.concatenate([[0.0], y_std]),
    )


def plot_comparison(
    curves_by_method: dict[str, dict[str, np.ndarray]],
    x_key: str,
    y_key: str,
    y_std_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
    annotate_final: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    y_max = 0.0

    for method, curves in curves_by_method.items():
        style = METHODS[method]
        x, y, y_std = prepend_origin(curves[x_key], curves[y_key], curves[y_std_key])
        ax.plot(x, y, color=style["color"], linewidth=2, label=style["label"])
        ax.fill_between(
            x,
            np.maximum(0.0, y - y_std),
            y + y_std,
            color=style["color"],
            alpha=0.16,
        )
        y_max = max(y_max, float(np.max(y + y_std)))

        if annotate_final:
            final_x = float(x[-1])
            final_y = float(y[-1])
            ax.plot(final_x, final_y, "o", color=style["color"], markersize=5, zorder=5)
            ax.annotate(
                f"{final_y:.1f}",
                xy=(final_x, final_y),
                xytext=(-8, 6),
                textcoords="offset points",
                fontsize=8,
                color=style["color"],
                ha="right",
                va="bottom",
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(1.0, y_max * 1.12))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print("Saved plot to:", out_path)


def save_stats(curves_by_method: dict[str, dict[str, np.ndarray]], out_path: Path) -> None:
    rows = []
    for method, curves in curves_by_method.items():
        rows.append(
            {
                "method": METHODS[method]["label"],
                "mean_final_recruits": float(np.mean(curves["final_recruits"])),
                "std_final_recruits": float(np.std(curves["final_recruits"])),
                "mean_final_discounted_reward": float(np.mean(curves["final_discounted"])),
                "std_final_discounted_reward": float(np.std(curves["final_discounted"])),
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("Saved stats to:", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot shared comparison curves for DQN, structured RL, and adaptive surrogate"
    )
    parser.add_argument("--std_name", type=str, default="HIV")
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--initial_frontier_size", type=int, default=10)
    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_episodes", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--surrogate_samples", type=int, default=1000)
    parser.add_argument("--max_rounds", type=int, default=50)
    parser.add_argument("--diffusion_results_dir", type=str, default=None)
    parser.add_argument("--adaptive_results_dir", type=str, default=None)
    parser.add_argument("--dqn_csv", type=str, default=None)
    parser.add_argument("--structured_csv", type=str, default=None)
    parser.add_argument("--adaptive_csv", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=str(REPO_DIR / "results" / "comparison"))
    args = parser.parse_args()

    paths = default_trajectory_paths(args)
    for method, path in paths.items():
        print(f"{METHODS[method]['label']}: {path}")

    curves_by_method = {
        method: build_curves(load_trajectory_csv(path), horizon=args.max_rounds, gamma=args.discount)
        for method, path in paths.items()
    }

    out_dir = resolve_path(args.out_dir, REPO_DIR / "results" / "comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"{args.std_name}"
        f"_B{args.budget}"
        f"_F{args.initial_frontier_size}"
        f"_disc{args.discount}"
        f"_seed{args.seed}"
        f"_train{args.train_episodes}"
        f"_hid{args.hidden_dim}"
        f"_surr{args.surrogate_samples}"
    )

    plot_comparison(
        curves_by_method,
        x_key="budget_x",
        y_key="recruits_y",
        y_std_key="recruits_y_std",
        xlabel="Budget spent",
        ylabel="Cumulative recruits",
        title=f"Cumulative recruits by budget (B={args.budget}, discount={args.discount})",
        out_path=out_dir / f"comparison_recruits_by_budget_{tag}.png",
    )
    plot_comparison(
        curves_by_method,
        x_key="time_x",
        y_key="recruits_y",
        y_std_key="recruits_y_std",
        xlabel="Time spent",
        ylabel="Cumulative recruits",
        title=f"Cumulative recruits by time (B={args.budget}, discount={args.discount})",
        out_path=out_dir / f"comparison_recruits_by_time_{tag}.png",
    )
    plot_comparison(
        curves_by_method,
        x_key="budget_x",
        y_key="discounted_y",
        y_std_key="discounted_y_std",
        xlabel="Budget spent",
        ylabel="Accumulated discounted reward",
        title=f"Discounted reward by budget (B={args.budget}, discount={args.discount})",
        out_path=out_dir / f"comparison_discounted_reward_by_budget_{tag}.png",
    )
    plot_comparison(
        curves_by_method,
        x_key="time_x",
        y_key="discounted_y",
        y_std_key="discounted_y_std",
        xlabel="Time spent",
        ylabel="Accumulated discounted reward",
        title=f"Discounted reward by time (B={args.budget}, discount={args.discount})",
        out_path=out_dir / f"comparison_discounted_reward_by_time_{tag}.png",
    )
    save_stats(curves_by_method, out_dir / f"comparison_stats_{tag}.csv")


if __name__ == "__main__":
    main()
