"""Shared evaluation and plotting utilities for recruiting drivers."""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.environment.recruiting_env import RecruitingEnv
from src.environment.state import RecruitingState


def evaluate_recruiting_curve(
    policy_fn: Callable[[RecruitingState], np.ndarray],
    env: RecruitingEnv,
    initial_frontier_fn: Callable[[], np.ndarray],
    n_episodes_eval: int,
    gamma: float,
    normalize: bool = False,
):
    """Evaluate a policy and produce recruitment curves.

    Args:
        policy_fn: Callable that takes a RecruitingState and returns
            an allocation vector (n_t,).
        env: The recruiting environment.
        initial_frontier_fn: Callable returning initial frontier covariates.
        n_episodes_eval: Number of evaluation episodes.
        gamma: Discount factor for computing discounted returns.
        normalize: If True, x is cumulative_budget_spent / initial_budget and
            y is cumulative_recruits / total_recruits_this_episode (both in [0,1]).
            Normalization is applied per-episode before horizon-padding.

    Returns:
        x: Mean cumulative budget spent by round.
        y: Mean cumulative recruits by round.
        y_std: Std of cumulative recruits by round.
        traj_rows: Per-round trajectory rows for CSV export.
        discounted_returns: Discounted returns per episode.
        total_rewards: Undiscounted total recruits per episode.
    """
    horizon = env.max_rounds
    initial_budget = env.initial_budget

    x_mat = []
    y_mat = []
    discounted_returns = []
    total_rewards = []
    traj_rows = []

    for ep in range(n_episodes_eval):
        state = env.reset(initial_frontier_fn(), seed=100000 + ep)
        cumulative_budget_spent = 0.0
        cumulative_recruits = 0.0
        rewards_this_ep = []

        x_ep = []
        y_ep = []

        while True:
            action_vec = policy_fn(state)
            next_state, reward, done, info = env.step(action_vec)

            rewards_this_ep.append(float(reward))
            cumulative_budget_spent += float(info["budget_spent"])
            cumulative_recruits += float(reward)

            x_ep.append(cumulative_budget_spent)
            y_ep.append(cumulative_recruits)

            traj_rows.append(
                {
                    "episode": ep,
                    "round": info["round"],
                    "frontier_size": int(state.frontier_size),
                    "budget_remaining_before": int(state.budget_remaining),
                    "budget_spent": int(info["budget_spent"]),
                    "cumulative_budget_spent": float(cumulative_budget_spent),
                    "budget_fraction": float(
                        cumulative_budget_spent / max(float(initial_budget), 1.0)
                    ),
                    "reward": float(reward),
                    "cumulative_recruits": float(cumulative_recruits),
                    "next_frontier_size": int(next_state.frontier_size),
                    "termination_reason": info["termination_reason"],
                }
            )

            state = next_state
            if done:
                break

        if normalize:
            budget_denom = max(float(initial_budget), 1.0)
            recruits_denom = max(float(cumulative_recruits), 1.0)
            x_ep = [v / budget_denom for v in x_ep]
            y_ep = [v / recruits_denom for v in y_ep]

        while len(x_ep) < horizon:
            x_ep.append(x_ep[-1] if len(x_ep) > 0 else 0.0)
            y_ep.append(y_ep[-1] if len(y_ep) > 0 else 0.0)

        x_ep = np.asarray(x_ep[:horizon], dtype=np.float32)
        y_ep = np.asarray(y_ep[:horizon], dtype=np.float32)

        x_mat.append(x_ep)
        y_mat.append(y_ep)

        discounts = gamma ** np.arange(len(rewards_this_ep))
        discounted_returns.append(float(np.sum(np.asarray(rewards_this_ep) * discounts)))
        total_rewards.append(float(np.sum(rewards_this_ep)))

    x_mat = np.asarray(x_mat, dtype=np.float32)
    y_mat = np.asarray(y_mat, dtype=np.float32)

    x = np.mean(x_mat, axis=0)
    y = np.mean(y_mat, axis=0)
    y_std = np.std(y_mat, axis=0)

    return x, y, y_std, traj_rows, np.asarray(discounted_returns), np.asarray(total_rewards)


def save_single_curve(
    x: np.ndarray,
    y: np.ndarray,
    y_std: np.ndarray,
    traj_rows: list[dict],
    results_dir: str,
    run_tag: str,
    gamma: float,
    label: str = "Policy",
) -> None:
    """Save a single-method evaluation: NPZ, CSV, and absolute-axes PNG."""
    os.makedirs(results_dir, exist_ok=True)

    npz_path = f"{results_dir}/eval_results_{run_tag}.npz"
    np.savez(npz_path, x=x, y=y, y_std=y_std)
    print("Saved eval vectors to:", npz_path)

    traj_path = f"{results_dir}/trajectories_{run_tag}.csv"
    pd.DataFrame(traj_rows).to_csv(traj_path, index=False)
    print("Saved trajectories to:", traj_path)

    x_plot = np.concatenate([[0.0], x])
    y_plot = np.concatenate([[0.0], y])
    y_std_plot = np.concatenate([[0.0], y_std])

    plt.figure(figsize=(8, 4))
    plt.plot(x_plot, y_plot, linestyle="-", color="tab:blue", label=label)
    plt.fill_between(
        x_plot,
        np.maximum(0.0, y_plot - y_std_plot),
        y_plot + y_std_plot,
        color="tab:blue",
        alpha=0.25,
    )
    plt.xlabel("Budget spent")
    plt.ylabel("Cumulative recruits")
    plt.title(f"Recruiting policy (discount = {gamma})")
    plt.legend()
    plt.tight_layout()

    png_path = f"{results_dir}/recruiting_curve_{run_tag}.png"
    plt.savefig(png_path, dpi=200)
    print("Saved plot to:", png_path)
    plt.close()


def _method_label(method: str) -> str:
    if method == "random":
        return "Random"
    if method == "dqn":
        return "Budget-DQN + GreedyAlloc"
    if method == "structured":
        return "Three-Head Structured RL"
    if method == "gfp":
        return "Generative Frontier Planning"
    return method


def _method_color(method: str) -> str:
    colors = {
        "random": "tab:gray",
        "dqn": "tab:blue",
        "structured": "tab:green",
        "gfp": "tab:purple",
    }
    return colors.get(method, "tab:orange")


def _build_raw_comparison_curves(
    traj_rows: list[dict],
    horizon: int,
    gamma: float,
) -> dict[str, np.ndarray]:
    df = pd.DataFrame(traj_rows)
    required = {
        "episode",
        "round",
        "reward",
        "cumulative_budget_spent",
        "cumulative_recruits",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"trajectory rows are missing columns: {sorted(missing)}")

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


def _prepend_origin(
    x: np.ndarray,
    y: np.ndarray,
    y_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.concatenate([[0.0], x]),
        np.concatenate([[0.0], y]),
        np.concatenate([[0.0], y_std]),
    )


def _plot_raw_comparison(
    curves_by_method: dict[str, dict[str, np.ndarray]],
    x_key: str,
    y_key: str,
    y_std_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    y_max = 0.0

    for method, curves in curves_by_method.items():
        color = _method_color(method)
        x, y, y_std = _prepend_origin(curves[x_key], curves[y_key], curves[y_std_key])
        ax.plot(x, y, color=color, linewidth=2, label=_method_label(method))
        ax.fill_between(
            x,
            np.maximum(0.0, y - y_std),
            y + y_std,
            color=color,
            alpha=0.16,
        )
        y_max = max(y_max, float(np.max(y + y_std)))

        ax.plot(float(x[-1]), float(y[-1]), "o", color=color, markersize=5, zorder=5)
        ax.annotate(
            f"{float(y[-1]):.1f}",
            xy=(float(x[-1]), float(y[-1])),
            xytext=(-8, 6),
            textcoords="offset points",
            fontsize=8,
            color=color,
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
    print(f"Saved plot to: {out_path}")


def save_four_panel_curve_outputs(
    bundles: dict[str, Any],
    results_dir: str,
    run_tag: str,
    gamma: float,
    horizon: int,
) -> None:
    """Save four raw-scale comparison plots and the curve data behind them."""
    os.makedirs(results_dir, exist_ok=True)
    curves_by_method = {
        method: _build_raw_comparison_curves(bundle.traj_rows, horizon=horizon, gamma=gamma)
        for method, bundle in bundles.items()
    }

    curve_rows = []
    for method, curves in curves_by_method.items():
        for idx in range(horizon):
            curve_rows.append(
                {
                    "method": method,
                    "method_label": _method_label(method),
                    "round": idx + 1,
                    "budget_x": float(curves["budget_x"][idx]),
                    "time_x": float(curves["time_x"][idx]),
                    "recruits_y": float(curves["recruits_y"][idx]),
                    "recruits_y_std": float(curves["recruits_y_std"][idx]),
                    "discounted_y": float(curves["discounted_y"][idx]),
                    "discounted_y_std": float(curves["discounted_y_std"][idx]),
                }
            )
    curve_csv_path = os.path.join(results_dir, f"comparison_curve_data_{run_tag}.csv")
    pd.DataFrame(curve_rows).to_csv(curve_csv_path, index=False)
    print(f"Saved comparison curve data to: {curve_csv_path}")

    npz_payload = {}
    for method, curves in curves_by_method.items():
        for key, value in curves.items():
            npz_payload[f"{method}_{key}"] = value
    curve_npz_path = os.path.join(results_dir, f"comparison_curve_data_{run_tag}.npz")
    np.savez(curve_npz_path, **npz_payload)
    print(f"Saved comparison curve vectors to: {curve_npz_path}")

    stats_rows = []
    for method, curves in curves_by_method.items():
        stats_rows.append(
            {
                "method": method,
                "method_label": _method_label(method),
                "mean_final_recruits": float(np.mean(curves["final_recruits"])),
                "std_final_recruits": float(np.std(curves["final_recruits"])),
                "mean_final_discounted_reward": float(np.mean(curves["final_discounted"])),
                "std_final_discounted_reward": float(np.std(curves["final_discounted"])),
            }
        )
    stats_path = os.path.join(results_dir, f"comparison_stats_{run_tag}.csv")
    pd.DataFrame(stats_rows).to_csv(stats_path, index=False)
    print(f"Saved comparison stats to: {stats_path}")

    _plot_raw_comparison(
        curves_by_method,
        x_key="budget_x",
        y_key="recruits_y",
        y_std_key="recruits_y_std",
        xlabel="Budget spent",
        ylabel="Cumulative recruits",
        title=f"Cumulative recruits by budget (discount={gamma})",
        out_path=os.path.join(results_dir, f"comparison_recruits_by_budget_{run_tag}.png"),
    )
    _plot_raw_comparison(
        curves_by_method,
        x_key="time_x",
        y_key="recruits_y",
        y_std_key="recruits_y_std",
        xlabel="Time spent",
        ylabel="Cumulative recruits",
        title=f"Cumulative recruits by time (discount={gamma})",
        out_path=os.path.join(results_dir, f"comparison_recruits_by_time_{run_tag}.png"),
    )
    _plot_raw_comparison(
        curves_by_method,
        x_key="budget_x",
        y_key="discounted_y",
        y_std_key="discounted_y_std",
        xlabel="Budget spent",
        ylabel="Accumulated discounted reward",
        title=f"Discounted reward by budget (discount={gamma})",
        out_path=os.path.join(results_dir, f"comparison_discounted_reward_by_budget_{run_tag}.png"),
    )
    _plot_raw_comparison(
        curves_by_method,
        x_key="time_x",
        y_key="discounted_y",
        y_std_key="discounted_y_std",
        xlabel="Time spent",
        ylabel="Accumulated discounted reward",
        title=f"Discounted reward by time (discount={gamma})",
        out_path=os.path.join(results_dir, f"comparison_discounted_reward_by_time_{run_tag}.png"),
    )


def save_comparison_curves(
    bundles: dict[str, Any],
    results_dir: str,
    run_tag: str,
    gamma: float,
) -> None:
    """Save per-method NPZ/CSV + a normalized-axes comparison PNG.

    Each value in `bundles` must duck-type on `.x`, `.y`, `.y_std`, `.traj_rows`.
    Curves are expected to already be normalized to [0, 1].
    """
    os.makedirs(results_dir, exist_ok=True)

    for method, bundle in bundles.items():
        npz_path = os.path.join(results_dir, f"eval_results_{run_tag}_{method}.npz")
        np.savez(npz_path, x=bundle.x, y=bundle.y, y_std=bundle.y_std)
        print(f"Saved eval vectors to: {npz_path}")

        traj_path = os.path.join(results_dir, f"trajectories_{run_tag}_{method}.csv")
        pd.DataFrame(bundle.traj_rows).to_csv(traj_path, index=False)
        print(f"Saved trajectories to: {traj_path}")

    combined_rows = []
    for method, bundle in bundles.items():
        for row in bundle.traj_rows:
            combined_rows.append({"method": method, **row})
    combined_path = os.path.join(results_dir, f"trajectories_{run_tag}_all_methods.csv")
    pd.DataFrame(combined_rows).to_csv(combined_path, index=False)
    print(f"Saved combined trajectories to: {combined_path}")

    plt.figure(figsize=(8, 4))
    for method, bundle in bundles.items():
        x_plot = np.concatenate([[0.0], bundle.x])
        y_plot = np.concatenate([[0.0], bundle.y])
        y_std_plot = np.concatenate([[0.0], bundle.y_std])
        label = _method_label(method)
        plt.plot(x_plot, y_plot, linestyle="-", label=label)
        plt.fill_between(
            x_plot,
            np.maximum(0.0, y_plot - y_std_plot),
            np.minimum(1.0, y_plot + y_std_plot),
            alpha=0.20,
        )

    plt.axvline(x=0.5, linestyle=":", color="gray", alpha=0.7)
    plt.xlabel("Fraction of budget spent")
    plt.ylabel("Fraction of total recruits obtained (normalized)")
    plt.title(f"Recruiting policies with discount = {gamma}")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()

    plot_name = (
        f"recruiting_curve_{run_tag}.png"
        if len(bundles) == 1
        else f"recruiting_curve_{run_tag}_comparison.png"
    )
    png_path = os.path.join(results_dir, plot_name)
    plt.savefig(png_path, dpi=200)
    plt.close()
    print(f"Saved plot to: {png_path}")
