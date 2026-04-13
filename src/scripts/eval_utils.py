"""Shared evaluation and plotting utilities for recruiting drivers."""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np
import pandas as pd
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
    if method == "dqn":
        return "Budget-DQN + GreedyAlloc"
    if method == "structured":
        return "Three-Head Structured RL"
    return method


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
    for bundle in bundles.values():
        combined_rows.extend(bundle.traj_rows)
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
