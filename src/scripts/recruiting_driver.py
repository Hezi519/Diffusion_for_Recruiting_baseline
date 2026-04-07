"""
Train + evaluate the recruiting environment with Budget-DQN.

This driver is designed to mirror the output style of driver_disease_dpmd.py:
- prints run summary
- saves evaluation vectors (.npz)
- saves trajectories (.csv)
- saves a final curve plot (.png)

Curve meaning in this recruiting setting:
    x-axis: fraction of budget spent
    y-axis: fraction of total recruits obtained

Usage:
    python -m src.scripts.train_rl_recruiting \
        --model_path checkpoints/diffusion/ddpm_HIV.pt \
        --data_dir ICPSR_22140 \
        --std_name HIV \
        --budget 100 \
        --initial_frontier_size 10 \
        --train_episodes 300 \
        --n_episodes_eval 10 \
        --seed 42
"""

from __future__ import annotations

import argparse
import os
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.data.icpsr_loader import ICPSRGraphData
from src.environment.recruiting_env import RecruitingEnv
from src.models.count_model.gaussian_count_model import GaussianCountModel
from src.models.covariate_model.ddpm_covariate_model import DDPMCovariateModel
from src.models.RL_model.dqn_estimator import (
    DQNConfig,
    run_budget_dqn,
)
from src.models.RL_model.greedy_allocator import greedy_allocator


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def reseed_all(seed: int) -> None:
    random.seed(seed)
    np_seed = seed % (2**32 - 1)
    np.random.seed(np_seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_final_eval_results(
    x: np.ndarray,
    y: np.ndarray,
    y_std: np.ndarray,
    traj_rows: list[dict],
    results_dir: str,
    run_tag: str,
    gamma: float,
) -> None:
    """Save final evaluation results: NPZ, CSV, and PNG."""
    os.makedirs(results_dir, exist_ok=True)

    npz_path = f"{results_dir}/eval_results_{run_tag}.npz"
    np.savez(npz_path, x=x, y=y, y_std=y_std)
    print("Saved eval vectors to:", npz_path)

    traj_path = f"{results_dir}/trajectories_{run_tag}.csv"
    pd.DataFrame(traj_rows).to_csv(traj_path, index=False)
    print("Saved trajectories to:", traj_path)

    # Add (0, 0) as the starting point
    x_plot = np.concatenate([[0.0], x])
    y_plot = np.concatenate([[0.0], y])
    y_std_plot = np.concatenate([[0.0], y_std])

    plt.figure(figsize=(8, 4))
    plt.plot(x_plot, y_plot, linestyle="-", color="tab:blue", label="Budget-DQN + GreedyAlloc")
    plt.fill_between(
        x_plot,
        np.maximum(0.0, y_plot - y_std_plot),
        np.minimum(1.0, y_plot + y_std_plot),
        color="tab:blue",
        alpha=0.25,
    )
    plt.axvline(x=0.5, linestyle=":", color="gray", alpha=0.7)
    plt.xlabel("Fraction of budget spent")
    plt.ylabel("Fraction of total recruits obtained (normalized)")
    plt.title(f"Recruiting policies with discount = {gamma}")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()

    png_path = f"{results_dir}/recruiting_curve_{run_tag}.png"
    plt.savefig(png_path, dpi=200)
    print("Saved plot to:", png_path)
    plt.close()


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate_recruiting_curve(
    learner,
    env: RecruitingEnv,
    initial_frontier_fn,
    n_episodes_eval: int,
    gamma: float,
):
    """
    Evaluate the trained policy and produce a curve analogous to the disease driver.

    Returns:
        x: mean cumulative budget fraction by round
        y: mean cumulative recruit fraction by round
        y_std: std of cumulative recruit fraction by round
        traj_rows: per-round trajectory rows for CSV
        discounted_returns: discounted returns for each evaluation episode
        total_rewards: undiscounted total recruits for each evaluation episode
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
        y_ep_raw = []

        while True:
            action_budget = learner.select_action(state, greedy=True)
            action_vec = learner.budget_allocator(state, action_budget)

            next_state, reward, done, info = env.step(action_vec)

            rewards_this_ep.append(float(reward))
            cumulative_budget_spent += float(info["budget_spent"])
            cumulative_recruits += float(reward)

            x_ep.append(
                cumulative_budget_spent / max(float(initial_budget), 1.0)
            )
            y_ep_raw.append(cumulative_recruits)

            traj_rows.append(
                {
                    "episode": ep,
                    "round": info["round"],
                    "frontier_size": int(state.frontier_size),
                    "budget_remaining_before": int(state.budget_remaining),
                    "action_budget": int(action_budget),
                    "budget_spent": int(info["budget_spent"]),
                    "cumulative_budget_spent": float(cumulative_budget_spent),
                    "budget_fraction": float(cumulative_budget_spent / max(float(initial_budget), 1.0)),
                    "reward": float(reward),
                    "cumulative_recruits": float(cumulative_recruits),
                    "next_frontier_size": int(next_state.frontier_size),
                    "termination_reason": info["termination_reason"],
                }
            )

            state = next_state
            if done:
                break

        total_final = max(cumulative_recruits, 1.0)
        y_ep = [v / total_final for v in y_ep_raw]

        # pad to horizon
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


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train + evaluate Budget-DQN on recruiting env")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained diffusion model checkpoint")
    parser.add_argument("--data_dir", type=str, default="ICPSR_22140")
    parser.add_argument("--std_name", type=str, default="HIV",
                        choices=["HIV", "Gonorrhea", "Chlamydia", "Syphilis", "Hepatitis"])
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--initial_frontier_size", type=int, default=10)
    parser.add_argument("--n_episodes_eval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    # env
    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--max_rounds", type=int, default=50)

    # data split / count model
    parser.add_argument("--test_fraction", type=float, default=0.2)

    # DQN hyperparams
    parser.add_argument("--train_episodes", type=int, default=300)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--buffer_capacity", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_decay", type=float, default=0.995)
    parser.add_argument("--target_update_interval", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)

    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--results_dir", type=str, default="results")

    args = parser.parse_args()

    reseed_all(args.seed)

    run_tag = (
        f"{args.std_name}"
        f"_B{args.budget}"
        f"_F{args.initial_frontier_size}"
        f"_disc{args.discount}"
        f"_seed{args.seed}"
        f"_train{args.train_episodes}"
        f"_hid{args.hidden_dim}"
    )

    print("--------------------------------------------------------")
    print("Load ICPSR Graph Data")
    print("--------------------------------------------------------")

    graph_data = ICPSRGraphData(args.data_dir, args.std_name)
    print(f"  disease: {args.std_name}")
    print(f"  total nodes: {graph_data.graph.number_of_nodes()}")
    print(f"  total directed edges: {graph_data.digraph.number_of_edges()}")
    print(f"  total edge pairs: {len(graph_data.edge_pairs)}")

    train_pairs, test_pairs = graph_data.train_test_split(
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    print(f"  train pairs: {len(train_pairs)}")
    print(f"  test pairs: {len(test_pairs)}")

    print("--------------------------------------------------------")
    print("Load Diffusion Model")
    print("--------------------------------------------------------")

    covariate_model = DDPMCovariateModel.load(args.model_path)
    print(f"  loaded diffusion model from: {args.model_path}")
    print(f"  device: {covariate_model.device}")

    print("--------------------------------------------------------")
    print("Fit Count Model")
    print("--------------------------------------------------------")

    common_nodes = sorted(set(graph_data.covariates) & set(graph_data.node_degrees))
    covariates_array = np.array([graph_data.covariates[n] for n in common_nodes])
    degrees_array = np.array([graph_data.node_degrees[n] for n in common_nodes])

    count_model = GaussianCountModel(seed=args.seed)
    count_model.fit(covariates_array, degrees_array)
    print(f"  fitted GaussianCountModel on {len(common_nodes)} nodes")

    print("--------------------------------------------------------")
    print("Create Recruiting Environment")
    print("--------------------------------------------------------")

    env = RecruitingEnv(
        covariate_model=covariate_model,
        count_model=count_model,
        initial_budget=args.budget,
        discount_factor=args.discount,
        max_rounds=args.max_rounds,
        seed=args.seed,
    )
    print(f"  budget={args.budget}, max_rounds={args.max_rounds}, discount={args.discount}")

    def initial_frontier_fn():
        return graph_data.sample_initial_frontier(
            n=args.initial_frontier_size,
            seed=np.random.randint(0, 10**9),
        )

    cfg = DQNConfig(
        gamma=args.gamma,
        lr=args.lr,
        tau=args.tau,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        warmup_steps=args.warmup_steps,
        train_episodes=args.train_episodes,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
        target_update_interval=args.target_update_interval,
        hidden_dim=args.hidden_dim,
        max_grad_norm=args.max_grad_norm,
        covariate_dim=72,
    )

    print("--------------------------------------------------------")
    print("Train + Evaluate Budget-DQN on recruiting env")
    print("--------------------------------------------------------")

    t0 = time.time()

    rewards, learner, best_eval_reward, best_eval_episode = run_budget_dqn(
        env=env,
        initial_frontier_fn=initial_frontier_fn,
        budget_allocator=lambda state, k: greedy_allocator(state, k, env.count_model),
        n_episodes_eval=args.n_episodes_eval,
        seed=args.seed,
        cfg=cfg,
        log_every_n_episodes=args.log_every,
    )

    elapsed = time.time() - t0

    os.makedirs(args.results_dir, exist_ok=True)

    x, y, y_std, traj_rows, discounted_returns_eval, total_rewards_eval = evaluate_recruiting_curve(
        learner=learner,
        env=env,
        initial_frontier_fn=initial_frontier_fn,
        n_episodes_eval=args.n_episodes_eval,
        gamma=args.discount,
    )

    save_final_eval_results(
        x=x,
        y=y,
        y_std=y_std,
        traj_rows=traj_rows,
        results_dir=args.results_dir,
        run_tag=run_tag,
        gamma=args.discount,
    )

    rewards = np.asarray(rewards, dtype=float)
    if rewards.size > 0:
        mean_total_reward = float(np.mean(rewards))
        std_total_reward = float(np.std(rewards))
    else:
        mean_total_reward = 0.0
        std_total_reward = 0.0

    if discounted_returns_eval.size > 0:
        mean_disc = float(np.mean(discounted_returns_eval))
        std_disc = float(np.std(discounted_returns_eval))
    else:
        mean_disc = 0.0
        std_disc = 0.0

    print("--------------------------------------------------------")
    print("Results")
    print("--------------------------------------------------------")
    print(
        f"{args.std_name} recruiting | "
        f"budget={args.budget}, init_frontier={args.initial_frontier_size}, "
        f"discount={args.discount}"
    )
    print(
        f"  eval episodes: {args.n_episodes_eval}, "
        f"mean total recruits = {mean_total_reward:.4f}, std = {std_total_reward:.4f}"
    )
    print(
        f"  mean discounted return = {mean_disc:.4f}, std = {std_disc:.4f}"
    )
    if best_eval_episode > 0:
        print(
            f"  best eval reward: {best_eval_reward:.4f} "
            f"(achieved at training episode {best_eval_episode})"
        )
    print(f"  runtime: {elapsed:.2f} seconds")
    print("[done]")


if __name__ == "__main__":
    main()