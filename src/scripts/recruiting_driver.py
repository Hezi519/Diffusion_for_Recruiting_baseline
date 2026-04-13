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
    python -m src.scripts.recruiting_driver \
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

import numpy as np
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
from src.scripts.eval_utils import evaluate_recruiting_curve, save_final_eval_results


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

    frontier_rng = np.random.default_rng(args.seed)

    def initial_frontier_fn():
        return graph_data.sample_initial_frontier(
            n=args.initial_frontier_size,
            seed=int(frontier_rng.integers(1 << 31)),
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

    def on_new_best(learner, episode, reward):
        def policy_fn(state):
            budget = learner.select_action(state, greedy=True)
            return learner.budget_allocator(state, budget)

        tag = f"{run_tag}_best_ep{episode}"
        x, y, y_std, traj_rows, _, _ = evaluate_recruiting_curve(
            policy_fn=policy_fn,
            env=env,
            initial_frontier_fn=initial_frontier_fn,
            n_episodes_eval=args.n_episodes_eval,
            gamma=args.discount,
        )
        save_final_eval_results(
            x, y, y_std, traj_rows,
            args.results_dir, tag, args.discount,
            label=f"DQN (ep {episode}, reward={reward:.1f})",
        )
        print(f"  [new best] ep={episode}, mean_reward={reward:.1f}")

    t0 = time.time()

    rewards, learner, best_eval_reward, best_eval_episode = run_budget_dqn(
        env=env,
        initial_frontier_fn=initial_frontier_fn,
        budget_allocator=lambda state, k: greedy_allocator(state, k, env.count_model),
        n_episodes_eval=args.n_episodes_eval,
        seed=args.seed,
        cfg=cfg,
        log_every_n_episodes=args.log_every,
        on_new_best=on_new_best,
    )

    elapsed = time.time() - t0

    os.makedirs(args.results_dir, exist_ok=True)

    def dqn_policy_fn(state):
        budget = learner.select_action(state, greedy=True)
        return learner.budget_allocator(state, budget)

    x, y, y_std, traj_rows, discounted_returns_eval, total_rewards_eval = evaluate_recruiting_curve(
        policy_fn=dqn_policy_fn,
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
        label="Budget-DQN + GreedyAlloc",
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