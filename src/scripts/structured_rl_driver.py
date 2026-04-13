"""
Train + evaluate the recruiting environment with Three-Head Structured RL.

Mirrors recruiting_driver.py in output shape:
- prints run summary
- saves evaluation vectors (.npz)
- saves trajectories (.csv)
- saves a final curve plot (.png)

Usage:
    python -m src.scripts.structured_rl_driver \
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
import random
import time

import numpy as np
import torch

from src.data.icpsr_loader import ICPSRGraphData
from src.environment.recruiting_env import RecruitingEnv
from src.models.count_model.gaussian_count_model import GaussianCountModel
from src.models.covariate_model.ddpm_covariate_model import DDPMCovariateModel
from src.models.RL_allocation_model.policy import StructuredValuePolicy
from src.models.RL_allocation_model.q_network import ThreeHeadQNetwork
from src.models.RL_allocation_model.state_encoder import StateEncoder
from src.models.RL_allocation_model.trainer import StructuredQTrainer, ValueTrainerConfig
from src.scripts.eval_utils import evaluate_recruiting_curve, save_single_curve


def reseed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(
        description="Train + evaluate Three-Head Structured RL on recruiting env"
    )

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained diffusion model checkpoint")
    parser.add_argument("--data_dir", type=str, default="ICPSR_22140")
    parser.add_argument("--std_name", type=str, default="HIV",
                        choices=["HIV", "Gonorrhea", "Chlamydia", "Syphilis", "Hepatitis"])
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--initial_frontier_size", type=int, default=10)
    parser.add_argument("--n_episodes_eval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    # env
    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--max_rounds", type=int, default=50)

    # data split / count model
    parser.add_argument("--test_fraction", type=float, default=0.2)

    # model size
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--max_k", type=int, default=None,
                        help="Defaults to min(budget, initial_frontier_size)")

    # structured RL hyperparams
    parser.add_argument("--train_episodes", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon_budget_start", type=float, default=0.20)
    parser.add_argument("--epsilon_budget_end", type=float, default=0.05)
    parser.add_argument("--epsilon_k_start", type=float, default=0.20)
    parser.add_argument("--epsilon_k_end", type=float, default=0.05)
    parser.add_argument("--score_noise_start", type=float, default=0.20)
    parser.add_argument("--score_noise_end", type=float, default=0.02)
    parser.add_argument("--node_score_loss_weight", type=float, default=1.0)
    parser.add_argument("--buffer_capacity", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--min_buffer_size", type=int, default=128)
    parser.add_argument("--updates_per_env_step", type=int, default=1)
    parser.add_argument("--target_update_interval", type=int, default=200)
    parser.add_argument("--node_target_num_samples", type=int, default=1)

    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--results_dir", type=str, default="results")

    args = parser.parse_args()
    reseed_all(args.seed)

    max_k = args.max_k if args.max_k is not None else min(args.budget, args.initial_frontier_size)

    run_tag = (
        f"{args.std_name}"
        f"_B{args.budget}"
        f"_F{args.initial_frontier_size}"
        f"_disc{args.discount}"
        f"_seed{args.seed}"
        f"_train{args.train_episodes}"
        f"_hid{args.hidden_dim}"
        f"_structured"
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
    covariate_model = DDPMCovariateModel.load(args.model_path, device=args.device)
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

    cfg = ValueTrainerConfig(
        gamma=args.gamma,
        lr=args.lr,
        train_episodes=args.train_episodes,
        max_steps_per_episode=args.max_rounds,
        epsilon_budget_start=args.epsilon_budget_start,
        epsilon_budget_end=args.epsilon_budget_end,
        epsilon_k_start=args.epsilon_k_start,
        epsilon_k_end=args.epsilon_k_end,
        score_noise_start=args.score_noise_start,
        score_noise_end=args.score_noise_end,
        node_score_loss_weight=args.node_score_loss_weight,
        replay_buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        min_buffer_size=args.min_buffer_size,
        updates_per_env_step=args.updates_per_env_step,
        target_update_interval=args.target_update_interval,
        node_target_num_samples=args.node_target_num_samples,
    )

    print("--------------------------------------------------------")
    print("Train + Evaluate Structured RL on recruiting env")
    print("--------------------------------------------------------")

    encoder = StateEncoder(covariate_dim=72, hidden_dim=args.hidden_dim)
    q_network = ThreeHeadQNetwork(
        state_dim=encoder.output_dim,
        covariate_dim=72,
        hidden_dim=args.hidden_dim,
        max_budget=env.initial_budget,
        max_k=max_k,
    )
    policy = StructuredValuePolicy(
        encoder=encoder,
        q_network=q_network,
        device=args.device,
        seed=args.seed,
    )

    best_reward_so_far = [-1.0]

    def on_eval_log(policy_obj, episode, reward):
        def policy_fn(state):
            return policy_obj.act_greedy(state).allocation

        x, y, y_std, traj_rows, _, _ = evaluate_recruiting_curve(
            policy_fn=policy_fn,
            env=env,
            initial_frontier_fn=initial_frontier_fn,
            n_episodes_eval=args.n_episodes_eval,
            gamma=args.discount,
        )

        periodic_tag = f"{run_tag}_ep{episode}"
        save_single_curve(
            x, y, y_std, traj_rows,
            args.results_dir, periodic_tag, args.discount,
            label=f"Structured RL (ep {episode}, reward={reward:.1f})",
        )

        is_new_best = reward > best_reward_so_far[0]
        if is_new_best:
            best_reward_so_far[0] = reward
            best_tag = f"{run_tag}_best_ep{episode}"
            save_single_curve(
                x, y, y_std, traj_rows,
                args.results_dir, best_tag, args.discount,
                label=f"Structured RL best (ep {episode}, reward={reward:.1f})",
            )
            print(f"  [new best] ep={episode}, mean_reward={reward:.1f}")
        else:
            print(f"  [eval] ep={episode}, mean_reward={reward:.1f}")

    trainer = StructuredQTrainer(
        env=env,
        policy=policy,
        initial_frontier_fn=initial_frontier_fn,
        count_model=env.count_model,
        covariate_model=env.covariate_model,
        cfg=cfg,
        device=args.device,
        seed=args.seed,
        on_eval_log=on_eval_log,
        n_episodes_eval=args.n_episodes_eval,
        log_every_n_episodes=args.log_every,
    )

    t0 = time.time()
    train_result = trainer.train()
    history = train_result["history"]
    best_eval_reward = train_result["best_eval_reward"]
    best_eval_episode = train_result["best_eval_episode"]
    policy.eval()
    elapsed = time.time() - t0

    os.makedirs(args.results_dir, exist_ok=True)

    def structured_policy_fn(state):
        return policy.act_greedy(state).allocation

    x, y, y_std, traj_rows, discounted_returns_eval, total_rewards_eval = evaluate_recruiting_curve(
        policy_fn=structured_policy_fn,
        env=env,
        initial_frontier_fn=initial_frontier_fn,
        n_episodes_eval=args.n_episodes_eval,
        gamma=args.discount,
    )

    save_single_curve(
        x=x,
        y=y,
        y_std=y_std,
        traj_rows=traj_rows,
        results_dir=args.results_dir,
        run_tag=run_tag,
        gamma=args.discount,
        label="Three-Head Structured RL",
    )

    if total_rewards_eval.size > 0:
        mean_total_reward = float(np.mean(total_rewards_eval))
        std_total_reward = float(np.std(total_rewards_eval))
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
        best_loss = history[best_eval_episode - 1]["avg_loss"]
        print(
            f"  best eval reward = {best_eval_reward:.4f} "
            f"at episode {best_eval_episode}, avg loss at that episode = {best_loss:.4f}"
        )
    if history:
        last = history[-1]
        print(
            f"  final training return = {last['episode_return']:.4f}, "
            f"final avg loss = {last['avg_loss']:.4f}"
        )
    print(f"  runtime: {elapsed:.2f} seconds")
    print("[done]")


if __name__ == "__main__":
    main()
