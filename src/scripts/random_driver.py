"""
Evaluate the recruiting environment with a random allocation policy.

Usage:
    python -m src.scripts.random_driver \
        --model_path checkpoints/diffusion/ddpm_HIV.pt \
        --data_dir ICPSR_22140 \
        --std_name HIV \
        --budget 100 \
        --initial_frontier_size 10 \
        --n_episodes_eval 10 \
        --seed 42
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from src.data.icpsr_loader import ICPSRGraphData
from src.environment.recruiting_env import RecruitingEnv
from src.models.count_model.gaussian_count_model import GaussianCountModel
from src.models.covariate_model.ddpm_covariate_model import DDPMCovariateModel
from src.models.policy.random_policy import RandomPolicy
from src.scripts.eval_utils import evaluate_recruiting_curve, save_single_curve


def main():
    parser = argparse.ArgumentParser(description="Evaluate random policy on recruiting env")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained diffusion model checkpoint")
    parser.add_argument("--data_dir", type=str, default="ICPSR_22140")
    parser.add_argument("--std_name", type=str, default="HIV",
                        choices=["HIV", "Gonorrhea", "Chlamydia", "Syphilis", "Hepatitis"])
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--initial_frontier_size", type=int, default=10)
    parser.add_argument("--n_episodes_eval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--max_rounds", type=int, default=50)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--results_dir", type=str, default="results")

    args = parser.parse_args()

    run_tag = (
        f"random_{args.std_name}"
        f"_B{args.budget}"
        f"_F{args.initial_frontier_size}"
        f"_disc{args.discount}"
        f"_seed{args.seed}"
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

    print("--------------------------------------------------------")
    print("Evaluate Random Policy")
    print("--------------------------------------------------------")

    policy = RandomPolicy(seed=args.seed)
    t0 = time.time()

    x, y, y_std, traj_rows, discounted_returns, total_rewards = evaluate_recruiting_curve(
        policy_fn=policy.act,
        env=env,
        initial_frontier_fn=initial_frontier_fn,
        n_episodes_eval=args.n_episodes_eval,
        gamma=args.discount,
    )

    elapsed = time.time() - t0

    save_single_curve(
        x=x, y=y, y_std=y_std,
        traj_rows=traj_rows,
        results_dir=args.results_dir,
        run_tag=run_tag,
        gamma=args.discount,
        label="Random Policy",
    )

    print("--------------------------------------------------------")
    print("Results")
    print("--------------------------------------------------------")
    print(
        f"{args.std_name} recruiting (random) | "
        f"budget={args.budget}, init_frontier={args.initial_frontier_size}, "
        f"discount={args.discount}"
    )
    print(
        f"  eval episodes: {args.n_episodes_eval}, "
        f"mean total recruits = {np.mean(total_rewards):.4f}, "
        f"std = {np.std(total_rewards):.4f}"
    )
    print(
        f"  mean discounted return = {np.mean(discounted_returns):.4f}, "
        f"std = {np.std(discounted_returns):.4f}"
    )
    print(f"  runtime: {elapsed:.2f} seconds")
    print("[done]")


if __name__ == "__main__":
    main()
