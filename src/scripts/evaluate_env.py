"""
Smoke test the recruiting environment with simple policies.

Usage:
    python -m src.scripts.evaluate_env \
        --model_path checkpoints/diffusion/ddpm_HIV.pt \
        --data_dir ICPSR_22140 \
        --std_name HIV \
        --budget 100 \
        --num_episodes 10
"""

import argparse

import numpy as np

from src.data.icpsr_loader import ICPSRGraphData
from src.environment.recruiting_env import RecruitingEnv
from src.models.count_model.gaussian_count_model import GaussianCountModel
from src.models.covariate_model.ddpm_covariate_model import DDPMCovariateModel


def constant_policy(frontier_size: int, budget_remaining: int, const_val: int = 5) -> np.ndarray:
    """Allocate a constant amount to each frontier member, respecting budget."""
    action = np.full(frontier_size, const_val, dtype=int)
    total = action.sum()
    if total > budget_remaining:
        # Scale down proportionally
        action = np.full(frontier_size, budget_remaining // max(frontier_size, 1), dtype=int)
        # Distribute remainder
        remainder = budget_remaining - action.sum()
        for i in range(int(remainder)):
            action[i] += 1
    return action


def random_policy(frontier_size: int, budget_remaining: int, rng: np.random.Generator) -> np.ndarray:
    """Randomly allocate budget across frontier members."""
    if frontier_size == 0:
        return np.array([], dtype=int)
    max_per_person = max(1, budget_remaining // frontier_size)
    action = rng.integers(0, max_per_person + 1, size=frontier_size)
    # Ensure we don't exceed budget
    while action.sum() > budget_remaining:
        idx = rng.integers(0, frontier_size)
        if action[idx] > 0:
            action[idx] -= 1
    return action


def run_episode(env: RecruitingEnv, policy: str, initial_frontier: np.ndarray,
                seed: int, const_val: int = 5) -> dict:
    """Run a single episode and return statistics."""
    rng = np.random.default_rng(seed)
    state = env.reset(initial_frontier, seed=seed)
    trajectory = [{
        "round": 0,
        "frontier_size": state.frontier_size,
        "budget_remaining": state.budget_remaining,
    }]

    total_reward = 0.0
    while True:
        if state.frontier_size == 0 or state.budget_remaining <= 0:
            break

        if policy == "constant":
            action = constant_policy(state.frontier_size, state.budget_remaining, const_val)
        elif policy == "random":
            action = random_policy(state.frontier_size, state.budget_remaining, rng)
        else:
            raise ValueError(f"Unknown policy: {policy}")

        state, reward, done, info = env.step(action)
        total_reward += reward

        trajectory.append({
            "round": info["round"],
            "frontier_size": state.frontier_size,
            "budget_remaining": state.budget_remaining,
            "reward": reward,
            "total_recruits": info["total_recruits"],
            "budget_spent": info["budget_spent"],
            "termination_reason": info["termination_reason"],
        })

        if done:
            break

    return {
        "total_reward": total_reward,
        "discounted_reward": env.cumulative_discounted_reward,
        "num_rounds": len(trajectory) - 1,
        "trajectory": trajectory,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate recruiting environment")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained diffusion model checkpoint")
    parser.add_argument("--data_dir", type=str, default="ICPSR_22140")
    parser.add_argument("--std_name", type=str, default="HIV")
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--initial_frontier_size", type=int, default=10)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--policy", type=str, default="constant",
                        choices=["constant", "random"])
    parser.add_argument("--const_val", type=int, default=5)
    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--max_rounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load data for initial frontiers and count model training
    print(f"Loading data from {args.data_dir}...")
    graph_data = ICPSRGraphData(args.data_dir, args.std_name)

    # Load diffusion model
    print(f"Loading diffusion model from {args.model_path}...")
    covariate_model = DDPMCovariateModel.load(args.model_path)

    # Fit count model on graph degrees
    print("Fitting count model...")
    covariates_array = np.array([graph_data.covariates[n] for n in sorted(graph_data.covariates)])
    degrees_array = np.array([graph_data.node_degrees[n] for n in sorted(graph_data.node_degrees)])
    count_model = GaussianCountModel(seed=args.seed)
    count_model.fit(covariates_array, degrees_array)

    # Create environment
    env = RecruitingEnv(
        covariate_model=covariate_model,
        count_model=count_model,
        initial_budget=args.budget,
        discount_factor=args.discount,
        max_rounds=args.max_rounds,
        seed=args.seed,
    )

    # Run episodes
    print(f"\nRunning {args.num_episodes} episodes with {args.policy} policy...")
    rewards = []
    rounds_list = []

    for ep in range(args.num_episodes):
        initial_frontier = graph_data.sample_initial_frontier(
            args.initial_frontier_size, seed=args.seed + ep
        )
        result = run_episode(
            env, args.policy, initial_frontier,
            seed=args.seed + ep, const_val=args.const_val,
        )
        rewards.append(result["total_reward"])
        rounds_list.append(result["num_rounds"])

        last = result["trajectory"][-1]
        print(f"  Episode {ep}: reward={result['total_reward']:.0f}, "
              f"rounds={result['num_rounds']}, "
              f"termination={last.get('termination_reason', 'N/A')}")

    print(f"\nSummary ({args.num_episodes} episodes):")
    print(f"  Mean reward:  {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
    print(f"  Mean rounds:  {np.mean(rounds_list):.1f} +/- {np.std(rounds_list):.1f}")


if __name__ == "__main__":
    main()
