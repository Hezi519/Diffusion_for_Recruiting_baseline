from __future__ import annotations

import argparse
import os
import time
import random
from dataclasses import dataclass
from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.icpsr_loader import ICPSRGraphData
from src.environment.recruiting_env import RecruitingEnv
from src.environment.state import RecruitingState
from src.models.count_model.gaussian_count_model import GaussianCountModel
from src.models.covariate_model.ddpm_covariate_model import DDPMCovariateModel
from src.models.RL_model.dqn_estimator import DQNConfig, run_budget_dqn
from src.models.RL_model.greedy_allocator import greedy_allocator
from src.models.RL_allocation_model.state_encoder import StateEncoder
from src.models.RL_allocation_model.q_network import ThreeHeadQNetwork
from src.models.RL_allocation_model.policy import StructuredValuePolicy
from src.models.RL_allocation_model.trainer import ValueTrainerConfig, StructuredQTrainer


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def reseed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class EvalBundle:
    method: str
    x: np.ndarray
    y: np.ndarray
    y_std: np.ndarray
    traj_rows: list[dict]
    discounted_returns: np.ndarray
    total_rewards: np.ndarray


class StructuredLearnerWrapper:
    """Light wrapper so evaluation code can treat both learners uniformly."""

    def __init__(self, policy: StructuredValuePolicy, trainer: StructuredQTrainer) -> None:
        self.policy = policy
        self.trainer = trainer


def make_initial_frontier_fn(
    graph_data: ICPSRGraphData,
    initial_frontier_size: int,
    base_seed: int,
) -> Callable[[], np.ndarray]:
    rng = np.random.default_rng(base_seed)

    def _fn() -> np.ndarray:
        return graph_data.sample_initial_frontier(
            n=initial_frontier_size,
            seed=int(rng.integers(1 << 31)),
        )

    return _fn


def get_eval_frontier(
    graph_data: ICPSRGraphData,
    initial_frontier_size: int,
    eval_seed: int,
) -> np.ndarray:
    return graph_data.sample_initial_frontier(
        n=initial_frontier_size,
        seed=eval_seed,
    )


# ------------------------------------------------------------------
# Training wrappers
# ------------------------------------------------------------------

def train_budget_dqn_baseline(
    env: RecruitingEnv,
    initial_frontier_fn: Callable[[], np.ndarray],
    n_episodes_eval: int,
    seed: int,
    cfg: DQNConfig,
    log_every_n_episodes: int,
):
    rewards, learner, best_eval_reward, best_eval_episode = run_budget_dqn(
        env=env,
        initial_frontier_fn=initial_frontier_fn,
        budget_allocator=lambda state, k: greedy_allocator(state, k, env.count_model),
        n_episodes_eval=n_episodes_eval,
        seed=seed,
        cfg=cfg,
        log_every_n_episodes=log_every_n_episodes,
    )
    return {
        "learner": learner,
        "rewards": np.asarray(rewards, dtype=float),
        "best_eval_reward": float(best_eval_reward),
        "best_eval_episode": int(best_eval_episode),
    }


def train_structured_allocation_baseline(
    env: RecruitingEnv,
    initial_frontier_fn: Callable[[], np.ndarray],
    seed: int,
    hidden_dim: int,
    max_k: int,
    cfg: ValueTrainerConfig,
    device: str,
):
    encoder = StateEncoder(covariate_dim=72, hidden_dim=hidden_dim)
    q_network = ThreeHeadQNetwork(
        state_dim=encoder.output_dim,
        covariate_dim=72,
        hidden_dim=hidden_dim,
        max_budget=env.initial_budget,
        max_k=max_k,
    )
    policy = StructuredValuePolicy(
        encoder=encoder,
        q_network=q_network,
        device=device,
        seed=seed,
    )
    trainer = StructuredQTrainer(
        env=env,
        policy=policy,
        initial_frontier_fn=initial_frontier_fn,
        count_model=env.count_model,
        covariate_model=env.covariate_model,
        cfg=cfg,
        device=device,
        seed=seed,
    )
    history = trainer.train()
    policy.eval()
    return {
        "learner": StructuredLearnerWrapper(policy=policy, trainer=trainer),
        "history": history,
    }


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate_recruiting_curve(
    method: Literal["dqn", "structured"],
    learner,
    env: RecruitingEnv,
    graph_data: ICPSRGraphData,
    initial_frontier_size: int,
    n_episodes_eval: int,
    gamma: float,
    eval_seed_base: int = 100000,
) -> EvalBundle:
    """
    Evaluate a trained policy and produce the normalized recruiting curve.

    x-axis: fraction of budget spent
    y-axis: fraction of total recruits obtained within that episode
    """
    horizon = env.max_rounds
    initial_budget = env.initial_budget

    x_mat = []
    y_mat = []
    discounted_returns = []
    total_rewards = []
    traj_rows: list[dict] = []

    for ep in range(n_episodes_eval):
        frontier_seed = eval_seed_base + ep
        env_seed = eval_seed_base + 10_000 + ep
        state = env.reset(
            get_eval_frontier(graph_data, initial_frontier_size, frontier_seed),
            seed=env_seed,
        )

        cumulative_budget_spent = 0.0
        cumulative_recruits = 0.0
        rewards_this_ep = []
        x_ep = []
        y_ep_raw = []

        while True:
            if method == "dqn":
                action_budget = learner.select_action(state, greedy=True)
                action_vec = learner.budget_allocator(state, action_budget)
                action_k = None
            else:
                step = learner.policy.act_greedy(state)
                action_budget = int(step.budget)
                action_k = int(step.k)
                action_vec = step.allocation

            next_state, reward, done, info = env.step(action_vec)

            rewards_this_ep.append(float(reward))
            cumulative_budget_spent += float(info["budget_spent"])
            cumulative_recruits += float(reward)

            x_ep.append(cumulative_budget_spent / max(float(initial_budget), 1.0))
            y_ep_raw.append(cumulative_recruits)

            row = {
                "method": method,
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
            if action_k is not None:
                row["action_k"] = action_k
                row["allocation_nonzero"] = int(np.count_nonzero(action_vec))
            traj_rows.append(row)

            state = next_state
            if done:
                break

        total_final = max(cumulative_recruits, 1.0)
        y_ep = [v / total_final for v in y_ep_raw]

        while len(x_ep) < horizon:
            x_ep.append(x_ep[-1] if x_ep else 0.0)
            y_ep.append(y_ep[-1] if y_ep else 0.0)

        x_ep = np.asarray(x_ep[:horizon], dtype=np.float32)
        y_ep = np.asarray(y_ep[:horizon], dtype=np.float32)

        x_mat.append(x_ep)
        y_mat.append(y_ep)

        discounts = gamma ** np.arange(len(rewards_this_ep))
        discounted_returns.append(float(np.sum(np.asarray(rewards_this_ep) * discounts)))
        total_rewards.append(float(np.sum(rewards_this_ep)))

    x = np.mean(np.asarray(x_mat, dtype=np.float32), axis=0)
    y = np.mean(np.asarray(y_mat, dtype=np.float32), axis=0)
    y_std = np.std(np.asarray(y_mat, dtype=np.float32), axis=0)

    return EvalBundle(
        method=method,
        x=x,
        y=y,
        y_std=y_std,
        traj_rows=traj_rows,
        discounted_returns=np.asarray(discounted_returns, dtype=float),
        total_rewards=np.asarray(total_rewards, dtype=float),
    )


# ------------------------------------------------------------------
# Saving / plotting
# ------------------------------------------------------------------

def _method_label(method: str) -> str:
    if method == "dqn":
        return "Budget-DQN + GreedyAlloc"
    if method == "structured":
        return "Three-Head Structured RL"
    return method


def save_eval_results(
    bundles: dict[str, EvalBundle],
    results_dir: str,
    run_tag: str,
    gamma: float,
) -> None:
    os.makedirs(results_dir, exist_ok=True)

    # Save per-method arrays and trajectories
    for method, bundle in bundles.items():
        npz_path = os.path.join(results_dir, f"eval_results_{run_tag}_{method}.npz")
        np.savez(npz_path, x=bundle.x, y=bundle.y, y_std=bundle.y_std)
        print(f"Saved eval vectors to: {npz_path}")

        traj_path = os.path.join(results_dir, f"trajectories_{run_tag}_{method}.csv")
        pd.DataFrame(bundle.traj_rows).to_csv(traj_path, index=False)
        print(f"Saved trajectories to: {traj_path}")

    # Save combined trajectories too
    combined_rows = []
    for bundle in bundles.values():
        combined_rows.extend(bundle.traj_rows)
    combined_path = os.path.join(results_dir, f"trajectories_{run_tag}_all_methods.csv")
    pd.DataFrame(combined_rows).to_csv(combined_path, index=False)
    print(f"Saved combined trajectories to: {combined_path}")

    # Plot single or multiple curves on one figure
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


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/evaluate recruiting env with Budget-DQN, structured RL, or both"
    )

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained diffusion model checkpoint")
    parser.add_argument("--data_dir", type=str, default="ICPSR_22140")
    parser.add_argument("--std_name", type=str, default="HIV",
                        choices=["HIV", "Gonorrhea", "Chlamydia", "Syphilis", "Hepatitis"])
    parser.add_argument("--method", type=str, default="dqn",
                        choices=["dqn", "structured", "both"],
                        help="Which RL baseline(s) to run")

    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--initial_frontier_size", type=int, default=10)
    parser.add_argument("--n_episodes_eval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # env
    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--max_rounds", type=int, default=50)

    # data split / count model
    parser.add_argument("--test_fraction", type=float, default=0.2)

    # shared-ish model size
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--log_every", type=int, default=10)

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
    parser.add_argument("--max_grad_norm", type=float, default=10.0)

    # structured RL hyperparams
    parser.add_argument("--structured_train_episodes", type=int, default=None,
                        help="Defaults to --train_episodes if omitted")
    parser.add_argument("--structured_lr", type=float, default=None,
                        help="Defaults to --lr if omitted")
    parser.add_argument("--structured_gamma", type=float, default=None,
                        help="Defaults to --gamma if omitted")
    parser.add_argument("--max_k", type=int, default=None,
                        help="Defaults to min(budget, initial_frontier_size)")
    parser.add_argument("--epsilon_budget_start", type=float, default=0.20)
    parser.add_argument("--epsilon_budget_end", type=float, default=0.05)
    parser.add_argument("--epsilon_k_start", type=float, default=0.20)
    parser.add_argument("--epsilon_k_end", type=float, default=0.05)
    parser.add_argument("--score_noise_start", type=float, default=0.20)
    parser.add_argument("--score_noise_end", type=float, default=0.02)
    parser.add_argument("--node_score_loss_weight", type=float, default=1.0)
    parser.add_argument("--structured_buffer_capacity", type=int, default=20000)
    parser.add_argument("--structured_batch_size", type=int, default=32)
    parser.add_argument("--structured_min_buffer_size", type=int, default=128)
    parser.add_argument("--updates_per_env_step", type=int, default=1)
    parser.add_argument("--structured_target_update_interval", type=int, default=200)
    parser.add_argument("--node_target_num_samples", type=int, default=1)

    args = parser.parse_args()
    reseed_all(args.seed)

    structured_train_episodes = args.structured_train_episodes or args.train_episodes
    structured_lr = args.structured_lr if args.structured_lr is not None else args.lr
    structured_gamma = args.structured_gamma if args.structured_gamma is not None else args.gamma
    max_k = args.max_k if args.max_k is not None else min(args.budget, args.initial_frontier_size)

    run_tag = (
        f"{args.std_name}"
        f"_B{args.budget}"
        f"_F{args.initial_frontier_size}"
        f"_disc{args.discount}"
        f"_seed{args.seed}"
        f"_method{args.method}"
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

    def build_env(seed_offset: int) -> RecruitingEnv:
        return RecruitingEnv(
            covariate_model=covariate_model,
            count_model=count_model,
            initial_budget=args.budget,
            discount_factor=args.discount,
            max_rounds=args.max_rounds,
            seed=args.seed + seed_offset,
        )

    dqn_cfg = DQNConfig(
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

    structured_cfg = ValueTrainerConfig(
        gamma=structured_gamma,
        lr=structured_lr,
        train_episodes=structured_train_episodes,
        max_steps_per_episode=args.max_rounds,
        epsilon_budget_start=args.epsilon_budget_start,
        epsilon_budget_end=args.epsilon_budget_end,
        epsilon_k_start=args.epsilon_k_start,
        epsilon_k_end=args.epsilon_k_end,
        score_noise_start=args.score_noise_start,
        score_noise_end=args.score_noise_end,
        node_score_loss_weight=args.node_score_loss_weight,
        replay_buffer_capacity=args.structured_buffer_capacity,
        batch_size=args.structured_batch_size,
        min_buffer_size=args.structured_min_buffer_size,
        updates_per_env_step=args.updates_per_env_step,
        target_update_interval=args.structured_target_update_interval,
        node_target_num_samples=args.node_target_num_samples,
    )

    print("--------------------------------------------------------")
    print("Train + Evaluate")
    print("--------------------------------------------------------")

    results: dict[str, dict] = {}
    eval_bundles: dict[str, EvalBundle] = {}

    if args.method in {"dqn", "both"}:
        print("[Run] Budget-DQN + GreedyAlloc")
        env_dqn = build_env(seed_offset=0)
        train_frontier_fn_dqn = make_initial_frontier_fn(
            graph_data=graph_data,
            initial_frontier_size=args.initial_frontier_size,
            base_seed=args.seed + 101,
        )
        t0 = time.time()
        results["dqn"] = train_budget_dqn_baseline(
            env=env_dqn,
            initial_frontier_fn=train_frontier_fn_dqn,
            n_episodes_eval=args.n_episodes_eval,
            seed=args.seed,
            cfg=dqn_cfg,
            log_every_n_episodes=args.log_every,
        )
        results["dqn"]["elapsed"] = time.time() - t0

        eval_env_dqn = build_env(seed_offset=1000)
        eval_bundles["dqn"] = evaluate_recruiting_curve(
            method="dqn",
            learner=results["dqn"]["learner"],
            env=eval_env_dqn,
            graph_data=graph_data,
            initial_frontier_size=args.initial_frontier_size,
            n_episodes_eval=args.n_episodes_eval,
            gamma=args.discount,
        )

    if args.method in {"structured", "both"}:
        print("[Run] Three-Head Structured RL")
        env_structured = build_env(seed_offset=2000)
        train_frontier_fn_structured = make_initial_frontier_fn(
            graph_data=graph_data,
            initial_frontier_size=args.initial_frontier_size,
            base_seed=args.seed + 202,
        )
        t0 = time.time()
        results["structured"] = train_structured_allocation_baseline(
            env=env_structured,
            initial_frontier_fn=train_frontier_fn_structured,
            seed=args.seed,
            hidden_dim=args.hidden_dim,
            max_k=max_k,
            cfg=structured_cfg,
            device=args.device,
        )
        results["structured"]["elapsed"] = time.time() - t0

        eval_env_structured = build_env(seed_offset=3000)
        eval_bundles["structured"] = evaluate_recruiting_curve(
            method="structured",
            learner=results["structured"]["learner"],
            env=eval_env_structured,
            graph_data=graph_data,
            initial_frontier_size=args.initial_frontier_size,
            n_episodes_eval=args.n_episodes_eval,
            gamma=args.discount,
        )

    os.makedirs(args.results_dir, exist_ok=True)
    save_eval_results(
        bundles=eval_bundles,
        results_dir=args.results_dir,
        run_tag=run_tag,
        gamma=args.discount,
    )

    print("--------------------------------------------------------")
    print("Results")
    print("--------------------------------------------------------")
    print(
        f"{args.std_name} recruiting | budget={args.budget}, "
        f"init_frontier={args.initial_frontier_size}, discount={args.discount}"
    )

    for method, bundle in eval_bundles.items():
        mean_total_reward = float(np.mean(bundle.total_rewards)) if bundle.total_rewards.size > 0 else 0.0
        std_total_reward = float(np.std(bundle.total_rewards)) if bundle.total_rewards.size > 0 else 0.0
        mean_disc = float(np.mean(bundle.discounted_returns)) if bundle.discounted_returns.size > 0 else 0.0
        std_disc = float(np.std(bundle.discounted_returns)) if bundle.discounted_returns.size > 0 else 0.0

        print(f"  [{_method_label(method)}]")
        print(
            f"    eval episodes: {args.n_episodes_eval}, "
            f"mean total recruits = {mean_total_reward:.4f}, std = {std_total_reward:.4f}"
        )
        print(
            f"    mean discounted return = {mean_disc:.4f}, std = {std_disc:.4f}"
        )
        print(f"    runtime: {results[method]['elapsed']:.2f} seconds")

        if method == "dqn":
            best_eval_episode = results[method].get("best_eval_episode", 0)
            if best_eval_episode > 0:
                print(
                    f"    best eval reward: {results[method]['best_eval_reward']:.4f} "
                    f"(achieved at training episode {best_eval_episode})"
                )
        else:
            history = results[method].get("history", [])
            if history:
                last = history[-1]
                print(
                    f"    final training return = {last['episode_return']:.4f}, "
                    f"final avg loss = {last['avg_loss']:.4f}"
                )

    print("[done]")


if __name__ == "__main__":
    main()
