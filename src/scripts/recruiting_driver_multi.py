from __future__ import annotations

import argparse
import time
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from src.data.icpsr_loader import ICPSRGraphData
from src.environment.recruiting_env import RecruitingEnv
from src.models.count_model.gaussian_count_model import GaussianCountModel
from src.models.covariate_model.ddpm_covariate_model import DDPMCovariateModel
from src.models.RL_model.dqn_estimator import DQNConfig, run_budget_dqn
from src.models.RL_model.greedy_allocator import greedy_allocator
from src.models.RL_allocation_model.state_encoder import StateEncoder
from src.models.RL_allocation_model.q_network import ThreeHeadQNetwork
from src.models.RL_allocation_model.policy import StructuredValuePolicy
from src.models.RL_allocation_model.trainer import ValueTrainerConfig, StructuredQTrainer
from src.scripts.eval_utils import evaluate_recruiting_curve, save_comparison_curves


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
    """Thin wrapper around the tuple returned by eval_utils.evaluate_recruiting_curve."""
    method: str
    x: np.ndarray
    y: np.ndarray
    y_std: np.ndarray
    traj_rows: list[dict]
    discounted_returns: np.ndarray
    total_rewards: np.ndarray


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
    on_new_best=None,
):
    rewards, learner, best_eval_reward, best_eval_episode = run_budget_dqn(
        env=env,
        initial_frontier_fn=initial_frontier_fn,
        budget_allocator=lambda state, k: greedy_allocator(state, k, env.count_model),
        n_episodes_eval=n_episodes_eval,
        seed=seed,
        cfg=cfg,
        log_every_n_episodes=log_every_n_episodes,
        on_new_best=on_new_best,
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
    # TODO: plumb on_new_best through StructuredQTrainer for parity with run_budget_dqn.
    return {
        "policy": policy,
        "trainer": trainer,
        "history": history,
    }


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

    eval_frontier_fn = make_initial_frontier_fn(
        graph_data=graph_data,
        initial_frontier_size=args.initial_frontier_size,
        base_seed=args.seed + 500,
    )

    def _run_eval(method: str, policy_fn, env_eval: RecruitingEnv) -> EvalBundle:
        (x, y, y_std, traj_rows,
         discounted_returns, total_rewards) = evaluate_recruiting_curve(
            policy_fn=policy_fn,
            env=env_eval,
            initial_frontier_fn=eval_frontier_fn,
            n_episodes_eval=args.n_episodes_eval,
            gamma=args.discount,
            normalize=True,
        )
        return EvalBundle(
            method=method,
            x=x,
            y=y,
            y_std=y_std,
            traj_rows=traj_rows,
            discounted_returns=discounted_returns,
            total_rewards=total_rewards,
        )

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

        dqn_learner = results["dqn"]["learner"]

        def dqn_policy_fn(state):
            return dqn_learner.budget_allocator(
                state,
                dqn_learner.select_action(state, greedy=True),
            )

        eval_env_dqn = build_env(seed_offset=1000)
        eval_bundles["dqn"] = _run_eval("dqn", dqn_policy_fn, eval_env_dqn)

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

        structured_policy = results["structured"]["policy"]

        def structured_policy_fn(state):
            return structured_policy.act_greedy(state).allocation

        eval_env_structured = build_env(seed_offset=3000)
        eval_bundles["structured"] = _run_eval(
            "structured", structured_policy_fn, eval_env_structured
        )

    save_comparison_curves(
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

    method_labels = {
        "dqn": "Budget-DQN + GreedyAlloc",
        "structured": "Three-Head Structured RL",
    }

    for method, bundle in eval_bundles.items():
        mean_total_reward = float(np.mean(bundle.total_rewards)) if bundle.total_rewards.size > 0 else 0.0
        std_total_reward = float(np.std(bundle.total_rewards)) if bundle.total_rewards.size > 0 else 0.0
        mean_disc = float(np.mean(bundle.discounted_returns)) if bundle.discounted_returns.size > 0 else 0.0
        std_disc = float(np.std(bundle.discounted_returns)) if bundle.discounted_returns.size > 0 else 0.0

        print(f"  [{method_labels.get(method, method)}]")
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
