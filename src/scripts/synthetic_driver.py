"""
Self-contained synthetic benchmark driver.

Runs recruiting strategies (Random, Budget-DQN, Structured RL, GFP)
on a fully synthetic environment — no external dataset or pretrained
diffusion model required.  The environment models are oracle parametric:

    count model   : SyntheticCountModel  (Poisson, no fitting)
    covariate model: SyntheticCovariateModel (categorical inheritance, no training)
    initial frontier: SyntheticGraphData (random one-hot covariate pool)

This makes the experiment runnable in minutes on CPU. GFP can either train
its own synthetic q_psi/G_theta/L_eta modules from simulator-generated data,
or use oracle adapters for planning ablations.

This driver now also supports constant-k and adaptive-surrogate baselines that
span the same oracle->diffusion->planning->testing workflow as the other
methods.

Usage:
    python -m src.scripts.synthetic_driver \\
        --budget 50 --initial_frontier_size 5 \\
        --train_episodes 100 --seed 42 \\
        --methods random,dqn,structured,gfp

    # Tunnel-vision experiment (Structured RL > Random > DQN)
    python -m src.scripts.synthetic_driver \\
        --env_type tunnel_vision \\
        --budget 50 --initial_frontier_size 10 \\
        --train_episodes 500 --seed 42 \\
        --methods random,dqn,structured

Output:
    results/synthetic/  — per-method NPZ + CSV + comparison PNG
"""

from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch

from src.data.synthetic_generator import SyntheticGraphData
from src.data.tunnel_vision_generator import TunnelVisionGraphData
from src.environment.recruiting_env import RecruitingEnv
from src.models.count_model.synthetic_count_model import SyntheticCountModel
from src.models.count_model.tunnel_vision_count_model import TunnelVisionCountModel
from src.models.covariate_model.synthetic_covariate_model import SyntheticCovariateModel
from src.models.covariate_model.ddpm_covariate_model import DDPMCovariateModel
from src.models.covariate_model.tunnel_vision_covariate_model import TunnelVisionCovariateModel
from src.models.policy.random_policy import RandomPolicy
from src.scripts.eval_utils import (
    evaluate_recruiting_curve,
    save_comparison_curves,
    save_four_panel_curve_outputs,
)

# Budget-DQN
from src.models.RL_model.dqn_estimator import DQNConfig, run_budget_dqn
from src.models.RL_model.greedy_allocator import greedy_allocator
from src.models.RL_model.tunnel_vision_greedy_allocator import type_a_only_allocator

# Structured RL
from src.models.RL_allocation_model.policy import StructuredValuePolicy
from src.models.RL_allocation_model.q_network import ThreeHeadQNetwork
from src.models.RL_allocation_model.state_encoder import StateEncoder
from src.models.RL_allocation_model.trainer import StructuredQTrainer, ValueTrainerConfig

# Generative Frontier Planning
from src.models.GFP_model import (
    AmortizedLaplaceProvider,
    CensoredCountFitConfig,
    FrontierValueSurrogate,
    GFPPlanner,
    GFPTrainer,
    GFPTrainerConfig,
    LearnedCategoricalOffspringModel,
    LearnedCensoredPoissonSurvival,
    MonteCarloLaplaceProvider,
    OffspringFitConfig,
    PoissonCountSurvival,
    make_censored_count_dataset,
    make_offspring_dataset,
)
from src.models.adaptive_surrogate import (
    AdaptiveSurrogatePolicy,
    PoissonCountDistributionAdapter,
    precompute_surrogate_from_population_pmf,
)


@dataclass
class EvalBundle:
    x: np.ndarray
    y: np.ndarray
    y_std: np.ndarray
    traj_rows: list


def reseed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_planning_env(
    count_model,
    covariate_model,
    args,
) -> RecruitingEnv:
    return RecruitingEnv(
        covariate_model=covariate_model,
        count_model=count_model,
        initial_budget=args.budget,
        discount_factor=args.discount,
        max_rounds=args.max_rounds,
        seed=args.seed,
    )


def build_testing_env(
    oracle_count_model,
    oracle_covariate_model,
    args,
) -> RecruitingEnv:
    return RecruitingEnv(
        covariate_model=oracle_covariate_model,
        count_model=oracle_count_model,
        initial_budget=args.budget,
        discount_factor=args.discount,
        max_rounds=args.max_rounds,
        seed=args.seed,
    )


def make_frontier_fn(graph_data: SyntheticGraphData, n: int, base_seed: int):
    frontier_rng = np.random.default_rng(base_seed)

    def fn():
        return graph_data.sample_initial_frontier(
            n=n,
            seed=int(frontier_rng.integers(1 << 31)),
        )

    return fn


def make_tunnel_frontier_fn(graph_data: TunnelVisionGraphData, n: int, base_seed: int):
    frontier_rng = np.random.default_rng(base_seed)

    def fn():
        return graph_data.sample_initial_frontier(
            n=n,
            seed=int(frontier_rng.integers(1 << 31)),
            balanced=True,
        )

    return fn


def covariate_pool_array(graph_data) -> np.ndarray:
    return np.asarray(list(graph_data.covariates.values()), dtype=np.float64)


def make_planning_population_pool(
    graph_data,
    covariate_model,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Build the population pool used by planners from the planning simulator."""
    base_pool = covariate_pool_array(graph_data)
    if n_samples <= 0:
        return base_pool

    rng = np.random.default_rng(seed)
    parent_idx = rng.integers(0, base_pool.shape[0], size=int(n_samples))
    parents = base_pool[parent_idx]
    sampled_children = covariate_model.sample(
        parents,
        seed=int(rng.integers(1 << 31)),
    )
    return np.concatenate([base_pool, sampled_children], axis=0)


# ------------------------------------------------------------------
# Per-method runners
# ------------------------------------------------------------------

def run_random(testing_env, frontier_fn, args, run_tag) -> EvalBundle:
    print("--- Random Policy ---")
    policy = RandomPolicy(seed=args.seed)
    t0 = time.time()
    x, y, y_std, traj_rows, _, total_rewards = evaluate_recruiting_curve(
        policy_fn=policy.act,
        env=testing_env,
        initial_frontier_fn=frontier_fn,
        n_episodes_eval=args.n_episodes_eval,
        gamma=args.discount,
        normalize=True,
    )
    elapsed = time.time() - t0
    print(
        f"  mean recruits = {np.mean(total_rewards):.2f} "
        f"± {np.std(total_rewards):.2f}  ({elapsed:.1f}s)"
    )
    return EvalBundle(x=x, y=y, y_std=y_std, traj_rows=traj_rows)


def run_constant(testing_env, frontier_fn, args, k: int, run_tag) -> EvalBundle:
    print(f"--- Constant-k Policy (k={k}) ---")

    def constant_policy(state):
        assignment = np.zeros(state.frontier_size, dtype=int)
        remaining = int(state.budget_remaining)
        for i in range(state.frontier_size):
            if remaining <= 0:
                break
            alloc = min(k, remaining)
            assignment[i] = alloc
            remaining -= alloc
        return assignment

    t0 = time.time()
    x, y, y_std, traj_rows, _, total_rewards = evaluate_recruiting_curve(
        policy_fn=constant_policy,
        env=testing_env,
        initial_frontier_fn=frontier_fn,
        n_episodes_eval=args.n_episodes_eval,
        gamma=args.discount,
        normalize=True,
    )
    elapsed = time.time() - t0
    print(
        f"  mean recruits = {np.mean(total_rewards):.2f} "
        f"± {np.std(total_rewards):.2f}  ({elapsed:.1f}s)"
    )
    return EvalBundle(x=x, y=y, y_std=y_std, traj_rows=traj_rows)


def run_adaptive_surrogate(
    planning_env,
    testing_env,
    frontier_fn,
    count_model,
    planning_population_pool,
    args,
    run_tag,
) -> EvalBundle:
    print("--- Adaptive Surrogate Policy ---")
    max_support = args.surrogate_max_support if args.surrogate_max_support is not None else args.budget
    distribution_adapter = PoissonCountDistributionAdapter(
        count_model=count_model,
        max_support=max_support,
    )
    population_pmf = distribution_adapter.population_pmf(
        planning_population_pool,
        sample_size=args.surrogate_sample_size,
        seed=args.seed,
    )
    surrogate = precompute_surrogate_from_population_pmf(
        r_max=args.budget,
        gamma=args.discount,
        population_pmf=population_pmf,
    )
    policy = AdaptiveSurrogatePolicy(distribution_adapter, surrogate)

    t0 = time.time()
    x, y, y_std, traj_rows, _, total_rewards = evaluate_recruiting_curve(
        policy_fn=policy.act,
        env=testing_env,
        initial_frontier_fn=frontier_fn,
        n_episodes_eval=args.n_episodes_eval,
        gamma=args.discount,
        normalize=True,
    )
    elapsed = time.time() - t0
    print(
        f"  mean recruits = {np.mean(total_rewards):.2f} "
        f"± {np.std(total_rewards):.2f}  ({elapsed:.1f}s)"
    )
    return EvalBundle(x=x, y=y, y_std=y_std, traj_rows=traj_rows)


def run_dqn(planning_env, testing_env, frontier_fn, count_model, args, run_tag) -> EvalBundle:
    is_tunnel = getattr(args, "env_type", "basic") == "tunnel_vision"
    label = "Budget-DQN (TypeA-only greedy)" if is_tunnel else "Budget-DQN"
    print(f"--- {label} ---")
    reseed_all(args.seed)

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

    cap = getattr(args, "max_budget_per_round", None)
    if is_tunnel:
        # In tunnel_vision mode, DQN uses a TypeA-only allocator to explicitly
        # demonstrate the tunnel-vision failure: always picks Type A (high
        # immediate rate) and ignores Type B (self-replicating, long-term value).
        # If a per-round budget cap is active, apply it here too.
        if cap is not None:
            allocator_fn = lambda state, k: type_a_only_allocator(
                state, min(k, cap), count_model
            )
        else:
            allocator_fn = lambda state, k: type_a_only_allocator(state, k, count_model)
    else:
        allocator_fn = lambda state, k: greedy_allocator(state, k, count_model)

    t0 = time.time()
    _, learner, _, _, _ = run_budget_dqn(
        env=planning_env,  # Use planning env for training
        initial_frontier_fn=frontier_fn,
        budget_allocator=allocator_fn,
        n_episodes_eval=args.n_episodes_eval,
        seed=args.seed,
        cfg=cfg,
        log_every_n_episodes=args.log_every,
        on_new_best=None,
    )
    elapsed = time.time() - t0

    def dqn_policy(state):
        budget = learner.select_action(state, greedy=True)
        return learner.budget_allocator(state, budget)

    x, y, y_std, traj_rows, _, total_rewards = evaluate_recruiting_curve(
        policy_fn=dqn_policy,
        env=testing_env,  # Use testing env for evaluation
        initial_frontier_fn=frontier_fn,
        n_episodes_eval=args.n_episodes_eval,
        gamma=args.discount,
        normalize=True,
    )
    print(
        f"  mean recruits = {np.mean(total_rewards):.2f} "
        f"± {np.std(total_rewards):.2f}  ({elapsed:.1f}s)"
    )
    return EvalBundle(x=x, y=y, y_std=y_std, traj_rows=traj_rows)


def run_structured(planning_env, testing_env, frontier_fn, count_model, covariate_model, args, run_tag) -> EvalBundle:
    print("--- Structured RL (Three-Head) ---")
    reseed_all(args.seed)

    max_k = min(args.budget, args.initial_frontier_size)

    is_tunnel = getattr(args, "env_type", "basic") == "tunnel_vision"

    if is_tunnel:
        # Tunnel-vision mode: use high initial exploration so the agent can
        # discover the TypeB self-replicating chain before exploitation.
        # High score_noise forces random node selection early, ensuring the
        # agent experiences TypeB-heavy episodes needed for Q-value bootstrap.
        eps_b_start, eps_k_start, sn_start = 0.80, 0.80, 0.80
        target_upd = 100
        n_samples = 3
    else:
        eps_b_start, eps_k_start, sn_start = 0.20, 0.20, 0.20
        target_upd = 200
        n_samples = 1

    cfg = ValueTrainerConfig(
        gamma=args.gamma,
        lr=args.lr,
        train_episodes=args.train_episodes,
        max_steps_per_episode=args.max_rounds,
        epsilon_budget_start=eps_b_start,
        epsilon_budget_end=0.05,
        epsilon_k_start=eps_k_start,
        epsilon_k_end=0.05,
        score_noise_start=sn_start,
        score_noise_end=0.02,
        node_score_loss_weight=1.0,
        replay_buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        min_buffer_size=min(args.min_buffer_size, args.buffer_capacity),
        updates_per_env_step=1,
        target_update_interval=target_upd,
        node_target_num_samples=n_samples,
    )

    encoder = StateEncoder(covariate_dim=72, hidden_dim=args.hidden_dim)
    q_network = ThreeHeadQNetwork(
        state_dim=encoder.output_dim,
        covariate_dim=72,
        hidden_dim=args.hidden_dim,
        max_budget=planning_env.initial_budget,
        max_k=max_k,
    )
    policy = StructuredValuePolicy(
        encoder=encoder,
        q_network=q_network,
        device=args.device,
        seed=args.seed,
    )

    trainer = StructuredQTrainer(
        env=planning_env,  # Use planning env for training
        policy=policy,
        initial_frontier_fn=frontier_fn,
        count_model=count_model,
        covariate_model=covariate_model,
        cfg=cfg,
        device=args.device,
        seed=args.seed,
        on_eval_log=None,
        n_episodes_eval=args.n_episodes_eval,
        log_every_n_episodes=args.log_every,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    policy.eval()

    def structured_policy(state):
        return policy.act_greedy(state).allocation

    x, y, y_std, traj_rows, _, total_rewards = evaluate_recruiting_curve(
        policy_fn=structured_policy,
        env=testing_env,  # Use testing env for evaluation
        initial_frontier_fn=frontier_fn,
        n_episodes_eval=args.n_episodes_eval,
        gamma=args.discount,
        normalize=True,
    )
    print(
        f"  mean recruits = {np.mean(total_rewards):.2f} "
        f"± {np.std(total_rewards):.2f}  ({elapsed:.1f}s)"
    )
    return EvalBundle(x=x, y=y, y_std=y_std, traj_rows=traj_rows)


def run_gfp(
    planning_env,
    testing_env,
    frontier_fn,
    count_model,
    covariate_model,
    planning_population_pool,
    args,
    run_tag,
) -> EvalBundle:
    print("--- Generative Frontier Planning (GFP) ---")
    reseed_all(args.seed)
    parent_pool = planning_population_pool

    value_surrogate = FrontierValueSurrogate(
        covariate_dim=72,
        latent_dim=args.gfp_latent_dim,
        hidden_dim=args.gfp_hidden_dim,
    ).to(args.device)

    if args.model_mode == "oracle":
        survival_model = PoissonCountSurvival(count_model)
        planning_covariate_model = covariate_model
        print("  GFP models: oracle count/covariate adapters")
    else:
        survival_model = count_model  # Already learned
        planning_covariate_model = covariate_model  # Already learned
        print("  GFP models: globally trained learned count/covariate models")

    if args.gfp_laplace_mode == "mc":
        laplace_provider = MonteCarloLaplaceProvider(
            covariate_model=planning_covariate_model,
            value_surrogate=value_surrogate,
            n_samples=args.gfp_laplace_samples,
            seed=args.seed,
            device=args.device,
        )
    else:
        laplace_provider = AmortizedLaplaceProvider(
            covariate_model=planning_covariate_model,
            value_surrogate=value_surrogate,
            parent_pool=parent_pool,
            n_train_parents=args.gfp_laplace_train_parents,
            n_child_samples=args.gfp_laplace_samples,
            train_steps=args.gfp_laplace_train_steps,
            batch_size=args.gfp_laplace_batch_size,
            lr=args.gfp_laplace_lr,
            hidden_dim=args.gfp_laplace_hidden_dim,
            seed=args.seed,
            device=args.device,
        )
        laplace_metrics = laplace_provider.refresh()
        print(
            f"  learned L_eta: final_loss={laplace_metrics['final_loss']:.4f}, "
            f"parents={laplace_metrics['n_train_parents']}"
        )

    trainer = GFPTrainer(
        env=planning_env,  # Use planning env for training
        initial_frontier_fn=frontier_fn,
        value_surrogate=value_surrogate,
        survival_model=survival_model,
        laplace_provider=laplace_provider,
        cfg=GFPTrainerConfig(
            train_iterations=args.gfp_train_iterations,
            batch_size=args.gfp_batch_size,
            lr=args.gfp_lr,
            target_update_interval=args.gfp_target_update_interval,
            max_steps_per_episode=args.max_rounds,
            state_pool_size=args.gfp_state_pool_size,
            random_rollout_episodes=args.gfp_random_rollout_episodes,
        ),
        gamma=args.gamma,
        max_budget_per_round=getattr(args, "max_budget_per_round", None),
        device=args.device,
        seed=args.seed,
    )

    t0 = time.time()
    train_metrics = trainer.train()

    planner = GFPPlanner(
        value_surrogate=value_surrogate,
        survival_model=survival_model,
        laplace_provider=laplace_provider,
        gamma=args.gamma,
        max_budget_per_round=getattr(args, "max_budget_per_round", None),
        device=args.device,
    )

    x, y, y_std, traj_rows, _, total_rewards = evaluate_recruiting_curve(
        policy_fn=planner.act,
        env=testing_env,  # Use testing env for evaluation
        initial_frontier_fn=frontier_fn,
        n_episodes_eval=args.n_episodes_eval,
        gamma=args.discount,
        normalize=True,
    )
    elapsed = time.time() - t0
    print(
        f"  mean recruits = {np.mean(total_rewards):.2f} "
        f"± {np.std(total_rewards):.2f}  ({elapsed:.1f}s)"
    )
    print(
        f"  GFP value training: final_loss={train_metrics['final_loss']:.4f}, "
        f"state_pool={train_metrics['state_pool_size']}"
    )
    return EvalBundle(x=x, y=y, y_std=y_std, traj_rows=traj_rows)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Synthetic benchmark driver for GFP on synthetic environments"
    )

    # Environment type
    parser.add_argument("--env_type", type=str, default="basic",
                        choices=["basic", "tunnel_vision"],
                        help="'basic': random covariate pool; 'tunnel_vision': 3-type A/B/C pool")

    # Synthetic env (basic only)
    parser.add_argument("--n_pool", type=int, default=300,
                        help="Number of covariate vectors in the synthetic pool")
    parser.add_argument("--mean_degree", type=float, default=2.5,
                        help="Expected recruits per unit of budget (Poisson rate mean)")
    parser.add_argument("--heterogeneity", type=float, default=1.0,
                        help="Spread of referral rates across covariates (weight std)")
    parser.add_argument("--inherit_prob", type=float, default=0.7,
                        help="Prob. a child inherits each covariate group from parent")

    # Tunnel-vision env params
    parser.add_argument("--rate_a", type=float, default=4.0,
                        help="[tunnel_vision] Poisson rate for Type A (boom-bust)")
    parser.add_argument("--rate_b", type=float, default=0.8,
                        help="[tunnel_vision] Poisson rate for Type B (slow-burn)")
    parser.add_argument("--rate_c", type=float, default=0.02,
                        help="[tunnel_vision] Poisson rate for Type C (dead-end)")
    parser.add_argument("--p_cross", type=float, default=0.85,
                        help="[tunnel_vision] Prob offspring follows designed type transition")
    parser.add_argument("--type_fractions", type=str, default="0.45,0.45,0.10",
                        help="[tunnel_vision] Comma-sep (frac_A,frac_B,frac_C) for pool")
    parser.add_argument("--max_budget_per_round", type=int, default=None,
                        help="[tunnel_vision] Per-round budget cap (None = uncapped). "
                             "Forces multi-round episodes so TypeB chain can compound.")

    # Episode settings
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--initial_frontier_size", type=int, default=10)
    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--max_rounds", type=int, default=50)

    # Experiment
    parser.add_argument("--methods", type=str, default="gfp",
                        help="Comma-separated list. Default: gfp. Optional baselines: random, dqn, structured, constant, adaptive")
    parser.add_argument("--constant_k", type=int, default=1,
                        help="Constant-k baseline allocation per frontier member when method is constant")
    parser.add_argument("--surrogate_sample_size", type=int, default=1024,
                        help="Number of covariate samples used to estimate the surrogate population PMF")
    parser.add_argument("--surrogate_max_support", type=int, default=None,
                        help="Max count support for Poisson-based surrogate distributions. Defaults to budget.")
    parser.add_argument("--model_mode", type=str, default="learned",
                        choices=["oracle", "learned"],
                        help="oracle: use synthetic parametric models; learned: train diffusion/count models on synthetic data")
    parser.add_argument("--train_episodes", type=int, default=300)
    parser.add_argument("--n_episodes_eval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--results_dir", type=str, default="results/synthetic")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    # Shared RL hyperparams
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--buffer_capacity", type=int, default=10000)
    parser.add_argument("--min_buffer_size", type=int, default=128,
                        help="Min transitions before RL updates start. Lower for short runs.")

    # DQN-specific
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_decay", type=float, default=0.995)
    parser.add_argument("--target_update_interval", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)

    # GFP-specific
    parser.add_argument("--gfp_train_iterations", type=int, default=200)
    parser.add_argument("--gfp_batch_size", type=int, default=16)
    parser.add_argument("--gfp_lr", type=float, default=1e-3)
    parser.add_argument("--gfp_hidden_dim", type=int, default=64)
    parser.add_argument("--gfp_latent_dim", type=int, default=32)
    parser.add_argument("--gfp_laplace_samples", type=int, default=64)
    parser.add_argument("--gfp_laplace_train_parents", type=int, default=256)
    parser.add_argument("--gfp_laplace_train_steps", type=int, default=200)
    parser.add_argument("--gfp_laplace_batch_size", type=int, default=128)
    parser.add_argument("--gfp_laplace_lr", type=float, default=1e-3)
    parser.add_argument("--gfp_laplace_hidden_dim", type=int, default=64)
    parser.add_argument("--gfp_state_pool_size", type=int, default=256)
    parser.add_argument("--gfp_random_rollout_episodes", type=int, default=64)
    parser.add_argument("--gfp_target_update_interval", type=int, default=25)
    parser.add_argument("--gfp_model_mode", type=str, default="learned",
                        choices=["learned", "oracle"],
                        help="learned trains q_psi/G_theta for GFP; oracle uses synthetic adapters")
    parser.add_argument("--gfp_laplace_mode", type=str, default="amortized",
                        choices=["amortized", "mc"],
                        help="amortized trains L_eta(x); mc uses cached Monte Carlo alpha(x)")
    parser.add_argument("--gfp_count_samples", type=int, default=2048)
    parser.add_argument("--gfp_count_max_allocation", type=int, default=10)
    parser.add_argument("--gfp_count_epochs", type=int, default=200)
    parser.add_argument("--gfp_count_batch_size", type=int, default=128)
    parser.add_argument("--gfp_count_lr", type=float, default=1e-3)
    parser.add_argument("--gfp_count_hidden_dim", type=int, default=64)
    parser.add_argument("--gfp_offspring_pairs", type=int, default=4096)
    parser.add_argument("--gfp_offspring_epochs", type=int, default=200)
    parser.add_argument("--gfp_offspring_batch_size", type=int, default=128)
    parser.add_argument("--gfp_offspring_lr", type=float, default=1e-3)
    parser.add_argument("--gfp_offspring_hidden_dim", type=int, default=64)
    parser.add_argument("--gfp_offspring_model", type=str, default="ddpm",
                        choices=["ddpm", "categorical"],
                        help="G_theta model for GFP: conditional DDPM or grouped categorical")
    parser.add_argument("--gfp_ddpm_hidden_dim", type=int, default=512)
    parser.add_argument("--gfp_ddpm_steps", type=int, default=100)
    # Learned model training params
    parser.add_argument("--learned_count_samples", type=int, default=2048)
    parser.add_argument("--learned_count_max_allocation", type=int, default=10)
    parser.add_argument("--learned_count_epochs", type=int, default=200)
    parser.add_argument("--learned_count_batch_size", type=int, default=128)
    parser.add_argument("--learned_count_lr", type=float, default=1e-3)
    parser.add_argument("--learned_count_hidden_dim", type=int, default=64)
    parser.add_argument("--learned_offspring_pairs", type=int, default=4096)
    parser.add_argument("--learned_offspring_epochs", type=int, default=200)
    parser.add_argument("--learned_offspring_batch_size", type=int, default=128)
    parser.add_argument("--learned_offspring_lr", type=float, default=1e-3)
    parser.add_argument("--learned_ddpm_hidden_dim", type=int, default=512)
    parser.add_argument("--learned_ddpm_steps", type=int, default=100)
    parser.add_argument("--planning_pool_samples", type=int, default=4096,
                        help="Number of covariates sampled from the planning diffusion model for GFP/adaptive planning statistics")

    args = parser.parse_args()
    reseed_all(args.seed)

    methods = [m.strip() for m in args.methods.split(",")]

    if args.env_type == "tunnel_vision":
        run_tag = (
            f"tunnel_vision"
            f"_B{args.budget}"
            f"_F{args.initial_frontier_size}"
            f"_rA{args.rate_a}_rB{args.rate_b}"
            f"_pc{args.p_cross}"
            f"_disc{args.discount}"
            f"_seed{args.seed}"
            f"_train{args.train_episodes}"
        )
    else:
        run_tag = (
            f"synthetic"
            f"_B{args.budget}"
            f"_F{args.initial_frontier_size}"
            f"_deg{args.mean_degree}"
            f"_het{args.heterogeneity}"
            f"_inh{args.inherit_prob}"
            f"_disc{args.discount}"
            f"_seed{args.seed}"
            f"_train{args.train_episodes}"
        )

    print("========================================================")
    print(f"Synthetic Recruiting Benchmark  [env_type={args.env_type}]")
    print("========================================================")
    print(f"  budget={args.budget}, frontier_size={args.initial_frontier_size}")
    print(f"  train_episodes={args.train_episodes}, seed={args.seed}")
    print(f"  methods: {methods}")
    print()

    # ------------------------------------------------------------------
    # Build shared models (branched by env_type)
    # ------------------------------------------------------------------
    print("Building models (no fitting/training required)...")

    if args.env_type == "tunnel_vision":
        type_fractions = tuple(float(x) for x in args.type_fractions.split(","))
        count_model = TunnelVisionCountModel(
            rate_a=args.rate_a,
            rate_b=args.rate_b,
            rate_c=args.rate_c,
            seed=args.seed,
        )
        covariate_model = TunnelVisionCovariateModel(
            p_cross=args.p_cross,
            inherit_prob=args.inherit_prob,
            seed=args.seed,
        )
        graph_data = TunnelVisionGraphData(
            n_nodes=args.n_pool,
            type_fractions=type_fractions,
            seed=args.seed,
        )
        print(f"  {graph_data.type_summary()}")
        print(f"  rates: A={args.rate_a}, B={args.rate_b}, C={args.rate_c}")
        print(f"  p_cross={args.p_cross} (type-transition probability)")
        print(f"  transitions: A→C (dead-end), B→A (productive), C→C (stays dead)")
    else:
        count_model = SyntheticCountModel(
            mean_degree=args.mean_degree,
            heterogeneity=args.heterogeneity,
            seed=args.seed,
        )
        covariate_model = SyntheticCovariateModel(
            inherit_prob=args.inherit_prob,
            seed=args.seed,
        )
        graph_data = SyntheticGraphData(n_nodes=args.n_pool, seed=args.seed)
        print(f"  mean_degree={args.mean_degree}, heterogeneity={args.heterogeneity}")
        print(f"  inherit_prob={args.inherit_prob}")
        print(f"  covariate pool: {args.n_pool} nodes, COVARIATE_DIM=72")

    print()

    # ------------------------------------------------------------------
    # Train learned models if needed
    # ------------------------------------------------------------------
    if args.model_mode == "learned":
        print("Training learned models on synthetic data...")
        parent_pool = covariate_pool_array(graph_data)

        # Train learned count model
        count_fit_cfg = CensoredCountFitConfig(
            n_samples=args.learned_count_samples,
            max_allocation=args.learned_count_max_allocation,
            epochs=args.learned_count_epochs,
            batch_size=args.learned_count_batch_size,
            lr=args.learned_count_lr,
        )
        cx, ck, cy = make_censored_count_dataset(
            base_count_model=count_model,
            covariate_pool=parent_pool,
            cfg=count_fit_cfg,
            seed=args.seed,
        )
        learned_count_model = LearnedCensoredPoissonSurvival(
            covariate_dim=72,
            hidden_dim=args.learned_count_hidden_dim,
            device=args.device,
            seed=args.seed,
        )
        count_metrics = learned_count_model.fit(
            cx,
            ck,
            cy,
            epochs=count_fit_cfg.epochs,
            batch_size=count_fit_cfg.batch_size,
            lr=count_fit_cfg.lr,
        )
        print(f"  learned count model: final_loss={count_metrics['final_loss']:.4f}, samples={count_metrics['n_samples']}")

        # Train learned covariate model (DDPM)
        offspring_fit_cfg = OffspringFitConfig(
            n_pairs=args.learned_offspring_pairs,
            epochs=args.learned_offspring_epochs,
            batch_size=args.learned_offspring_batch_size,
            lr=args.learned_offspring_lr,
        )
        px, py = make_offspring_dataset(
            base_covariate_model=covariate_model,
            covariate_pool=parent_pool,
            cfg=offspring_fit_cfg,
            seed=args.seed + 17,
        )
        from torch.utils.data import TensorDataset
        learned_covariate_model = DDPMCovariateModel(
            hidden_dim=args.learned_ddpm_hidden_dim,
            num_steps=args.learned_ddpm_steps,
            device=args.device,
        )
        pair_tensor = torch.tensor(
            np.concatenate([px, py], axis=1), dtype=torch.float32,
        )
        offspring_metrics = learned_covariate_model.train(
            TensorDataset(pair_tensor),
            epochs=offspring_fit_cfg.epochs,
            batch_size=offspring_fit_cfg.batch_size,
            learning_rate=offspring_fit_cfg.lr,
            seed=args.seed,
            log_interval=max(1, offspring_fit_cfg.epochs // 5),
        )
        print(f"  learned covariate model (DDPM): final_loss={offspring_metrics['final_loss']:.4f}, pairs={offspring_metrics['n_pairs']}")
        print()
    else:
        learned_count_model = count_model
        learned_covariate_model = covariate_model

    planning_population_pool = make_planning_population_pool(
        graph_data=graph_data,
        covariate_model=learned_covariate_model,
        n_samples=args.planning_pool_samples if args.model_mode == "learned" else 0,
        seed=args.seed + 31,
    )

    # ------------------------------------------------------------------
    # Run each method
    # ------------------------------------------------------------------
    bundles: dict[str, EvalBundle] = {}

    for method in methods:
        planning_env = build_planning_env(learned_count_model, learned_covariate_model, args)
        testing_env = build_testing_env(count_model, covariate_model, args)
        if args.env_type == "tunnel_vision":
            frontier_fn = make_tunnel_frontier_fn(graph_data, args.initial_frontier_size, args.seed)
        else:
            frontier_fn = make_frontier_fn(graph_data, args.initial_frontier_size, args.seed)

        if method == "random":
            bundles["random"] = run_random(testing_env, frontier_fn, args, run_tag)
        elif method == "dqn":
            bundles["dqn"] = run_dqn(planning_env, testing_env, frontier_fn, learned_count_model, args, run_tag)
        elif method == "structured":
            bundles["structured"] = run_structured(planning_env, testing_env, frontier_fn, learned_count_model, learned_covariate_model, args, run_tag)
        elif method == "gfp":
            bundles["gfp"] = run_gfp(
                planning_env,
                testing_env,
                frontier_fn,
                learned_count_model,
                learned_covariate_model,
                planning_population_pool,
                args,
                run_tag,
            )
        elif method == "adaptive":
            bundles["adaptive"] = run_adaptive_surrogate(
                planning_env,
                testing_env,
                frontier_fn,
                learned_count_model,
                planning_population_pool,
                args,
                run_tag,
            )
        elif method == "constant":
            bundles[f"constant{args.constant_k}"] = run_constant(
                testing_env,
                frontier_fn,
                args,
                args.constant_k,
                run_tag,
            )
        elif method.startswith("constant") and len(method) > len("constant"):
            try:
                const_k = int(method[len("constant"):])
            except ValueError:
                print(f"  [WARNING] Unknown method '{method}', skipping.")
                continue
            bundles[f"constant{const_k}"] = run_constant(
                testing_env,
                frontier_fn,
                args,
                const_k,
                run_tag,
            )
        else:
            print(f"  [WARNING] Unknown method '{method}', skipping.")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    print()
    print("Saving results...")
    os.makedirs(args.results_dir, exist_ok=True)
    save_comparison_curves(bundles, args.results_dir, run_tag, args.discount)
    save_four_panel_curve_outputs(
        bundles,
        args.results_dir,
        run_tag,
        gamma=args.discount,
        horizon=args.max_rounds,
    )

    print()
    print("========================================================")
    print("Summary")
    print("========================================================")
    if args.env_type == "tunnel_vision":
        print(f"  budget={args.budget}, frontier={args.initial_frontier_size}, "
              f"rate_A={args.rate_a}, rate_B={args.rate_b}, p_cross={args.p_cross}, seed={args.seed}")
    else:
        print(f"  budget={args.budget}, frontier={args.initial_frontier_size}, "
              f"mean_degree={args.mean_degree}, het={args.heterogeneity}, seed={args.seed}")
    print()
    print(f"  {'Method':<16}  {'Final Recruits (mean ± std)':>30}")
    print(f"  {'-'*16}  {'-'*30}")
    for method, bundle in bundles.items():
        traj_df = {r["episode"]: r for r in bundle.traj_rows}
        ep_totals = {}
        for row in bundle.traj_rows:
            ep = row["episode"]
            ep_totals[ep] = row["cumulative_recruits"]
        vals = list(ep_totals.values())
        print(f"  {method:<16}  {np.mean(vals):>12.1f} ± {np.std(vals):<8.1f}")
    print("[done]")


if __name__ == "__main__":
    main()
