"""Evaluate adaptive surrogate on the diffusion recruiting environment.

Example:
    python /home/u/diffusion_surrogatet/evaluate_adaptive_surrogate.py \
        --model_path /home/u/diffusion_surrogatet/Diffusion_for_Recruiting_baseline-leakage-free-split/checkpoints/diffusion/ddpm_HIV.pt \
        --budget 100 \
        --initial_frontier_size 10 \
        --n_episodes_eval 10
"""

from __future__ import annotations

import argparse
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

try:
    from .adaptive_surrogate import (
        AdaptiveSurrogatePolicy,
        GaussianCountDistributionAdapter,
        precompute_surrogate_from_population_pmf,
    )
except ImportError:
    from adaptive_surrogate import (
        AdaptiveSurrogatePolicy,
        GaussianCountDistributionAdapter,
        precompute_surrogate_from_population_pmf,
    )


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_DIFFUSION_PROJECT_DIR = (
    REPO_DIR / "Diffusion_for_Recruiting_baseline-leakage-free-split"
)
DEFAULT_DATA_DIR = REPO_DIR / "recruit_baseline" / "data" / "ICPSR_22140"


def reseed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_or_build_surrogate(
    cache_path: Path,
    adapter: GaussianCountDistributionAdapter,
    train_covariates: np.ndarray,
    budget: int,
    gamma: float,
    surrogate_samples: int,
    seed: int,
    force_recompute: bool,
):
    if cache_path.exists() and not force_recompute:
        with cache_path.open("rb") as f:
            return pickle.load(f)

    population_pmf = adapter.population_pmf(
        train_covariates,
        sample_size=surrogate_samples,
        seed=seed,
    )
    surrogate = precompute_surrogate_from_population_pmf(
        r_max=budget,
        gamma=gamma,
        population_pmf=population_pmf,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(surrogate, f, protocol=pickle.HIGHEST_PROTOCOL)
    return surrogate


def resolve_existing_path(path: str | None, default: Path, base_dir: Path | None = None) -> Path:
    candidate = Path(path).expanduser() if path else default
    if not candidate.is_absolute() and base_dir is not None:
        candidate = base_dir / candidate
    return candidate.resolve()


def sample_frontier_from_nodes(graph_data, candidate_nodes: set[int], n: int, seed: int) -> np.ndarray:
    nodes_with_covariates = sorted(candidate_nodes & set(graph_data.covariates))
    if len(nodes_with_covariates) == 0:
        raise ValueError("held-out split has no nodes with covariates for initial frontier sampling")

    held_out_roots = sorted(set(graph_data.roots) & set(nodes_with_covariates))
    candidates = held_out_roots if len(held_out_roots) >= n else nodes_with_covariates

    rng = np.random.default_rng(seed)
    chosen = rng.choice(candidates, size=n, replace=len(candidates) < n)
    return np.asarray([graph_data.covariates[int(node)] for node in chosen], dtype=np.float64)


def build_discounted_reward_curves(
    traj_rows: list[dict],
    n_episodes: int,
    horizon: int,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    budget_mat = []
    discounted_mat = []

    rows_by_episode: dict[int, list[dict]] = {ep: [] for ep in range(n_episodes)}
    for row in traj_rows:
        rows_by_episode[int(row["episode"])].append(row)

    for ep in range(n_episodes):
        rows = sorted(rows_by_episode.get(ep, []), key=lambda row: int(row["round"]))
        budget_ep = []
        discounted_ep = []
        cumulative_discounted = 0.0

        for row in rows:
            round_idx = int(row["round"]) - 1
            cumulative_discounted += float(row["reward"]) * (gamma ** round_idx)
            budget_ep.append(float(row["cumulative_budget_spent"]))
            discounted_ep.append(cumulative_discounted)

        while len(budget_ep) < horizon:
            budget_ep.append(budget_ep[-1] if budget_ep else 0.0)
            discounted_ep.append(discounted_ep[-1] if discounted_ep else 0.0)

        budget_mat.append(np.asarray(budget_ep[:horizon], dtype=np.float32))
        discounted_mat.append(np.asarray(discounted_ep[:horizon], dtype=np.float32))

    budget_mat_arr = np.asarray(budget_mat, dtype=np.float32)
    discounted_mat_arr = np.asarray(discounted_mat, dtype=np.float32)

    budget_x = np.mean(budget_mat_arr, axis=0)
    time_x = np.arange(1, horizon + 1, dtype=np.float32)
    discounted_y = np.mean(discounted_mat_arr, axis=0)
    discounted_y_std = np.std(discounted_mat_arr, axis=0)
    return budget_x, time_x, discounted_y, discounted_y_std, discounted_mat_arr


def save_discounted_reward_curves(
    traj_rows: list[dict],
    n_episodes: int,
    horizon: int,
    gamma: float,
    results_dir: Path,
    run_tag: str,
    label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_dir.mkdir(parents=True, exist_ok=True)
    budget_x, time_x, discounted_y, discounted_y_std, discounted_mat = (
        build_discounted_reward_curves(
            traj_rows=traj_rows,
            n_episodes=n_episodes,
            horizon=horizon,
            gamma=gamma,
        )
    )

    npz_path = results_dir / f"discounted_reward_curves_{run_tag}.npz"
    np.savez(
        npz_path,
        budget_x=budget_x,
        time_x=time_x,
        discounted_y=discounted_y,
        discounted_y_std=discounted_y_std,
        discounted_mat=discounted_mat,
    )
    print("Saved discounted reward vectors to:", npz_path)

    def plot_curve(x: np.ndarray, xlabel: str, filename: str) -> None:
        x_plot = np.concatenate([[0.0], x])
        y_plot = np.concatenate([[0.0], discounted_y])
        y_std_plot = np.concatenate([[0.0], discounted_y_std])

        plt.figure(figsize=(8, 4))
        plt.plot(x_plot, y_plot, linestyle="-", color="tab:green", label=label)
        plt.fill_between(
            x_plot,
            np.maximum(0.0, y_plot - y_std_plot),
            y_plot + y_std_plot,
            color="tab:green",
            alpha=0.25,
        )
        plt.xlabel(xlabel)
        plt.ylabel("Accumulated discounted reward")
        plt.title(f"Discounted recruiting reward (discount = {gamma})")
        plt.legend()
        plt.tight_layout()

        png_path = results_dir / filename
        plt.savefig(png_path, dpi=200)
        print("Saved plot to:", png_path)
        plt.close()

    plot_curve(
        budget_x,
        xlabel="Budget spent",
        filename=f"discounted_reward_by_budget_{run_tag}.png",
    )
    plot_curve(
        time_x,
        xlabel="Time spent",
        filename=f"discounted_reward_by_time_{run_tag}.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate adaptive surrogate on diffusion recruiting env"
    )
    parser.add_argument(
        "--project_dir",
        type=str,
        default=str(DEFAULT_DIFFUSION_PROJECT_DIR),
        help="Path to the diffusion recruiting project root",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to trained diffusion model checkpoint. Defaults to project_dir/checkpoints/diffusion/ddpm_<std_name>.pt",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Path to ICPSR_22140 data directory",
    )
    parser.add_argument(
        "--std_name",
        type=str,
        default="HIV",
        choices=["HIV", "Gonorrhea", "Chlamydia", "Syphilis", "Hepatitis"],
    )
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--initial_frontier_size", type=int, default=10)
    parser.add_argument("--n_episodes_eval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--max_rounds", type=int, default=50)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--surrogate_samples", type=int, default=1000)
    parser.add_argument("--count_min_std", type=float, default=1e-3)
    parser.add_argument("--count_std_scale", type=float, default=1.0)
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(REPO_DIR / "results"),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=str(REPO_DIR / "cache"),
    )
    parser.add_argument("--force_recompute_surrogate", action="store_true")
    args = parser.parse_args()

    reseed_all(args.seed)

    project_dir = resolve_existing_path(args.project_dir, DEFAULT_DIFFUSION_PROJECT_DIR)
    data_dir = resolve_existing_path(args.data_dir, DEFAULT_DATA_DIR, base_dir=project_dir)
    default_model_path = project_dir / "checkpoints" / "diffusion" / f"ddpm_{args.std_name}.pt"
    model_path = resolve_existing_path(args.model_path, default_model_path, base_dir=project_dir)
    results_dir = resolve_existing_path(args.results_dir, REPO_DIR / "results")
    cache_dir = resolve_existing_path(args.cache_dir, REPO_DIR / "cache")

    if not project_dir.exists():
        raise FileNotFoundError(f"diffusion project_dir does not exist: {project_dir}")
    if not data_dir.exists():
        raise FileNotFoundError(f"ICPSR data_dir does not exist: {data_dir}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"diffusion checkpoint does not exist: {model_path}. "
            "Pass --model_path or train the diffusion model first."
        )

    sys.path.insert(0, str(project_dir))

    from src.data.icpsr_loader import ICPSRGraphData
    from src.environment.recruiting_env import RecruitingEnv
    from src.models.count_model.gaussian_count_model import GaussianCountModel
    from src.models.covariate_model.ddpm_covariate_model import DDPMCovariateModel
    from src.scripts.eval_utils import evaluate_recruiting_curve, save_single_curve

    print("--------------------------------------------------------")
    print("Load ICPSR Graph Data")
    print("--------------------------------------------------------")
    graph_data = ICPSRGraphData(str(data_dir), args.std_name)
    train_nodes, test_nodes = graph_data.train_test_node_split(
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    common_nodes = sorted(
        train_nodes & set(graph_data.covariates) & set(graph_data.node_degrees)
    )
    train_covariates = np.asarray(
        [graph_data.covariates[n] for n in common_nodes],
        dtype=np.float64,
    )
    train_degrees = np.asarray(
        [graph_data.node_degrees[n] for n in common_nodes],
        dtype=np.float64,
    )
    print(f"  disease: {args.std_name}")
    print(f"  data_dir: {data_dir}")
    print(f"  train nodes for count/surrogate: {len(common_nodes)}")
    print(f"  held-out nodes: {len(test_nodes)}")

    print("--------------------------------------------------------")
    print("Load Diffusion Model")
    print("--------------------------------------------------------")
    if args.device is None:
        covariate_model = DDPMCovariateModel.load(str(model_path))
    else:
        covariate_model = DDPMCovariateModel.load(str(model_path), device=args.device)
    print(f"  loaded diffusion model from: {model_path}")
    print(f"  device: {covariate_model.device}")

    print("--------------------------------------------------------")
    print("Fit Count Model")
    print("--------------------------------------------------------")
    count_model = GaussianCountModel(seed=args.seed)
    count_model.fit(train_covariates, train_degrees)
    print("  fitted GaussianCountModel with leakage-free node split")

    print("--------------------------------------------------------")
    print("Build Adaptive Surrogate")
    print("--------------------------------------------------------")
    adapter = GaussianCountDistributionAdapter(
        count_model=count_model,
        max_support=args.budget + 1,
        min_std=args.count_min_std,
        std_scale=args.count_std_scale,
    )
    cache_name = (
        f"surrogate_{args.std_name}"
        f"_B{args.budget}"
        f"_disc{args.discount}"
        f"_surr{args.surrogate_samples}"
        f"_seed{args.seed}"
        f"_stdscale{args.count_std_scale}.pkl"
    )
    cache_path = cache_dir / cache_name
    t0 = time.time()
    surrogate = load_or_build_surrogate(
        cache_path=cache_path,
        adapter=adapter,
        train_covariates=train_covariates,
        budget=args.budget,
        gamma=args.discount,
        surrogate_samples=args.surrogate_samples,
        seed=args.seed,
        force_recompute=args.force_recompute_surrogate,
    )
    print(f"  surrogate r_max={surrogate.r_max}, cache={cache_path}")
    print(f"  surrogate ready in {time.time() - t0:.1f}s")

    policy = AdaptiveSurrogatePolicy(
        distribution_adapter=adapter,
        surrogate=surrogate,
    )

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
    frontier_rng = np.random.default_rng(args.seed)

    def initial_frontier_fn():
        return sample_frontier_from_nodes(
            graph_data=graph_data,
            candidate_nodes=test_nodes,
            n=args.initial_frontier_size,
            seed=int(frontier_rng.integers(1 << 31)),
        )

    def policy_fn(state):
        return policy.act(state)

    print("--------------------------------------------------------")
    print("Evaluate Adaptive Surrogate")
    print("--------------------------------------------------------")
    x, y, y_std, traj_rows, discounted_returns, total_rewards = evaluate_recruiting_curve(
        policy_fn=policy_fn,
        env=env,
        initial_frontier_fn=initial_frontier_fn,
        n_episodes_eval=args.n_episodes_eval,
        gamma=args.discount,
    )

    run_tag = (
        f"adaptive_surrogate_{args.std_name}"
        f"_B{args.budget}"
        f"_F{args.initial_frontier_size}"
        f"_disc{args.discount}"
        f"_seed{args.seed}"
        f"_surr{args.surrogate_samples}"
    )
    save_single_curve(
        x=x,
        y=y,
        y_std=y_std,
        traj_rows=traj_rows,
        results_dir=str(results_dir),
        run_tag=run_tag,
        gamma=args.discount,
        label="Adaptive Surrogate",
    )
    save_discounted_reward_curves(
        traj_rows=traj_rows,
        n_episodes=args.n_episodes_eval,
        horizon=args.max_rounds,
        gamma=args.discount,
        results_dir=results_dir,
        run_tag=run_tag,
        label="Adaptive Surrogate",
    )

    print("--------------------------------------------------------")
    print("Summary")
    print("--------------------------------------------------------")
    print(f"  mean total recruits:       {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}")
    print(f"  mean discounted recruits:  {np.mean(discounted_returns):.2f} +/- {np.std(discounted_returns):.2f}")
    print(f"  results_dir: {results_dir}")


if __name__ == "__main__":
    main()
