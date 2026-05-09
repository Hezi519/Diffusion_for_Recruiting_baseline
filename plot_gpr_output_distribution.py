"""Plot Gaussian process count-model output diagnostics.

This is a lightweight diagnostic for the decision-making/count-model side of
the diffusion recruiting pipeline. It fits ``GaussianCountModel`` on ICPSR node
covariates and plots the GPR predictive mean/std distributions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_DIFFUSION_PROJECT_DIR = (
    REPO_DIR / "Diffusion_for_Recruiting_baseline-leakage-free-split"
)
DEFAULT_DATA_DIR = REPO_DIR / "recruit_baseline" / "data" / "ICPSR_22140"
DEFAULT_OUT_DIR = REPO_DIR / "results" / "gpr_diagnostics"


def resolve_path(path: str | None, default: Path, base_dir: Path | None = None) -> Path:
    candidate = Path(path).expanduser() if path else default
    if not candidate.is_absolute() and base_dir is not None:
        candidate = base_dir / candidate
    return candidate.resolve()


def load_node_arrays(graph_data, nodes: set[int]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    common_nodes = sorted(nodes & set(graph_data.covariates) & set(graph_data.node_degrees))
    covariates = np.asarray([graph_data.covariates[n] for n in common_nodes], dtype=np.float64)
    degrees = np.asarray([graph_data.node_degrees[n] for n in common_nodes], dtype=np.float64)
    return covariates, degrees, common_nodes


def predict_gpr(count_model, covariates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means, stds = count_model.gpr.predict(covariates, return_std=True)
    return np.asarray(means, dtype=float), np.asarray(stds, dtype=float)


def maybe_sample_points(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=max_points, replace=False)
    return x[idx], y[idx]


def plot_hist_overlay(
    train_values: np.ndarray,
    test_values: np.ndarray | None,
    xlabel: str,
    title: str,
    out_path: Path,
    bins: int,
) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.hist(train_values, bins=bins, alpha=0.55, density=True, label="Train", color="tab:blue")
    if test_values is not None:
        plt.hist(test_values, bins=bins, alpha=0.45, density=True, label="Held-out", color="tab:orange")
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved plot to:", out_path)


def plot_mean_vs_std(
    train_mean: np.ndarray,
    train_std: np.ndarray,
    test_mean: np.ndarray | None,
    test_std: np.ndarray | None,
    out_path: Path,
    max_points: int,
    seed: int,
) -> None:
    train_x, train_y = maybe_sample_points(train_mean, train_std, max_points, seed)
    plt.figure(figsize=(7, 5))
    plt.scatter(train_x, train_y, s=12, alpha=0.45, label="Train", color="tab:blue")
    if test_mean is not None and test_std is not None:
        test_x, test_y = maybe_sample_points(test_mean, test_std, max_points, seed + 1)
        plt.scatter(test_x, test_y, s=12, alpha=0.45, label="Held-out", color="tab:orange")
    plt.xlabel("GPR predicted mean count")
    plt.ylabel("GPR predicted std")
    plt.title("GPR predictive uncertainty")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved plot to:", out_path)


def plot_predicted_vs_true(
    train_mean: np.ndarray,
    train_degrees: np.ndarray,
    test_mean: np.ndarray | None,
    test_degrees: np.ndarray | None,
    out_path: Path,
    max_points: int,
    seed: int,
) -> None:
    train_x, train_y = maybe_sample_points(train_degrees, train_mean, max_points, seed)
    plt.figure(figsize=(6, 6))
    plt.scatter(train_x, train_y, s=12, alpha=0.45, label="Train", color="tab:blue")
    all_x = [train_degrees]
    all_y = [train_mean]
    if test_mean is not None and test_degrees is not None:
        test_x, test_y = maybe_sample_points(test_degrees, test_mean, max_points, seed + 1)
        plt.scatter(test_x, test_y, s=12, alpha=0.45, label="Held-out", color="tab:orange")
        all_x.append(test_degrees)
        all_y.append(test_mean)
    max_axis = max(float(np.max(np.concatenate(all_x))), float(np.max(np.concatenate(all_y))), 1.0)
    plt.plot([0, max_axis], [0, max_axis], linestyle="--", color="black", linewidth=1, alpha=0.65)
    plt.xlabel("True out-degree")
    plt.ylabel("GPR predicted mean count")
    plt.title("GPR predicted count vs true out-degree")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved plot to:", out_path)


def save_summary(
    rows: list[dict],
    out_path: Path,
) -> None:
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("Saved summary to:", out_path)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_true - y_pred
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))

    if np.std(y_true) <= 1e-12 or np.std(y_pred) <= 1e-12:
        pearson = np.nan
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])

    spearman = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    return {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "pearson_corr": pearson,
        "spearman_corr": float(spearman) if pd.notna(spearman) else np.nan,
    }


def constant_baseline_predictions(train_degrees: np.ndarray, eval_degrees: np.ndarray) -> np.ndarray:
    return np.full_like(eval_degrees, fill_value=float(np.mean(train_degrees)), dtype=float)


def fit_permutation_baselines(
    GaussianCountModel,
    train_covariates: np.ndarray,
    train_degrees: np.ndarray,
    eval_covariates: np.ndarray,
    eval_degrees: np.ndarray,
    n_permutations: int,
    seed: int,
    alpha: float,
    n_restarts_optimizer: int,
    max_train_size: int | None,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for i in range(n_permutations):
        shuffled_degrees = np.asarray(train_degrees, dtype=float).copy()
        rng.shuffle(shuffled_degrees)
        shuffled_model = GaussianCountModel(
            seed=seed + i + 1,
            alpha=alpha,
            n_restarts_optimizer=n_restarts_optimizer,
            max_train_size=max_train_size,
        )
        shuffled_model.fit(train_covariates, shuffled_degrees)
        shuffled_pred = np.asarray(shuffled_model.gpr.predict(eval_covariates), dtype=float)
        row = regression_metrics(eval_degrees, shuffled_pred)
        row["permutation"] = i
        rows.append(row)
    return pd.DataFrame(rows)


def plot_permutation_metric(
    permutation_df: pd.DataFrame,
    observed_value: float,
    metric: str,
    out_path: Path,
) -> None:
    values = permutation_df[metric].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0 or not np.isfinite(observed_value):
        return

    plt.figure(figsize=(7, 4.5))
    plt.hist(values, bins=min(20, max(5, len(values))), alpha=0.75, color="tab:gray", label="Shuffled covariates")
    plt.axvline(observed_value, color="tab:red", linewidth=2, label="Observed GPR")
    plt.xlabel(metric)
    plt.ylabel("Count")
    plt.title(f"Permutation baseline for {metric}")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved plot to:", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GPR output diagnostics for GaussianCountModel")
    parser.add_argument("--project_dir", type=str, default=str(DEFAULT_DIFFUSION_PROJECT_DIR))
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--std_name", type=str, default="HIV")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--n_restarts_optimizer", type=int, default=5)
    parser.add_argument("--max_train_size", type=int, default=2000)
    parser.add_argument("--n_permutations", type=int, default=10)
    parser.add_argument("--permutation_n_restarts_optimizer", type=int, default=0)
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--max_plot_points", type=int, default=3000)
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--no_node_split",
        action="store_true",
        help="Fit and plot on all nodes without a train/held-out node split.",
    )
    args = parser.parse_args()

    project_dir = resolve_path(args.project_dir, DEFAULT_DIFFUSION_PROJECT_DIR)
    data_dir = resolve_path(args.data_dir, DEFAULT_DATA_DIR, base_dir=project_dir)
    out_dir = resolve_path(args.out_dir, DEFAULT_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not project_dir.exists():
        raise FileNotFoundError(f"diffusion project_dir does not exist: {project_dir}")
    if not data_dir.exists():
        raise FileNotFoundError(f"ICPSR data_dir does not exist: {data_dir}")

    sys.path.insert(0, str(project_dir))
    from src.data.icpsr_loader import ICPSRGraphData
    from src.models.count_model.gaussian_count_model import GaussianCountModel

    print("--------------------------------------------------------")
    print("Load ICPSR Graph Data")
    print("--------------------------------------------------------")
    graph_data = ICPSRGraphData(str(data_dir), args.std_name)
    all_nodes = set(graph_data.covariates)

    if args.no_node_split:
        train_nodes = all_nodes
        test_nodes: set[int] = set()
        split_label = "all_nodes"
    else:
        train_nodes, test_nodes = graph_data.train_test_node_split(
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
        split_label = f"split{args.test_fraction}"

    train_covariates, train_degrees, train_common_nodes = load_node_arrays(graph_data, train_nodes)
    if args.no_node_split:
        test_covariates = None
        test_degrees = None
        test_common_nodes: list[int] = []
    else:
        test_covariates, test_degrees, test_common_nodes = load_node_arrays(graph_data, test_nodes)

    print(f"  disease: {args.std_name}")
    print(f"  data_dir: {data_dir}")
    print(f"  train/all nodes used for GPR fit: {len(train_common_nodes)}")
    if not args.no_node_split:
        print(f"  held-out nodes for diagnostics: {len(test_common_nodes)}")

    print("--------------------------------------------------------")
    print("Fit GaussianCountModel")
    print("--------------------------------------------------------")
    count_model = GaussianCountModel(
        seed=args.seed,
        alpha=args.alpha,
        n_restarts_optimizer=args.n_restarts_optimizer,
        max_train_size=args.max_train_size,
    )
    count_model.fit(train_covariates, train_degrees)
    print("  fitted GPR count model")
    print(f"  kernel: {count_model.gpr.kernel_}")

    print("--------------------------------------------------------")
    print("Predict GPR mean/std")
    print("--------------------------------------------------------")
    train_mean, train_std = predict_gpr(count_model, train_covariates)
    if test_covariates is not None and len(test_covariates) > 0:
        test_mean, test_std = predict_gpr(count_model, test_covariates)
    else:
        test_mean = None
        test_std = None

    tag = (
        f"gpr_{args.std_name}"
        f"_{split_label}"
        f"_seed{args.seed}"
        f"_maxtrain{args.max_train_size}"
    )

    plot_hist_overlay(
        train_mean,
        test_mean,
        xlabel="GPR predicted mean count",
        title=f"GPR predicted mean distribution ({args.std_name})",
        out_path=out_dir / f"{tag}_predicted_mean_hist.png",
        bins=args.bins,
    )
    plot_hist_overlay(
        train_std,
        test_std,
        xlabel="GPR predicted std",
        title=f"GPR predicted std distribution ({args.std_name})",
        out_path=out_dir / f"{tag}_predicted_std_hist.png",
        bins=args.bins,
    )
    plot_mean_vs_std(
        train_mean,
        train_std,
        test_mean,
        test_std,
        out_path=out_dir / f"{tag}_mean_vs_std_scatter.png",
        max_points=args.max_plot_points,
        seed=args.seed,
    )
    plot_predicted_vs_true(
        train_mean,
        train_degrees,
        test_mean,
        test_degrees,
        out_path=out_dir / f"{tag}_predicted_vs_true.png",
        max_points=args.max_plot_points,
        seed=args.seed,
    )

    if test_mean is not None and test_degrees is not None and test_covariates is not None:
        eval_split = "held_out"
        eval_covariates = test_covariates
        eval_degrees = test_degrees
        eval_mean = test_mean
    else:
        eval_split = "all"
        eval_covariates = train_covariates
        eval_degrees = train_degrees
        eval_mean = train_mean

    metric_rows = []
    observed_metrics = regression_metrics(eval_degrees, eval_mean)
    metric_rows.append({"model": "gpr_observed", "eval_split": eval_split, **observed_metrics})

    constant_pred = constant_baseline_predictions(train_degrees, eval_degrees)
    constant_metrics = regression_metrics(eval_degrees, constant_pred)
    metric_rows.append({"model": "constant_train_mean", "eval_split": eval_split, **constant_metrics})

    metrics_path = out_dir / f"{tag}_dependence_metrics.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    print("Saved dependence metrics to:", metrics_path)

    if args.n_permutations > 0:
        print("--------------------------------------------------------")
        print("Fit shuffled-degree permutation baselines")
        print("--------------------------------------------------------")
        permutation_df = fit_permutation_baselines(
            GaussianCountModel=GaussianCountModel,
            train_covariates=train_covariates,
            train_degrees=train_degrees,
            eval_covariates=eval_covariates,
            eval_degrees=eval_degrees,
            n_permutations=args.n_permutations,
            seed=args.seed,
            alpha=args.alpha,
            n_restarts_optimizer=args.permutation_n_restarts_optimizer,
            max_train_size=args.max_train_size,
        )
        permutation_path = out_dir / f"{tag}_permutation_metrics.csv"
        permutation_df.to_csv(permutation_path, index=False)
        print("Saved permutation metrics to:", permutation_path)

        permutation_summary = permutation_df[["r2", "mae", "rmse", "pearson_corr", "spearman_corr"]].agg(
            ["mean", "std", "min", "max"]
        )
        permutation_summary_path = out_dir / f"{tag}_permutation_summary.csv"
        permutation_summary.to_csv(permutation_summary_path)
        print("Saved permutation summary to:", permutation_summary_path)

        for metric in ("r2", "rmse", "pearson_corr", "spearman_corr"):
            plot_permutation_metric(
                permutation_df,
                observed_value=observed_metrics[metric],
                metric=metric,
                out_path=out_dir / f"{tag}_permutation_{metric}.png",
            )

    rows = [
        {
            "split": "train" if not args.no_node_split else "all",
            "n": len(train_mean),
            "true_degree_mean": float(np.mean(train_degrees)),
            "true_degree_std": float(np.std(train_degrees)),
            "predicted_mean_mean": float(np.mean(train_mean)),
            "predicted_mean_std": float(np.std(train_mean)),
            "predicted_std_mean": float(np.mean(train_std)),
            "predicted_std_std": float(np.std(train_std)),
        }
    ]
    if test_mean is not None and test_std is not None and test_degrees is not None:
        rows.append(
            {
                "split": "held_out",
                "n": len(test_mean),
                "true_degree_mean": float(np.mean(test_degrees)),
                "true_degree_std": float(np.std(test_degrees)),
                "predicted_mean_mean": float(np.mean(test_mean)),
                "predicted_mean_std": float(np.std(test_mean)),
                "predicted_std_mean": float(np.mean(test_std)),
                "predicted_std_std": float(np.std(test_std)),
            }
        )
    save_summary(rows, out_dir / f"{tag}_summary.csv")

    print("--------------------------------------------------------")
    print("Summary")
    print("--------------------------------------------------------")
    for row in rows:
        print(
            f"  {row['split']}: n={row['n']}, "
            f"true_degree={row['true_degree_mean']:.3f}+/-{row['true_degree_std']:.3f}, "
            f"pred_mean={row['predicted_mean_mean']:.3f}+/-{row['predicted_mean_std']:.3f}, "
            f"pred_std={row['predicted_std_mean']:.3f}+/-{row['predicted_std_std']:.3f}"
        )
    print(
        f"  dependence check on {eval_split}: "
        f"GPR R2={observed_metrics['r2']:.3f}, MAE={observed_metrics['mae']:.3f}, "
        f"RMSE={observed_metrics['rmse']:.3f}, Pearson={observed_metrics['pearson_corr']:.3f}, "
        f"Spearman={observed_metrics['spearman_corr']:.3f}"
    )
    print(
        f"  constant baseline: "
        f"R2={constant_metrics['r2']:.3f}, MAE={constant_metrics['mae']:.3f}, "
        f"RMSE={constant_metrics['rmse']:.3f}"
    )
    print(f"  out_dir: {out_dir}")


if __name__ == "__main__":
    main()
