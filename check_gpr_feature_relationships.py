"""Quick relationship checks for Gaussian count-model outputs.

This standalone diagnostic fits ``GaussianCountModel`` on ICPSR node
covariates, predicts GPR mean/std, and checks whether those outputs relate to
graph structure, parent/child covariate geometry, and local covariate density.
It writes compact CSV summaries plus a few overview plots.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx
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
DEFAULT_OUT_DIR = REPO_DIR / "results" / "gpr_feature_relationships"


def resolve_path(path: str | None, default: Path, base_dir: Path | None = None) -> Path:
    candidate = Path(path).expanduser() if path else default
    if not candidate.is_absolute() and base_dir is not None:
        candidate = base_dir / candidate
    return candidate.resolve()


def finite_or_nan(value: float) -> float:
    return float(value) if np.isfinite(value) else np.nan


def summarize_values(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def pairwise_distances_to(
    center: np.ndarray,
    other_nodes: list[int],
    covariates: dict[int, np.ndarray],
) -> list[float]:
    distances = []
    for other in other_nodes:
        if other in covariates:
            distances.append(float(np.linalg.norm(covariates[other] - center)))
    return distances


def covariance_spread(vectors: list[np.ndarray]) -> dict[str, float]:
    if len(vectors) < 2:
        return {
            "spread_trace": np.nan,
            "spread_max_eig": np.nan,
            "spread_anisotropy": np.nan,
        }
    x = np.asarray(vectors, dtype=float)
    cov = np.cov(x, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    max_eig = float(np.max(eigvals))
    min_positive = float(np.min(eigvals[eigvals > 1e-12])) if np.any(eigvals > 1e-12) else np.nan
    return {
        "spread_trace": float(np.trace(cov)),
        "spread_max_eig": max_eig,
        "spread_anisotropy": finite_or_nan(max_eig / min_positive) if min_positive else np.nan,
    }


def graph_depths(digraph: nx.DiGraph, roots: np.ndarray) -> dict[int, int]:
    root_nodes = [int(r) for r in roots if r in digraph]
    if not root_nodes:
        return {}
    depths = {root: 0 for root in root_nodes}
    queue = list(root_nodes)
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        for child in digraph.successors(node):
            if child not in depths:
                depths[child] = depths[node] + 1
                queue.append(child)
    return depths


def compute_local_density_features(
    covariate_matrix: np.ndarray,
    k_values: tuple[int, ...],
    radius_quantile: float,
) -> pd.DataFrame:
    from sklearn.neighbors import NearestNeighbors

    if covariate_matrix.shape[0] < 2:
        return pd.DataFrame(index=np.arange(covariate_matrix.shape[0]))

    max_k = min(max(k_values) + 1, covariate_matrix.shape[0])
    nn = NearestNeighbors(n_neighbors=max_k)
    nn.fit(covariate_matrix)
    distances, _ = nn.kneighbors(covariate_matrix)
    nonself = distances[:, 1:]

    features: dict[str, np.ndarray] = {}
    features["nn_dist"] = nonself[:, 0]
    for k in k_values:
        if nonself.shape[1] >= k:
            features[f"knn{k}_mean_dist"] = np.mean(nonself[:, :k], axis=1)
            features[f"knn{k}_max_dist"] = np.max(nonself[:, :k], axis=1)

    radius = float(np.quantile(nonself[:, 0], radius_quantile))
    if radius > 0.0:
        radius_nn = NearestNeighbors(radius=radius)
        radius_nn.fit(covariate_matrix)
        radius_neighbors = radius_nn.radius_neighbors(covariate_matrix, return_distance=False)
        features[f"neighbors_within_q{radius_quantile:g}_nn_radius"] = np.asarray(
            [max(0, len(idx) - 1) for idx in radius_neighbors],
            dtype=float,
        )
    return pd.DataFrame(features)


def build_feature_table(graph_data, nodes: list[int]) -> pd.DataFrame:
    digraph = graph_data.digraph
    graph = graph_data.graph
    covariates = graph_data.covariates
    statuses = graph_data.statuses
    depths = graph_depths(digraph, graph_data.roots)

    rows = []
    for node in nodes:
        cov = np.asarray(covariates[node], dtype=float)
        children = [v for v in digraph.successors(node) if v in covariates]
        parents = [u for u in digraph.predecessors(node) if u in covariates]
        neighbors = [v for v in graph.neighbors(node) if v in covariates] if node in graph else []

        child_dists = pairwise_distances_to(cov, children, covariates)
        parent_dists = pairwise_distances_to(cov, parents, covariates)
        neighbor_dists = pairwise_distances_to(cov, neighbors, covariates)
        child_dist_stats = summarize_values(child_dists)
        parent_dist_stats = summarize_values(parent_dists)
        neighbor_dist_stats = summarize_values(neighbor_dists)
        child_spread = covariance_spread([covariates[c] for c in children if c in covariates])

        parent_out_degrees = [digraph.out_degree(p) for p in parents if p in digraph]
        child_out_degrees = [digraph.out_degree(c) for c in children if c in digraph]
        sibling_count = sum(max(0, digraph.out_degree(p) - 1) for p in parents if p in digraph)

        try:
            descendant_count = len(nx.descendants(digraph, node))
        except nx.NetworkXError:
            descendant_count = np.nan

        row = {
            "node": node,
            "status": statuses.get(node, np.nan),
            "is_root": int(node in set(graph_data.roots)),
            "depth": depths.get(node, np.nan),
            "out_degree_children": digraph.out_degree(node) if node in digraph else 0,
            "in_degree_parents": digraph.in_degree(node) if node in digraph else 0,
            "undirected_degree": graph.degree(node) if node in graph else np.nan,
            "sibling_count": sibling_count,
            "descendant_count": descendant_count,
            "parent_out_degree_mean": summarize_values(parent_out_degrees)["mean"],
            "child_out_degree_mean": summarize_values(child_out_degrees)["mean"],
            "child_cov_dist_mean": child_dist_stats["mean"],
            "child_cov_dist_std": child_dist_stats["std"],
            "child_cov_dist_max": child_dist_stats["max"],
            "parent_cov_dist_mean": parent_dist_stats["mean"],
            "neighbor_cov_dist_mean": neighbor_dist_stats["mean"],
            "neighbor_cov_dist_std": neighbor_dist_stats["std"],
            **{f"child_cov_{k}": v for k, v in child_spread.items()},
        }
        rows.append(row)

    return pd.DataFrame(rows)


def add_covariate_dimension_features(df: pd.DataFrame, covariate_matrix: np.ndarray) -> pd.DataFrame:
    cov_df = pd.DataFrame(
        covariate_matrix,
        columns=[f"cov_dim_{i:02d}" for i in range(covariate_matrix.shape[1])],
    )
    cov_df["cov_sum"] = np.sum(covariate_matrix, axis=1)
    cov_df["cov_l2_norm"] = np.linalg.norm(covariate_matrix, axis=1)
    cov_df["cov_nonzero_count"] = np.count_nonzero(covariate_matrix, axis=1)
    return pd.concat([df.reset_index(drop=True), cov_df], axis=1)


def correlation_table(df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number])
    feature_cols = [col for col in numeric.columns if col not in targets and col != "node"]
    rows = []
    for target in targets:
        for feature in feature_cols:
            valid = numeric[[target, feature]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid) < 3 or valid[target].nunique() < 2 or valid[feature].nunique() < 2:
                continue
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "n": len(valid),
                    "pearson": valid[target].corr(valid[feature], method="pearson"),
                    "spearman": valid[target].corr(valid[feature], method="spearman"),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_spearman"] = out["spearman"].abs()
    out["abs_pearson"] = out["pearson"].abs()
    return out.sort_values(["target", "abs_spearman", "abs_pearson"], ascending=[True, False, False])


def plot_top_scatter(
    df: pd.DataFrame,
    corr_df: pd.DataFrame,
    target: str,
    out_path: Path,
    top_n: int,
    max_points: int,
    seed: int,
) -> None:
    top = corr_df[corr_df["target"] == target].head(top_n)
    if top.empty:
        return

    rng = np.random.default_rng(seed)
    plot_df = df.copy()
    if len(plot_df) > max_points:
        plot_df = plot_df.iloc[rng.choice(len(plot_df), size=max_points, replace=False)]

    ncols = min(3, len(top))
    nrows = int(np.ceil(len(top) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for ax, (_, row) in zip(axes.ravel(), top.iterrows()):
        feature = row["feature"]
        valid = plot_df[[feature, target]].replace([np.inf, -np.inf], np.nan).dropna()
        ax.scatter(valid[feature], valid[target], s=12, alpha=0.45)
        ax.set_xlabel(feature)
        ax.set_ylabel(target)
        ax.set_title(f"rho={row['spearman']:.3f}, r={row['pearson']:.3f}")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    for ax in axes.ravel()[len(top) :]:
        ax.axis("off")
    fig.suptitle(f"Top relationships for {target}", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved plot to:", out_path)


def plot_correlation_heatmap(corr_df: pd.DataFrame, out_path: Path, top_n_per_target: int) -> None:
    if corr_df.empty:
        return
    top = corr_df.groupby("target", group_keys=False).head(top_n_per_target)
    pivot = top.pivot(index="feature", columns="target", values="spearman").fillna(0.0)
    if pivot.empty:
        return

    fig_height = max(5, 0.32 * len(pivot))
    fig, ax = plt.subplots(figsize=(8, fig_height))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax.set_title("Spearman correlations")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print("Saved plot to:", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check simple relationships with GPR mean/std")
    parser.add_argument("--project_dir", type=str, default=str(DEFAULT_DIFFUSION_PROJECT_DIR))
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--std_name", type=str, default="HIV")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--no_node_split", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--n_restarts_optimizer", type=int, default=5)
    parser.add_argument("--max_train_size", type=int, default=2000)
    parser.add_argument("--knn", type=int, nargs="+", default=[5, 10, 25])
    parser.add_argument("--radius_quantile", type=float, default=0.25)
    parser.add_argument("--top_n", type=int, default=12)
    parser.add_argument("--max_plot_points", type=int, default=3000)
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    project_dir = resolve_path(args.project_dir, DEFAULT_DIFFUSION_PROJECT_DIR)
    data_dir = resolve_path(args.data_dir, DEFAULT_DATA_DIR, base_dir=project_dir)
    out_dir = resolve_path(args.out_dir, DEFAULT_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(project_dir))
    from src.data.icpsr_loader import ICPSRGraphData
    from src.models.count_model.gaussian_count_model import GaussianCountModel

    print("--------------------------------------------------------")
    print("Load data")
    print("--------------------------------------------------------")
    graph_data = ICPSRGraphData(str(data_dir), args.std_name)
    all_nodes = set(graph_data.covariates) & set(graph_data.node_degrees)

    if args.no_node_split:
        train_nodes = all_nodes
        eval_nodes = sorted(all_nodes)
        split_label = "all_nodes"
    else:
        train_nodes, test_nodes = graph_data.train_test_node_split(
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
        train_nodes = train_nodes & all_nodes
        eval_nodes = sorted(test_nodes & all_nodes)
        split_label = f"heldout{args.test_fraction:g}"

    train_nodes_sorted = sorted(train_nodes)
    train_covariates = np.asarray([graph_data.covariates[n] for n in train_nodes_sorted], dtype=np.float64)
    train_degrees = np.asarray([graph_data.node_degrees[n] for n in train_nodes_sorted], dtype=np.float64)
    eval_covariates = np.asarray([graph_data.covariates[n] for n in eval_nodes], dtype=np.float64)
    eval_degrees = np.asarray([graph_data.node_degrees[n] for n in eval_nodes], dtype=np.float64)

    print(f"  disease: {args.std_name}")
    print(f"  fit nodes: {len(train_nodes_sorted)}")
    print(f"  relationship-check nodes: {len(eval_nodes)} ({split_label})")

    print("--------------------------------------------------------")
    print("Fit and predict GaussianCountModel")
    print("--------------------------------------------------------")
    count_model = GaussianCountModel(
        seed=args.seed,
        alpha=args.alpha,
        n_restarts_optimizer=args.n_restarts_optimizer,
        max_train_size=args.max_train_size,
    )
    count_model.fit(train_covariates, train_degrees)
    gpr_mean, gpr_std = count_model.gpr.predict(eval_covariates, return_std=True)

    print("--------------------------------------------------------")
    print("Build feature table")
    print("--------------------------------------------------------")
    feature_df = build_feature_table(graph_data, eval_nodes)
    density_df = compute_local_density_features(
        eval_covariates,
        k_values=tuple(sorted(set(args.knn))),
        radius_quantile=args.radius_quantile,
    )
    feature_df = pd.concat([feature_df.reset_index(drop=True), density_df.reset_index(drop=True)], axis=1)
    feature_df = add_covariate_dimension_features(feature_df, eval_covariates)
    feature_df.insert(1, "true_out_degree", eval_degrees)
    feature_df.insert(2, "gpr_mean", np.asarray(gpr_mean, dtype=float))
    feature_df.insert(3, "gpr_std", np.asarray(gpr_std, dtype=float))
    feature_df.insert(4, "gpr_residual_true_minus_mean", eval_degrees - np.asarray(gpr_mean, dtype=float))
    feature_df.insert(5, "gpr_abs_residual", np.abs(eval_degrees - np.asarray(gpr_mean, dtype=float)))

    tag = (
        f"gpr_features_{args.std_name}"
        f"_{split_label}"
        f"_seed{args.seed}"
        f"_maxtrain{args.max_train_size}"
    )
    features_path = out_dir / f"{tag}_node_features.csv"
    feature_df.to_csv(features_path, index=False)
    print("Saved node feature table to:", features_path)

    targets = ["gpr_mean", "gpr_std", "gpr_abs_residual", "true_out_degree"]
    corr_df = correlation_table(feature_df, targets=targets)
    corr_path = out_dir / f"{tag}_correlations.csv"
    corr_df.to_csv(corr_path, index=False)
    print("Saved correlations to:", corr_path)

    top_corr_path = out_dir / f"{tag}_top_correlations.csv"
    corr_df.groupby("target", group_keys=False).head(args.top_n).to_csv(top_corr_path, index=False)
    print("Saved top correlations to:", top_corr_path)

    plot_correlation_heatmap(
        corr_df,
        out_path=out_dir / f"{tag}_spearman_heatmap.png",
        top_n_per_target=args.top_n,
    )
    for target in targets:
        plot_top_scatter(
            feature_df,
            corr_df,
            target=target,
            out_path=out_dir / f"{tag}_{target}_top_scatter.png",
            top_n=min(9, args.top_n),
            max_points=args.max_plot_points,
            seed=args.seed,
        )

    print("--------------------------------------------------------")
    print("Top relationships by absolute Spearman correlation")
    print("--------------------------------------------------------")
    for target in targets:
        print(f"\n{target}")
        cols = ["feature", "n", "spearman", "pearson"]
        print(corr_df[corr_df["target"] == target][cols].head(args.top_n).to_string(index=False))


if __name__ == "__main__":
    main()
