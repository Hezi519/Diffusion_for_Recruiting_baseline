"""
Overlay comparison plots across methods (random / dqn / structured / gfp / adaptive).

Each method's driver writes a per-episode trajectories CSV
``trajectories_{run_tag}_{method}.csv`` into ``<base_dir>/<method>/``.
This script reads those, computes per-episode (budget, recruits,
discounted-reward) curves, interpolates onto a common budget grid
(holding each episode's last value flat after it terminates, so early
terminations don't drag the mean leftward), and overlays the methods:

  - overlay_recruits_by_budget_{run_tag}.png
  - overlay_discounted_reward_by_time_{run_tag}.png

Methods whose subdirectory is missing or empty are skipped (so an
unfinished method like gfp is silently ignored).

If a method's subdirectory has a non-canonical name (e.g.
``structured_0.9_t500`` instead of ``structured``), pass it via
``--method_dirs structured=structured_0.9_t500``. The script also
auto-falls-back to any ``<base_dir>/<method>*`` match when
``<base_dir>/<method>`` is missing.

Usage:

    python -m src.scripts.plot_overlay \\
        --base_dir results/synthetic/basic_b100 \\
        --gamma 0.9 \\
        --method_dirs structured=structured_0.9_t500
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_COLORS = {
    "random": "tab:gray",
    "dqn": "tab:blue",
    "structured": "tab:green",
    "gfp": "tab:purple",
    "adaptive": "tab:red",
}

METHOD_LABELS = {
    "random": "Random",
    "dqn": "Budget-DQN + GreedyAlloc",
    "structured": "Three-Head Structured RL",
    "gfp": "Generative Frontier Planning",
    "adaptive": "Adaptive (Poisson + DP)",
}

DEFAULT_METHODS = ("random", "dqn", "structured", "gfp", "adaptive")


def _resolve_method_dir(
    base_dir: str,
    method: str,
    overrides: dict[str, str],
) -> str | None:
    """Resolve the subdirectory for ``method`` under ``base_dir``.

    Priority: explicit override > ``<base_dir>/<method>`` > first
    ``<base_dir>/<method>*`` glob match. Returns None if nothing found.
    """
    if method in overrides:
        candidate = os.path.join(base_dir, overrides[method])
        return candidate if os.path.isdir(candidate) else None
    canonical = os.path.join(base_dir, method)
    if os.path.isdir(canonical):
        return canonical
    # Fallback: pick the first sibling directory whose name starts with
    # "<method>_" (e.g. structured_0.9_t500).
    if os.path.isdir(base_dir):
        prefix = f"{method}_"
        matches = sorted(
            d for d in os.listdir(base_dir)
            if d.startswith(prefix) and os.path.isdir(os.path.join(base_dir, d))
        )
        if matches:
            return os.path.join(base_dir, matches[0])
    return None


def _infer_run_tag(method_dir: str, method: str) -> str | None:
    if not os.path.isdir(method_dir):
        return None
    prefix = "trajectories_"
    suffix = f"_{method}.csv"
    for f in os.listdir(method_dir):
        if f.startswith(prefix) and f.endswith(suffix) and "_all_methods" not in f:
            return f[len(prefix):-len(suffix)]
    return None


def _traj_csv_in(method_dir: str, method: str, run_tag: str) -> str | None:
    path = os.path.join(method_dir, f"trajectories_{run_tag}_{method}.csv")
    return path if os.path.exists(path) else None


def _episode_curves(
    df: pd.DataFrame,
    gamma: float,
) -> list[dict[str, np.ndarray]]:
    """Return per-episode curves with a (0,0,0) origin prepended.

    Each entry has keys: budget, recruits, discounted, round (1..T_ep).
    """
    required = {"episode", "round", "reward",
                "cumulative_budget_spent", "cumulative_recruits"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"trajectory CSV missing columns: {sorted(missing)}")

    eps: list[dict[str, np.ndarray]] = []
    for _, ep_df in df.groupby("episode"):
        ep_df = ep_df.sort_values("round")
        rounds = ep_df["round"].to_numpy(dtype=np.int64)
        rewards = ep_df["reward"].to_numpy(dtype=np.float64)
        discounted = np.cumsum(rewards * (gamma ** (rounds - 1)))
        budget = ep_df["cumulative_budget_spent"].to_numpy(dtype=np.float64)
        recruits = ep_df["cumulative_recruits"].to_numpy(dtype=np.float64)

        eps.append({
            "round": np.concatenate([[0], rounds]),
            "budget": np.concatenate([[0.0], budget]),
            "recruits": np.concatenate([[0.0], recruits]),
            "discounted": np.concatenate([[0.0], discounted]),
        })
    return eps


def _mean_on_budget_grid(
    eps: list[dict[str, np.ndarray]],
    y_key: str,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate each episode's (budget -> y) onto ``grid``.

    Past each episode's terminal budget the curve is held flat at the
    episode's final y (right-extension), which is what we want for early-
    terminating episodes.
    """
    stack = []
    for ep in eps:
        # np.interp linearly interpolates and uses right=last for x > xp[-1]
        y_grid = np.interp(grid, ep["budget"], ep[y_key], left=0.0, right=ep[y_key][-1])
        stack.append(y_grid)
    arr = np.vstack(stack)
    return arr.mean(axis=0), arr.std(axis=0)


def _mean_on_round_grid(
    eps: list[dict[str, np.ndarray]],
    y_key: str,
    max_round: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-round mean/std, holding each episode's last value past its
    terminal round (so episodes that ended early still contribute the
    final reached y to later rounds).

    Returns (rounds_axis, mean, std) where rounds_axis is 0..max_round.
    """
    grid = np.arange(0, max_round + 1)
    stack = []
    for ep in eps:
        ep_rounds = ep["round"]
        ep_y = ep[y_key]
        # Map round -> y; for rounds beyond ep's max, hold last value.
        # rounds in ep_rounds are 0,1,...,T_ep already.
        y_grid = np.empty_like(grid, dtype=np.float64)
        last = ep_y[0]
        idx = 0
        for r in grid:
            while idx < len(ep_rounds) and ep_rounds[idx] <= r:
                last = ep_y[idx]
                idx += 1
            y_grid[r] = last
        stack.append(y_grid)
    arr = np.vstack(stack)
    return grid.astype(np.float64), arr.mean(axis=0), arr.std(axis=0)


def _load_method_curves(
    base_dir: str,
    methods: list[str],
    run_tag: str | None,
    gamma: float,
    method_dirs: dict[str, str] | None = None,
) -> tuple[dict[str, list[dict]], str]:
    method_dirs = method_dirs or {}
    resolved: dict[str, str] = {}
    for m in methods:
        d = _resolve_method_dir(base_dir, m, method_dirs)
        if d is None:
            print(f"[skip] {m}: no directory under {base_dir}")
            continue
        resolved[m] = d

    if run_tag is None:
        tag_votes: Counter[str] = Counter()
        for m, d in resolved.items():
            inferred = _infer_run_tag(d, m)
            if inferred:
                tag_votes[inferred] += 1
        if not tag_votes:
            raise FileNotFoundError(
                f"No trajectories_*_{{method}}.csv under any of "
                f"{list(resolved.values())}"
            )
        run_tag = tag_votes.most_common(1)[0][0]

    out: dict[str, list[dict]] = {}
    for m, d in resolved.items():
        path = _traj_csv_in(d, m, run_tag)
        if path is None:
            print(f"[skip] {m}: no trajectories CSV for run_tag={run_tag} in {d}")
            continue
        df = pd.read_csv(path)
        eps = _episode_curves(df, gamma=gamma)
        out[m] = eps
        print(f"[load] {m}: {path} ({len(eps)} episodes)")
    if not out:
        raise FileNotFoundError(
            f"No trajectories found for run_tag={run_tag} under {base_dir}"
        )
    return out, run_tag


def _overlay_lines(
    series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    y_max = 0.0
    for method, (x, y, s) in series.items():
        color = METHOD_COLORS.get(method, "tab:orange")
        label = METHOD_LABELS.get(method, method)
        ax.plot(x, y, color=color, linewidth=2, label=label)
        ax.fill_between(x, np.maximum(0.0, y - s), y + s, color=color, alpha=0.16)
        ax.plot(float(x[-1]), float(y[-1]), "o", color=color, markersize=5, zorder=5)
        ax.annotate(
            f"{float(y[-1]):.1f}",
            xy=(float(x[-1]), float(y[-1])),
            xytext=(-8, 6),
            textcoords="offset points",
            fontsize=8,
            color=color,
            ha="right",
            va="bottom",
        )
        y_max = max(y_max, float(np.max(y + s)))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(1.0, y_max * 1.12))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[save] {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_dir", required=True,
                        help="Directory containing one subdir per method "
                             "(e.g. results/synthetic/basic)")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS),
                        help="Method subdir names to overlay (missing ones are skipped).")
    parser.add_argument("--run_tag", default=None,
                        help="Run tag suffix, inferred if omitted.")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (defaults to base_dir).")
    parser.add_argument("--gamma", type=float, default=0.9,
                        help="Discount factor used to recompute cumulative discounted reward.")
    parser.add_argument("--budget_grid", type=int, default=200,
                        help="Number of points on the shared budget grid.")
    parser.add_argument("--method_dirs", nargs="*", default=[],
                        help="Optional method=subdir overrides for non-canonical "
                             "directory names (e.g. structured=structured_0.9_t500).")
    args = parser.parse_args()

    method_dirs: dict[str, str] = {}
    for spec in args.method_dirs:
        if "=" not in spec:
            raise ValueError(f"--method_dirs entry must be method=subdir, got {spec!r}")
        k, v = spec.split("=", 1)
        method_dirs[k.strip()] = v.strip()

    eps_by_method, run_tag = _load_method_curves(
        args.base_dir, args.methods, args.run_tag,
        gamma=args.gamma, method_dirs=method_dirs,
    )
    out_dir = args.out_dir or args.base_dir
    os.makedirs(out_dir, exist_ok=True)
    gamma_suffix = f" (discount={args.gamma})"

    # --- Budget axis: shared grid spanning [0, max final budget across methods]
    max_budget = max(
        max(float(ep["budget"][-1]) for ep in eps)
        for eps in eps_by_method.values()
    )
    budget_grid = np.linspace(0.0, max_budget, args.budget_grid)

    recruits_by_budget = {}
    for method, eps in eps_by_method.items():
        r_mean, r_std = _mean_on_budget_grid(eps, "recruits", budget_grid)
        recruits_by_budget[method] = (budget_grid, r_mean, r_std)

    # --- Time axis: per-round mean across episodes, holding flat after terminal.
    max_round = max(int(ep["round"][-1]) for eps in eps_by_method.values() for ep in eps)
    discounted_by_time = {}
    recruits_by_time = {}
    for method, eps in eps_by_method.items():
        t, d_mean, d_std = _mean_on_round_grid(eps, "discounted", max_round)
        discounted_by_time[method] = (t, d_mean, d_std)
        t_r, r_mean, r_std = _mean_on_round_grid(eps, "recruits", max_round)
        recruits_by_time[method] = (t_r, r_mean, r_std)

    _overlay_lines(
        recruits_by_budget,
        xlabel="Budget spent",
        ylabel="Cumulative recruits",
        title=f"Cumulative recruits by budget{gamma_suffix}",
        out_path=os.path.join(out_dir, f"overlay_recruits_by_budget_{run_tag}.png"),
    )
    _overlay_lines(
        discounted_by_time,
        xlabel="Round",
        ylabel="Accumulated discounted reward",
        title=f"Discounted reward by time{gamma_suffix}",
        out_path=os.path.join(out_dir, f"overlay_discounted_reward_by_time_{run_tag}.png"),
    )
    _overlay_lines(
        recruits_by_time,
        xlabel="Round",
        ylabel="Cumulative recruits",
        title="Cumulative recruits by time",
        out_path=os.path.join(out_dir, f"overlay_recruits_by_time_{run_tag}.png"),
    )


if __name__ == "__main__":
    main()
