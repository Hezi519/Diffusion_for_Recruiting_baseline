"""Combined 3-panel overlay: recruits-by-budget, discounted-reward-by-time,
recruits-by-time — for one (env, discount) setting, saved as a single PNG.

Reuses the loading + averaging helpers from src.scripts.plot_overlay so the
per-panel curves are identical to what that script writes individually.

Usage:

    python -m src.scripts.plot_overlay_combined \\
        --base_dir results/synthetic/basic_b100 \\
        --gamma 1.0 \\
        --out_path results/synthetic/basic_b100/overlay_combined_disc1.0.png \\
        --method_dirs random=random_disc1.0 dqn=dqn_disc1.0 \\
                      structured=structured_disc1.0 gfp=gfp_disc1.0 \\
                      adaptive=adaptive_disc1.0
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.scripts.plot_overlay import (
    DEFAULT_METHODS,
    METHOD_COLORS,
    METHOD_LABELS,
    _load_method_curves,
    _mean_on_budget_grid,
    _mean_on_round_grid,
)


def _plot_overlay_ax(ax, series, xlabel, ylabel, title):
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--run_tag", default=None)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--gamma", type=float, default=0.9,
                        help="Discount factor used to recompute cumulative discounted reward.")
    parser.add_argument("--budget_grid", type=int, default=200)
    parser.add_argument("--method_dirs", nargs="*", default=[],
                        help="Optional method=subdir overrides "
                             "(e.g. structured=structured_disc1.0).")
    parser.add_argument("--suptitle", default=None,
                        help="Top-level figure title. Defaults to base_dir + run_tag.")
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

    max_budget = max(
        max(float(ep["budget"][-1]) for ep in eps)
        for eps in eps_by_method.values()
    )
    budget_grid = np.linspace(0.0, max_budget, args.budget_grid)
    max_round = max(int(ep["round"][-1]) for eps in eps_by_method.values() for ep in eps)

    recruits_by_budget: dict = {}
    discounted_by_time: dict = {}
    recruits_by_time: dict = {}
    for method, eps in eps_by_method.items():
        r_mean, r_std = _mean_on_budget_grid(eps, "recruits", budget_grid)
        recruits_by_budget[method] = (budget_grid, r_mean, r_std)
        t, d_mean, d_std = _mean_on_round_grid(eps, "discounted", max_round)
        discounted_by_time[method] = (t, d_mean, d_std)
        t_r, r2_mean, r2_std = _mean_on_round_grid(eps, "recruits", max_round)
        recruits_by_time[method] = (t_r, r2_mean, r2_std)

    gamma_suffix = f" (discount={args.gamma})"
    fig, axes = plt.subplots(1, 3, figsize=(21, 5.0))
    _plot_overlay_ax(
        axes[0], recruits_by_budget,
        xlabel="Budget spent", ylabel="Cumulative recruits",
        title=f"Cumulative recruits by budget{gamma_suffix}",
    )
    _plot_overlay_ax(
        axes[1], discounted_by_time,
        xlabel="Round", ylabel="Accumulated discounted reward",
        title=f"Discounted reward by time{gamma_suffix}",
    )
    _plot_overlay_ax(
        axes[2], recruits_by_time,
        xlabel="Round", ylabel="Cumulative recruits",
        title="Cumulative recruits by time",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=len(labels),
        bbox_to_anchor=(0.5, -0.02), fontsize=10, frameon=False,
    )
    suptitle = args.suptitle or f"{os.path.basename(args.base_dir.rstrip('/'))} — {run_tag}"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    fig.savefig(args.out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {args.out_path}")


if __name__ == "__main__":
    main()
