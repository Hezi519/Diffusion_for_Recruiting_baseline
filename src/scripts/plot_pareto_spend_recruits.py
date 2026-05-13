"""
Per-dollar efficiency vs. cumulative budget spent.

For each method, plots cumulative recruits / cumulative budget against
cumulative budget. A flat-high curve means "efficient per dollar at every
spend level"; a curve that decays means "early spend was high yield, later
spend wasn't." Trajectories are interpolated onto a shared budget grid and
held flat past each episode's terminal spend so early terminations don't
drag the mean leftward.

Reuses helpers from ``plot_overlay`` to load per-episode curves.

Usage:

    python -m src.scripts.plot_pareto_spend_recruits \
        --base_dir results/synthetic/basic_b100 \
        --gamma 0.9 \
        --method_dirs structured=structured_0.9_t500
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
)


def make_pareto_figure(
    eps_by_method: dict[str, list[dict[str, np.ndarray]]],
    out_path: str,
    budget_grid_n: int = 200,
    gamma: float | None = None,
) -> None:
    max_budget = max(
        max(float(ep["budget"][-1]) for ep in eps)
        for eps in eps_by_method.values()
    )
    grid = np.linspace(0.0, max_budget, budget_grid_n)

    fig, ax = plt.subplots(figsize=(8, 5.0))

    for method, eps in eps_by_method.items():
        color = METHOD_COLORS.get(method, "tab:orange")
        label = METHOD_LABELS.get(method, method)
        r_mean, _ = _mean_on_budget_grid(eps, "recruits", grid)

        # Avoid divide-by-zero near the origin; mask budget < 1 unit.
        mask = grid >= 1.0
        eff = np.zeros_like(r_mean)
        eff[mask] = r_mean[mask] / grid[mask]
        ax.plot(grid[mask], eff[mask], color=color, linewidth=2.0, label=label)

    title_suffix = f" (γ={gamma})" if gamma is not None else ""

    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.0, alpha=0.5,
               label="1 recruit per unit budget (ref)")
    ax.set_xlabel("Cumulative budget spent")
    ax.set_ylabel("Recruits per unit budget (cumulative)")
    ax.set_title(f"Per-dollar efficiency vs. spend{title_suffix}")
    ax.set_ylim(0, None)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, loc="lower left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[save] {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--run_tag", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--budget_grid", type=int, default=200)
    parser.add_argument("--method_dirs", nargs="*", default=[])
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
    out_path = os.path.join(out_dir, f"pareto_spend_vs_recruits_{run_tag}.png")
    make_pareto_figure(
        eps_by_method, out_path,
        budget_grid_n=args.budget_grid, gamma=args.gamma,
    )


if __name__ == "__main__":
    main()
