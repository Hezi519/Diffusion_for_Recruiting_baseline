"""
Plot a comparison of recruiting curves from per-method NPZ/CSV results.

Reads `eval_results_<tag>.npz` and `trajectories_<tag>.csv` for each requested
method and emits the same normalized-axes comparison figure that the old
multi-driver produced.

Usage:
    python -m src.scripts.plot_comparison results/run1 \
        --methods dqn,structured,random \
        --run_tag HIV_B100_F10_disc0.9_seed42 \
        --gamma 0.9
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.scripts.eval_utils import save_comparison_curves


@dataclass
class LoadedBundle:
    method: str
    x: np.ndarray
    y: np.ndarray
    y_std: np.ndarray
    traj_rows: list[dict]


def load_bundle(results_dir: str, run_tag: str, method: str) -> LoadedBundle:
    npz_path = os.path.join(results_dir, f"eval_results_{run_tag}_{method}.npz")
    csv_path = os.path.join(results_dir, f"trajectories_{run_tag}_{method}.csv")

    data = np.load(npz_path)
    traj_rows = pd.read_csv(csv_path).to_dict("records") if os.path.exists(csv_path) else []

    return LoadedBundle(
        method=method,
        x=data["x"],
        y=data["y"],
        y_std=data["y_std"],
        traj_rows=traj_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot comparison of recruiting curves from disk")
    parser.add_argument("results_dir", type=str, help="Directory containing per-method NPZ/CSV files")
    parser.add_argument("--methods", type=str, required=True,
                        help="Comma-separated list of methods to overlay (e.g. dqn,structured,random)")
    parser.add_argument("--run_tag", type=str, required=True,
                        help="Shared run tag prefix (everything before _<method>)")
    parser.add_argument("--gamma", type=float, default=0.9,
                        help="Discount factor used in the plot title")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    bundles = {m: load_bundle(args.results_dir, args.run_tag, m) for m in methods}

    save_comparison_curves(
        bundles=bundles,
        results_dir=args.results_dir,
        run_tag=args.run_tag,
        gamma=args.gamma,
    )


if __name__ == "__main__":
    main()
