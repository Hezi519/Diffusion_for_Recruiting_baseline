"""
Generate per-method and combined comparison figures from saved NPZ results.

Accepts explicit paths to each method's NPZ file (as produced by the individual
driver scripts) and outputs:
  - random_{tag}.png
  - dqn_{tag}.png
  - structured_rl_{tag}.png
  - comparison_{tag}.png
  - stats_{tag}.csv

Usage:
    python -m src.scripts.plot_comparison \
        --random_npz results/eval_results_random_HIV_B20_F10_disc0.9_seed42.npz \
        --dqn_npz    results/eval_results_HIV_B20_F10_disc0.9_seed42_train50_hid128.npz \
        --rl_npz     results/eval_results_HIV_B20_F10_disc0.9_seed42_train50_hid128_structured.npz \
        --out_dir    results \
        --tag        B20 \
        --budget     20 \
        --discount   0.9
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


COLORS = {
    "random": "tab:orange",
    "dqn":    "tab:blue",
    "rl":     "tab:green",
}
LABELS = {
    "random": "Random",
    "dqn":    "Budget-DQN",
    "rl":     "Structured RL",
}


def _load(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(path)
    return d["x"], d["y"], d["y_std"]


def _prep_curve(
    x: np.ndarray,
    y: np.ndarray,
    y_std: np.ndarray,
    budget: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    - Prepend (0, 0, 0) origin.
    - Deduplicate: keep only the first occurrence of each unique x value
      so the plateau doesn't produce a dense repeated segment.
    - Extend to budget: if the last x < budget, append (budget, final_y, final_std)
      so all curves reach the same right edge.
    """
    x = np.concatenate([[0.0], x])
    y = np.concatenate([[0.0], y])
    s = np.concatenate([[0.0], y_std])

    # Deduplicate on x (keep first occurrence of each x)
    _, idx = np.unique(x, return_index=True)
    x, y, s = x[idx], y[idx], s[idx]

    # Extend to full budget
    if x[-1] < budget:
        x = np.append(x, float(budget))
        y = np.append(y, y[-1])
        s = np.append(s, s[-1])

    return x, y, s


def _style_ax(ax, budget: int) -> None:
    ax.set_xlabel("Cumulative budget spent", fontsize=11)
    ax.set_ylabel("Cumulative recruits", fontsize=11)
    ax.set_xlim(0, budget)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def _single_plot(
    x: np.ndarray,
    y: np.ndarray,
    y_std: np.ndarray,
    method: str,
    budget: int,
    discount: float,
    out_path: str,
) -> None:
    xp, yp, sp = _prep_curve(x, y, y_std, budget)
    color = COLORS[method]
    final_val = yp[-1]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xp, yp, color=color, linewidth=2, label=f"{LABELS[method]}  (final: {final_val:.1f})")
    ax.fill_between(
        xp,
        np.maximum(0.0, yp - sp),
        yp + sp,
        color=color,
        alpha=0.20,
        label="_nolegend_",
    )
    # Annotate final value
    ax.annotate(
        f"{final_val:.1f}",
        xy=(xp[-1], yp[-1]),
        xytext=(-8, 6),
        textcoords="offset points",
        fontsize=9,
        color=color,
        ha="right",
    )
    _style_ax(ax, budget)
    ax.set_title(f"{LABELS[method]}  (B={budget}, γ={discount})", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _comparison_plot(
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    budget: int,
    discount: float,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, (x, y, y_std) in curves.items():
        xp, yp, sp = _prep_curve(x, y, y_std, budget)
        color = COLORS[method]
        final_val = yp[-1]
        label = f"{LABELS[method]}  ({final_val:.1f} recruits)"

        ax.plot(xp, yp, color=color, linewidth=2, label=label)
        ax.fill_between(
            xp,
            np.maximum(0.0, yp - sp),
            yp + sp,
            color=color,
            alpha=0.15,
        )
        # Mark final value with a dot + label
        ax.plot(xp[-1], yp[-1], "o", color=color, markersize=6, zorder=5)
        ax.annotate(
            f"{final_val:.1f}",
            xy=(xp[-1], yp[-1]),
            xytext=(-10, 6),
            textcoords="offset points",
            fontsize=9,
            color=color,
            ha="right",
        )

    _style_ax(ax, budget)
    ax.set_title(f"Recruiting performance comparison  (B={budget}, γ={discount})", fontsize=12)
    ax.legend(fontsize=10, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _stats_table(
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    budget: int,
    out_path: str,
) -> None:
    rows = []
    for method, (x, y, y_std) in curves.items():
        xp, yp, sp = _prep_curve(x, y, y_std, budget)

        final_recruits = yp[-1]
        final_std      = sp[-1]
        budget_used    = x[x > 0].max() if (x > 0).any() else 0.0
        budget_pct     = 100.0 * budget_used / budget
        efficiency     = final_recruits / budget_used if budget_used > 0 else 0.0
        # AUC via trapezoidal rule (normalised by budget so it's comparable)
        auc            = float(np.trapz(yp, xp)) / budget

        rows.append({
            "Method":               LABELS[method],
            "Final Recruits (mean)": f"{final_recruits:.1f}",
            "Std":                   f"{final_std:.1f}",
            "Budget Used":           f"{budget_used:.0f} / {budget}  ({budget_pct:.1f}%)",
            "Recruits / Budget":     f"{efficiency:.3f}",
            "AUC (normalised)":      f"{auc:.1f}",
        })

    # Print table
    col_order = ["Method", "Final Recruits (mean)", "Std",
                 "Budget Used", "Recruits / Budget", "AUC (normalised)"]
    col_w = {c: max(len(c), max(len(r[c]) for r in rows)) + 2 for c in col_order}

    header = "  ".join(c.ljust(col_w[c]) for c in col_order)
    sep    = "  ".join("-" * col_w[c] for c in col_order)
    print("\n" + header)
    print(sep)
    for r in rows:
        print("  ".join(r[c].ljust(col_w[c]) for c in col_order))
    print()

    # Save CSV
    import csv
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=col_order)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Stats saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_npz", required=True, help="Path to random policy NPZ")
    parser.add_argument("--dqn_npz",    required=True, help="Path to DQN NPZ")
    parser.add_argument("--rl_npz",     required=True, help="Path to structured RL NPZ")
    parser.add_argument("--out_dir",    default="results")
    parser.add_argument("--tag",        default="", help="Suffix for output filenames, e.g. B20")
    parser.add_argument("--budget",     type=int,   default=100)
    parser.add_argument("--discount",   type=float, default=0.9)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""

    curves: dict[str, tuple] = {}
    for method, path in [("random", args.random_npz),
                          ("dqn",    args.dqn_npz),
                          ("rl",     args.rl_npz)]:
        x, y, y_std = _load(path)
        curves[method] = (x, y, y_std)
        _single_plot(
            x, y, y_std, method,
            budget=args.budget,
            discount=args.discount,
            out_path=os.path.join(args.out_dir, f"{method}{suffix}.png"),
        )

    _comparison_plot(
        curves,
        budget=args.budget,
        discount=args.discount,
        out_path=os.path.join(args.out_dir, f"comparison{suffix}.png"),
    )

    _stats_table(
        curves,
        budget=args.budget,
        out_path=os.path.join(args.out_dir, f"stats{suffix}.csv"),
    )


if __name__ == "__main__":
    main()
