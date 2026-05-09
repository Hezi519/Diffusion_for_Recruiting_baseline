"""
Visualization for the tunnel-vision synthetic experiment.

Four-panel figure:
  A) Node-type transition diagram (schematic)
  B) Final-recruit bar chart (mean ± std across eval episodes)
  C) Cumulative recruits vs. round (median + IQR)
  D) Frontier size vs. round (median + IQR)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from pathlib import Path

RESULTS = Path("results/tunnel_vision")
TAG = "tunnel_vision_B100_F10_rA5.0_rB2.0_pc0.9_disc0.9_seed42_train2000"
METHODS = ["random", "dqn", "structured"]
LABELS = {"random": "Random", "dqn": "Budget-DQN\n(TypeA-only)", "structured": "Structured RL"}
COLORS = {"random": "#888888", "dqn": "#E45C3A", "structured": "#4A90D9"}
FILLS  = {"random": "#CCCCCC", "dqn": "#F7C4B8", "structured": "#B8D4F5"}

# ── load data ────────────────────────────────────────────────────────────────

trajs = {}
for m in METHODS:
    trajs[m] = pd.read_csv(RESULTS / f"trajectories_{TAG}_{m}.csv")

evals = {}
for m in METHODS:
    d = np.load(RESULTS / f"eval_results_{TAG}_{m}.npz")
    evals[m] = {"x": d["x"], "y": d["y"], "y_std": d["y_std"]}

# per-method final-episode total recruits (last row of each episode)
finals = {}
for m in METHODS:
    df = trajs[m]
    finals[m] = df.groupby("episode")["cumulative_recruits"].last().values


def round_stats(df, col, max_round=12):
    """Median + IQR of `col` per round across episodes."""
    rows = []
    for r in range(1, max_round + 1):
        vals = df[df["round"] == r][col].dropna().values
        if len(vals) == 0:
            rows.append((r, np.nan, np.nan, np.nan))
        else:
            rows.append((r, np.median(vals), np.percentile(vals, 25), np.percentile(vals, 75)))
    return np.array(rows)  # (max_round, 4): round, med, q25, q75


# ── figure layout ─────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
              left=0.07, right=0.97, top=0.93, bottom=0.07)

ax_diag  = fig.add_subplot(gs[0, 0])   # A – node-type diagram
ax_bar   = fig.add_subplot(gs[0, 1])   # B – bar chart
ax_cum   = fig.add_subplot(gs[1, 0])   # C – cumulative recruits vs round
ax_front = fig.add_subplot(gs[1, 1])   # D – frontier size vs round

fig.suptitle("Tunnel-Vision Synthetic Experiment\n"
             r"TypeA: rate=5, offspring→C  |  TypeB: rate=2, offspring→B  |  Budget=100, cap=10/round",
             fontsize=11, y=0.985)

# ── Panel A: node-type transition diagram ─────────────────────────────────────

ax_diag.set_xlim(0, 10)
ax_diag.set_ylim(0, 8)
ax_diag.axis("off")
ax_diag.set_title("A   Node-Type Transition Diagram", fontsize=10, loc="left", pad=6)

node_defs = {
    "A": dict(xy=(2, 6.0), color="#E45C3A", label="Type A\n(boom-bust)\nrate = 5"),
    "B": dict(xy=(2, 2.5), color="#4A90D9", label="Type B\n(sustainable)\nrate = 2"),
    "C": dict(xy=(7, 4.2), color="#888888", label="Type C\n(dead-end)\nrate ≈ 0"),
}

R = 0.85
for key, nd in node_defs.items():
    circle = plt.Circle(nd["xy"], R, color=nd["color"], zorder=3, alpha=0.85)
    ax_diag.add_patch(circle)
    ax_diag.text(*nd["xy"], nd["label"], ha="center", va="center",
                 fontsize=7.5, color="white", fontweight="bold", zorder=4,
                 linespacing=1.4)

def arrow(ax, src, dst, label, color, rad=0.0, offset=(0, 0)):
    x0, y0 = node_defs[src]["xy"]
    x1, y1 = node_defs[dst]["xy"]
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=2)
    mx, my = (x0 + x1) / 2 + offset[0], (y0 + y1) / 2 + offset[1]
    ax.text(mx, my, label, ha="center", va="center", fontsize=7.5,
            color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

arrow(ax_diag, "A", "C", "offspring →", "#E45C3A", rad=-0.25, offset=(0.5, 0.3))
arrow(ax_diag, "B", "B", "", "#4A90D9")   # self-loop placeholder
arrow(ax_diag, "C", "C", "", "#888888")   # self-loop placeholder

# Self-loops drawn manually
for key, (cx, cy), color in [("B", (2, 2.5), "#4A90D9"), ("C", (7, 4.2), "#888888")]:
    loop = mpatches.Arc((cx - 0.1, cy - R * 1.2), 1.3, 1.3, angle=0,
                        theta1=200, theta2=340, color=color, lw=1.8)
    ax_diag.add_patch(loop)
    ax_diag.annotate("", xy=(cx - 0.55, cy - R * 1.55),
                     xytext=(cx + 0.45, cy - R * 1.45),
                     arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8))
    lbl = "offspring →\n(self-replicating)" if key == "B" else "offspring →\n(stays dead)"
    ax_diag.text(cx - 0.1, cy - R * 2.4, lbl, ha="center", va="top",
                 fontsize=7, color=color, fontweight="bold")

ax_diag.text(5, 7.4,
             "DQN targets A (high rate)\n→ frontier collapses to C after 1 round",
             ha="center", va="top", fontsize=7.5, color="#E45C3A",
             style="italic",
             bbox=dict(boxstyle="round,pad=0.3", fc="#FFF0ED", ec="#E45C3A", lw=0.8))
ax_diag.text(5, 0.4,
             "Structured RL invests in B\n→ self-sustaining chain, many rounds",
             ha="center", va="bottom", fontsize=7.5, color="#4A90D9",
             style="italic",
             bbox=dict(boxstyle="round,pad=0.3", fc="#EDF3FC", ec="#4A90D9", lw=0.8))

# ── Panel B: bar chart ────────────────────────────────────────────────────────

ax_bar.set_title("B   Final Recruits (30 eval episodes)", fontsize=10, loc="left", pad=6)
xs = np.arange(len(METHODS))
means = [finals[m].mean() for m in METHODS]
stds  = [finals[m].std()  for m in METHODS]
bars = ax_bar.bar(xs, means, color=[COLORS[m] for m in METHODS],
                  width=0.5, zorder=3, alpha=0.85)
ax_bar.errorbar(xs, means, yerr=stds, fmt="none", color="black",
                capsize=5, lw=1.5, zorder=4)
for i, (m, mn, sd) in enumerate(zip(METHODS, means, stds)):
    ax_bar.text(i, mn + sd + 1.5, f"{mn:.1f}\n±{sd:.1f}", ha="center",
                va="bottom", fontsize=8.5, color=COLORS[m], fontweight="bold")
ax_bar.set_xticks(xs)
ax_bar.set_xticklabels([LABELS[m] for m in METHODS], fontsize=9)
ax_bar.set_ylabel("Total recruits", fontsize=9)
ax_bar.set_ylim(0, max(means) + max(stds) + 18)
ax_bar.yaxis.grid(True, alpha=0.3, zorder=0)
ax_bar.set_axisbelow(True)

# ── Panel C: cumulative recruits vs round ────────────────────────────────────

ax_cum.set_title("C   Cumulative Recruits vs. Round", fontsize=10, loc="left", pad=6)
max_round = 12
for m in METHODS:
    st = round_stats(trajs[m], "cumulative_recruits", max_round)
    rr, med, q25, q75 = st[:, 0], st[:, 1], st[:, 2], st[:, 3]
    mask = ~np.isnan(med)
    ax_cum.plot(rr[mask], med[mask], color=COLORS[m], lw=2, label=LABELS[m].replace("\n", " "))
    ax_cum.fill_between(rr[mask], q25[mask], q75[mask],
                        color=FILLS[m], alpha=0.5)

ax_cum.set_xlabel("Round", fontsize=9)
ax_cum.set_ylabel("Cumulative recruits (median + IQR)", fontsize=9)
ax_cum.set_xlim(1, max_round)
ax_cum.set_xticks(range(1, max_round + 1))
ax_cum.yaxis.grid(True, alpha=0.3)
ax_cum.set_axisbelow(True)
ax_cum.legend(fontsize=8, loc="upper left")

# ── Panel D: frontier size vs round ──────────────────────────────────────────

ax_front.set_title("D   Frontier Size vs. Round (next-round)", fontsize=10, loc="left", pad=6)
for m in METHODS:
    st = round_stats(trajs[m], "next_frontier_size", max_round)
    rr, med, q25, q75 = st[:, 0], st[:, 1], st[:, 2], st[:, 3]
    mask = ~np.isnan(med)
    ax_front.plot(rr[mask], med[mask], color=COLORS[m], lw=2, label=LABELS[m].replace("\n", " "))
    ax_front.fill_between(rr[mask], q25[mask], q75[mask],
                          color=FILLS[m], alpha=0.5)

ax_front.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)
ax_front.set_xlabel("Round", fontsize=9)
ax_front.set_ylabel("Frontier size (median + IQR)", fontsize=9)
ax_front.set_xlim(1, max_round)
ax_front.set_xticks(range(1, max_round + 1))
ax_front.yaxis.grid(True, alpha=0.3)
ax_front.set_axisbelow(True)
ax_front.legend(fontsize=8, loc="upper right")

# ── save ─────────────────────────────────────────────────────────────────────

out = RESULTS / "tunnel_vision_overview.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
