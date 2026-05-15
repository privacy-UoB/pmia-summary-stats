"""Visual comparison of candidate legend conventions.

Renders the same data under several legend schemes side-by-side so we can pick
a convention before retrofitting it into the real figure scripts
(experiment_nsweep.py, Noise.py, Ordered_*.py, etc.).

Two recurring four-line scenarios are exercised in parallel:
  A: metric x pool   (AUC, single axes)         -- e.g. nsweep_auc.pdf
  B: metric x score  (dual-axis vs noise level) -- e.g. fig2b_D3_random.pdf

Data sources:
  Scenario A: nsweep_results.csv (D3 only).
  Scenario B: results/fig2b_D3_random.npz (Noise.py output).

Run:  python plot_legend_options.py
Out:  legend_comparison.pdf, legend_comparison.png
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from experiment_io import resolve_output_path


# --------------------------------------------------------------------------- #
# Data loaders
# --------------------------------------------------------------------------- #

def load_scenario_A() -> dict:
    df = pd.read_csv("results/nsweep_results.csv")
    d3 = df[df["disease"] == "D3"].sort_values("n").reset_index(drop=True)
    return {
        "x":          d3["n"].to_numpy(float),
        "L1_case":    d3["case_auc_l1"].to_numpy(float),
        "L1_random":  d3["random_auc_l1"].to_numpy(float),
        "LLR_case":   d3["case_auc_llr"].to_numpy(float),
        "LLR_random": d3["random_auc_llr"].to_numpy(float),
    }


SCEN_B_NPZ = "results/fig2b_D3_random.npz"


def load_scenario_B() -> dict:
    if not os.path.exists(SCEN_B_NPZ):
        raise FileNotFoundError(
            f"Missing {SCEN_B_NPZ}. Drop the npz next to the existing pdf "
            f"or run: bash run_noise_experiments.sh d"
        )
    with np.load(SCEN_B_NPZ, allow_pickle=True) as nz:
        return {
            "x":       nz["multiplier"].astype(float),
            "auc_L1":  nz["auc_L1"].astype(float),
            "auc_LLR": nz["auc_LLR"].astype(float),
            "tpr_L1":  nz["tpr_at_fpr_L1"].astype(float),
            "tpr_LLR": nz["tpr_at_fpr_LLR"].astype(float),
        }


# --------------------------------------------------------------------------- #
# Legend option specs
# --------------------------------------------------------------------------- #

# Default tab10-ish red/blue
BLUE   = "#1f77b4"
RED    = "#d62728"
ORANGE = "#ff7f0e"
DBLUE  = "#0b3d80"
LBLUE  = "#7eb6dd"
DRED   = "#8b0000"
LRED   = "#e88d8d"
BLACK  = "#000000"
GREY   = "#888888"
# Colorblind-safe Okabe-Ito-ish pair
CB_BLUE = "#0072B2"   # blue
CB_RED  = "#D55E00"   # vermillion (distinguishable in deutero/protan)


def kw(color, marker, linestyle, **extra):
    base = dict(color=color, marker=marker, linestyle=linestyle,
                markersize=6, linewidth=1.6)
    base.update(extra)
    return base


OPTIONS = [
    {
        "name": "Option 1 - Proposed canonical",
        "desc": "L1=blue / LLR=red ; case=o / random=sq ; AUC=solid / TPR=dashed",
        "A": {
            "L1_case":    kw(BLUE, "o", "-"),
            "L1_random":  kw(BLUE, "s", "-"),
            "LLR_case":   kw(RED,  "o", "-"),
            "LLR_random": kw(RED,  "s", "-"),
        },
        "B": {
            "auc_L1":  kw(BLUE, "o", "-"),
            "auc_LLR": kw(RED,  "o", "-"),
            "tpr_L1":  kw(BLUE, "o", "--"),
            "tpr_LLR": kw(RED,  "o", "--"),
        },
    },
    {
        "name": "Option 2 - Inverted: pool=linestyle, score=marker",
        "desc": "L1=blue / LLR=red ; case=solid / random=dashed ; AUC=o / TPR=sq",
        "A": {
            "L1_case":    kw(BLUE, "o", "-"),
            "L1_random":  kw(BLUE, "o", "--"),
            "LLR_case":   kw(RED,  "o", "-"),
            "LLR_random": kw(RED,  "o", "--"),
        },
        "B": {
            "auc_L1":  kw(BLUE, "o", "-"),
            "auc_LLR": kw(RED,  "o", "-"),
            "tpr_L1":  kw(BLUE, "s", "-"),
            "tpr_LLR": kw(RED,  "s", "-"),
        },
    },
    {
        "name": "Option 3 - Pool-as-colour",
        "desc": "case=blue / random=orange ; L1=solid / LLR=dashed ; AUC=o / TPR=sq",
        "A": {
            "L1_case":    kw(BLUE,   "o", "-"),
            "L1_random":  kw(ORANGE, "o", "-"),
            "LLR_case":   kw(BLUE,   "o", "--"),
            "LLR_random": kw(ORANGE, "o", "--"),
        },
        "B": {
            "auc_L1":  kw(BLUE, "o", "-"),
            "auc_LLR": kw(BLUE, "o", "--"),
            "tpr_L1":  kw(BLUE, "s", "-"),
            "tpr_LLR": kw(BLUE, "s", "--"),
        },
    },
    {
        "name": "Option 4 - Hue pair within metric family",
        "desc": "L1=dark/light blue ; LLR=dark/light red ; dark=case/AUC ; marker redundant",
        "A": {
            "L1_case":    kw(DBLUE, "o", "-"),
            "L1_random":  kw(LBLUE, "s", "-"),
            "LLR_case":   kw(DRED,  "o", "-"),
            "LLR_random": kw(LRED,  "s", "-"),
        },
        "B": {
            "auc_L1":  kw(DBLUE, "o", "-"),
            "auc_LLR": kw(DRED,  "o", "-"),
            "tpr_L1":  kw(LBLUE, "s", "-"),
            "tpr_LLR": kw(LRED,  "s", "-"),
        },
    },
    {
        "name": "Option 5 - Marker-only (drops linestyle dim)",
        "desc": "All solid ; L1=blue / LLR=red ; pool/score by marker o vs sq",
        "A": {
            "L1_case":    kw(BLUE, "o", "-"),
            "L1_random":  kw(BLUE, "s", "-"),
            "LLR_case":   kw(RED,  "o", "-"),
            "LLR_random": kw(RED,  "s", "-"),
        },
        "B": {
            "auc_L1":  kw(BLUE, "o", "-"),
            "auc_LLR": kw(RED,  "o", "-"),
            "tpr_L1":  kw(BLUE, "s", "-"),
            "tpr_LLR": kw(RED,  "s", "-"),
        },
    },
    {
        "name": "Option 6 - Filled/hollow markers",
        "desc": "L1=blue / LLR=red ; case/AUC filled, random/TPR hollow ; AUC=solid / TPR=dashed",
        "A": {
            "L1_case":    kw(BLUE, "o", "-",  markerfacecolor=BLUE,    markeredgecolor=BLUE),
            "L1_random":  kw(BLUE, "o", "-",  markerfacecolor="white", markeredgecolor=BLUE),
            "LLR_case":   kw(RED,  "o", "-",  markerfacecolor=RED,     markeredgecolor=RED),
            "LLR_random": kw(RED,  "o", "-",  markerfacecolor="white", markeredgecolor=RED),
        },
        "B": {
            "auc_L1":  kw(BLUE, "o", "-",  markerfacecolor=BLUE,    markeredgecolor=BLUE),
            "auc_LLR": kw(RED,  "o", "-",  markerfacecolor=RED,     markeredgecolor=RED),
            "tpr_L1":  kw(BLUE, "o", "--", markerfacecolor="white", markeredgecolor=BLUE),
            "tpr_LLR": kw(RED,  "o", "--", markerfacecolor="white", markeredgecolor=RED),
        },
    },
    {
        "name": "Option 7 - Greyscale fallback of #1",
        "desc": "L1=black / LLR=grey ; case=o / random=sq ; AUC=solid / TPR=dashed (B&W check)",
        "A": {
            "L1_case":    kw(BLACK, "o", "-"),
            "L1_random":  kw(BLACK, "s", "-"),
            "LLR_case":   kw(GREY,  "o", "-"),
            "LLR_random": kw(GREY,  "s", "-"),
        },
        "B": {
            "auc_L1":  kw(BLACK, "o", "-"),
            "auc_LLR": kw(GREY,  "o", "-"),
            "tpr_L1":  kw(BLACK, "o", "--"),
            "tpr_LLR": kw(GREY,  "o", "--"),
        },
    },
    {
        "name": "Option 8 - Bold contrast (turned-up #1)",
        "desc": "Option 1 with thicker lines & larger markers - readability stress test",
        "A": {
            "L1_case":    kw(BLUE, "o", "-",  linewidth=2.4, markersize=8),
            "L1_random":  kw(BLUE, "s", "-",  linewidth=2.4, markersize=8),
            "LLR_case":   kw(RED,  "o", "-",  linewidth=2.4, markersize=8),
            "LLR_random": kw(RED,  "s", "-",  linewidth=2.4, markersize=8),
        },
        "B": {
            "auc_L1":  kw(BLUE, "o", "-",  linewidth=2.4, markersize=8),
            "auc_LLR": kw(RED,  "o", "-",  linewidth=2.4, markersize=8),
            "tpr_L1":  kw(BLUE, "o", "--", linewidth=2.4, markersize=8),
            "tpr_LLR": kw(RED,  "o", "--", linewidth=2.4, markersize=8),
        },
    },
    {
        # No legend box - each line labelled inline at its rightmost point.
        "name": "Option 9 - Direct end-of-line labels (no legend box)",
        "desc": "Same colours/markers as #1, but lines labelled at the right endpoint",
        "legend": "endpoint",
        "A": {
            "L1_case":    kw(BLUE, "o", "-"),
            "L1_random":  kw(BLUE, "s", "-"),
            "LLR_case":   kw(RED,  "o", "-"),
            "LLR_random": kw(RED,  "s", "-"),
        },
        "B": {
            "auc_L1":  kw(BLUE, "o", "-"),
            "auc_LLR": kw(RED,  "o", "-"),
            "tpr_L1":  kw(BLUE, "o", "--"),
            "tpr_LLR": kw(RED,  "o", "--"),
        },
    },
    {
        # Scenario B is split into two stacked sub-axes (AUC top, TPR bottom)
        # instead of using a twin axis. Scenario A is unchanged from #1.
        "name": "Option 10 - Stacked AUC/TPR sub-plots (Scen B)",
        "desc": "Scen B: AUC top / TPR bottom (no twin axis) ; L1=blue / LLR=red",
        "B_layout": "stacked",
        "A": {
            "L1_case":    kw(BLUE, "o", "-"),
            "L1_random":  kw(BLUE, "s", "-"),
            "LLR_case":   kw(RED,  "o", "-"),
            "LLR_random": kw(RED,  "s", "-"),
        },
        "B": {
            "auc_L1":  kw(BLUE, "o", "-"),
            "auc_LLR": kw(RED,  "o", "-"),
            "tpr_L1":  kw(BLUE, "o", "-"),
            "tpr_LLR": kw(RED,  "o", "-"),
        },
    },
    {
        # Linestyle carries every secondary dimension; no markers.
        "name": "Option 11 - Linestyle-only, no markers",
        "desc": "L1=blue / LLR=red ; case=solid / random=dotted (A) ; AUC=solid / TPR=dashed (B)",
        "A": {
            "L1_case":    kw(BLUE, None, "-",  linewidth=2.0),
            "L1_random":  kw(BLUE, None, ":",  linewidth=2.0),
            "LLR_case":   kw(RED,  None, "-",  linewidth=2.0),
            "LLR_random": kw(RED,  None, ":",  linewidth=2.0),
        },
        "B": {
            "auc_L1":  kw(BLUE, None, "-",  linewidth=2.0),
            "auc_LLR": kw(RED,  None, "-",  linewidth=2.0),
            "tpr_L1":  kw(BLUE, None, "--", linewidth=2.0),
            "tpr_LLR": kw(RED,  None, "--", linewidth=2.0),
        },
    },
    {
        # Okabe-Ito-ish palette for deutero/protanopia.
        "name": "Option 12 - Colour-blind safe palette",
        "desc": "L1=blue (#0072B2) / LLR=vermillion (#D55E00) ; case=o / random=sq ; AUC=solid / TPR=dashed",
        "A": {
            "L1_case":    kw(CB_BLUE, "o", "-"),
            "L1_random":  kw(CB_BLUE, "s", "-"),
            "LLR_case":   kw(CB_RED,  "o", "-"),
            "LLR_random": kw(CB_RED,  "s", "-"),
        },
        "B": {
            "auc_L1":  kw(CB_BLUE, "o", "-"),
            "auc_LLR": kw(CB_RED,  "o", "-"),
            "tpr_L1":  kw(CB_BLUE, "o", "--"),
            "tpr_LLR": kw(CB_RED,  "o", "--"),
        },
    },
    {
        # Markers only in A (sparse pool sizes); B uses linestyle alone.
        "name": "Option 13 - Markers in A only; pure lines in B",
        "desc": "L1=blue / LLR=red ; A: case=o / random=sq ; B: no markers, AUC=solid / TPR=dashed",
        "A": {
            "L1_case":    kw(BLUE, "o", "-"),
            "L1_random":  kw(BLUE, "s", "-"),
            "LLR_case":   kw(RED,  "o", "-"),
            "LLR_random": kw(RED,  "s", "-"),
        },
        "B": {
            "auc_L1":  kw(BLUE, None, "-",  linewidth=2.0),
            "auc_LLR": kw(RED,  None, "-",  linewidth=2.0),
            "tpr_L1":  kw(BLUE, None, "--", linewidth=2.0),
            "tpr_LLR": kw(RED,  None, "--", linewidth=2.0),
        },
    },
    {
        # Same markers as #1 but only every 6th point in B, so they read as
        # annotations rather than a solid bar of glyphs.
        "name": "Option 14 - Sparse markers in B (every 6th)",
        "desc": "L1=blue / LLR=red ; A: case=o / random=sq ; B: same as A, markevery=6",
        "A": {
            "L1_case":    kw(BLUE, "o", "-"),
            "L1_random":  kw(BLUE, "s", "-"),
            "LLR_case":   kw(RED,  "o", "-"),
            "LLR_random": kw(RED,  "s", "-"),
        },
        "B": {
            "auc_L1":  kw(BLUE, "o", "-",  markevery=6),
            "auc_LLR": kw(RED,  "o", "-",  markevery=6),
            "tpr_L1":  kw(BLUE, "o", "--", markevery=6),
            "tpr_LLR": kw(RED,  "o", "--", markevery=6),
        },
    },
    {
        # No markers in B at all; linewidth is bumped so the curves carry
        # the visual weight that markers would otherwise contribute.
        "name": "Option 15 - Bold lines, no markers in B",
        "desc": "L1=blue / LLR=red ; A: case=o / random=sq ; B: linewidth=3, AUC=solid / TPR=dashed",
        "A": {
            "L1_case":    kw(BLUE, "o", "-"),
            "L1_random":  kw(BLUE, "s", "-"),
            "LLR_case":   kw(RED,  "o", "-"),
            "LLR_random": kw(RED,  "s", "-"),
        },
        "B": {
            "auc_L1":  kw(BLUE, None, "-",  linewidth=3.0),
            "auc_LLR": kw(RED,  None, "-",  linewidth=3.0),
            "tpr_L1":  kw(BLUE, None, "--", linewidth=3.0),
            "tpr_LLR": kw(RED,  None, "--", linewidth=3.0),
        },
    },
    {
        # Distinct linestyles per series in B - four matplotlib styles
        # (solid, dashed, dashdot, dotted), no markers.
        "name": "Option 16 - Four linestyles in B (solid/dashed/dashdot/dotted)",
        "desc": "L1=blue / LLR=red ; A: case=o / random=sq ; B: linestyle per series, no markers",
        "A": {
            "L1_case":    kw(BLUE, "o", "-"),
            "L1_random":  kw(BLUE, "s", "-"),
            "LLR_case":   kw(RED,  "o", "-"),
            "LLR_random": kw(RED,  "s", "-"),
        },
        "B": {
            "auc_L1":  kw(BLUE, None, "-",   linewidth=2.0),
            "auc_LLR": kw(RED,  None, "-",   linewidth=2.0),
            "tpr_L1":  kw(BLUE, None, "-.",  linewidth=2.0),
            "tpr_LLR": kw(RED,  None, "--",  linewidth=2.0),
        },
    },
    {
        # The chosen convention. Stacked B, redundant fill+linestyle for pool
        # in A so the legend reads two ways at once.
        "name": "Option 18 - FINAL: stacked B + filled/hollow + solid/dashed in A",
        "desc": "L1=blue / LLR=red ; A: case=solid filled-o, random=dashed hollow-o ; B: stacked AUC/TPR, no markers",
        "B_layout": "stacked",
        "A": {
            "L1_case":    kw(BLUE, "o", "-",  markerfacecolor=BLUE,    markeredgecolor=BLUE, linewidth=1.8),
            "L1_random":  kw(BLUE, "o", "--", markerfacecolor="white", markeredgecolor=BLUE, linewidth=1.8),
            "LLR_case":   kw(RED,  "o", "-",  markerfacecolor=RED,     markeredgecolor=RED,  linewidth=1.8),
            "LLR_random": kw(RED,  "o", "--", markerfacecolor="white", markeredgecolor=RED,  linewidth=1.8),
        },
        "B": {
            "auc_L1":  kw(BLUE, None, "-", linewidth=2.0),
            "auc_LLR": kw(RED,  None, "-", linewidth=2.0),
            "tpr_L1":  kw(BLUE, None, "-", linewidth=2.0),
            "tpr_LLR": kw(RED,  None, "-", linewidth=2.0),
        },
    },
    {
        # Stacked B (AUC/TPR by panel position) and Scen A uses linestyle for
        # pool, dropping markers entirely. Colour is metric in both scenarios.
        "name": "Option 17 - Stacked B + linestyle for pool (no markers)",
        "desc": "L1=blue / LLR=red ; A: case=solid / random=dashed (no markers) ; B: stacked AUC/TPR, solid lines",
        "B_layout": "stacked",
        "A": {
            "L1_case":    kw(BLUE, None, "-",  linewidth=2.0),
            "L1_random":  kw(BLUE, None, "--", linewidth=2.0),
            "LLR_case":   kw(RED,  None, "-",  linewidth=2.0),
            "LLR_random": kw(RED,  None, "--", linewidth=2.0),
        },
        "B": {
            "auc_L1":  kw(BLUE, None, "-",  linewidth=2.0),
            "auc_LLR": kw(RED,  None, "-",  linewidth=2.0),
            "tpr_L1":  kw(BLUE, None, "-",  linewidth=2.0),
            "tpr_LLR": kw(RED,  None, "-",  linewidth=2.0),
        },
    },
]


LABELS_A = {
    "L1_case":    "L1, case",
    "L1_random":  "L1, random",
    "LLR_case":   "LLR, case",
    "LLR_random": "LLR, random",
}
LABELS_B = {
    "auc_L1":  "AUC L1",
    "auc_LLR": "AUC LLR",
    "tpr_L1":  "TPR L1",
    "tpr_LLR": "TPR LLR",
}


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def _endpoint_label(ax, x, y, label, color):
    ax.annotate(label, xy=(x[-1], y[-1]), xytext=(4, 0),
                textcoords="offset points", color=color,
                fontsize=8, va="center", clip_on=False)


def plot_A(ax, data, styles, opt):
    keys = ["L1_case", "L1_random", "LLR_case", "LLR_random"]
    for key in keys:
        ax.plot(data["x"], data[key], label=LABELS_A[key], **styles[key])
    ax.set_xlabel("pool size n")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.65, 1.02)
    ax.set_title(f"{opt['name']}\nA: metric x pool (AUC)  |  {opt['desc']}",
                 fontsize=9, loc="left")
    ax.grid(True, alpha=0.3)
    if opt.get("legend") == "endpoint":
        for key in keys:
            _endpoint_label(ax, data["x"], data[key], LABELS_A[key],
                            styles[key]["color"])
    else:
        ax.legend(fontsize=8, loc="lower left", framealpha=0.85)


def plot_B_dual(ax1, data, styles, opt):
    ax2 = ax1.twinx()
    handles, labels = [], []
    for key, ax in [("auc_L1", ax1), ("auc_LLR", ax1),
                    ("tpr_L1", ax2), ("tpr_LLR", ax2)]:
        line, = ax.plot(data["x"], data[key], label=LABELS_B[key], **styles[key])
        handles.append(line)
        labels.append(LABELS_B[key])
    ax1.set_xscale("log")
    ax1.set_xlabel("noise multiplier (sigma)")
    ax1.set_ylabel("AUC")
    ax2.set_ylabel("TPR @ 1% FPR")
    ax1.set_ylim(0.45, 1.02)
    ax2.set_ylim(0, max(0.3, np.max([np.max(data[k]) for k in ("tpr_L1", "tpr_LLR")]) * 1.1))
    ax1.set_title("B: metric x score (dual-axis, fig2b_D3_random)",
                  fontsize=9, loc="left")
    ax1.grid(True, alpha=0.3)
    if opt.get("legend") == "endpoint":
        for key, ax in [("auc_L1", ax1), ("auc_LLR", ax1),
                        ("tpr_L1", ax2), ("tpr_LLR", ax2)]:
            _endpoint_label(ax, data["x"], data[key], LABELS_B[key],
                            styles[key]["color"])
    else:
        ax1.legend(handles, labels, fontsize=8, loc="lower left",
                   framealpha=0.85)


def plot_B_stacked(ax_top, ax_bot, data, styles, opt):
    ax_top.plot(data["x"], data["auc_L1"],  label=LABELS_B["auc_L1"],  **styles["auc_L1"])
    ax_top.plot(data["x"], data["auc_LLR"], label=LABELS_B["auc_LLR"], **styles["auc_LLR"])
    ax_bot.plot(data["x"], data["tpr_L1"],  label=LABELS_B["tpr_L1"],  **styles["tpr_L1"])
    ax_bot.plot(data["x"], data["tpr_LLR"], label=LABELS_B["tpr_LLR"], **styles["tpr_LLR"])

    ax_top.set_xscale("log")
    ax_bot.set_xscale("log")
    ax_top.set_xticklabels([])
    ax_top.set_ylabel("AUC")
    ax_bot.set_ylabel("TPR @ 1% FPR")
    ax_bot.set_xlabel("noise multiplier (sigma)")
    ax_top.set_ylim(0.45, 1.02)
    ax_top.grid(True, alpha=0.3)
    ax_bot.grid(True, alpha=0.3)
    ax_top.set_title("B: metric x score (stacked sub-plots, fig2b_D3_random)",
                     fontsize=9, loc="left")
    ax_top.legend(fontsize=8, loc="lower left", framealpha=0.85)
    ax_bot.legend(fontsize=8, loc="upper right", framealpha=0.85)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    A = load_scenario_A()
    B = load_scenario_B()

    # Each option owns 1 grid row; stacked options own 2 so the AUC/TPR
    # sub-panels get full-sized rows rather than a squashed subgridspec.
    row_starts: list[int] = []
    total_rows = 0
    for opt in OPTIONS:
        row_starts.append(total_rows)
        total_rows += 2 if opt.get("B_layout") == "stacked" else 1

    fig = plt.figure(figsize=(13, 3.0 * total_rows), constrained_layout=True)
    gs = fig.add_gridspec(total_rows, 2)

    for opt, r in zip(OPTIONS, row_starts):
        if opt.get("B_layout") == "stacked":
            ax_A   = fig.add_subplot(gs[r:r + 2, 0])
            ax_top = fig.add_subplot(gs[r,     1])
            ax_bot = fig.add_subplot(gs[r + 1, 1])
            plot_A(ax_A, A, opt["A"], opt)
            plot_B_stacked(ax_top, ax_bot, B, opt["B"], opt)
        else:
            ax_A = fig.add_subplot(gs[r, 0])
            ax_B = fig.add_subplot(gs[r, 1])
            plot_A(ax_A, A, opt["A"], opt)
            plot_B_dual(ax_B, B, opt["B"], opt)

    fig.suptitle(
        "Legend convention comparison  "
        "(Scen A: D3 nsweep   |   Scen B: D3 random noise sweep)",
        fontsize=12,
    )
    pdf_path = resolve_output_path("legend_comparison.pdf")
    png_path = resolve_output_path("legend_comparison.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=150)
    print(f"Wrote {pdf_path} and {png_path}")


if __name__ == "__main__":
    main()
