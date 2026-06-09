#!/usr/bin/env python3
"""Figure 1 — theoretical minimum-error bound from Theorem 1.

Plots the membership-inference lower bound

    min error >= 1/2 * ( 1 - sqrt( m/4 * log(n/(n-1)) + 1/4 * sum_j delta_j^2/sigma_j^2 ) )

as a function of pool size n, for the two parameter sweeps shown in the paper:

  (a) MinimumError_evolution_delta-n.pdf
        vary the per-feature shift delta in {0, 0.1, ..., 0.5}
        fixed m = 500, |A| = 5, sigma^2 = 1
  (b) MinimumError_evolution_A-n.pdf
        vary the number of affected features |A| in {0, 5, 10, 20}
        fixed m = 500, delta = 0.5, sigma^2 = 1

This script only plots closed-form formulas, so it is deterministic and fast
(no datasets involved). It mirrors the repo convention of saving the figure
data alongside the PDF and supporting a replot mode.

Usage:
    python fig_min_error.py                 # compute + plot
    python fig_min_error.py --plot-only      # replot from saved data only
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_io import save_figdata, load_figdata, resolve_output_path

# ── Fixed parameters from the paper (Figure 1) ───────────────────────────────
M = 500            # total feature dimensionality
SIGMA2 = 1.0       # per-feature variance
N = np.arange(2, 502)            # pool size n = 2 .. 501

# Curve set matches the published Figure 1 (and the original disease_filter.py):
# delta in {0, 0.1, 0.2, 0.3, 0.5} (0.4 is intentionally omitted).
DELTAS = [0.0, 0.1, 0.2, 0.3, 0.5]         # panel (a): |A| fixed at 5
A_FIXED = 5

A_SIZES = [0, 5, 10, 20]                   # panel (b): delta fixed at 0.5
DELTA_FIXED = 0.5

DELTA_BASENAME = "MinimumError_evolution_delta-n.pdf"
A_BASENAME = "MinimumError_evolution_A-n.pdf"


def min_error(n, m, signal_term, sigma2=SIGMA2):
    """Theorem 1 lower bound on the minimum inference error.

    signal_term is 1/4 * sum_{j in A} delta_j^2 / sigma_j^2. The argument of
    the square root is clamped to [0, 1]: values above 1 make the bound vacuous
    (min error = 0), matching the paper's small-n behaviour.
    """
    stat_term = (m / 4.0) * np.log(n / (n - 1.0))
    total = np.clip(stat_term + signal_term, 0.0, 1.0)
    return 0.5 * (1.0 - np.sqrt(total))


def compute():
    """Evaluate both sweeps. Returns a flat dict suitable for save_figdata."""
    delta_curves = np.stack([
        min_error(N, M, 0.25 * A_FIXED * (d ** 2) / SIGMA2)
        for d in DELTAS
    ])
    a_curves = np.stack([
        min_error(N, M, 0.25 * a * (DELTA_FIXED ** 2) / SIGMA2)
        for a in A_SIZES
    ])
    return {
        "n": N,
        "deltas": np.array(DELTAS),
        "delta_curves": delta_curves,       # shape (len(DELTAS), len(N))
        "a_sizes": np.array(A_SIZES),
        "a_curves": a_curves,               # shape (len(A_SIZES), len(N))
        "m": M,
        "sigma2": SIGMA2,
        "a_fixed": A_FIXED,
        "delta_fixed": DELTA_FIXED,
    }


def _style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
    })


def _plot_panel(n, curves, labels, label_prefix, title, output_pdf):
    _style()
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(curves)))
    for curve, lab, col in zip(curves, labels, colors):
        ax.plot(n, curve, color=col, linewidth=1.8, label=f"{label_prefix}{lab}")
    ax.set_xlabel("Pool size $n$")
    ax.set_ylabel("Minimum error")
    ax.set_ylim(0.0, 0.3)        # matches the published Figure 1 axis range
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    pdf_path = resolve_output_path(output_pdf)
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf_path}")


def make_figures(data):
    """Render both panels from a data dict (freshly computed or replotted)."""
    n = data["n"]
    _plot_panel(
        n, data["delta_curves"], [f"{d:.1f}" for d in data["deltas"]],
        r"$\delta=$",
        rf"$m={int(data['m'])},\ |A|={int(data['a_fixed'])},\ \sigma^2={data['sigma2']:g}$",
        DELTA_BASENAME,
    )
    _plot_panel(
        n, data["a_curves"], [str(int(a)) for a in data["a_sizes"]],
        r"$|A|=$",
        rf"$m={int(data['m'])},\ \delta={data['delta_fixed']:g},\ \sigma^2={data['sigma2']:g}$",
        A_BASENAME,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot Theorem 1 minimum-error bound (Figure 1)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Replot from saved fig data instead of recomputing")
    args = parser.parse_args()

    if args.plot_only:
        data, _ = load_figdata(DELTA_BASENAME)
        # the A-sweep is saved in the same dump
        make_figures(data)
    else:
        data = compute()
        save_figdata(DELTA_BASENAME, data,
                     meta={"description": "Theorem 1 minimum-error bound, "
                                          "delta-n and A-n sweeps"})
        make_figures(data)
