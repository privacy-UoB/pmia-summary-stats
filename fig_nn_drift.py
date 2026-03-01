#!/usr/bin/env python3
"""Drift model comparison experiment.

Tests four conditions against real biological drift at t=1:
1. Real drift (ground truth)
2. Independent Gaussian (per-feature variance matched)
3. LOO conditional mean (linear regression, held-out prediction)
4. NN k=1 (nearest neighbor's real drift, Euclidean)

Produces fig_nn_drift.csv.
"""

import contextlib
import io
import multiprocessing as mp
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"utils")

from utils_datasets import load_timestamp_dataset, drop_timestamp_index
from utils import auc_scores

NUM_ITERATIONS = 2000
NUM_WORKERS = max(1, os.cpu_count() - 1)
K_VALUES = [1, 2, 5, 10, 20, 50]
NN_METRICS = ["euclidean", "cosine", "correlation"]


def _nn_shuffle(baseline, diff, k, rng, metric="euclidean"):
    """Assign each individual the drift of a random one of their k nearest neighbors.

    Returns (nn_diff, mean_percentile) where mean_percentile is the average
    percentile rank of the chosen neighbor's distance within the full pairwise
    distance distribution.
    """
    dist = cdist(baseline, baseline, metric=metric)
    np.fill_diagonal(dist, np.inf)
    n = len(baseline)
    k_eff = min(k, n - 1)
    triu_idx = np.triu_indices(n, k=1)
    all_dists = dist[triu_idx]
    nn_diff = np.empty_like(diff)
    chosen_percentiles = np.empty(n)
    for i in range(n):
        neighbors = np.argsort(dist[i])[:k_eff]
        chosen = rng.choice(neighbors)
        nn_diff[i] = diff[chosen]
        chosen_percentiles[i] = np.mean(all_dists <= dist[i, chosen]) * 100
    return nn_diff, chosen_percentiles.mean()


def _loo_conditional_mean(baseline, diff):
    """Leave-one-out conditional mean: for each individual, fit regression
    on all others and predict that individual's drift from the held-out model."""
    n, d = baseline.shape
    loo_diff = np.empty_like(diff)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_train = baseline[mask]
        Y_train = diff[mask]
        mu_base = X_train.mean(axis=0)
        mu_diff = Y_train.mean(axis=0)
        X_c = X_train - mu_base
        Y_c = Y_train - mu_diff
        B, _, _, _ = np.linalg.lstsq(X_c, Y_c, rcond=None)
        loo_diff[i] = mu_diff + (baseline[i] - mu_base) @ B
    return loo_diff


def _one_iteration(seed):
    """Run one iteration: score four drift conditions with both LLR and L1."""
    rng = np.random.RandomState(seed)
    warnings.filterwarnings("ignore")

    with contextlib.redirect_stdout(io.StringIO()):
        ti_pop, ti_pool, _sample, _ = load_timestamp_dataset(
            with_independent_miRNAs=True)
        ti_pop, ti_pool = drop_timestamp_index(ti_pop, ti_pool)

    pop0 = np.array(ti_pop[0], dtype=float)
    pool0 = np.array(ti_pool[0], dtype=float)
    n_pop = len(pop0)
    n_pool = len(pool0)

    t = 1
    if t >= len(ti_pop):
        return {}

    t_pop = np.array(ti_pop[t], dtype=float)
    t_pool = np.array(ti_pool[t], dtype=float)

    if len(t_pop) != n_pop or len(t_pool) != n_pool:
        return {}

    pop_diff = t_pop - pop0
    pool_diff = t_pool - pool0

    results = {}

    def _score_both(vp, vpl, p, pl, prefix):
        with contextlib.redirect_stdout(io.StringIO()):
            roc_llr, _, _ = auc_scores(vp, vpl, p, pl, LR=True)
            roc_l1, _, _ = auc_scores(vp, vpl, p, pl, LR=False)
        results[f"{prefix}_llr"] = roc_llr
        results[f"{prefix}_l1"] = roc_l1

    # 1. Real drift
    _score_both(t_pop, t_pool, pop0, pool0, "real")

    # 2. Independent Gaussian: N(0, sigma_j^2) per feature
    pop_std = pop_diff.std(axis=0)
    pool_std = pool_diff.std(axis=0)
    _score_both(pop0 + rng.randn(n_pop, pop0.shape[1]) * pop_std,
                pool0 + rng.randn(n_pool, pool0.shape[1]) * pool_std,
                pop0, pool0, "indep_gauss")

    # 3. LOO conditional mean
    loo_pop_diff = _loo_conditional_mean(pop0, pop_diff)
    loo_pool_diff = _loo_conditional_mean(pool0, pool_diff)
    _score_both(pop0 + loo_pop_diff, pool0 + loo_pool_diff,
                pop0, pool0, "loo_cond_mean")

    # 4. NN k=1 (Euclidean) — for main table
    nn_pop_diff, _ = _nn_shuffle(pop0, pop_diff, 1, rng, metric="euclidean")
    nn_pool_diff, _ = _nn_shuffle(pool0, pool_diff, 1, rng, metric="euclidean")
    _score_both(pop0 + nn_pop_diff, pool0 + nn_pool_diff,
                pop0, pool0, "nn_k1")

    # Appendix: NN k-sweep across all metrics
    for metric in NN_METRICS:
        for k in K_VALUES:
            nn_pop_diff, pop_pctl = _nn_shuffle(pop0, pop_diff, k, rng, metric=metric)
            nn_pool_diff, pool_pctl = _nn_shuffle(pool0, pool_diff, k, rng, metric=metric)
            _score_both(pop0 + nn_pop_diff, pool0 + nn_pool_diff,
                        pop0, pool0, f"{metric}_k{k}")
            results[f"{metric}_k{k}_pctl"] = (pop_pctl + pool_pctl) / 2

    return results


def run():
    print(f"Running {NUM_ITERATIONS} iterations across {NUM_WORKERS} workers...",
          flush=True)

    all_results = []
    done = 0
    with mp.Pool(NUM_WORKERS) as pool:
        for result in pool.imap_unordered(_one_iteration, range(NUM_ITERATIONS)):
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{NUM_ITERATIONS}", flush=True)
            if result:
                all_results.append(result)

    df_all = pd.DataFrame(all_results)

    # ---- Main table (4 rows) ----
    CONDITIONS = [
        ("real", "Real drift"),
        ("indep_gauss", "Independent Gaussian"),
        ("loo_cond_mean", "LOO conditional mean"),
        ("nn_k1", "NN (k=1)"),
    ]

    rows = []
    for key, label in CONDITIONS:
        rows.append({
            "model": label,
            "llr": df_all[f"{key}_llr"].mean(),
            "l1": df_all[f"{key}_l1"].mean(),
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv("fig_nn_drift.csv", index=False)

    print(f"\n{'Model':<25s} {'LLR':>8s} {'L1':>8s}")
    print("-" * 43)
    for _, row in df_out.iterrows():
        print(f"{row['model']:<25s} {row['llr']:8.3f} {row['l1']:8.3f}")

    # ---- Appendix table: NN k-sweep by metric ----
    app_rows = []
    # k=0 (real drift) reference row
    app_row = {"k": 0, "pctl": 0.0}
    for metric in NN_METRICS:
        app_row[f"{metric}_llr"] = df_all["real_llr"].mean()
        app_row[f"{metric}_l1"] = df_all["real_l1"].mean()
    app_rows.append(app_row)

    for k in K_VALUES:
        app_row = {"k": k}
        pctl_col = f"euclidean_k{k}_pctl"
        app_row["pctl"] = df_all[pctl_col].mean() if pctl_col in df_all.columns else 0
        for metric in NN_METRICS:
            for score in ["llr", "l1"]:
                col = f"{metric}_k{k}_{score}"
                if col in df_all.columns:
                    app_row[f"{metric}_{score}"] = df_all[col].mean()
        app_rows.append(app_row)

    df_app = pd.DataFrame(app_rows)
    df_app.to_csv("fig_nn_drift_appendix.csv", index=False)

    print(f"\nAppendix: NN k-sweep (Euclidean LLR shown; all metrics in CSV)")
    print(f"{'k':>4s} {'pctl':>6s} {'LLR':>8s} {'L1':>8s}")
    print("-" * 30)
    for _, row in df_app.iterrows():
        print(f"{int(row['k']):4d} {row['pctl']:5.0f}% {row['euclidean_llr']:8.3f} {row['euclidean_l1']:8.3f}")

    print(f"\nSaved fig_nn_drift.csv and fig_nn_drift_appendix.csv")

    # ---- Appendix plot: k-sweep with reference lines ----
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    nn_rows = df_app[df_app["k"] > 0]
    k_plot = nn_rows["k"].astype(int).values
    pctl_vals = nn_rows["pctl"].values

    metric_colors = {"euclidean": "C0", "cosine": "C1", "correlation": "C2"}
    metric_labels = {"euclidean": "Euclidean", "cosine": "Cosine",
                     "correlation": "Correlation"}

    # Reference line values from main table
    ref_lines = [
        ("real",          "Real drift",           "k",  "--", 2.0),
        ("indep_gauss",   "Independent Gaussian",  "C4", ":",  2.0),
        ("loo_cond_mean", "LOO conditional mean",  "C5", "-.", 2.0),
    ]
    ref_vals = {key: {s: df_all[f"{key}_{s}"].mean() for s in ["llr", "l1"]}
                for key, _, _, _, _ in ref_lines}

    for ax, score, score_label in zip(axes, ["llr", "l1"], ["LLR", "L1"]):
        # NN curves per metric
        for metric in NN_METRICS:
            col = f"{metric}_{score}"
            if col in nn_rows.columns:
                ax.plot(k_plot, nn_rows[col].values, "o-", ms=6, lw=1.5,
                        color=metric_colors[metric], label=metric_labels[metric])

        # Reference lines
        for key, label, color, ls, lw in ref_lines:
            ax.axhline(ref_vals[key][score], ls=ls, lw=lw, color=color,
                       alpha=0.8, label=label)

        ax.set_xscale("log")
        ax.set_xlabel("k (number of nearest neighbors)")
        ax.set_ylabel(f"AUC ({score_label})")
        ax.set_ylim(0.4, 1.05)
        ax.set_title(f"({chr(97 + list(axes).index(ax))}) {score_label} test")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

        # Secondary x-axis: percentile
        ax_top = ax.twiny()
        ax_top.set_xscale("log")
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(k_plot)
        ax_top.set_xticklabels([f"{p:.0f}%" for p in pctl_vals], fontsize=8)
        ax_top.set_xlabel("Percentile of pairwise distances", fontsize=10)

    plt.tight_layout()
    fig.savefig("fig_nn_drift.pdf", dpi=300, bbox_inches="tight")
    print("Saved fig_nn_drift.pdf")


if __name__ == "__main__":
    run()
