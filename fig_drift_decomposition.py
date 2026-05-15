#!/usr/bin/env python3
"""Drift decomposition experiment.

Compares 4 drift models to identify which property of real biological drift
kills the LLR:
  1. Real drift (ground truth)
  2. Gaussian pop-std scaled (correct total magnitude, wrong profile)
  3. Gaussian drift-profile matched (correct per-feature magnitude, no correlation)
  4. Regression-to-mean emulation (correct magnitude + baseline-drift correlation)

Produces fig_drift_decomposition.pdf and fig_drift_decomposition.csv.
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

from sklearn.metrics import roc_auc_score
from experiment_io import resolve_output_path
from utils_datasets import load_timestamp_dataset, drop_timestamp_index
from utils import auc_scores

warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"utils")

NUM_ITERATIONS = 500
NUM_WORKERS = max(1, os.cpu_count() - 1)


# ---------------------------------------------------------------------------
# LLR helpers (copied from fig_correlation_preserving.py to avoid coupling)
# ---------------------------------------------------------------------------

def _llr_scores(victims, pop, pool):
    """Compute per-individual LLR scores (sum over features)."""
    victims = np.array(victims, dtype=float)
    pop = np.array(pop, dtype=float)
    pool = np.array(pool, dtype=float)

    mu = np.mean(pop, axis=0)
    mu_hat = np.mean(pool, axis=0)
    var = np.var(pop, axis=0, ddof=0)
    var_hat = np.var(pool, axis=0, ddof=0)
    sigma = np.std(pop, axis=0, ddof=0)
    sigma_hat = np.std(pool, axis=0, ddof=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        w_pop = 1.0 / (2.0 * var)
        w_pool = 1.0 / (2.0 * var_hat)
        term_pop = np.square(victims - mu) * w_pop
        term_pool = np.square(victims - mu_hat) * w_pool
        term_log = np.log(sigma / sigma_hat)
        scores = np.nansum(term_pop - term_pool + term_log, axis=1)

    return scores


def _auc_from_scores(pop_scores, pool_scores):
    """AUC from pre-computed score arrays."""
    y_true = np.concatenate([np.zeros(len(pop_scores)),
                             np.ones(len(pool_scores))])
    y_score = np.concatenate([pop_scores, pool_scores])
    return roc_auc_score(y_true, y_score)


# ---------------------------------------------------------------------------
# Drift model constructors
# ---------------------------------------------------------------------------

def _apply_gauss_pop(baseline, real_drift, sigma_pop, rng):
    """Model 2: Gaussian noise scaled by population std, energy-matched."""
    target_energy = np.mean(np.sum(real_drift ** 2, axis=1))
    pop_var_sum = np.sum(sigma_pop ** 2)
    if pop_var_sum == 0:
        return baseline.copy()
    c = np.sqrt(target_energy / pop_var_sum)
    noise = rng.normal(0, 1, size=baseline.shape) * (c * sigma_pop)
    return baseline + noise


def _apply_gauss_drift(baseline, real_drift, rng):
    """Model 3: Gaussian noise with per-feature drift std (iid, no correlation)."""
    sigma_drift = np.std(real_drift, axis=0, ddof=0)
    noise = rng.normal(0, 1, size=baseline.shape) * sigma_drift
    return baseline + noise


def _apply_rtm(baseline, all_baseline, all_drift, rng):
    """Model 4: Regression-to-mean emulation.

    Fit δ_kj = β · x_kj(0) + α + ε_kj per individual (OLS across features),
    average β and α, then generate synthetic drift preserving the
    baseline-drift correlation.
    """
    n_all = len(all_baseline)
    betas = np.empty(n_all)
    intercepts = np.empty(n_all)
    for k in range(n_all):
        slope, intercept = np.polyfit(all_baseline[k], all_drift[k], 1)
        betas[k] = slope
        intercepts[k] = intercept
    beta_avg = np.mean(betas)
    alpha_avg = np.mean(intercepts)

    # Per-feature residual std
    predicted = beta_avg * all_baseline + alpha_avg
    residuals = all_drift - predicted
    sigma_residual = np.std(residuals, axis=0, ddof=0)

    # Generate synthetic drift for the target baseline
    synth_drift = beta_avg * baseline + alpha_avg + rng.normal(0, 1, size=baseline.shape) * sigma_residual
    return baseline + synth_drift


# ---------------------------------------------------------------------------
# Single iteration
# ---------------------------------------------------------------------------

def _one_iteration(seed):
    """Run one partition. Returns dict of arrays keyed by model_metric."""
    warnings.filterwarnings("ignore")
    rng = np.random.RandomState(seed)

    with contextlib.redirect_stdout(io.StringIO()):
        ti_pop, ti_pool, _sample, _ = load_timestamp_dataset(
            with_independent_miRNAs=True)
        ti_pop, ti_pool = drop_timestamp_index(ti_pop, ti_pool)

    pop0 = np.array(ti_pop[0], dtype=float)
    pool0 = np.array(ti_pool[0], dtype=float)
    n_pop = len(pop0)
    n_pool = len(pool0)
    n_timepoints = len(ti_pop)

    # Population std (for model 2)
    sigma_pop = np.std(np.vstack([pop0, pool0]), axis=0, ddof=0)

    # Output arrays
    keys = [
        "real_LLR", "gauss_pop_LLR", "gauss_drift_LLR", "rtm_LLR",
        "real_L1", "gauss_pop_L1", "gauss_drift_L1", "rtm_L1",
    ]
    result = {k: np.full(n_timepoints, np.nan) for k in keys}

    for t in range(n_timepoints):
        t_pop = np.array(ti_pop[t], dtype=float)
        t_pool = np.array(ti_pool[t], dtype=float)

        # --- Model 1: Real drift ---
        real_pop_llr = _llr_scores(t_pop, pop0, pool0)
        real_pool_llr = _llr_scores(t_pool, pop0, pool0)
        result["real_LLR"][t] = _auc_from_scores(real_pop_llr, real_pool_llr)

        roc_l1, _, _ = auc_scores(ti_pop[t], ti_pool[t], ti_pop[0], ti_pool[0])
        result["real_L1"][t] = roc_l1

        # For synthetic models we need matched sizes (element-wise drift)
        if len(t_pop) != n_pop or len(t_pool) != n_pool:
            continue

        # Compute real drift vectors
        pop_drift = t_pop - pop0
        pool_drift = t_pool - pool0
        all_baseline = np.vstack([pop0, pool0])
        all_drift = np.vstack([pop_drift, pool_drift])

        # --- Model 2: Gaussian pop-std scaled ---
        synth_pop2 = _apply_gauss_pop(pop0, all_drift, sigma_pop, rng)
        synth_pool2 = _apply_gauss_pop(pool0, all_drift, sigma_pop, rng)
        result["gauss_pop_LLR"][t] = _auc_from_scores(
            _llr_scores(synth_pop2, pop0, pool0),
            _llr_scores(synth_pool2, pop0, pool0))
        roc2, _, _ = auc_scores(
            pd.DataFrame(synth_pop2), pd.DataFrame(synth_pool2),
            ti_pop[0], ti_pool[0])
        result["gauss_pop_L1"][t] = roc2

        # --- Model 3: Gaussian drift-profile matched ---
        synth_pop3 = _apply_gauss_drift(pop0, all_drift, rng)
        synth_pool3 = _apply_gauss_drift(pool0, all_drift, rng)
        result["gauss_drift_LLR"][t] = _auc_from_scores(
            _llr_scores(synth_pop3, pop0, pool0),
            _llr_scores(synth_pool3, pop0, pool0))
        roc3, _, _ = auc_scores(
            pd.DataFrame(synth_pop3), pd.DataFrame(synth_pool3),
            ti_pop[0], ti_pool[0])
        result["gauss_drift_L1"][t] = roc3

        # --- Model 4: RTM emulation ---
        synth_pop4 = _apply_rtm(pop0, all_baseline, all_drift, rng)
        synth_pool4 = _apply_rtm(pool0, all_baseline, all_drift, rng)
        result["rtm_LLR"][t] = _auc_from_scores(
            _llr_scores(synth_pop4, pop0, pool0),
            _llr_scores(synth_pool4, pop0, pool0))
        roc4, _, _ = auc_scores(
            pd.DataFrame(synth_pop4), pd.DataFrame(synth_pool4),
            ti_pop[0], ti_pool[0])
        result["rtm_L1"][t] = roc4

    return result


# ---------------------------------------------------------------------------
# Plot setup
# ---------------------------------------------------------------------------

def _setup_rc():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
            all_results.append(result)

    # Aggregate: nanmean over iterations for each key
    keys = list(all_results[0].keys())
    n_timepoints = len(all_results[0][keys[0]])
    avg = {}
    for k in keys:
        stacked = np.array([r[k] for r in all_results])
        avg[k] = np.nanmean(stacked, axis=0)

    ts = np.arange(n_timepoints)

    # Save CSV
    df = pd.DataFrame({"timepoint": ts})
    for k in keys:
        df[k] = avg[k]
    df.to_csv(resolve_output_path("fig_drift_decomposition.csv"), index=False)
    print(f"Saved fig_drift_decomposition.csv ({len(all_results)} iterations)")

    # Print key comparisons
    print("\nMean AUC by model:")
    print(df.to_string(index=False, float_format="%.4f"))

    # Plot: 1x2 subplots (LLR left, L1 right)
    _setup_rc()
    fig, (ax_llr, ax_l1) = plt.subplots(1, 2, figsize=(12, 5))

    styles = [
        ("real", "b-o", "Real drift"),
        ("gauss_pop", "g--^", "Gauss (pop-std)"),
        ("gauss_drift", "m--s", "Gauss (drift-matched)"),
        ("rtm", "r-D", "RTM emulation"),
    ]

    for ax, metric, title in [(ax_llr, "LLR", "LLR"), (ax_l1, "L1", "L1")]:
        for prefix, style, label in styles:
            key = f"{prefix}_{metric}"
            ax.plot(ts, avg[key], style, ms=5, lw=1.5, label=label)
        ax.set_xlabel("Timepoint")
        ax.set_ylabel("AUC")
        ax.set_ylim(0.5, 1)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    handles, labels = ax_llr.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(resolve_output_path("fig_drift_decomposition.pdf"), dpi=300, bbox_inches="tight")
    print("Saved fig_drift_decomposition.pdf")


if __name__ == "__main__":
    run()
