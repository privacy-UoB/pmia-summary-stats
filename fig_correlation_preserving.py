#!/usr/bin/env python3
"""Correlation-preserving noise plot.

Compares real longitudinal AUC against a correlation-preserving noise model
that shuffles full difference vectors across individuals (preserving
inter-feature structure).

Produces fig_correlation_preserving.pdf and fig_correlation_preserving.csv.
"""

import argparse
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

warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"utils")

from sklearn.metrics import roc_auc_score
from experiment_io import resolve_output_path
from utils_datasets import load_timestamp_dataset, drop_timestamp_index
from utils import auc_scores, LLR

NUM_ITERATIONS = 2000
NUM_WORKERS = max(1, os.cpu_count() - 1)  # leave one core free


def diagnose():
    """Load one partition and report variance / LLR diagnostics."""
    print("=== Loading dataset (with_independent_miRNAs=True) ===")
    ti_pop, ti_pool, _sample, _ = load_timestamp_dataset(
        with_independent_miRNAs=True)
    ti_pop, ti_pool = drop_timestamp_index(ti_pop, ti_pool)

    # 1. Feature counts
    n_features = ti_pop[0].shape[1]
    print(f"\n--- Feature counts ---")
    print(f"Total features (after independent miRNA filter): {n_features}")

    # 2. Pop/pool sizes at each timepoint
    print(f"\n--- Pop / pool sizes per timepoint ---")
    for i, (p, q) in enumerate(zip(ti_pop, ti_pool)):
        print(f"  t={i}: pop={len(p):3d}  pool={len(q):3d}")

    # 3. Near-zero variance at timepoint 0
    pop0 = np.array(ti_pop[0], dtype=float)
    pool0 = np.array(ti_pool[0], dtype=float)

    var_pop = np.var(pop0, axis=0, ddof=0)
    var_pool = np.var(pool0, axis=0, ddof=0)

    thresh = 1e-10
    nzv_pop = np.sum(var_pop < thresh)
    nzv_pool = np.sum(var_pool < thresh)
    zero_pop = np.sum(var_pop == 0)
    zero_pool = np.sum(var_pool == 0)

    print(f"\n--- Near-zero variance (t=0) ---")
    print(f"Pop  var < {thresh}: {nzv_pop}/{n_features}  (exactly 0: {zero_pop})")
    print(f"Pool var < {thresh}: {nzv_pool}/{n_features}  (exactly 0: {zero_pool})")
    print(f"Pop  var range:  min={var_pop.min():.2e}  median={np.median(var_pop):.2e}  max={var_pop.max():.2e}")
    print(f"Pool var range:  min={var_pool.min():.2e}  median={np.median(var_pool):.2e}  max={var_pool.max():.2e}")

    # 4. LLR inf/nan check at timepoint 0
    print(f"\n--- LLR inf/nan check (t=0) ---")
    all_individuals = np.vstack([pop0, pool0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(io.StringIO()):
            scores = LLR(all_individuals, pop0, pool0)
    scores = scores.ravel()
    n_inf = np.sum(np.isinf(scores))
    n_nan = np.sum(np.isnan(scores))
    n_finite = np.sum(np.isfinite(scores))
    print(f"Individuals tested: {len(scores)}")
    print(f"  finite: {n_finite}  inf: {n_inf}  nan: {n_nan}")
    if n_finite > 0:
        finite_scores = scores[np.isfinite(scores)]
        print(f"  finite range: [{finite_scores.min():.4f}, {finite_scores.max():.4f}]")

    # 5. Feature contribution: top-5 features by |term_j| for first pop individual
    print(f"\n--- Top-5 features by |LLR term| (first pop individual, t=0) ---")
    x = pop0[0:1]  # shape (1, n_features)
    mu = np.mean(pop0, axis=0)
    mu_hat = np.mean(pool0, axis=0)
    sigma = np.std(pop0, axis=0, ddof=0)
    sigma_hat = np.std(pool0, axis=0, ddof=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        term_pop = np.square(x - mu) / (2 * var_pop)
        term_pool = np.square(x - mu_hat) / (2 * var_pool)
        term_log = np.log(sigma / sigma_hat)
        term_j = (term_pop - term_pool + term_log).ravel()

    feature_names = list(ti_pop[0].columns) if hasattr(ti_pop[0], "columns") else [
        f"f{i}" for i in range(n_features)]
    abs_term = np.abs(term_j)
    # Replace inf with large value for sorting, keep track
    abs_term_sort = np.where(np.isfinite(abs_term), abs_term, 1e30)
    top5_idx = np.argsort(abs_term_sort)[-5:][::-1]

    print(f"  {'Feature':<20s} {'|term_j|':>12s} {'var_pop':>12s} {'var_pool':>12s}")
    for idx in top5_idx:
        t_str = f"{abs_term[idx]:.4e}" if np.isfinite(abs_term[idx]) else "inf"
        print(f"  {feature_names[idx]:<20s} {t_str:>12s} {var_pop[idx]:>12.4e} {var_pool[idx]:>12.4e}")

    print("\n=== Diagnostic complete ===")


def _llr_scores(victims, pop, pool, var_cap_percentile=None):
    """Compute per-individual LLR scores (sum over features).

    If var_cap_percentile is set (0-100), cap 1/(2*var) weights at the
    given percentile to reduce the influence of low-variance super-features.
    Returns shape (n_individuals,).
    """
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

        if var_cap_percentile is not None:
            # Cap weights at the given percentile (only finite values)
            finite_w_pop = w_pop[np.isfinite(w_pop)]
            finite_w_pool = w_pool[np.isfinite(w_pool)]
            if len(finite_w_pop) > 0:
                cap_pop = np.percentile(finite_w_pop, var_cap_percentile)
                w_pop = np.minimum(w_pop, cap_pop)
            if len(finite_w_pool) > 0:
                cap_pool = np.percentile(finite_w_pool, var_cap_percentile)
                w_pool = np.minimum(w_pool, cap_pool)

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


def diagnose2():
    """Targeted LLR diagnostic: why shuffled LLR stays high."""
    from scipy.stats import pearsonr

    print("=== diagnose2: targeted LLR diagnostic ===")
    print("Loading dataset (with_independent_miRNAs=True)...")
    with contextlib.redirect_stdout(io.StringIO()):
        ti_pop, ti_pool, _sample, _ = load_timestamp_dataset(
            with_independent_miRNAs=True)
        ti_pop, ti_pool = drop_timestamp_index(ti_pop, ti_pool)

    n_timepoints = len(ti_pop)
    pop0 = np.array(ti_pop[0], dtype=float)
    pool0 = np.array(ti_pool[0], dtype=float)
    n_pool = len(pool0)
    n_pop = len(pop0)
    n_features = pop0.shape[1]
    print(f"pop={n_pop}, pool={n_pool}, features={n_features}, timepoints={n_timepoints}")

    # Pick timepoints to examine (0, 1, and last available > 1)
    test_ts = [0, 1, min(4, n_timepoints - 1)]
    # Remove duplicates
    test_ts = sorted(set(test_ts))

    # ---- Test A: Score distribution comparison ----
    print("\n" + "=" * 60)
    print("TEST A: Pool member LLR score distributions (real vs shuffled)")
    print("=" * 60)
    print(f"Examining timepoints: {test_ts}")

    for t in test_ts:
        t_pool = np.array(ti_pool[t], dtype=float)
        n_t = len(t_pool)

        # Real scores: LLR(pool_k(t), pop0, pool0) for each pool member
        real_scores = _llr_scores(t_pool, pop0, pool0)

        # Shuffled: pool_k(0) + delta_{pi(k)} where pi is a random permutation
        if n_t == n_pool:
            delta = t_pool - pool0
            perm = np.random.permutation(n_pool)
            shuffled_pool = pool0 + delta[perm]
            shuf_scores = _llr_scores(shuffled_pool, pop0, pool0)
        else:
            shuf_scores = np.full(n_t, np.nan)

        print(f"\n  t={t} (n_pool_at_t={n_t}):")
        print(f"    Real pool scores:     mean={np.nanmean(real_scores):8.2f}  "
              f"std={np.nanstd(real_scores):8.2f}  "
              f"min={np.nanmin(real_scores):8.2f}  "
              f"max={np.nanmax(real_scores):8.2f}")
        if not np.all(np.isnan(shuf_scores)):
            print(f"    Shuffled pool scores: mean={np.nanmean(shuf_scores):8.2f}  "
                  f"std={np.nanstd(shuf_scores):8.2f}  "
                  f"min={np.nanmin(shuf_scores):8.2f}  "
                  f"max={np.nanmax(shuf_scores):8.2f}")
        else:
            print(f"    Shuffled pool scores: SKIPPED (size mismatch at t={t})")

        # Also show pop scores for reference (overlap context)
        pop_scores_t = _llr_scores(np.array(ti_pop[t], dtype=float), pop0, pool0)
        print(f"    Pop scores (ref):     mean={np.nanmean(pop_scores_t):8.2f}  "
              f"std={np.nanstd(pop_scores_t):8.2f}  "
              f"min={np.nanmin(pop_scores_t):8.2f}  "
              f"max={np.nanmax(pop_scores_t):8.2f}")

    # ---- Test B: Variance-capped LLR ----
    print("\n" + "=" * 60)
    print("TEST B: Variance-capped LLR (cap at 90th, 75th, 50th percentile)")
    print("=" * 60)

    caps = [None, 90, 75, 50]
    header = f"  {'cap':>5s}"
    for t in test_ts:
        header += f"  {'t=' + str(t) + ' real':>10s}  {'t=' + str(t) + ' shuf':>10s}  {'gap':>6s}"
    print(header)

    for cap in caps:
        cap_label = "none" if cap is None else f"p{cap}"
        row = f"  {cap_label:>5s}"
        for t in test_ts:
            t_pop_arr = np.array(ti_pop[t], dtype=float)
            t_pool_arr = np.array(ti_pool[t], dtype=float)

            # Real AUC
            real_pop_s = _llr_scores(t_pop_arr, pop0, pool0, var_cap_percentile=cap)
            real_pool_s = _llr_scores(t_pool_arr, pop0, pool0, var_cap_percentile=cap)
            real_auc = _auc_from_scores(real_pop_s, real_pool_s)

            # Shuffled AUC
            if len(t_pop_arr) == n_pop and len(t_pool_arr) == n_pool:
                pop_diff = t_pop_arr - pop0
                pool_diff = t_pool_arr - pool0
                shuf_pop = pop0 + pop_diff[np.random.permutation(n_pop)]
                shuf_pool = pool0 + pool_diff[np.random.permutation(n_pool)]
                shuf_pop_s = _llr_scores(shuf_pop, pop0, pool0, var_cap_percentile=cap)
                shuf_pool_s = _llr_scores(shuf_pool, pop0, pool0, var_cap_percentile=cap)
                shuf_auc = _auc_from_scores(shuf_pop_s, shuf_pool_s)
            else:
                shuf_auc = float("nan")

            gap = shuf_auc - real_auc
            row += f"  {real_auc:10.4f}  {shuf_auc:10.4f}  {gap:+6.3f}"
        print(row)

    # ---- Test C: Baseline-drift correlation ----
    print("\n" + "=" * 60)
    print("TEST C: Baseline-drift correlation (Pearson r across features)")
    print("=" * 60)
    print("  For each pool member k, correlate pool_k(0) with delta_k = pool_k(t) - pool_k(0)")
    print("  across all features, then average over individuals.\n")

    for t in test_ts:
        if t == 0:
            print(f"  t={t}: delta=0, skipped")
            continue
        t_pool_arr = np.array(ti_pool[t], dtype=float)
        if len(t_pool_arr) != n_pool:
            print(f"  t={t}: size mismatch, skipped")
            continue

        delta = t_pool_arr - pool0
        corrs = []
        for k in range(n_pool):
            r, p = pearsonr(pool0[k], delta[k])
            corrs.append(r)
        corrs = np.array(corrs)
        print(f"  t={t}: mean_r={np.mean(corrs):+.4f}  std_r={np.std(corrs):.4f}  "
              f"min_r={np.min(corrs):+.4f}  max_r={np.max(corrs):+.4f}  "
              f"(n={n_pool} individuals)")

    # Also do the same for pop members
    print("\n  Same for pop members (for comparison):")
    for t in test_ts:
        if t == 0:
            continue
        t_pop_arr = np.array(ti_pop[t], dtype=float)
        if len(t_pop_arr) != n_pop:
            print(f"  t={t}: size mismatch, skipped")
            continue
        delta = t_pop_arr - pop0
        corrs = []
        for k in range(n_pop):
            r, p = pearsonr(pop0[k], delta[k])
            corrs.append(r)
        corrs = np.array(corrs)
        print(f"  t={t}: mean_r={np.mean(corrs):+.4f}  std_r={np.std(corrs):.4f}  "
              f"min_r={np.min(corrs):+.4f}  max_r={np.max(corrs):+.4f}  "
              f"(n={n_pop} individuals)")

    # ---- Bonus: variance weight distribution ----
    print("\n" + "=" * 60)
    print("BONUS: Variance weight distribution (1/(2*var_hat))")
    print("=" * 60)
    var_hat = np.var(pool0, axis=0, ddof=0)
    with np.errstate(divide="ignore"):
        weights = 1.0 / (2.0 * var_hat)
    finite_w = weights[np.isfinite(weights)]
    print(f"  Total features: {len(weights)}")
    print(f"  Finite weights: {len(finite_w)}")
    if len(finite_w) > 0:
        pcts = [50, 75, 90, 95, 99, 100]
        vals = np.percentile(finite_w, pcts)
        print(f"  Percentiles: " + "  ".join(f"p{p}={v:.2f}" for p, v in zip(pcts, vals)))
        # What fraction of total weight comes from top 10% of features?
        threshold_90 = np.percentile(finite_w, 90)
        top10_mask = finite_w >= threshold_90
        frac = np.sum(finite_w[top10_mask]) / np.sum(finite_w)
        print(f"  Top 10% of features carry {frac:.1%} of total weight")

    print("\n=== diagnose2 complete ===")


def _setup_rc():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
    })


def _one_iteration(_seed):
    """Run a single iteration. Returns (real_L1, real_LLR, corr_L1, corr_LLR)."""
    warnings.filterwarnings("ignore")

    with contextlib.redirect_stdout(io.StringIO()):
        ti_pop, ti_pool, _sample, _ = load_timestamp_dataset(
            with_independent_miRNAs=True)
        ti_pop, ti_pool = drop_timestamp_index(ti_pop, ti_pool)

    pop = ti_pop[0]
    pool = ti_pool[0]
    n_timepoints = len(ti_pop)

    iter_real_L1 = np.full(n_timepoints, np.nan)
    iter_real_LLR = np.full(n_timepoints, np.nan)
    iter_corr_L1 = np.full(n_timepoints, np.nan)
    iter_corr_LLR = np.full(n_timepoints, np.nan)

    for i, (t_pop, t_pool) in enumerate(zip(ti_pop, ti_pool)):
        # Real longitudinal AUC (works regardless of size match)
        roc_L1, _, _ = auc_scores(t_pop, t_pool, pop, pool)
        with contextlib.redirect_stdout(io.StringIO()):
            roc_LLR, _, _ = auc_scores(t_pop, t_pool, pop, pool, LR=True)
        iter_real_L1[i] = roc_L1
        iter_real_LLR[i] = roc_LLR

        # Correlation-preserving needs element-wise subtraction → sizes must match
        local_pop = np.array(pop)
        local_pool = np.array(pool)
        local_t_pop = np.array(t_pop)
        local_t_pool = np.array(t_pool)

        if (len(local_pop) != len(local_t_pop) or
                len(local_pool) != len(local_t_pool)):
            continue

        pop_diff = local_t_pop - local_pop
        pool_diff = local_t_pool - local_pool

        shuffled_pop_diff = pop_diff[np.random.permutation(len(pop_diff))]
        shuffled_pool_diff = pool_diff[np.random.permutation(len(pool_diff))]

        noisy_pop = local_pop + shuffled_pop_diff
        noisy_pool = local_pool + shuffled_pool_diff

        roc_corr_L1, _, _ = auc_scores(noisy_pop, noisy_pool, pop, pool)
        with contextlib.redirect_stdout(io.StringIO()):
            roc_corr_LLR, _, _ = auc_scores(
                noisy_pop, noisy_pool, pop, pool, LR=True)
        iter_corr_L1[i] = roc_corr_L1
        iter_corr_LLR[i] = roc_corr_LLR

    return (iter_real_L1, iter_real_LLR, iter_corr_L1, iter_corr_LLR)


def run():
    print(f"Running {NUM_ITERATIONS} iterations across {NUM_WORKERS} workers...",
          flush=True)

    auc_real_L1, auc_real_LLR = [], []
    auc_corr_L1, auc_corr_LLR = [], []

    done = 0
    with mp.Pool(NUM_WORKERS) as pool:
        for result in pool.imap_unordered(_one_iteration, range(NUM_ITERATIONS)):
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{NUM_ITERATIONS}", flush=True)
            real_L1, real_LLR, corr_L1, corr_LLR = result
            auc_real_L1.append(real_L1)
            auc_real_LLR.append(real_LLR)
            auc_corr_L1.append(corr_L1)
            auc_corr_LLR.append(corr_LLR)

    # Average over iterations (nanmean handles timepoints with size mismatches)
    avg_real_L1 = np.nanmean(auc_real_L1, axis=0)
    avg_real_LLR = np.nanmean(auc_real_LLR, axis=0)
    avg_corr_L1 = np.nanmean(auc_corr_L1, axis=0)
    avg_corr_LLR = np.nanmean(auc_corr_LLR, axis=0)

    n_timepoints = len(avg_real_L1)
    ts = np.arange(n_timepoints)

    # Save CSV
    df = pd.DataFrame({
        "timepoint": ts,
        "real_L1": avg_real_L1,
        "real_LLR": avg_real_LLR,
        "corr_preserving_L1": avg_corr_L1,
        "corr_preserving_LLR": avg_corr_LLR,
    })
    df.to_csv(resolve_output_path("fig_correlation_preserving.csv"), index=False)
    print(f"Saved fig_correlation_preserving.csv ({len(auc_real_L1)} iterations)")

    # Plot: 1x2 subplots (L1 left, LLR right)
    _setup_rc()
    fig, (ax_l1, ax_llr) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, real, corr, label in [
        (ax_l1, avg_real_L1, avg_corr_L1, "L1"),
        (ax_llr, avg_real_LLR, avg_corr_LLR, "LLR"),
    ]:
        ax.plot(ts, real, "b-o", ms=5, lw=1.5, label="Real longitudinal")
        ax.plot(ts, corr, "r--s", ms=5, lw=1.5, label="Correlation-preserving")
        ax.set_xlabel("Timepoint")
        ax.set_ylabel("AUC")
        ax.set_ylim(0.5, 1)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    handles, labels = ax_l1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(resolve_output_path("fig_correlation_preserving.pdf"), dpi=300, bbox_inches="tight")
    print("Saved fig_correlation_preserving.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnose", action="store_true",
                        help="Print LLR variance diagnostics and exit")
    parser.add_argument("--diagnose2", action="store_true",
                        help="Targeted LLR diagnostic: why shuffled LLR stays high")
    args = parser.parse_args()
    if args.diagnose:
        diagnose()
    elif args.diagnose2:
        diagnose2()
    else:
        run()
