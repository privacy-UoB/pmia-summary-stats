#!/usr/bin/env python3
"""Pool-size sweep experiment: validate Theorem 1's lower bound by sweeping n.

Measures how minimum classification error and AUC change as pool size n
varies for three diseases (D1, D3, D14), comparing against the theoretical
prediction from Theorem 1.

Usage:
    python experiment_nsweep.py [--iterations 2000] [--plot-only]
"""

import argparse
import contextlib
import io
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Suppress divide-by-zero warnings from LLR at small pool sizes
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module=r"utils")

from plot_style import line_kwargs
from utils import auc_scores, tpr_at_fpr
from utils_datasets import D1, D3, D14


# ── Disease metadata ─────────────────────────────────────────────────────────

DISEASE_META = {
    D1:  {"tag": "D1",  "name": "Wilms Tumor",    "short": "WT"},
    D3:  {"tag": "D3",  "name": "Prostate Cancer", "short": "PC"},
    D14: {"tag": "D14", "name": "Ovarian Cancer",  "short": "OC"},
}

# Pool sizes per disease (16 + 11 + 6 = 33 rows)
POOL_SIZES = {
    D1:  [5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 124],
    D3:  [5, 8, 10, 15, 20, 25, 30, 35, 40, 50, 65],
    D14: [5, 8, 10, 15, 20, 24],
}

# Ordered by increasing signal term for visual argument in figures
TARGET_DISEASES = [D3, D1, D14]


# ── Data loading (copied from experiment_19disease.py — not a library) ───────

def load_miRNA_data_once():
    """Load and preprocess miRNA dataset once (avoids repeated disk reads)."""
    df = pd.read_csv("Datasets/GSE61741_series_matrix.csv",
                     skiprows=52, skipfooter=1, sep="\t",
                     index_col=0, engine="python")

    df_median = df.median(axis=1)
    filter_population = df[df_median >= 50].transpose()

    with open("Datasets/GSE61741_series_matrix.csv", "rt") as f:
        lines = f.readlines()

    diseases_line = ""
    for line in lines:
        if line.startswith("!Sample_characteristics_ch1"):
            diseases_line = line.strip()

    diseases = [d.strip('"') for d in diseases_line.split("\t")[1:]]
    filter_population.insert(0, "diseases", diseases)
    filter_population = filter_population.sort_values("diseases")
    return filter_population


# ── Metrics (copied from experiment_19disease.py) ────────────────────────────

def compute_empirical_min_error_fast(scores_pop, scores_pool):
    """O(n log n) threshold sweep for minimum classification error."""
    sorted_non = np.sort(np.ravel(scores_pop))
    sorted_mem = np.sort(np.ravel(scores_pool))
    thresholds = np.unique(np.concatenate([sorted_non, sorted_mem]))
    fp = len(sorted_non) - np.searchsorted(sorted_non, thresholds, side="left")
    fn = np.searchsorted(sorted_mem, thresholds, side="left")
    errors = 0.5 * (fp / len(sorted_non) + fn / len(sorted_mem))
    return errors.min()


def compute_signal_term(pop_cpool, case_pool):
    """Signal term: sum_j delta_j^2 / (4 * sigma_j^2) over non-constant features."""
    mu = np.mean(pop_cpool, axis=0)
    mu_hat = np.mean(case_pool, axis=0)
    sigma = np.std(pop_cpool, axis=0, ddof=0)
    delta = mu_hat - mu
    mask = sigma > 1e-10
    return np.sum(delta[mask] ** 2 / (4 * sigma[mask] ** 2))


def compute_theoretical_min_error(m, pool_size, signal_term):
    """Theorem 1 bounds.

    Returns (stat_term, min_error_stat, min_error_full).
    """
    stat_term = m / 4 * np.log(pool_size / (pool_size - 1))
    full_term = stat_term + signal_term
    min_error_stat = max(0.0, 0.5 * (1 - np.sqrt(min(stat_term, 1.0))))
    min_error_full = max(0.0, 0.5 * (1 - np.sqrt(min(full_term, 1.0))))
    return stat_term, min_error_stat, min_error_full


# ── Experiment ───────────────────────────────────────────────────────────────

def run_experiment(iterations=2000):
    """Run pool-size sweep and save results to nsweep_results.csv."""
    print("Loading miRNA dataset ...")
    filter_pop = load_miRNA_data_once()
    n_total = filter_pop.shape[0]
    m = filter_pop.shape[1] - 1  # 465 features (minus 'diseases' column)
    print(f"Dataset: {n_total} individuals, {m} features")

    rng = np.random.default_rng(42)
    results = []

    for disease in TARGET_DISEASES:
        meta = DISEASE_META[disease]
        tag, name = meta["tag"], meta["name"]
        print(f"\n{'='*60}")
        print(f"{tag}: {name}")
        print(f"{'='*60}")

        # Split: case pool vs non-case population
        case_pool_df = filter_pop[filter_pop["diseases"] == disease]
        pop_cpool_df = filter_pop[filter_pop["diseases"] != disease]
        case_pool_full = case_pool_df.drop(columns="diseases").values
        pop_cpool = pop_cpool_df.drop(columns="diseases").values

        A_full = case_pool_full.shape[0]
        print(f"  Full case pool |A|={A_full}, pop_cpool={pop_cpool.shape[0]}, m={m}")

        # Signal terms (computed once from full case pool)
        signal_term_full = compute_signal_term(pop_cpool, case_pool_full)
        half = A_full // 2
        signal_term_half = compute_signal_term(pop_cpool, case_pool_full[:half])
        print(f"  signal_full={signal_term_full:.4f}  signal_half={signal_term_half:.4f}")

        # Count diseased miRNAs for panel titles
        mu = np.mean(pop_cpool, axis=0)
        mu_hat = np.mean(case_pool_full, axis=0)
        sigma = np.std(pop_cpool, axis=0, ddof=0)
        delta = mu_hat - mu
        mask = sigma > 1e-10
        A_diseased = int(np.sum(np.abs(delta[mask]) / sigma[mask] > 0.5))

        pool_sizes = POOL_SIZES[disease]
        for n in pool_sizes:
            print(f"\n  n={n}  ", end="", flush=True)

            # Accumulators
            c_llr_me, c_l1_me = [], []
            c_llr_auc, c_l1_auc = [], []
            c_llr_tpr, c_l1_tpr = [], []
            r_llr_me, r_l1_me = [], []
            r_llr_auc, r_l1_auc = [], []
            r_llr_tpr, r_l1_tpr = [], []

            for it in range(iterations):
                # ── Case pool: subsample n from full case pool ──
                idx_case = rng.choice(A_full, size=n, replace=False)
                case_sub = case_pool_full[idx_case]

                with contextlib.redirect_stdout(io.StringIO()):
                    try:
                        auc_c_llr, sp_c_llr, sm_c_llr = auc_scores(
                            pop_cpool, case_sub, pop_cpool, case_sub, LR=True)
                        me_c_llr = compute_empirical_min_error_fast(
                            np.ravel(sp_c_llr), np.ravel(sm_c_llr))
                        c_llr_auc.append(auc_c_llr)
                        c_llr_me.append(me_c_llr)
                        c_llr_tpr.append(tpr_at_fpr(sp_c_llr, sm_c_llr))
                    except Exception:
                        pass

                    try:
                        auc_c_l1, sp_c_l1, sm_c_l1 = auc_scores(
                            pop_cpool, case_sub, pop_cpool, case_sub, LR=False)
                        me_c_l1 = compute_empirical_min_error_fast(
                            np.ravel(sp_c_l1), np.ravel(sm_c_l1))
                        c_l1_auc.append(auc_c_l1)
                        c_l1_me.append(me_c_l1)
                        c_l1_tpr.append(tpr_at_fpr(sp_c_l1, sm_c_l1))
                    except Exception:
                        pass

                # ── Random pool: subsample n from non-case pop ──
                idx_all = rng.permutation(len(pop_cpool))
                random_sub = pop_cpool[idx_all[:n]]
                pop_rpool = pop_cpool[idx_all[n:]]

                with contextlib.redirect_stdout(io.StringIO()):
                    try:
                        auc_r_llr, sp_r_llr, sm_r_llr = auc_scores(
                            pop_rpool, random_sub, pop_rpool, random_sub, LR=True)
                        me_r_llr = compute_empirical_min_error_fast(
                            np.ravel(sp_r_llr), np.ravel(sm_r_llr))
                        r_llr_auc.append(auc_r_llr)
                        r_llr_me.append(me_r_llr)
                        r_llr_tpr.append(tpr_at_fpr(sp_r_llr, sm_r_llr))
                    except Exception:
                        pass

                    try:
                        auc_r_l1, sp_r_l1, sm_r_l1 = auc_scores(
                            pop_rpool, random_sub, pop_rpool, random_sub, LR=False)
                        me_r_l1 = compute_empirical_min_error_fast(
                            np.ravel(sp_r_l1), np.ravel(sm_r_l1))
                        r_l1_auc.append(auc_r_l1)
                        r_l1_me.append(me_r_l1)
                        r_l1_tpr.append(tpr_at_fpr(sp_r_l1, sm_r_l1))
                    except Exception:
                        pass

                if (it + 1) % 100 == 0:
                    print(".", end="", flush=True)

            # Theoretical bounds for this n
            stat_term, theory_min_err_stat, theory_min_err_full = \
                compute_theoretical_min_error(m, n, signal_term_full)
            _, _, theory_min_err_half = \
                compute_theoretical_min_error(m, n, signal_term_half)

            avg = lambda lst: np.mean(lst) if lst else np.nan
            std = lambda lst: np.std(lst, ddof=1) if len(lst) > 1 else np.nan

            row = {
                "disease": tag,
                "label": name,
                "n": n,
                "A_full": A_full,
                "A_diseased_miRNAs": A_diseased,
                "signal_term": signal_term_full,
                "signal_term_half": signal_term_half,
                "theory_min_err_full": theory_min_err_full,
                "theory_min_err_stat": theory_min_err_stat,
                "theory_min_err_half": theory_min_err_half,
                "case_min_err_llr": avg(c_llr_me),
                "case_min_err_l1": avg(c_l1_me),
                "random_min_err_llr": avg(r_llr_me),
                "random_min_err_l1": avg(r_l1_me),
                "case_auc_llr": avg(c_llr_auc),
                "case_auc_l1": avg(c_l1_auc),
                "random_auc_llr": avg(r_llr_auc),
                "random_auc_l1": avg(r_l1_auc),
                "case_tpr_llr": avg(c_llr_tpr),
                "case_tpr_l1": avg(c_l1_tpr),
                "random_tpr_llr": avg(r_llr_tpr),
                "random_tpr_l1": avg(r_l1_tpr),
                "case_min_err_llr_std": std(c_llr_me),
                "case_min_err_l1_std": std(c_l1_me),
                "random_min_err_llr_std": std(r_llr_me),
                "random_min_err_l1_std": std(r_l1_me),
                "case_auc_llr_std": std(c_llr_auc),
                "case_auc_l1_std": std(c_l1_auc),
                "random_auc_llr_std": std(r_llr_auc),
                "random_auc_l1_std": std(r_l1_auc),
                "case_tpr_llr_std": std(c_llr_tpr),
                "case_tpr_l1_std": std(c_l1_tpr),
                "random_tpr_llr_std": std(r_llr_tpr),
                "random_tpr_l1_std": std(r_l1_tpr),
                "n_ok_case_llr": len(c_llr_me),
                "n_ok_case_l1": len(c_l1_me),
                "n_ok_random_llr": len(r_llr_me),
                "n_ok_random_l1": len(r_l1_me),
                "iterations": iterations,
            }
            results.append(row)

            print(f"\n    case_err LLR={row['case_min_err_llr']:.4f} "
                  f"L1={row['case_min_err_l1']:.4f}  "
                  f"rand_err LLR={row['random_min_err_llr']:.4f} "
                  f"L1={row['random_min_err_l1']:.4f}  "
                  f"theory_full={theory_min_err_full:.4f}  "
                  f"ok={len(c_llr_me)}/{len(r_llr_me)}")

            # Save incrementally
            pd.DataFrame(results).to_csv("results/nsweep_results.csv", index=False)

    print(f"\nDone. Saved {len(results)} rows to nsweep_results.csv")
    return pd.DataFrame(results)


# ── Figures ──────────────────────────────────────────────────────────────────

def _setup_rc():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
    })


def _make_nsweep_figure(df, case_llr_col, case_l1_col, random_llr_col,
                         random_l1_col, ylabel, ylim, output_path,
                         baseline=None):
    """Render the 1×3 nsweep panel for any metric (AUC or TPR@1%FPR)."""
    _setup_rc()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, disease in zip(axes, TARGET_DISEASES):
        meta = DISEASE_META[disease]
        tag = meta["tag"]
        ddf = df[df["disease"] == tag].sort_values("n")
        ns = ddf["n"].values
        sig = ddf["signal_term"].iloc[0]

        if baseline is not None:
            value, label = baseline
            ax.axhline(value, color="grey", ls=":", lw=0.8, label=label)

        ax.plot(ns, ddf[case_llr_col].values, label="Case LLR",
                **line_kwargs("LLR", "case", markersize=4, linewidth=1.2))
        ax.plot(ns, ddf[case_l1_col].values, label="Case L1",
                **line_kwargs("L1", "case", markersize=4, linewidth=1.2))
        ax.plot(ns, ddf[random_llr_col].values, label="Random LLR",
                **line_kwargs("LLR", "random", markersize=4, linewidth=1.2))
        ax.plot(ns, ddf[random_l1_col].values, label="Random L1",
                **line_kwargs("L1", "random", markersize=4, linewidth=1.2))

        ax.set_xlabel("Pool size n")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_title(f"{tag}: {meta['name']}  (signal = {sig:.1f})")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    ncol = 5 if baseline is not None else 4
    fig.legend(handles, labels, loc="lower center", ncol=ncol, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def make_auc_figure(csv_path="results/nsweep_results.csv"):
    """nsweep_auc.pdf — AUC vs n, 1x3 panels ordered by increasing signal."""
    df = pd.read_csv(csv_path)
    _make_nsweep_figure(
        df,
        case_llr_col="case_auc_llr", case_l1_col="case_auc_l1",
        random_llr_col="random_auc_llr", random_l1_col="random_auc_l1",
        ylabel="AUC", ylim=(0.5, 1),
        output_path="nsweep_auc.pdf",
        baseline=(0.5, "AUC = 0.5"),
    )


def make_tpr_figure(csv_path="results/nsweep_results.csv"):
    """nsweep_tpr.pdf — TPR@1%FPR vs n, 1x3 panels ordered by increasing signal."""
    df = pd.read_csv(csv_path)
    if "case_tpr_llr" not in df.columns:
        print("nsweep_tpr.pdf skipped: CSV is missing TPR columns "
              "(rerun without --plot-only).")
        return
    _make_nsweep_figure(
        df,
        case_llr_col="case_tpr_llr", case_l1_col="case_tpr_l1",
        random_llr_col="random_tpr_llr", random_l1_col="random_tpr_l1",
        ylabel="TPR at 0.01 FPR", ylim=(0, 1),
        output_path="nsweep_tpr.pdf",
        baseline=(0.01, "TPR = FPR = 0.01"),
    )


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pool-size sweep experiment (Theorem 1 validation)")
    parser.add_argument("--iterations", type=int, default=2000,
                        help="Iterations per (disease, n) pair (default 2000)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Generate figures from existing CSV only")
    args = parser.parse_args()

    if args.plot_only:
        make_auc_figure()
        make_tpr_figure()
    else:
        run_experiment(iterations=args.iterations)
        make_auc_figure()
        make_tpr_figure()
