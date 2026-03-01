#!/usr/bin/env python3
"""Membership inference experiment across 19 diseases in GSE61741 miRNA dataset.

Computes AUC and empirical minimum error for case/random pools, computes
theoretical minimum error bounds from Theorem 1, and produces a 2-panel
figure.

Usage:
    python experiment_19disease.py [--iterations N] [--plot-only]
"""

import argparse
import io
import sys
import contextlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import ShuffleSplit

from utils import auc_scores
from utils_datasets import (D1, D2, D3, D4, D5, D6, D7, D8, D9, D10,
                             D11, D12, D13, D14, D15, D16, D17, D18, D19)

# ── Disease metadata ─────────────────────────────────────────────────────────

DISEASES = [D1, D2, D3, D4, D5, D6, D7, D8, D9, D10,
            D11, D12, D13, D14, D15, D16, D17, D18, D19]

DISEASE_NAMES = {
    D1: "Wilms Tumor", D2: "Lung Cancer", D3: "Prostate Cancer",
    D4: "MI", D5: "COPD", D6: "Sarcoidosis",
    D7: "Ductal Adeno.", D8: "Psoriasis", D9: "Pancreatitis",
    D10: "BPH", D11: "Melanoma", D12: "Heart Failure",
    D13: "Colon Cancer", D14: "Ovarian Cancer", D15: "MS",
    D16: "Glioma", D17: "Renal Cancer", D18: "Periodontitis",
    D19: "Stomach Tumor",
}

SHORT_NAMES = {
    D1: "WT", D2: "LC", D3: "PC", D4: "MI", D5: "COPD",
    D6: "Sarc", D7: "DA", D8: "Psor", D9: "Panc", D10: "BPH",
    D11: "Mel", D12: "HF", D13: "CC", D14: "OC", D15: "MS",
    D16: "Glio", D17: "RC", D18: "Perio", D19: "TS",
}


# ── Optimised data loading ───────────────────────────────────────────────────

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


def get_disease_splits(filter_population, disease_label, n_random=None):
    """Extract case pool (deterministic) and random pool (fresh each call).

    Random pool size defaults to the case pool size for fair comparison.
    Returns (pop_cpool, case_pool, pop_rpool, random_pool) as numpy arrays.
    """
    case_pool_df = filter_population[filter_population["diseases"] == disease_label]
    pop_cpool_df = filter_population[filter_population["diseases"] != disease_label]

    a = case_pool_df.shape[0] if n_random is None else n_random
    rs = ShuffleSplit(n_splits=1, test_size=a)
    train, test = next(rs.split(filter_population))
    pop_rpool_df = filter_population.iloc[train]
    random_pool_df = filter_population.iloc[test]

    drop = lambda df: df.drop(columns="diseases").values
    return drop(pop_cpool_df), drop(case_pool_df), drop(pop_rpool_df), drop(random_pool_df)


# ── Metrics ──────────────────────────────────────────────────────────────────

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
    """Run the 19-disease experiment and save results to CSV."""
    print("Loading miRNA dataset ...")
    filter_pop = load_miRNA_data_once()
    n_total = filter_pop.shape[0]
    m = filter_pop.shape[1] - 1  # 465 features (minus 'diseases' column)
    print(f"Dataset: {n_total} individuals, {m} features")

    results = []

    for idx, disease in enumerate(DISEASES):
        name = DISEASE_NAMES[disease]
        short = SHORT_NAMES[disease]
        print(f"\n[{idx + 1}/19] {name} ({short})")

        # ── deterministic case-pool split ──
        pop_cpool, case_pool, _, _ = get_disease_splits(filter_pop, disease)
        A = case_pool.shape[0]
        print(f"  |A|={A}, pop_cpool={pop_cpool.shape[0]}, m={m}")

        # ── theoretical bounds ──
        signal_term = compute_signal_term(pop_cpool, case_pool)
        stat_term, min_err_stat, min_err_full = compute_theoretical_min_error(
            m, A, signal_term)
        print(f"  signal={signal_term:.4f}  stat={stat_term:.4f}  "
              f"theory_full={min_err_full:.4f}  theory_stat={min_err_stat:.4f}")

        # ── case-pool empirical metrics (computed once — deterministic) ──
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                c_auc_llr, c_sp_llr, c_sm_llr = auc_scores(
                    pop_cpool, case_pool, pop_cpool, case_pool, LR=True)
                c_me_llr = compute_empirical_min_error_fast(
                    np.ravel(c_sp_llr), np.ravel(c_sm_llr))
            except Exception as e:
                print(f"  WARN case LLR: {e}", file=sys.stderr)
                c_auc_llr = c_me_llr = np.nan

            try:
                c_auc_l1, c_sp_l1, c_sm_l1 = auc_scores(
                    pop_cpool, case_pool, pop_cpool, case_pool, LR=False)
                c_me_l1 = compute_empirical_min_error_fast(
                    np.ravel(c_sp_l1), np.ravel(c_sm_l1))
            except Exception as e:
                print(f"  WARN case L1: {e}", file=sys.stderr)
                c_auc_l1 = c_me_l1 = np.nan

        print(f"  Case  AUC  LLR={c_auc_llr:.4f}  L1={c_auc_l1:.4f}")
        print(f"  Case  err  LLR={c_me_llr:.4f}  L1={c_me_l1:.4f}")

        # ── random-pool iterations ──
        r_llr_auc, r_l1_auc = [], []
        r_llr_me, r_l1_me = [], []

        for it in range(iterations):
            _, _, pop_rpool, rpool = get_disease_splits(filter_pop, disease)

            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    ra_llr, rsp_llr, rsm_llr = auc_scores(
                        pop_rpool, rpool, pop_rpool, rpool, LR=True)
                    rme_llr = compute_empirical_min_error_fast(
                        np.ravel(rsp_llr), np.ravel(rsm_llr))
                    r_llr_auc.append(ra_llr)
                    r_llr_me.append(rme_llr)
                except Exception:
                    pass

                try:
                    ra_l1, rsp_l1, rsm_l1 = auc_scores(
                        pop_rpool, rpool, pop_rpool, rpool, LR=False)
                    rme_l1 = compute_empirical_min_error_fast(
                        np.ravel(rsp_l1), np.ravel(rsm_l1))
                    r_l1_auc.append(ra_l1)
                    r_l1_me.append(rme_l1)
                except Exception:
                    pass

            if (it + 1) % 100 == 0:
                print(f"  iter {it + 1}/{iterations}", end="\r")

        print()
        avg = lambda lst: np.mean(lst) if lst else np.nan
        print(f"  Rand  AUC  LLR={avg(r_llr_auc):.4f}  L1={avg(r_l1_auc):.4f}")
        print(f"  Rand  err  LLR={avg(r_llr_me):.4f}  L1={avg(r_l1_me):.4f}")

        results.append({
            "disease": f"D{idx + 1}",
            "label": name,
            "n": n_total,
            "A": A,
            "signal_term": signal_term,
            "stat_term": stat_term,
            "theory_min_error_full": min_err_full,
            "theory_min_error_stat": min_err_stat,
            "case_auc_llr": c_auc_llr,
            "random_auc_llr": avg(r_llr_auc),
            "case_auc_l1": c_auc_l1,
            "random_auc_l1": avg(r_l1_auc),
            "case_emp_min_err_llr": c_me_llr,
            "random_emp_min_err_llr": avg(r_llr_me),
            "case_emp_min_err_l1": c_me_l1,
            "random_emp_min_err_l1": avg(r_l1_me),
        })

        # save incrementally so progress is preserved on interruption
        pd.DataFrame(results).to_csv("19disease_results.csv", index=False)
        print(f"  Saved ({len(results)}/19)")

    return pd.DataFrame(results)


# ── Two-panel figure ─────────────────────────────────────────────────────────

def make_figure(csv_path="19disease_results.csv"):
    """Generate 2-panel figure from results CSV.

    Panel A: Pool size vs AUC (case and random curves).
    Panel B: Case AUC vs Random AUC scatter with identity line.
    """
    df = pd.read_csv(csv_path)
    d2s = {f"D{i + 1}": s for i, (_, s) in enumerate(SHORT_NAMES.items())}
    df = df.sort_values("A").reset_index(drop=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
    })
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Panel A: Pool size vs AUC (both curves) ─────────────────────────
    ax = axes[0]
    ax.scatter(df["A"], df["case_auc_llr"], marker="o", color="steelblue",
               s=30, edgecolors="k", linewidths=0.5, label="Case pool", zorder=3)
    ax.plot(df["A"], df["random_auc_llr"], "s-", color="coral",
            markersize=5, linewidth=1.2, label="Random pool", zorder=3)

    for _, r in df.iterrows():
        ax.annotate(d2s[r["disease"]], (r["A"], r["case_auc_llr"]),
                    fontsize=7, xytext=(3, 4), textcoords="offset points")

    ax.set_xlabel("Pool size $n$")
    ax.set_ylabel("AUC (LLR)")
    ax.set_title("A. Pool size vs. AUC")
    ax.legend(fontsize=9, loc="lower left")

    # ── Panel B: Case AUC vs Random AUC ──────────────────────────────────
    ax = axes[1]
    lo = min(df["random_auc_llr"].min(), df["case_auc_llr"].min()) - 0.02
    hi = max(df["random_auc_llr"].max(), df["case_auc_llr"].max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="$y = x$", zorder=1)

    ax.scatter(df["random_auc_llr"], df["case_auc_llr"],
               s=50, edgecolors="k", linewidths=0.5, c="steelblue", zorder=3)
    for _, r in df.iterrows():
        ax.annotate(d2s[r["disease"]],
                    (r["random_auc_llr"], r["case_auc_llr"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")

    mean_gap = (df["case_auc_llr"] - df["random_auc_llr"]).mean()
    ax.text(0.05, 0.95, f"Mean gap = {mean_gap:.3f}",
            transform=ax.transAxes, fontsize=9, va="top")

    ax.set_xlabel("Random-pool AUC (LLR)")
    ax.set_ylabel("Case-pool AUC (LLR)")
    ax.set_title("B. Case vs. random AUC")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    fig.savefig("fig_19disease.pdf", dpi=300, bbox_inches="tight")
    fig.savefig("fig_19disease.png", dpi=300, bbox_inches="tight")
    print("Saved fig_19disease.pdf and fig_19disease.png")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="19-disease membership inference experiment")
    parser.add_argument("--iterations", type=int, default=2000,
                        help="Random-pool iterations per disease (default 2000)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Generate figure from existing CSV only")
    args = parser.parse_args()

    if args.plot_only:
        make_figure()
    else:
        run_experiment(iterations=args.iterations)
        make_figure()
