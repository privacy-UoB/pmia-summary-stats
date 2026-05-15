#!/usr/bin/env python3
"""Linear/residual drift factorial experiment.

Decomposes each individual's drift into a per-individual linear component
(β_k · x_k + α_k) and a residual (ε_k), then swaps each component
independently using the k=1 nearest neighbor.

Tests whether the destructive baseline-drift coupling lives in the
per-individual regression slope or in the nonlinear residual.

Produces fig_linear_residual.csv and fig_linear_residual.pdf (a 2×2
panel: AUC on top, TPR @ 1% FPR on bottom; LLR / L1 across the columns).
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

from experiment_io import resolve_output_path
from utils_datasets import load_timestamp_dataset, drop_timestamp_index
from utils import auc_scores, LLR, tpr_at_fpr

NUM_ITERATIONS = 500
NUM_WORKERS = max(1, os.cpu_count() - 1)
TEST_TIMEPOINTS = [1, 4]
CONDITIONS = ["self_self", "self_nn", "nn_self", "nn_nn"]


def _get_nn_indices(baseline):
    """Return array where entry i is the index of i's nearest neighbor."""
    dist = cdist(baseline, baseline, metric='euclidean')
    np.fill_diagonal(dist, np.inf)
    return np.argmin(dist, axis=1)


def _decompose_drift(baseline, diff):
    """Fit per-individual OLS: δ_kj = β_k · x_kj(0) + α_k + ε_kj.

    Returns (betas, alphas, residuals) where betas and alphas are 1-d arrays
    of length n, and residuals has the same shape as diff.
    """
    n = len(baseline)
    betas = np.empty(n)
    alphas = np.empty(n)
    for k in range(n):
        betas[k], alphas[k] = np.polyfit(baseline[k], diff[k], 1)
    linear_pred = betas[:, None] * baseline + alphas[:, None]
    residuals = diff - linear_pred
    return betas, alphas, residuals


def _reconstruct(baseline, betas, alphas, residuals, nn_idx, condition):
    """Build synthetic follow-up under a factorial condition.

    condition is one of: self_self, self_nn, nn_self, nn_nn.
    The first token controls whose (β, α) to use; the second controls whose ε.
    NN's β is always applied to the individual's own baseline.
    """
    lin_src, res_src = condition.split('_')
    b = betas if lin_src == 'self' else betas[nn_idx]
    a = alphas if lin_src == 'self' else alphas[nn_idx]
    e = residuals if res_src == 'self' else residuals[nn_idx]
    synth_diff = b[:, None] * baseline + a[:, None] + e
    return baseline + synth_diff


def _one_iteration(seed):
    """Run one iteration across all conditions and timepoints."""
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

    results = {}

    for t in TEST_TIMEPOINTS:
        if t >= len(ti_pop):
            continue
        t_pop = np.array(ti_pop[t], dtype=float)
        t_pool = np.array(ti_pool[t], dtype=float)

        if len(t_pop) != n_pop or len(t_pool) != n_pool:
            continue

        pop_diff = t_pop - pop0
        pool_diff = t_pool - pool0

        # NN indices
        nn_pop = _get_nn_indices(pop0)
        nn_pool = _get_nn_indices(pool0)

        # Decompose drift
        pop_betas, pop_alphas, pop_resid = _decompose_drift(pop0, pop_diff)
        pool_betas, pool_alphas, pool_resid = _decompose_drift(pool0, pool_diff)

        # Factorial conditions
        for cond in CONDITIONS:
            synth_pop = _reconstruct(
                pop0, pop_betas, pop_alphas, pop_resid, nn_pop, cond)
            synth_pool = _reconstruct(
                pool0, pool_betas, pool_alphas, pool_resid, nn_pool, cond)
            with contextlib.redirect_stdout(io.StringIO()):
                roc_llr, sp_llr, sm_llr = auc_scores(
                    synth_pop, synth_pool, pop0, pool0, LR=True)
                roc_l1, sp_l1, sm_l1 = auc_scores(
                    synth_pop, synth_pool, pop0, pool0, LR=False)
            results[f"t{t}_{cond}_LLR"] = roc_llr
            results[f"t{t}_{cond}_L1"] = roc_l1
            results[f"t{t}_{cond}_LLR_tpr"] = tpr_at_fpr(sp_llr, sm_llr)
            results[f"t{t}_{cond}_L1_tpr"] = tpr_at_fpr(sp_l1, sm_l1)

        # Full-NN reference: assign each individual their NN's entire drift
        nn_pop_diff = pop_diff[nn_pop]
        nn_pool_diff = pool_diff[nn_pool]
        noisy_pop = pop0 + nn_pop_diff
        noisy_pool = pool0 + nn_pool_diff
        with contextlib.redirect_stdout(io.StringIO()):
            roc_nn_llr, sp_nn_llr, sm_nn_llr = auc_scores(
                noisy_pop, noisy_pool, pop0, pool0, LR=True)
            roc_nn_l1, sp_nn_l1, sm_nn_l1 = auc_scores(
                noisy_pop, noisy_pool, pop0, pool0, LR=False)
        results[f"t{t}_full_nn_LLR"] = roc_nn_llr
        results[f"t{t}_full_nn_L1"] = roc_nn_l1
        results[f"t{t}_full_nn_LLR_tpr"] = tpr_at_fpr(sp_nn_llr, sm_nn_llr)
        results[f"t{t}_full_nn_L1_tpr"] = tpr_at_fpr(sp_nn_l1, sm_nn_l1)

        # Random permutation shuffle reference
        perm_pop_diff = pop_diff[rng.permutation(n_pop)]
        perm_pool_diff = pool_diff[rng.permutation(n_pool)]
        noisy_pop = pop0 + perm_pop_diff
        noisy_pool = pool0 + perm_pool_diff
        with contextlib.redirect_stdout(io.StringIO()):
            roc_perm_llr, sp_perm_llr, sm_perm_llr = auc_scores(
                noisy_pop, noisy_pool, pop0, pool0, LR=True)
            roc_perm_l1, sp_perm_l1, sm_perm_l1 = auc_scores(
                noisy_pop, noisy_pool, pop0, pool0, LR=False)
        results[f"t{t}_perm_LLR"] = roc_perm_llr
        results[f"t{t}_perm_L1"] = roc_perm_l1
        results[f"t{t}_perm_LLR_tpr"] = tpr_at_fpr(sp_perm_llr, sm_perm_llr)
        results[f"t{t}_perm_L1_tpr"] = tpr_at_fpr(sp_perm_l1, sm_perm_l1)

    return results


def _setup_rc():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
    })


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

    df_all = pd.DataFrame(all_results)

    # Build output CSV
    all_conditions = CONDITIONS + ["full_nn", "perm"]
    rows = []
    for cond in all_conditions:
        row = {"condition": cond}
        for t in TEST_TIMEPOINTS:
            for metric in ("LLR", "L1"):
                col = f"t{t}_{cond}_{metric}"
                if col in df_all.columns:
                    row[f"t{t}_{metric}"] = df_all[col].mean()
                tpr_col = f"t{t}_{cond}_{metric}_tpr"
                if tpr_col in df_all.columns:
                    row[f"t{t}_{metric}_tpr"] = df_all[tpr_col].mean()
        rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(resolve_output_path("fig_linear_residual.csv"), index=False)
    print(f"\nSaved fig_linear_residual.csv")
    print(df_out.to_string(index=False))

    # ---- Plot: 2×2 grid (rows: AUC / TPR, cols: LLR / L1) ----
    _setup_rc()

    def _val(cond, metric_col):
        return df_out.loc[df_out["condition"] == cond, metric_col].values[0]

    group_labels = ["Self residual", "NN residual"]
    x = np.arange(len(group_labels))
    width = 0.35

    output_path = resolve_output_path("fig_linear_residual.pdf")
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex='col')

    rows = [("",     "AUC",          (0.5, 1.0)),
            ("_tpr", "TPR @ 1% FPR", (0.0, 1.0))]
    cols = [("LLR", 0), ("L1", 1)]

    for r, (col_suffix, ylabel, ylim) in enumerate(rows):
        for metric, c in cols:
            ax = axes[r, c]
            col = f"t1_{metric}{col_suffix}"
            self_lin = [_val("self_self", col), _val("self_nn", col)]
            nn_lin = [_val("nn_self", col), _val("nn_nn", col)]

            ax.bar(x - width / 2, self_lin, width, label="Self linear",
                   color="tab:blue")
            ax.bar(x + width / 2, nn_lin, width, label="NN linear",
                   color="tab:orange")

            real_val = _val("self_self", col)
            ax.axhline(real_val, ls=":", lw=1, color="grey", alpha=0.6)
            ax.text(0.02, real_val, "real drift", va="bottom", fontsize=8,
                    color="grey", alpha=0.8, ha="left",
                    transform=ax.get_yaxis_transform())

            ax.set_xticks(x)
            ax.set_xticklabels(group_labels)
            ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.3, axis="y")
            if r == 0:
                ax.set_title(metric)
            if c == 0:
                ax.set_ylabel(ylabel)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    run()
