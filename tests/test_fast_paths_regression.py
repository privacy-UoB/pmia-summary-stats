"""Regression test: new fast paths in fast_paths.py must match the existing
auc_scores-based pipeline numerically.

Run with:  uv run python tests/test_fast_paths_regression.py
"""

import contextlib
import io
import os
import random
import sys

import numpy as np

# Make the project root importable when invoked as a script from anywhere.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from utils_datasets import load_dataset, D3                              # noqa: E402
from utils import auc_scores, Gaussian_noise                             # noqa: E402
from fast_paths import (                                                  # noqa: E402
    ordered_curves_llr, ordered_curves_l1, noise_curves,
)


def _seed(s):
    np.random.seed(s)
    random.seed(s)


def baseline_ordered_curves(pop, pool, multipliers, miRNA_counts,
                            num_orders, sigma_j, *, metric, target_fpr=1e-2):
    """Reproduces the inner loop of Ordered_Noise.py for one pool."""
    miRNAs = list(pop.keys())
    shuffled = []
    for _ in range(num_orders):
        s = list(miRNAs)
        random.shuffle(s)
        shuffled.append(s)

    auc_out = np.zeros((len(multipliers), len(miRNA_counts)))
    tpr_out = np.zeros((len(multipliers), len(miRNA_counts)))

    for m_idx, m in enumerate(multipliers):
        nopops = []
        nopools = []
        for _ in range(num_orders):
            nopop, nopool = Gaussian_noise(pop, pool, 0, m * sigma_j, clip=True)
            nopops.append(nopop)
            nopools.append(nopool)

        for k, i in enumerate(miRNA_counts):
            aucs, tprs = [], []
            for j in range(num_orders):
                sel = shuffled[j][:i]
                lnp = nopops[j][sel]; lnpp = nopools[j][sel]
                lp = pop[sel]; lpp = pool[sel]
                with contextlib.redirect_stdout(io.StringIO()):
                    auc = auc_scores(lnp, lnpp, lp, lpp,
                                     LR=(metric == "LLR"), p_values=False)
                    fpr, tpr, _ = auc_scores(lnp, lnpp, lp, lpp,
                                             LR=(metric == "LLR"), FPR=True)
                aucs.append(auc)
                tprs.append(np.interp(target_fpr, fpr, tpr))
            auc_out[m_idx, k] = np.mean(aucs)
            tpr_out[m_idx, k] = np.mean(tprs)

    return auc_out, tpr_out


def baseline_noise_curves_dual(pop, pool, multipliers, num_orders, *,
                               include_deviations, sigma_j, target_fpr=1e-2):
    """Reproduces the inner loop of Noise.py (miRNA path) for one pool.
    Returns (auc_L1, auc_LLR, tpr_L1, tpr_LLR), each length n_mults."""
    nm = len(multipliers)
    aL1, aLLR = np.zeros(nm), np.zeros(nm)
    tL1, tLLR = np.zeros(nm), np.zeros(nm)
    for m_idx, m in enumerate(multipliers):
        dev = (m * sigma_j) if include_deviations else m
        pa1, pa2, pt1, pt2 = [], [], [], []
        for _ in range(num_orders):
            nopop, nopool = Gaussian_noise(pop, pool, 0, dev, clip=True)
            with contextlib.redirect_stdout(io.StringIO()):
                a1 = auc_scores(nopop, nopool, pop, pool, LR=False, p_values=False)
                a2 = auc_scores(nopop, nopool, pop, pool, LR=True, p_values=False)
                f1, t1, _ = auc_scores(nopop, nopool, pop, pool, LR=False, FPR=True)
                f2, t2, _ = auc_scores(nopop, nopool, pop, pool, LR=True, FPR=True)
            pa1.append(a1); pa2.append(a2)
            pt1.append(np.interp(target_fpr, f1, t1))
            pt2.append(np.interp(target_fpr, f2, t2))
        aL1[m_idx] = np.mean(pa1); aLLR[m_idx] = np.mean(pa2)
        tL1[m_idx] = np.mean(pt1); tLLR[m_idx] = np.mean(pt2)
    return aL1, aLLR, tL1, tLLR


def run_ordered_test(metric):
    print(f"\n=== Ordered_Noise — {metric} ===", flush=True)
    NUM_ORDERS = 8
    miRNA_counts = [4, 50, 200, 400]
    multipliers = [0.0, 0.25, 0.5, 1.0]

    _seed(42)
    pop_l, pool_l = load_dataset(miRNA=True, disease_case_sample=D3, random_sample_size=None)
    pop = pop_l[0]; pool = pool_l[0]
    sigma_j = np.std(pop, axis=0)

    _seed(42)
    # Recreate the dataset load to consume RNG identically (it advances state).
    _ = load_dataset(miRNA=True, disease_case_sample=D3, random_sample_size=None)
    auc_old, tpr_old = baseline_ordered_curves(
        pop, pool, multipliers, miRNA_counts, NUM_ORDERS, sigma_j, metric=metric)

    _seed(42)
    _ = load_dataset(miRNA=True, disease_case_sample=D3, random_sample_size=None)
    if metric == "LLR":
        auc_new, tpr_new = ordered_curves_llr(
            np.asarray(pop), np.asarray(pool), multipliers, miRNA_counts, NUM_ORDERS,
            np.asarray(sigma_j), np.random, random)
    else:
        auc_new, tpr_new = ordered_curves_l1(
            np.asarray(pop), np.asarray(pool), multipliers, miRNA_counts, NUM_ORDERS,
            np.asarray(sigma_j), np.random, random)

    auc_diff = np.abs(auc_old - auc_new).max()
    tpr_diff = np.abs(tpr_old - tpr_new).max()
    print(f"  max |AUC_old - AUC_new| = {auc_diff:.6f}", flush=True)
    print(f"  max |TPR_old - TPR_new| = {tpr_diff:.6f}", flush=True)
    print(f"  AUC_old peak (m=0): {auc_old[0].max():.4f}    AUC_new peak (m=0): {auc_new[0].max():.4f}", flush=True)
    ok = auc_diff < 5e-3 and tpr_diff < 5e-3
    print(f"  {'OK' if ok else 'FAIL'}", flush=True)
    return ok


def run_noise_test():
    print("\n=== Noise — dual L1+LLR (miRNA random pool) ===", flush=True)
    NUM_ORDERS = 16
    multipliers = [0.01, 0.1, 1.0, 10.0]
    include_deviations = True

    _seed(42)
    pop_l, pool_l = load_dataset(miRNA=True, disease_case_sample=D3, random_sample_size=None)
    pop = pop_l[0]; pool = pool_l[0]
    sigma_j = np.std(pop, axis=0)

    _seed(42)
    _ = load_dataset(miRNA=True, disease_case_sample=D3, random_sample_size=None)
    aL1_old, aLLR_old, tL1_old, tLLR_old = baseline_noise_curves_dual(
        pop, pool, multipliers, NUM_ORDERS,
        include_deviations=include_deviations, sigma_j=sigma_j)

    _seed(42)
    _ = load_dataset(miRNA=True, disease_case_sample=D3, random_sample_size=None)
    new = noise_curves(
        np.asarray(pop), np.asarray(pool), multipliers, NUM_ORDERS,
        include_deviations=include_deviations,
        sigma_j=np.asarray(sigma_j), rng_np=np.random)

    diffs = {
        "auc_L1": np.abs(aL1_old - new["auc_L1"]).max(),
        "auc_LLR": np.abs(aLLR_old - new["auc_LLR"]).max(),
        "tpr_L1": np.abs(tL1_old - new["tpr_L1"]).max(),
        "tpr_LLR": np.abs(tLLR_old - new["tpr_LLR"]).max(),
    }
    print(f"  AUC_L1 old : {np.round(aL1_old, 4)}    new : {np.round(new['auc_L1'], 4)}", flush=True)
    print(f"  AUC_LLR old: {np.round(aLLR_old, 4)}    new: {np.round(new['auc_LLR'], 4)}", flush=True)
    for k, v in diffs.items():
        print(f"  max |{k}_old - {k}_new| = {v:.6f}", flush=True)
    ok = all(v < 5e-3 for v in diffs.values())
    print(f"  {'OK' if ok else 'FAIL'}", flush=True)
    return ok


if __name__ == "__main__":
    results = []
    results.append(("ordered LLR", run_ordered_test("LLR")))
    results.append(("ordered L1", run_ordered_test("L1")))
    results.append(("noise dual", run_noise_test()))
    print()
    for name, ok in results:
        print(f"  {name}: {'OK' if ok else 'FAIL'}")
    sys.exit(0 if all(ok for _, ok in results) else 1)
