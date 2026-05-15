import os
import sys
import random

import numpy as np
import matplotlib

from experiment_io import parse_flags, seed_all, save_figdata, load_figdata, resolve_output_path

_flags = parse_flags(sys.argv)
seed_all(_flags["seed"])

if len(sys.argv) >= 4 or _flags["replot"]:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from plot_style import stacked_auc_tpr, noise_sequential  # noqa: E402
from utils_datasets import load_dataset, separate_diseased_miRNAs, D3, D17  # noqa: E402
from fast_paths import ordered_curves_llr, ordered_curves_l1  # noqa: E402


# Random-pool sampling variance at n=65 is large (~0.07 in AUC), so we average
# AUC curves over K independent pool draws. Case pool (POOL_IDX=1) is
# disease-deterministic, so K=1 there.
NUM_POOLS_DEFAULT = 20


def make_figure(data: dict, output_path: str | None) -> None:
    multiplier = list(np.asarray(data["multiplier"]))
    num_miRNAs = list(np.asarray(data["num_miRNAs"]))
    fixed_FPR = bool(np.asarray(data["_fixed_FPR"]).item())
    L1_or_LLR = str(np.asarray(data["_L1_or_LLR"]).item())

    if fixed_FPR:
        fig, ax_auc, ax_tpr = stacked_auc_tpr()
    else:
        fig, ax_auc = plt.subplots()
        ax_tpr = None

    # Light-to-dark ramp within the metric's hue family. Same palette used on
    # both panels so a given noise level is the same shade in AUC and TPR.
    palette = noise_sequential(L1_or_LLR, len(multiplier))

    if L1_or_LLR == "L1":
        noise_fraction_L1 = [list(row) for row in np.asarray(data["noise_fraction_L1"], dtype=object)]
        if fixed_FPR:
            noise_fraction_tpr_at_fpr_L1 = [list(row) for row in np.asarray(data["noise_fraction_tpr_at_fpr_L1"], dtype=object)]
    else:
        noise_fraction_LLR = [list(row) for row in np.asarray(data["noise_fraction_LLR"], dtype=object)]
        if fixed_FPR:
            noise_fraction_tpr_at_fpr_LLR = [list(row) for row in np.asarray(data["noise_fraction_tpr_at_fpr_LLR"], dtype=object)]

    for index, noise in enumerate(multiplier):
        c = palette[index]
        if L1_or_LLR == "L1":
            ax_auc.plot(num_miRNAs, noise_fraction_L1[index],
                        color=c, linewidth=2.0,
                        label=f"std. dev. = {noise}")
            if fixed_FPR:
                ax_tpr.plot(num_miRNAs, noise_fraction_tpr_at_fpr_L1[index],
                            color=c, linewidth=2.0,
                            label=f"std. dev. = {noise}")
        elif L1_or_LLR == "LLR":
            ax_auc.plot(num_miRNAs, noise_fraction_LLR[index],
                        color=c, linewidth=2.0,
                        label=f"std. dev. = {noise}")
            if fixed_FPR:
                ax_tpr.plot(num_miRNAs, noise_fraction_tpr_at_fpr_LLR[index],
                            color=c, linewidth=2.0,
                            label=f"std. dev. = {noise}")

    ax_auc.invert_xaxis()
    ax_auc.legend(loc='upper right')
    ax_auc.set_ylabel("AUC")
    ax_auc.set_ylim([0.5, 1])
    ax_auc.grid(True)
    if fixed_FPR:
        ax_tpr.set_xlabel("number miRNAs")
        ax_tpr.set_ylabel("TPR at 0.01 FPR")
        ax_tpr.set_ylim([0, 1])
        ax_tpr.grid(True)
    else:
        ax_auc.set_xlabel("number miRNAs")

    if output_path:
        plt.savefig(output_path)
        print(f"Saved to {output_path}")
    else:
        plt.show()


# CLI: python Ordered_Noise.py <disease> <metric> <pool_idx> [random_sample_size] [output.pdf]
# Falls back to interactive defaults when no args given.
DISEASES = {"D3": D3, "D17": D17}
if len(sys.argv) >= 4:
    DISEASE = DISEASES[sys.argv[1]]
    L1_or_LLR = sys.argv[2]
    POOL_IDX = int(sys.argv[3])
    RANDOM_SAMPLE_SIZE = int(sys.argv[4]) if len(sys.argv) >= 5 and sys.argv[4] != "_" else None
    OUTPUT_FILE = resolve_output_path(sys.argv[5] if len(sys.argv) >= 6 else None)
else:
    DISEASE = D3
    L1_or_LLR = "L1"
    POOL_IDX = 1
    RANDOM_SAMPLE_SIZE = None
    OUTPUT_FILE = None

if _flags["replot"]:
    data, _meta = load_figdata(_flags["replot"])
    make_figure(data, OUTPUT_FILE)
    sys.exit(0)


stratifying = False  # Not enough pool miRNAs in True case
fixed_FPR = True
# Env overrides for cheap smoke tests and runtime tuning.
num_orders = int(os.environ.get("NUM_ORDERS", 2000))
_num_pools_default = int(os.environ.get("NUM_POOLS", NUM_POOLS_DEFAULT))
NUM_POOLS = _num_pools_default if POOL_IDX == 0 else 1   # case pool is deterministic
base_seed = _flags["seed"]


def _compute_one_pool(pool_seed: int):
    """Reseed, draw a fresh random pool of n=65 (or use the deterministic case
    pool), and return (auc, tpr) curves of shape (n_multipliers, n_miRNA_counts)
    plus the miRNA-count x-axis."""
    np.random.seed(pool_seed)
    random.seed(pool_seed)

    if stratifying:
        only_pop, wo_pop, only_pool, wo_pool = separate_diseased_miRNAs(DISEASE, "miRNA")
        pop_df = wo_pop
        pool_df = wo_pool
    else:
        populations, pools = load_dataset(
            miRNA=True, disease_case_sample=DISEASE,
            random_sample_size=RANDOM_SAMPLE_SIZE)
        pop_df = populations[POOL_IDX]
        pool_df = pools[POOL_IDX]

    sigma_j = np.std(pop_df, axis=0).to_numpy()
    n_feat = pop_df.shape[1]
    miRNA_counts = list(range(2, n_feat, 2))   # 2, 4, ..., n_feat-1 (or n_feat-2 for even)

    multipliers = [0, 0.25, 0.5, 0.75, 1]
    pop_arr = pop_df.to_numpy(dtype=np.float64)
    pool_arr = pool_df.to_numpy(dtype=np.float64)

    if L1_or_LLR == "LLR":
        auc, tpr = ordered_curves_llr(
            pop_arr, pool_arr, multipliers, miRNA_counts, num_orders,
            sigma_j, np.random, random, target_fpr=1e-2)
    elif L1_or_LLR == "L1":
        auc, tpr = ordered_curves_l1(
            pop_arr, pool_arr, multipliers, miRNA_counts, num_orders,
            sigma_j, np.random, random, target_fpr=1e-2)
    else:
        raise ValueError(f"Unknown metric: {L1_or_LLR}")
    return auc, tpr, multipliers, miRNA_counts


pool_aucs = []
pool_tprs = []
pool_seeds = []
multiplier = None
num_miRNAs = None

average_over_pools = (POOL_IDX == 0)  # case pool is disease-deterministic
for k in range(NUM_POOLS):
    pool_seed = (base_seed * 1000 + k) if average_over_pools else base_seed
    pool_seeds.append(pool_seed)
    print(f"\n=== pool {k + 1}/{NUM_POOLS}  (seed={pool_seed}) ===", flush=True)
    auc, tpr, multipliers, miRNA_counts = _compute_one_pool(pool_seed)
    pool_aucs.append(auc)
    pool_tprs.append(tpr)
    if multiplier is None:
        multiplier = multipliers
        num_miRNAs = miRNA_counts

per_pool_auc = np.stack(pool_aucs, axis=0)         # (K, n_mults, n_cnts)
per_pool_tpr = np.stack(pool_tprs, axis=0)
mean_auc = per_pool_auc.mean(axis=0)
mean_tpr = per_pool_tpr.mean(axis=0)


# Build the figure-data dict and save before plotting.
data = {
    "multiplier": np.asarray(multiplier),
    "num_miRNAs": np.asarray(num_miRNAs),
    "_fixed_FPR": fixed_FPR,
    "_L1_or_LLR": L1_or_LLR,
}
if L1_or_LLR == "L1":
    data["noise_fraction_L1"] = mean_auc
    data["noise_fraction_L1_per_pool"] = per_pool_auc
    if fixed_FPR:
        data["noise_fraction_tpr_at_fpr_L1"] = mean_tpr
        data["noise_fraction_tpr_at_fpr_L1_per_pool"] = per_pool_tpr
else:
    data["noise_fraction_LLR"] = mean_auc
    data["noise_fraction_LLR_per_pool"] = per_pool_auc
    if fixed_FPR:
        data["noise_fraction_tpr_at_fpr_LLR"] = mean_tpr
        data["noise_fraction_tpr_at_fpr_LLR_per_pool"] = per_pool_tpr

meta = {
    "seed": base_seed,
    "disease": sys.argv[1] if len(sys.argv) >= 2 else None,
    "L1_or_LLR": L1_or_LLR,
    "pool_idx": POOL_IDX,
    "random_sample_size": RANDOM_SAMPLE_SIZE,
    "fixed_FPR": fixed_FPR,
    "num_orders": num_orders,
    "num_pools": NUM_POOLS,
    "pool_seeds": pool_seeds,
    "stratifying": stratifying,
}

if OUTPUT_FILE:
    save_figdata(OUTPUT_FILE, data, meta)

make_figure(data, OUTPUT_FILE)
