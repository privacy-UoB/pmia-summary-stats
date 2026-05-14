import os
import sys
import random

import numpy as np
import matplotlib

from experiment_io import parse_flags, seed_all, save_figdata, load_figdata

_flags = parse_flags(sys.argv)
seed_all(_flags["seed"])

if len(sys.argv) >= 3 or _flags["replot"]:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from plot_style import line_kwargs, stacked_auc_tpr  # noqa: E402
from utils_datasets import load_dataset, D3, D17  # noqa: E402
from fast_paths import ordered_curves_l1, ordered_curves_llr  # noqa: E402


# Random-pool sampling variance at n=65 is large (~0.07 in AUC); average AUC/TPR
# curves over K independent pool draws. Case pool (POOL_IDX=1) is
# disease-deterministic, so K=1 there. Matches Ordered_Noise.py.
NUM_POOLS_DEFAULT = 20

POOL_NAME = {0: "random", 1: "case"}


def make_figure(data: dict, output_path: str | None) -> None:
    num_miRNAs = list(np.asarray(data["num_miRNAs"]))
    fixed_FPR = bool(np.asarray(data["_fixed_FPR"]).item())
    pool_idx = int(np.asarray(data["_pool_idx"]).item())
    pool_name = POOL_NAME[pool_idx]

    auc_L1 = np.asarray(data["auc_L1"], dtype=float)
    auc_LLR = np.asarray(data["auc_LLR"], dtype=float)
    if fixed_FPR:
        tpr_at_fpr_L1 = np.asarray(data["tpr_at_fpr_L1"], dtype=float)
        tpr_at_fpr_LLR = np.asarray(data["tpr_at_fpr_LLR"], dtype=float)

    if fixed_FPR:
        fig, ax_auc, ax_tpr = stacked_auc_tpr()
    else:
        fig, ax_auc = plt.subplots()
        ax_tpr = None

    ax_auc.plot(num_miRNAs, auc_L1, label="L1",
                **line_kwargs("L1", pool_name, marker=None, linewidth=2.0))
    ax_auc.plot(num_miRNAs, auc_LLR, label="LLR",
                **line_kwargs("LLR", pool_name, marker=None, linewidth=2.0))
    if fixed_FPR:
        ax_tpr.plot(num_miRNAs, tpr_at_fpr_L1, label="L1",
                    **line_kwargs("L1", pool_name, marker=None, linewidth=2.0))
        ax_tpr.plot(num_miRNAs, tpr_at_fpr_LLR, label="LLR",
                    **line_kwargs("LLR", pool_name, marker=None, linewidth=2.0))

    ax_auc.invert_xaxis()
    ax_auc.legend(loc='upper right')
    ax_auc.set_ylabel("AUC")
    ax_auc.set_ylim([0.5, 1])
    ax_auc.grid(True)
    if fixed_FPR:
        ax_tpr.legend(loc='upper right')
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


# CLI: python Ordered.py <disease> <pool_idx> [random_sample_size] [output.pdf]
# Falls back to interactive defaults when no args given.
# In --replot mode, only the optional [output.pdf] positional is consulted.
DISEASES = {"D3": D3, "D17": D17}
if _flags["replot"]:
    OUTPUT_FILE = sys.argv[1] if len(sys.argv) >= 2 else None
    data, _meta = load_figdata(_flags["replot"])
    make_figure(data, OUTPUT_FILE)
    sys.exit(0)

if len(sys.argv) >= 3:
    DISEASE = DISEASES[sys.argv[1]]
    POOL_IDX = int(sys.argv[2])
    RANDOM_SAMPLE_SIZE = int(sys.argv[3]) if len(sys.argv) >= 4 and sys.argv[3] != "_" else None
    OUTPUT_FILE = sys.argv[4] if len(sys.argv) >= 5 else None
else:
    DISEASE = D3
    POOL_IDX = 0
    RANDOM_SAMPLE_SIZE = None
    OUTPUT_FILE = None


fixed_FPR = True
# Env overrides for cheap smoke tests and runtime tuning.
num_orders = int(os.environ.get("NUM_ORDERS", 5000))
_num_pools_default = int(os.environ.get("NUM_POOLS", NUM_POOLS_DEFAULT))
NUM_POOLS = _num_pools_default if POOL_IDX == 0 else 1   # case pool is deterministic
base_seed = _flags["seed"]


def _compute_one_pool(pool_seed: int):
    """Reseed, draw a fresh random pool (or use the deterministic case pool),
    and return (auc_L1, auc_LLR, tpr_L1, tpr_LLR) curves of shape (n_miRNA_counts,)
    plus the miRNA-count x-axis. L1 and LLR share identical feature orderings
    (re-seed between calls so the cumulative-prefix permutations match)."""
    np.random.seed(pool_seed)
    random.seed(pool_seed)

    populations, pools = load_dataset(
        miRNA=True, disease_case_sample=DISEASE,
        random_sample_size=RANDOM_SAMPLE_SIZE)
    pop_df = populations[POOL_IDX]
    pool_df = pools[POOL_IDX]

    sigma_j = np.std(pop_df, axis=0).to_numpy()
    n_feat = pop_df.shape[1]
    miRNA_counts = list(range(2, n_feat + 1))   # 2, 3, ..., n_feat

    multipliers = [0]   # no synthetic noise (Figure 3)
    pop_arr = pop_df.to_numpy(dtype=np.float64)
    pool_arr = pool_df.to_numpy(dtype=np.float64)

    # Re-seed before each metric so both consume the same permutation sequence.
    np.random.seed(pool_seed)
    random.seed(pool_seed)
    auc_l1_mat, tpr_l1_mat = ordered_curves_l1(
        pop_arr, pool_arr, multipliers, miRNA_counts, num_orders,
        sigma_j, np.random, random, target_fpr=1e-2)

    np.random.seed(pool_seed)
    random.seed(pool_seed)
    auc_llr_mat, tpr_llr_mat = ordered_curves_llr(
        pop_arr, pool_arr, multipliers, miRNA_counts, num_orders,
        sigma_j, np.random, random, target_fpr=1e-2)

    return (auc_l1_mat[0], auc_llr_mat[0],
            tpr_l1_mat[0], tpr_llr_mat[0], miRNA_counts)


pool_auc_l1 = []
pool_auc_llr = []
pool_tpr_l1 = []
pool_tpr_llr = []
pool_seeds = []
num_miRNAs = None

average_over_pools = (POOL_IDX == 0)   # case pool is disease-deterministic
for k in range(NUM_POOLS):
    pool_seed = (base_seed * 1000 + k) if average_over_pools else base_seed
    pool_seeds.append(pool_seed)
    print(f"\n=== pool {k + 1}/{NUM_POOLS}  (seed={pool_seed}) ===", flush=True)
    auc_l1, auc_llr, tpr_l1, tpr_llr, miRNA_counts = _compute_one_pool(pool_seed)
    pool_auc_l1.append(auc_l1)
    pool_auc_llr.append(auc_llr)
    pool_tpr_l1.append(tpr_l1)
    pool_tpr_llr.append(tpr_llr)
    if num_miRNAs is None:
        num_miRNAs = miRNA_counts

per_pool_auc_l1 = np.stack(pool_auc_l1, axis=0)   # (K, n_cnts)
per_pool_auc_llr = np.stack(pool_auc_llr, axis=0)
per_pool_tpr_l1 = np.stack(pool_tpr_l1, axis=0)
per_pool_tpr_llr = np.stack(pool_tpr_llr, axis=0)
mean_auc_l1 = per_pool_auc_l1.mean(axis=0)
mean_auc_llr = per_pool_auc_llr.mean(axis=0)
mean_tpr_l1 = per_pool_tpr_l1.mean(axis=0)
mean_tpr_llr = per_pool_tpr_llr.mean(axis=0)


data = {
    "num_miRNAs": np.asarray(num_miRNAs),
    "_fixed_FPR": fixed_FPR,
    "_pool_idx": POOL_IDX,
    "auc_L1": mean_auc_l1,
    "auc_LLR": mean_auc_llr,
    "auc_L1_per_pool": per_pool_auc_l1,
    "auc_LLR_per_pool": per_pool_auc_llr,
}
if fixed_FPR:
    data["tpr_at_fpr_L1"] = mean_tpr_l1
    data["tpr_at_fpr_LLR"] = mean_tpr_llr
    data["tpr_at_fpr_L1_per_pool"] = per_pool_tpr_l1
    data["tpr_at_fpr_LLR_per_pool"] = per_pool_tpr_llr

meta = {
    "seed": base_seed,
    "disease": sys.argv[1] if len(sys.argv) >= 2 else None,
    "pool_idx": POOL_IDX,
    "random_sample_size": RANDOM_SAMPLE_SIZE,
    "fixed_FPR": fixed_FPR,
    "num_orders": num_orders,
    "num_pools": NUM_POOLS,
    "pool_seeds": pool_seeds,
}

if OUTPUT_FILE:
    save_figdata(OUTPUT_FILE, data, meta)

make_figure(data, OUTPUT_FILE)
