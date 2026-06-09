import os
import random
import sys

import numpy as np
import matplotlib

from experiment_io import parse_flags, seed_all, save_figdata, load_figdata, resolve_output_path

_flags = parse_flags(sys.argv)
seed_all(_flags["seed"])

if len(sys.argv) >= 6 or _flags["replot"]:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from plot_style import line_kwargs, stacked_auc_tpr, METRIC_COLOR  # noqa: E402
from utils_datasets import load_dataset, D3, D17  # noqa: E402
from fast_paths import noise_curves  # noqa: E402


# Random-pool sampling variance at n=65 is large, so the miRNA random-pool
# panel (Fig 2b / 9b) averages over K independent pool draws. All other
# configurations (case pool, longitudinal datasets) keep K=1.
NUM_POOLS_DEFAULT = 20


def make_figure(data: dict, output_path: str | None) -> None:
    multiplier = np.asarray(data["multiplier"])
    fixed_FPR = bool(np.asarray(data["_fixed_FPR"]).item())
    include_longitudinals = bool(np.asarray(data["_include_longitudinals"]).item())
    error_bands = bool(np.asarray(data["_error_bands"]).item())
    L1_or_LLR = str(np.asarray(data["_L1_or_LLR"]).item())
    iterations = int(np.asarray(data["_iterations"]).item())
    pool_idx = int(np.asarray(data["_pool_idx"]).item())
    dataset_name = str(np.asarray(data["_dataset"]).item())
    # Case/random distinction only applies to the miRNA pool comparison;
    # longitudinal datasets have no case-vs-random axis so we stay at
    # pool=None (solid).
    pool_name = (("random" if pool_idx == 0 else "case")
                 if dataset_name == "miRNA" else None)

    if fixed_FPR:
        fig, ax_auc, ax_tpr = stacked_auc_tpr()
    else:
        fig, ax_auc = plt.subplots()
        ax_tpr = None

    if not include_longitudinals:
        auc_L1 = np.asarray(data["auc_L1"])
        auc_LLR = np.asarray(data["auc_LLR"])
        ax_auc.plot(multiplier, auc_L1, label="L1",
                    **line_kwargs("L1", pool_name, marker=None, linewidth=2.0))
        ax_auc.plot(multiplier, auc_LLR, label="LLR",
                    **line_kwargs("LLR", pool_name, marker=None, linewidth=2.0))
        if fixed_FPR:
            tpr_at_fpr_L1 = np.asarray(data["tpr_at_fpr_L1"])
            tpr_at_fpr_LLR = np.asarray(data["tpr_at_fpr_LLR"])
            ax_tpr.plot(multiplier, tpr_at_fpr_L1, label="L1",
                        **line_kwargs("L1", pool_name, marker=None, linewidth=2.0))
            ax_tpr.plot(multiplier, tpr_at_fpr_LLR, label="LLR",
                        **line_kwargs("LLR", pool_name, marker=None, linewidth=2.0))
    else:
        noisy_longitudinals_L1 = [list(row) for row in np.asarray(data["noisy_longitudinals_L1"], dtype=object)]
        noisy_longitudinals_LLR = [list(row) for row in np.asarray(data["noisy_longitudinals_LLR"], dtype=object)]
        if fixed_FPR:
            noisy_longitudinals_tpr_L1 = [list(row) for row in np.asarray(data["noisy_longitudinals_tpr_L1"], dtype=object)]
            noisy_longitudinals_tpr_LLR = [list(row) for row in np.asarray(data["noisy_longitudinals_tpr_LLR"], dtype=object)]

        if error_bands:
            transposed_L1 = [list(s) for s in zip(*noisy_longitudinals_L1)]
            transposed_LLR = [list(s) for s in zip(*noisy_longitudinals_LLR)]
            mm_L1 = [[np.average(i) for i in transposed_L1],
                     [np.min(j) for j in transposed_L1],
                     [np.max(k) for k in transposed_L1]]
            mm_LLR = [[np.average(i) for i in transposed_LLR],
                      [np.min(j) for j in transposed_LLR],
                      [np.max(k) for k in transposed_LLR]]

            ax_auc.plot(multiplier, mm_L1[0], label="L1",
                        **line_kwargs("L1", pool_name, marker=None, linewidth=2.0))
            ax_auc.fill_between(multiplier, mm_L1[1], mm_L1[2],
                                alpha=0.2, color=METRIC_COLOR["L1"])
            ax_auc.plot(multiplier, mm_LLR[0], label="LLR",
                        **line_kwargs("LLR", pool_name, marker=None, linewidth=2.0))
            ax_auc.fill_between(multiplier, mm_LLR[1], mm_LLR[2],
                                alpha=0.2, color=METRIC_COLOR["LLR"])

            if fixed_FPR:
                tt_L1 = [list(s) for s in zip(*noisy_longitudinals_tpr_L1)]
                tt_LLR = [list(s) for s in zip(*noisy_longitudinals_tpr_LLR)]
                mm_tpr_L1 = [[np.average(i) for i in tt_L1],
                             [np.min(j) for j in tt_L1],
                             [np.max(k) for k in tt_L1]]
                mm_tpr_LLR = [[np.average(i) for i in tt_LLR],
                              [np.min(j) for j in tt_LLR],
                              [np.max(k) for k in tt_LLR]]
                ax_tpr.plot(multiplier, mm_tpr_L1[0], label="L1",
                            **line_kwargs("L1", pool_name, marker=None, linewidth=2.0))
                ax_tpr.fill_between(multiplier, mm_tpr_L1[1], mm_tpr_L1[2],
                                    alpha=0.2, color=METRIC_COLOR["L1"])
                ax_tpr.plot(multiplier, mm_tpr_LLR[0], label="LLR",
                            **line_kwargs("LLR", pool_name, marker=None, linewidth=2.0))
                ax_tpr.fill_between(multiplier, mm_tpr_LLR[1], mm_tpr_LLR[2],
                                    alpha=0.2, color=METRIC_COLOR["LLR"])
        else:
            for l in range(iterations):
                if L1_or_LLR == "L1":
                    ax_auc.plot(multiplier, noisy_longitudinals_L1[l],
                                label=f"timestamp {l}",
                                **line_kwargs("L1", pool_name, marker=None, linewidth=2.0))
                    if fixed_FPR:
                        ax_tpr.plot(multiplier, noisy_longitudinals_tpr_L1[l],
                                    label=f"timestamp {l}",
                                    **line_kwargs("L1", pool_name, marker=None, linewidth=2.0))
                elif L1_or_LLR == "LLR":
                    ax_auc.plot(multiplier, noisy_longitudinals_LLR[l],
                                label=f"timestamp {l}",
                                **line_kwargs("LLR", pool_name, marker=None, linewidth=2.0))
                    if fixed_FPR:
                        ax_tpr.plot(multiplier, noisy_longitudinals_tpr_LLR[l],
                                    label=f"timestamp {l}",
                                    **line_kwargs("LLR", pool_name, marker=None, linewidth=2.0))

    ax_auc.legend(loc='upper right')
    ax_auc.set_xscale("log")
    ax_auc.set_ylabel("AUC")
    ax_auc.set_ylim([0.5, 1])
    ax_auc.grid(True)
    if fixed_FPR:
        ax_tpr.legend(loc='upper right')
        ax_tpr.set_xscale("log")
        ax_tpr.set_xlabel("noise scale")
        ax_tpr.set_ylabel("TPR @ 1% FPR")
        ax_tpr.set_ylim([0, 1])
        ax_tpr.grid(True)
    else:
        ax_auc.set_xlabel("noise scale")

    if output_path:
        plt.savefig(output_path)
        print(f"Saved to {output_path}")
    else:
        plt.show()


# CLI: python Noise.py <dataset> <include_deviations> <disease> <pop_idx> <pool_idx> [random_sample_size] [output.pdf]
# Falls back to interactive defaults when no args given.
DISEASES = {"D3": D3, "D17": D17}
if len(sys.argv) >= 6:
    dataset = sys.argv[1]             # miRNA, Timestamp, FitBit, Electricity
    include_deviations = sys.argv[2].lower() == "true"
    DISEASE = DISEASES.get(sys.argv[3], D17)
    POP_IDX = int(sys.argv[4])
    POOL_IDX = int(sys.argv[5])
    RANDOM_SAMPLE_SIZE = int(sys.argv[6]) if len(sys.argv) >= 7 and sys.argv[6] != "_" else None
    OUTPUT_FILE = resolve_output_path(sys.argv[7] if len(sys.argv) >= 8 else None)
else:
    dataset = "miRNA"
    include_deviations = True
    DISEASE = D17
    POP_IDX = 1
    POOL_IDX = 1
    RANDOM_SAMPLE_SIZE = None
    OUTPUT_FILE = None

# Replot mode: skip computation, redraw from a saved dump.
if _flags["replot"]:
    data, _meta = load_figdata(_flags["replot"])
    # Back-compat: older .npz files don't carry _pool_idx / _dataset in `data`;
    # recover from the sidecar meta so the linestyle convention is honored.
    if "_pool_idx" not in data:
        data["_pool_idx"] = _meta.get("pool_idx", 0)
    if "_dataset" not in data:
        data["_dataset"] = _meta.get("dataset", "miRNA")
    make_figure(data, OUTPUT_FILE)
    sys.exit(0)

fixed_FPR = True
include_longitudinals = True if dataset != "miRNA" else False
# Env overrides for cheap smoke tests and runtime tuning.
num_orders = int(os.environ.get("NUM_ORDERS", 2000))
_num_pools_default = int(os.environ.get("NUM_POOLS", NUM_POOLS_DEFAULT))
# Only the miRNA + random-pool config suffers single-pool sampling variance.
average_over_pools = (dataset == "miRNA" and POOL_IDX == 0)
NUM_POOLS = _num_pools_default if average_over_pools else 1
base_seed = _flags["seed"]

# noise ranges:
ranges = [[0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1], # 0
              np.arange(0, 8, 0.04), # 1, sufficient for m * sigma_j
              np.arange(0, 20, 0.1), # 2
              np.arange(0, 10000, 10), # 3
              np.logspace(0, 15, base=2), # 4, used for standard noise values (raw)
              np.geomspace(0.1, 10000, 100), # 5
              np.concatenate(([0], np.geomspace(0.1, 10000, 100)))] # 6
# when include_deviations=True, m is in units of sigma (per-feature std dev),
# so the same range works across all datasets:
deviation_range = np.logspace(-2, 2, 50) # 0.01*sigma to 100*sigma

# Choose multiplier axis from dataset (deferred — actual pop/pool loads happen
# inside the pool loop below for the miRNA path).
if dataset == "miRNA":
    include_longitudinals = False  # failsafe
    multiplier = deviation_range if include_deviations else ranges[4]
elif dataset == "Timestamp":
    multiplier = ranges[2]
elif dataset == "FitBit":
    multiplier = deviation_range if include_deviations else ranges[6]
elif dataset == "Electricity":
    multiplier = ranges[2]

L1_or_LLR = locals().get("L1_or_LLR", "LLR")
error_bands = include_longitudinals  # min/max bands only meaningful when iterating timepoints


def _load_for_pool(pool_seed: int):
    """Reseed + load dataset for one pool draw (miRNA random-pool path only)."""
    np.random.seed(pool_seed)
    random.seed(pool_seed)
    populations, pools = load_dataset(
        miRNA=True, disease_case_sample=DISEASE, random_sample_size=RANDOM_SAMPLE_SIZE)
    return populations[POP_IDX], pools[POOL_IDX]


if dataset == "miRNA":
    # miRNA path: K-pool averaging (K=1 for case pool, K=NUM_POOLS for random pool).
    pool_auc_L1, pool_auc_LLR, pool_tpr_L1, pool_tpr_LLR = [], [], [], []
    pool_seeds = []
    for k in range(NUM_POOLS):
        pool_seed = base_seed * 1000 + k if average_over_pools else base_seed
        pool_seeds.append(pool_seed)
        print(f"\n=== pool {k + 1}/{NUM_POOLS}  (seed={pool_seed}) ===", flush=True)

        pop, pool = _load_for_pool(pool_seed)
        sigma_j = np.std(pop, axis=0).to_numpy() if include_deviations else None

        out = noise_curves(
            pop.to_numpy(np.float64), pool.to_numpy(np.float64),
            list(multiplier), num_orders,
            include_deviations=include_deviations,
            sigma_j=sigma_j, rng_np=np.random, clip=True,
            target_fpr=1e-2)
        pool_auc_L1.append(out["auc_L1"])
        pool_auc_LLR.append(out["auc_LLR"])
        pool_tpr_L1.append(out["tpr_L1"])
        pool_tpr_LLR.append(out["tpr_LLR"])

    per_pool_auc_L1 = np.stack(pool_auc_L1, axis=0)
    per_pool_auc_LLR = np.stack(pool_auc_LLR, axis=0)
    per_pool_tpr_L1 = np.stack(pool_tpr_L1, axis=0)
    per_pool_tpr_LLR = np.stack(pool_tpr_LLR, axis=0)
    auc_L1 = per_pool_auc_L1.mean(axis=0)
    auc_LLR = per_pool_auc_LLR.mean(axis=0)
    tpr_at_fpr_L1 = per_pool_tpr_L1.mean(axis=0)
    tpr_at_fpr_LLR = per_pool_tpr_LLR.mean(axis=0)
    iterations = 1
else:
    # Longitudinal datasets: each "iteration" is a different timepoint snapshot.
    if dataset == "Timestamp":
        population, chosen_pool = load_dataset(timestamp=True)
    elif dataset == "FitBit":
        population, chosen_pool = load_dataset(FitBit=True)
    elif dataset == "Electricity":
        population, chosen_pool = load_dataset(electricity=True)
    iterations = 8 if dataset == "FitBit" else len(population)
    pool_seeds = [base_seed]

    noisy_longitudinals_L1 = []
    noisy_longitudinals_LLR = []
    noisy_longitudinals_tpr_L1 = []
    noisy_longitudinals_tpr_LLR = []

    for i in range(iterations):
        pop = population[i]
        pool = chosen_pool[i]
        sigma_j = (np.std(pop, axis=0)
                   if include_deviations else None)
        if hasattr(sigma_j, "to_numpy"):
            sigma_j = sigma_j.to_numpy()

        print(f"\n=== timepoint {i + 1}/{iterations} ===", flush=True)
        out = noise_curves(
            np.asarray(pop, dtype=np.float64), np.asarray(pool, dtype=np.float64),
            list(multiplier), num_orders,
            include_deviations=include_deviations,
            sigma_j=sigma_j, rng_np=np.random, clip=False,
            target_fpr=1e-2)
        noisy_longitudinals_L1.append(out["auc_L1"])
        noisy_longitudinals_LLR.append(out["auc_LLR"])
        noisy_longitudinals_tpr_L1.append(out["tpr_L1"])
        noisy_longitudinals_tpr_LLR.append(out["tpr_LLR"])


# Build the figure-data dict and save before plotting.
data = {
    "multiplier": np.asarray(multiplier),
    "_fixed_FPR": fixed_FPR,
    "_include_longitudinals": include_longitudinals,
    "_error_bands": error_bands,
    "_L1_or_LLR": L1_or_LLR,
    "_iterations": iterations,
    "_pool_idx": POOL_IDX,
    "_dataset": dataset,
}
if not include_longitudinals:
    data["auc_L1"] = np.asarray(auc_L1)
    data["auc_LLR"] = np.asarray(auc_LLR)
    data["auc_L1_per_pool"] = per_pool_auc_L1
    data["auc_LLR_per_pool"] = per_pool_auc_LLR
    if fixed_FPR:
        data["tpr_at_fpr_L1"] = np.asarray(tpr_at_fpr_L1)
        data["tpr_at_fpr_LLR"] = np.asarray(tpr_at_fpr_LLR)
        data["tpr_at_fpr_L1_per_pool"] = per_pool_tpr_L1
        data["tpr_at_fpr_LLR_per_pool"] = per_pool_tpr_LLR
else:
    data["noisy_longitudinals_L1"] = np.asarray(noisy_longitudinals_L1, dtype=object)
    data["noisy_longitudinals_LLR"] = np.asarray(noisy_longitudinals_LLR, dtype=object)
    if fixed_FPR:
        data["noisy_longitudinals_tpr_L1"] = np.asarray(noisy_longitudinals_tpr_L1, dtype=object)
        data["noisy_longitudinals_tpr_LLR"] = np.asarray(noisy_longitudinals_tpr_LLR, dtype=object)

meta = {
    "seed": base_seed,
    "dataset": dataset,
    "disease": sys.argv[3] if len(sys.argv) >= 4 else None,
    "pop_idx": POP_IDX,
    "pool_idx": POOL_IDX,
    "random_sample_size": RANDOM_SAMPLE_SIZE,
    "include_deviations": include_deviations,
    "fixed_FPR": fixed_FPR,
    "include_longitudinals": include_longitudinals,
    "num_orders": num_orders,
    "num_pools": NUM_POOLS,
    "pool_seeds": pool_seeds,
    "iterations": iterations,
}

if OUTPUT_FILE:
    save_figdata(OUTPUT_FILE, data, meta)

make_figure(data, OUTPUT_FILE)
