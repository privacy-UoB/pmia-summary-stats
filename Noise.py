import sys
import numpy as np
import matplotlib

from experiment_io import parse_flags, seed_all, save_figdata, load_figdata

_flags = parse_flags(sys.argv)
seed_all(_flags["seed"])

if len(sys.argv) >= 6 or _flags["replot"]:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils_datasets import load_dataset, D3, D17
from utils import auc_scores, Gaussian_noise


def make_figure(data: dict, output_path: str | None) -> None:
    multiplier = np.asarray(data["multiplier"])
    fixed_FPR = bool(np.asarray(data["_fixed_FPR"]).item())
    include_longitudinals = bool(np.asarray(data["_include_longitudinals"]).item())
    error_bands = bool(np.asarray(data["_error_bands"]).item())
    L1_or_LLR = str(np.asarray(data["_L1_or_LLR"]).item())
    iterations = int(np.asarray(data["_iterations"]).item())

    fig, ax1 = plt.subplots()
    colours1 = ["cornflowerblue", "gold"]
    if fixed_FPR:
        ax2 = ax1.twinx()
        colours2 = ["mediumblue", "orange"]

    if not include_longitudinals:
        auc_L1 = np.asarray(data["auc_L1"])
        auc_LLR = np.asarray(data["auc_LLR"])
        ax1.plot(multiplier, auc_L1, colours1[0], linewidth=2.0, label="L1 AUC")
        ax1.plot(multiplier, auc_LLR, colours1[1], linewidth=2.0, label="LLR AUC")
        if fixed_FPR:
            tpr_at_fpr_L1 = np.asarray(data["tpr_at_fpr_L1"])
            tpr_at_fpr_LLR = np.asarray(data["tpr_at_fpr_LLR"])
            ax2.plot(multiplier, tpr_at_fpr_L1, colours2[0], linewidth=2.0, label="L1 tpr")
            ax2.plot(multiplier, tpr_at_fpr_LLR, colours2[1], linewidth=2.0, label="LLR tpr")
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

            ax1.plot(multiplier, mm_L1[0], colours1[0], linewidth=2.0, label="L1 AUC")
            ax1.fill_between(multiplier, mm_L1[1], mm_L1[2], alpha=0.2)
            ax1.plot(multiplier, mm_LLR[0], colours1[1], linewidth=2.0, label="LLR AUC")
            ax1.fill_between(multiplier, mm_LLR[1], mm_LLR[2], alpha=0.2)

            if fixed_FPR:
                tt_L1 = [list(s) for s in zip(*noisy_longitudinals_tpr_L1)]
                tt_LLR = [list(s) for s in zip(*noisy_longitudinals_tpr_LLR)]
                mm_tpr_L1 = [[np.average(i) for i in tt_L1],
                             [np.min(j) for j in tt_L1],
                             [np.max(k) for k in tt_L1]]
                mm_tpr_LLR = [[np.average(i) for i in tt_LLR],
                              [np.min(j) for j in tt_LLR],
                              [np.max(k) for k in tt_LLR]]
                ax2.plot(multiplier, mm_tpr_L1[0], colours2[0], linewidth=2.0, label="L1 tpr")
                ax2.fill_between(multiplier, mm_tpr_L1[1], mm_tpr_L1[2], alpha=0.2)
                ax2.plot(multiplier, mm_tpr_LLR[0], colours2[1], linewidth=2.0, label="LLR tpr")
                ax2.fill_between(multiplier, mm_tpr_LLR[1], mm_tpr_LLR[2], alpha=0.2)
        else:
            for l in range(iterations):
                if L1_or_LLR == "L1":
                    ax1.plot(multiplier, noisy_longitudinals_L1[l], colours1[0], linewidth=2.0, label=f"L1 AUC timestamp {l}")
                    if fixed_FPR:
                        ax2.plot(multiplier, noisy_longitudinals_tpr_L1[l], colours2[0], linewidth=2.0, label=f"L1 tpr timestamp {l}")
                elif L1_or_LLR == "LLR":
                    ax1.plot(multiplier, noisy_longitudinals_LLR[l], colours1[1], linewidth=2.0, label=f"LLR AUC timestamp {l}")
                    if fixed_FPR:
                        ax2.plot(multiplier, noisy_longitudinals_tpr_LLR[l], colours2[1], linewidth=2.0, label=f"LLR tpr timestamp {l}")

    ax1.legend(loc='upper right')
    if fixed_FPR:
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc='upper right')
        ax2.set_ylabel("TPR at 0.01 FPR")

    ax1.set_xscale("log")
    ax1.set_xlabel("noise scale")
    ax1.set_ylabel("AUC scores")
    ax1.set_ylim([0.45, 1])
    ax1.grid(True)

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
    OUTPUT_FILE = sys.argv[7] if len(sys.argv) >= 8 else None
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
    make_figure(data, OUTPUT_FILE)
    sys.exit(0)

fixed_FPR = True
include_longitudinals = True if dataset != "miRNA" else False
iterations = 1 # to exclude repeating without additional longitudinal entries
num_orders = 2000 # number of iterations to average over
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

# load partitioned dataset
if dataset == "miRNA":
    include_longitudinals = False # failsafe
    populations, pools = load_dataset(miRNA=True, disease_case_sample=DISEASE, random_sample_size=RANDOM_SAMPLE_SIZE)
    # 0 = random, 1 = case:
    pop = populations[POP_IDX]
    pool = pools[POOL_IDX]
    multiplier = ranges[4]
    if include_deviations == True:
        multiplier = deviation_range

elif dataset == "Timestamp":
    population, chosen_pool = load_dataset(timestamp=True)
    multiplier = ranges[2]

elif dataset == "FitBit":
    population, chosen_pool = load_dataset(FitBit=True)
    multiplier = ranges[6]
    if include_deviations == True:
        multiplier = deviation_range

elif dataset == "Electricity":
    population, chosen_pool = load_dataset(electricity=True)
    multiplier = ranges[2]

if include_longitudinals:
    noisy_longitudinals_L1 = []
    noisy_longitudinals_LLR = []
    if fixed_FPR == True:
        noisy_longitudinals_tpr_L1 = []
        noisy_longitudinals_tpr_LLR = []

    error_bands = True # if we wish to include min/max AUC scores over all iterations
    if error_bands == False:
        L1_or_LLR = "LLR" # graph is too messy with all longitudinals over both L1 and LLR

    if dataset == "FitBit":
        iterations = 8
    else:
        iterations = len(population)
else:
    error_bands = False

L1_or_LLR = locals().get("L1_or_LLR", "LLR")

for i in range(iterations): # multiple if include_longitudinals, 1 otherwise
    auc_L1 = []
    auc_LLR = []
    if fixed_FPR == True:
        tpr_at_fpr_L1 = []
        tpr_at_fpr_LLR = []

    # configuring the reference pop & pool to match the dataframe of a particular data entry
    if include_longitudinals:
        pop = population[i]
        pool = chosen_pool[i]
    elif dataset != "miRNA":
        pop = population[0]
        pool = chosen_pool[0]

    if include_deviations == True:
        sigma_j = np.std(pop, axis=0) # this is doing it over all the features (e.g. miRNAs)

    for count, m in enumerate(multiplier):
        aucs_L1 = []
        aucs_LLR = []
        if fixed_FPR == True:
            tpr_at_fprs_L1 = []
            tpr_at_fprs_LLR = []
        print("iteration", count)

        # for loop for numorder lots of train/test, then average at end
        for j in range (num_orders):

            # the 'noise' increases throughout each of the later timepoints the data is collected from
            # changed from m * sigmaj so it's not tailored variance to each feature
            deviation = m if include_deviations == False else m * sigma_j

            if dataset == "miRNA":
                noisy_pop, noisy_pool = Gaussian_noise(pop, pool, 0, deviation, clip=True) # make noise non-negative
            else:
                noisy_pop, noisy_pool = Gaussian_noise(pop, pool, 0, deviation)

            # get performance/accuracy for L1 & LLR statistics over the noisy stat inputs compared to the 'original' pop & pool
            roc_L1 = auc_scores(noisy_pop, noisy_pool, pop, pool, p_values=False)
            roc_LLR = auc_scores(noisy_pop, noisy_pool, pop, pool, LR=True, p_values=False)

            aucs_L1.append(roc_L1)
            aucs_LLR.append(roc_LLR)

            if fixed_FPR == True:
                fpr_L1, tpr_L1, thresholds_L1 = auc_scores(noisy_pop, noisy_pool, pop, pool, FPR=True)
                fpr_LLR, tpr_LLR, thresholds_LLR = auc_scores(noisy_pop, noisy_pool, pop, pool, LR=True, FPR=True)

                # TPR at a fixed FPR (e.g., 0.01 = 1%)
                target_fpr = 1e-2
                tpr_at_fprs_L1.append(np.interp(target_fpr, fpr_L1, tpr_L1))
                tpr_at_fprs_LLR.append(np.interp(target_fpr, fpr_LLR, tpr_LLR))

        # num_order rows of datasets, columns are each longitudinal entry
        if len(aucs_L1) >0:
            auc_L1.append(np.average(aucs_L1))

        if len(aucs_LLR) >0:
            auc_LLR.append(np.average(aucs_LLR))

        if fixed_FPR == True:
            if len(tpr_at_fprs_L1) >0:
                tpr_at_fpr_L1.append(np.average(tpr_at_fprs_L1))

            if len(tpr_at_fprs_LLR) >0:
                tpr_at_fpr_LLR.append(np.average(tpr_at_fprs_LLR))

    if include_longitudinals:
        noisy_longitudinals_L1.append(auc_L1)
        noisy_longitudinals_LLR.append(auc_LLR)
        if fixed_FPR == True:
            noisy_longitudinals_tpr_L1.append(tpr_at_fpr_L1)
            noisy_longitudinals_tpr_LLR.append(tpr_at_fpr_LLR)

# Build the figure-data dict and save before plotting.
data = {
    "multiplier": np.asarray(multiplier),
    "_fixed_FPR": fixed_FPR,
    "_include_longitudinals": include_longitudinals,
    "_error_bands": error_bands,
    "_L1_or_LLR": L1_or_LLR,
    "_iterations": iterations,
}
if not include_longitudinals:
    data["auc_L1"] = np.asarray(auc_L1)
    data["auc_LLR"] = np.asarray(auc_LLR)
    if fixed_FPR:
        data["tpr_at_fpr_L1"] = np.asarray(tpr_at_fpr_L1)
        data["tpr_at_fpr_LLR"] = np.asarray(tpr_at_fpr_LLR)
else:
    data["noisy_longitudinals_L1"] = np.asarray(noisy_longitudinals_L1, dtype=object)
    data["noisy_longitudinals_LLR"] = np.asarray(noisy_longitudinals_LLR, dtype=object)
    if fixed_FPR:
        data["noisy_longitudinals_tpr_L1"] = np.asarray(noisy_longitudinals_tpr_L1, dtype=object)
        data["noisy_longitudinals_tpr_LLR"] = np.asarray(noisy_longitudinals_tpr_LLR, dtype=object)

meta = {
    "seed": _flags["seed"],
    "dataset": dataset,
    "disease": sys.argv[3] if len(sys.argv) >= 4 else None,
    "pop_idx": POP_IDX,
    "pool_idx": POOL_IDX,
    "random_sample_size": RANDOM_SAMPLE_SIZE,
    "include_deviations": include_deviations,
    "fixed_FPR": fixed_FPR,
    "include_longitudinals": include_longitudinals,
    "num_orders": num_orders,
    "iterations": iterations,
}

if OUTPUT_FILE:
    save_figdata(OUTPUT_FILE, data, meta)

make_figure(data, OUTPUT_FILE)
