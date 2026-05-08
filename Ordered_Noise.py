import sys
import numpy as np
import matplotlib
if len(sys.argv) >= 4:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random
from utils_datasets import load_dataset, separate_diseased_miRNAs, D3, D17
from utils import auc_scores, Gaussian_noise

# CLI: python Ordered_Noise.py <disease> <metric> <pool_idx> [random_sample_size] [output.pdf]
# Falls back to interactive defaults when no args given.
DISEASES = {"D3": D3, "D17": D17}
if len(sys.argv) >= 4:
    DISEASE = DISEASES[sys.argv[1]]
    L1_or_LLR = sys.argv[2]
    POOL_IDX = int(sys.argv[3])
    RANDOM_SAMPLE_SIZE = int(sys.argv[4]) if len(sys.argv) >= 5 and sys.argv[4] != "_" else None
    OUTPUT_FILE = sys.argv[5] if len(sys.argv) >= 6 else None
else:
    DISEASE = D3
    L1_or_LLR = "L1"
    POOL_IDX = 1
    RANDOM_SAMPLE_SIZE = None
    OUTPUT_FILE = None

# paper: the demonstrated graphs showing roc curves
    # 1st: 50 subsets of n/1049 different individuals (n = 35, 65, 124)
    # 2nd: 6 case groups D19, D17, D10, D7, D3, D1

stratifying = False # Not enough pool miRNAs in True case
fixed_FPR = True

if stratifying == False:
    # load dataset
    population, pool = load_dataset(miRNA=True, disease_case_sample=DISEASE, random_sample_size=RANDOM_SAMPLE_SIZE)

    # 0 = random, 1 = case
    pop = population[POOL_IDX]
    pool = pool[POOL_IDX]

else:
    # diseased case sample pop/pool only
    only_diseased_miRNAs_pop, without_diseased_miRNAs_pop, only_diseased_miRNAs_pool, without_diseased_miRNAs_pool = separate_diseased_miRNAs(DISEASE, "miRNA")
    pop = without_diseased_miRNAs_pop
    pool = without_diseased_miRNAs_pool

sigma_j = np.std(pop, axis=0) # this is doing it over all the columns (miRNAs)
sigma_j_pool = np.std(pool, axis=0)
ranges = [[0, 0.25, 0.5, 0.75, 1], # 0, fractions of standard deviation applied to the dataset
              [0, 100, 200, 300, 400], # 1, static values produce similar noise to one observed case
              np.concatenate(([0], np.logspace(1, 4, num=4)))] # 2
multiplier = ranges[2]

miRNAs = list(pop.keys()) # get the list of miRNAs ["miRNA_1234", "miRNA_1235", ...]
num_orders = 2000 # number of different samples of MiRNAs

# Pre-generate shuffled orderings (minimal memory)
shuffled_lists = []
for j in range(num_orders):
    current_miRNA_list = list(miRNAs)
    random.shuffle(current_miRNA_list)
    shuffled_lists.append(current_miRNA_list)

noise_fraction_L1 = []
noise_fraction_LLR = []
if fixed_FPR == True:
    noise_fraction_tpr_at_fpr_L1 = []
    noise_fraction_tpr_at_fpr_LLR = []

# Process one multiplier at a time to reduce memory (~8 GB instead of ~37 GB)
for count, m in enumerate(multiplier):
    # Generate noise for all orders at this multiplier only
    nonneg_noised_pops = []
    nonneg_noised_pools = []
    for j in range(num_orders):
        nonneg_noisy_pop, nonneg_noisy_pool = Gaussian_noise(pop, pool, 0, m, clip=True)
        nonneg_noised_pops.append(nonneg_noisy_pop)
        nonneg_noised_pools.append(nonneg_noisy_pool)

    auc_L1 = []
    auc_LLR = []
    if fixed_FPR == True:
        tpr_at_fpr_L1 = []
        tpr_at_fpr_LLR = []
    num_miRNAs = []

    for i in range(2, len(miRNAs), 2): # MiRNAs range from 1 to 466 in paper
        aucs_L1 = []
        aucs_LLR = []
        if fixed_FPR == True:
            tpr_at_fprs_L1 = []
            tpr_at_fprs_LLR = []
        num_miRNAs.append(i)

        print("miRNA", i, "on iteration", count)

        for j in range (num_orders):
            current_shuffled_list = shuffled_lists[j]
            selected_miRNAs = current_shuffled_list[:i]

            local_noised_pop = nonneg_noised_pops[j][selected_miRNAs]
            local_noised_pool = nonneg_noised_pools[j][selected_miRNAs]

            local_pop = pop[selected_miRNAs]
            local_pool = pool[selected_miRNAs]

            # local_pop & local_pool instead of pop & pool due to attacker's lacking access to full information
            if L1_or_LLR == "L1":
                roc_L1 = auc_scores(local_noised_pop, local_noised_pool, local_pop, local_pool, p_values=False)
                aucs_L1.append(roc_L1)

                if fixed_FPR == True:
                    fpr_L1, tpr_L1, thresholds_L1 = auc_scores(local_noised_pop, local_noised_pool, local_pop, local_pool, FPR=True)

                    # TPR at a fixed FPR (e.g., 0.01 = 1%)
                    target_fpr = 1e-2
                    tpr_at_fprs_L1.append(np.interp(target_fpr, fpr_L1, tpr_L1))
            elif L1_or_LLR == "LLR":
                roc_LLR = auc_scores(local_noised_pop, local_noised_pool, local_pop, local_pool, LR=True, p_values=False)
                aucs_LLR.append(roc_LLR)

                if fixed_FPR == True:
                    fpr_LLR, tpr_LLR, thresholds_LLR = auc_scores(local_noised_pop, local_noised_pool, local_pop, local_pool, LR=True, FPR=True)

                    # TPR at a fixed FPR (e.g., 0.01 = 1%)
                    target_fpr = 1e-2
                    tpr_at_fprs_LLR.append(np.interp(target_fpr, fpr_LLR, tpr_LLR))

        if L1_or_LLR == "L1":
            if len(aucs_L1) >0:
                auc_L1.append(np.average(aucs_L1))

            if fixed_FPR == True:
                if len(tpr_at_fprs_L1) >0:
                    tpr_at_fpr_L1.append(np.average(tpr_at_fprs_L1))

        elif L1_or_LLR == "LLR":
            if len(aucs_LLR) >0:
                auc_LLR.append(np.average(aucs_LLR))

            if fixed_FPR == True:
                if len(tpr_at_fprs_LLR) >0:
                    tpr_at_fpr_LLR.append(np.average(tpr_at_fprs_LLR))

    noise_fraction_L1.append(auc_L1) if L1_or_LLR == "L1" else noise_fraction_LLR.append(auc_LLR)
    if fixed_FPR == True:
        noise_fraction_tpr_at_fpr_L1.append(tpr_at_fpr_L1) if L1_or_LLR == "L1" else noise_fraction_tpr_at_fpr_LLR.append(tpr_at_fpr_LLR)

    # Free memory before next multiplier
    del nonneg_noised_pops, nonneg_noised_pools

# plots!
fig, ax1 = plt.subplots()
colours1 = ["cornflowerblue", "gold", "springgreen", "red", "mediumpurple"]
if fixed_FPR == True:
    ax2 = ax1.twinx()
    colours2 = ["mediumblue", "orange", "green", "brown", "purple"]

for index, noise in enumerate(multiplier):
    if L1_or_LLR == "L1":
        # Left-hand x axis for AUC scores
        ax1.plot(num_miRNAs, noise_fraction_L1[index], colours1[index], linewidth=2.0, label=f"AUC Std. dev. = {noise}")

        # Right hand x axis for TPR at fixed FPR
        if fixed_FPR == True:
            ax2.plot(num_miRNAs, noise_fraction_tpr_at_fpr_L1[index], colours2[index], linewidth=2.0, label=f"AUC Std. dev. = {noise}")

    elif L1_or_LLR == "LLR":
        # Left-hand x axis for AUC scores
        ax1.plot(num_miRNAs, noise_fraction_LLR[index], colours1[index], linewidth=2.0, label=f"fpr Std. dev. = {noise}")

        # Right hand x axis for TPR at fixed FPR
        if fixed_FPR == True:
            ax2.plot(num_miRNAs, noise_fraction_tpr_at_fpr_LLR[index], colours2[index], linewidth=2.0, label=f"fpr Std. dev. = {noise}")

ax1.legend(loc='upper right')
if fixed_FPR == True:
    # Merge handles and labels
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    # Add combined legend to one axis
    ax1.legend(h1 + h2, l1 + l2, loc='upper right')
    ax2.set_ylabel("TPR at 0.01 FPR")

ax1.invert_xaxis()
ax1.set_xlabel("number miRNAs")
ax1.set_ylabel("AUC scores")
ax1.set_ylim([0.3,1]) # enables comparable auc scores between L1 and LLR
ax1.grid(True)

if OUTPUT_FILE:
    plt.savefig(OUTPUT_FILE)
    print(f"Saved to {OUTPUT_FILE}")
else:
    plt.show()
