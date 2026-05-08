import numpy as np
import matplotlib.pyplot as plt
from utils_datasets import load_dataset, D17
from utils import auc_scores, Gaussian_noise

include_longitudinals = True # if we wish to create one graph with all longitudinal entries, excludes "miRNA"
include_deviations = True # if we want the x axis dependant on standard deviation of features
fixed_FPR = True
dataset = "miRNA" # choices are "miRNA", "Timestamp", "FitBit", "Electricity"
iterations = 1 # to exclude repeating without additional longitudinal entries
num_orders = 2000 # number of iterations to average over
# fractions of standard deviation applied to the dataset:
ranges = [[0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1], # 0
              np.arange(0, 8, 0.04), # 1, sufficient for m * sigma_j
              np.arange(0, 20, 0.1), # 2
              np.arange(0, 10000, 10), # 3
              np.logspace(0, 15, base=2), # 4, used for standard noise values
              np.geomspace(0.1, 10000, 100), # 5
              np.concatenate(([0], np.geomspace(0.1, 10000, 100)))] # 6

# load partitioned dataset
if dataset == "miRNA":
    include_longitudinals = False # failsafe
    populations, pools = load_dataset(miRNA=True, disease_case_sample=D17)
    # 0 = random, 1 = case:
    pop = populations[0] # make pop configurable
    pool = pools[0] # make pool configurable
    multiplier = ranges[4]
    if include_deviations == True:
        multiplier = ranges[4]/100

elif dataset == "Timestamp":
    population, chosen_pool = load_dataset(timestamp=True)
    multiplier = ranges[2]

elif dataset == "FitBit":
    population, chosen_pool = load_dataset(FitBit=True)
    multiplier = ranges[6]
    if include_deviations == True:
        multiplier = ranges[6]/10

elif dataset == "Electricity":
    population, chosen_pool = load_dataset(electricity=True)
    multiplier = ranges[2]

# for x, y in zip(ti_pop, ti_pool):
    # for row in range(len(x)):
    #     (x.iloc[row]).dropna(inplace=True) # remove NaN rows from the dataframe
    # print(x)
    # for row in range(len(y)):
    #     (y.iloc[row]).dropna() # remove NaN rows from the dataframe

# for i in range(8):
#     for row in range(len(ti_pop[i])):
#         (ti_pop[i].iloc[row]).dropna(inplace=True) # remove NaN rows from the dataframe
#         print(ti_pop[i])
#     for row in range(len(ti_pool[i])):
#         (ti_pool[i].iloc[row]).dropna() # remove NaN rows from the dataframe
    
# This isn't working because it's 'a value is trying to be set on a copy of a slice from a DataFrame'?????
    # so why is the above for x, y in zip() working?!?!??

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

# plots!
fig, ax1 = plt.subplots()
colours1 = ["cornflowerblue", "gold"]
if fixed_FPR == True:
    ax2 = ax1.twinx()
    colours2 = ["mediumblue", "orange"]

# plotting the performance of the inference for one timestamp over noise
if not include_longitudinals:
    ax1.plot(multiplier, auc_L1, colours1[0], linewidth=2.0, label="L1 AUC")
    ax1.plot(multiplier, auc_LLR, colours1[1], linewidth=2.0, label="LLR AUC")
    if fixed_FPR == True:
        ax2.plot(multiplier, tpr_at_fpr_L1, colours2[0], linewidth=2.0, label="L1 tpr")
        ax2.plot(multiplier, tpr_at_fpr_LLR, colours2[1], linewidth=2.0, label="LLR tpr")

# plotting the performance of the inference for each of the longitudinal datasets over noise
else:
    if error_bands:
        zipped_L1 = zip(*noisy_longitudinals_L1)
        zipped_LLR = zip(*noisy_longitudinals_LLR)
        transposed_L1 = [list(sublist) for sublist in zipped_L1]
        transposed_LLR = [list(sublist) for sublist in zipped_LLR]
        noisy_longitudinals_meanminmax_L1 = [[np.average(i) for i in transposed_L1], 
                                    [np.min(j) for j in transposed_L1],
                                    [np.max(k) for k in transposed_L1]]
        noisy_longitudinals_meanminmax_LLR = [[np.average(i) for i in transposed_LLR], 
                                    [np.min(j) for j in transposed_LLR],
                                    [np.max(k) for k in transposed_LLR]]

        ax1.plot(multiplier, noisy_longitudinals_meanminmax_L1[0], colours1[0], linewidth=2.0, label=f"L1 AUC")
        ax1.fill_between(multiplier, noisy_longitudinals_meanminmax_L1[1], noisy_longitudinals_meanminmax_L1[2], alpha=0.2)
        ax1.plot(multiplier, noisy_longitudinals_meanminmax_LLR[0], colours1[1], linewidth=2.0, label=f"LLR AUC")
        ax1.fill_between(multiplier, noisy_longitudinals_meanminmax_LLR[1], noisy_longitudinals_meanminmax_LLR[2], alpha=0.2)

        if fixed_FPR == True:
            zipped_tpr_L1 = zip(*noisy_longitudinals_tpr_L1)
            zipped_tpr_LLR = zip(*noisy_longitudinals_tpr_LLR)
            transposed_tpr_L1 = [list(sublist) for sublist in zipped_tpr_L1]
            transposed_tpr_LLR = [list(sublist) for sublist in zipped_tpr_LLR]
            noisy_longitudinals_meanminmax_tpr_L1 = [[np.average(i) for i in transposed_tpr_L1], 
                                        [np.min(j) for j in transposed_tpr_L1],
                                        [np.max(k) for k in transposed_tpr_L1]]
            noisy_longitudinals_meanminmax_tpr_LLR = [[np.average(i) for i in transposed_tpr_LLR], 
                                        [np.min(j) for j in transposed_tpr_LLR],
                                        [np.max(k) for k in transposed_tpr_LLR]]

            ax2.plot(multiplier, noisy_longitudinals_meanminmax_tpr_L1[0], colours2[0], linewidth=2.0, label=f"L1 tpr")
            ax2.fill_between(multiplier, noisy_longitudinals_meanminmax_tpr_L1[1], noisy_longitudinals_meanminmax_tpr_L1[2], alpha=0.2)
            ax2.plot(multiplier, noisy_longitudinals_meanminmax_tpr_LLR[0], colours2[1], linewidth=2.0, label=f"LLR tpr")
            ax2.fill_between(multiplier, noisy_longitudinals_meanminmax_tpr_LLR[1], noisy_longitudinals_meanminmax_tpr_LLR[2], alpha=0.2)
    
    else:
        for l in range(iterations):
            if L1_or_LLR == "L1":
                ax1.plot(multiplier, noisy_longitudinals_L1[l], colours1[0], linewidth=2.0, label=f"L1 AUC timestamp {l}")
                if fixed_FPR == True:
                    ax2.plot(multiplier, noisy_longitudinals_tpr_L1[l], colours2[0], linewidth=2.0, label=f"L1 tpr timestamp {l}")
            elif L1_or_LLR == "LLR":
                ax1.plot(multiplier, noisy_longitudinals_LLR[l], colours1[1], linewidth=2.0, label=f"LLR AUC timestamp {l}")
                if fixed_FPR == True:
                    ax2.plot(multiplier, noisy_longitudinals_tpr_L1[l], colours2[1], linewidth=2.0, label=f"LLR tpr timestamp {l}")

ax1.legend(loc='upper right')
if fixed_FPR == True:
    # Merge handles and labels
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    # Add combined legend to one axis
    ax1.legend(h1 + h2, l1 + l2, loc='upper right')
    ax2.set_ylabel("TPR at 0.01 FPR")

ax1.set_xscale("log")
ax1.set_xlabel("noise scale")
ax1.set_ylabel("AUC scores")
ax1.set_ylim([0.45, 1]) # enables comparable auc scores between L1 and LLR
ax1.grid(True)

plt.show()
