import numpy as np
import matplotlib.pyplot as plt
from utils_datasets import load_dataset, D3
from utils import auc_scores, Gaussian_noise

include_longitudinals = True # if we wish to create one graph with all longitudinal entries, excludes "miRNA"
dataset = "miRNA" # choices are "miRNA", "Timestamp", "Fitbit", "Electricity"
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
    populations, pools = load_dataset(miRNA=True, disease_case_sample=D3)
    # 0 = random, 1 = case:
    pop = populations[0] # make pop configurable
    pool = pools[0] # make pool configurable
    multiplier = ranges[4]

elif dataset == "Timestamp":
    population, chosen_pool = load_dataset(timestamp=True)
    multiplier = ranges[2]

elif dataset == "FitBit":
    population, chosen_pool = load_dataset(FitBit=True)
    multiplier = ranges[6]

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

    # configuring the reference pop & pool to match the dataframe of a particular data entry
    if include_longitudinals:
        pop = population[i]
        pool = chosen_pool[i]
    elif dataset != "miRNA":
        pop = population[0]
        pool = chosen_pool[0]

    sigma_j = np.std(pop, axis=0) # this is doing it over all the features (e.g. miRNAs)
    for m in multiplier:
        aucs_L1 = []
        aucs_LLR = []

        # for loop for numorder lots of train/test, then average at end
        for j in range (num_orders):
            print("Run Number", j)

            # the 'noise' increases throughout each of the later timepoints the data is collected from
            deviation = m # changed from m * sigmaj so it's not tailored variance to each feature

            if dataset == "miRNA":
                noisy_pop, noisy_pool = Gaussian_noise(pop, pool, 0, deviation, clip=True) # make noise non-negative
            else:
                noisy_pop, noisy_pool = Gaussian_noise(pop, pool, 0, deviation)

            # get performance/accuracy for L1 & LLR statistics over the noisy stat inputs compared to the 'original' pop & pool
            roc_L1 = auc_scores(noisy_pop, noisy_pool, pop, pool, p_values=False)
            roc_LLR = auc_scores(noisy_pop, noisy_pool, pop, pool, LR=True, p_values=False)
            
            aucs_L1.append(roc_L1)
            aucs_LLR.append(roc_LLR)

        # num_order rows of datasets, columns are each longitudinal entry
        if len(aucs_L1) >0:
            auc_L1.append(np.average(aucs_L1))

        if len(aucs_LLR) >0:
            auc_LLR.append(np.average(aucs_LLR))
        
    if include_longitudinals:
        noisy_longitudinals_L1.append(auc_L1)
        noisy_longitudinals_LLR.append(auc_LLR)

# plots!
fig, ax = plt.subplots()

# plotting the performance of the inference for one timestamp over noise
if not include_longitudinals:
    ax.plot(multiplier, auc_L1, "-b", linewidth=2.0, label="L1")
    ax.plot(multiplier, auc_LLR, "-r", linewidth=2.0, label="LLR")

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

        ax.plot(multiplier, noisy_longitudinals_meanminmax_L1[0], linewidth=2.0, label=f"L1")
        ax.fill_between(multiplier, noisy_longitudinals_meanminmax_L1[1], noisy_longitudinals_meanminmax_L1[2], alpha=0.2)
        ax.plot(multiplier, noisy_longitudinals_meanminmax_LLR[0], linewidth=2.0, label=f"LLR")
        ax.fill_between(multiplier, noisy_longitudinals_meanminmax_LLR[1], noisy_longitudinals_meanminmax_LLR[2], alpha=0.2)
    
    else:
        for l in range(iterations):
            if L1_or_LLR == "L1":
                ax.plot(multiplier, noisy_longitudinals_L1[l], linewidth=2.0, label=f"L1 timestamp {l}")
            elif L1_or_LLR == "LLR":
                ax.plot(multiplier, noisy_longitudinals_LLR[l], linewidth=2.0, label=f"LLR timestamp {l}")

ax.set_ylim([0.45, 1]) # enables comparable auc scores between L1 and LLR
ax.set_xscale("log")
plt.xlabel("noise scale")
plt.ylabel("AUC scores")
plt.legend(loc="upper right")
plt.show()
