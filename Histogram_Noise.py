import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils_datasets import load_dataset, D3, split_pool, drop_dataset_index
from utils import auc_scores, Gaussian_noise, LLR, L1_ttest

# load dataset
pop_rpool, pop_cpool, rpool, cpool = load_dataset(case_sample=D3)
pop_rpool, pop_cpool, rpool, cpool = drop_dataset_index(pop_rpool, pop_cpool, rpool, cpool)

pop = pop_cpool # make pop configurable
pool = cpool # make pool configurable

split_pop, split_cpool, cpool_into_pop = split_pool(pop, pool, include_transfer=False)
pop = split_pop
pool = split_cpool

sigma_j = np.std(pop, axis=0) # this is doing it over all the columns (miRNAs)

# fractions of standard deviation applied to the dataset
multiplier = [0, 0.25, 0.5, 0.75, 1] # fractions of standard deviation applied to the dataset
# multiplier = [0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# multiplier = np.logspace(0, 15, base=2) # used for standard noise values
# multiplier = np.arange(0, 8, 0.04) # sufficient for m * sigma_j

p_values_pop_L1 = []
p_values_pool_L1 = []
p_values_cpoolintopop_L1 = []
p_values_pop_LLR = []
p_values_pool_LLR = []
p_values_cpoolintopop_LLR = []

for m in multiplier:
    # make deviation A LOT bigger, then plot on np.logspace scale
    deviation = m * sigma_j # sometimes replace m * sigma_j with m
    nonneg_noisy_pop, nonneg_noisy_pool = Gaussian_noise(pop, pool, 0, deviation, clip=True)

    cpoolintopop_noise = np.random.normal(0, deviation, cpool_into_pop.shape)
    noisy_cpoolintopop = cpool_into_pop + cpoolintopop_noise
    nonneg_noisy_cpoolintopop = np.clip(noisy_cpoolintopop, 0, None)

    roc_L1, pvalue_pop_L1, pvalue_pool_L1 = auc_scores(nonneg_noisy_pop, nonneg_noisy_pool, pop, pool)
    roc_LLR, pvalue_pop_LLR, pvalue_pool_LLR = auc_scores(nonneg_noisy_pop, nonneg_noisy_pool, pop, pool, LR=True)
    pvalue_cpoolintopop_L1 = L1_ttest(nonneg_noisy_cpoolintopop, pop, pool)
    pvalue_cpoolintopop_LLR = LLR(nonneg_noisy_cpoolintopop, pop, pool)

    p_values_pop_L1.append(pvalue_pop_L1)
    p_values_pool_L1.append(pvalue_pool_L1)
    p_values_cpoolintopop_L1.append(pvalue_cpoolintopop_L1)

    p_values_pop_LLR.append((pvalue_pop_LLR.ravel()))
    p_values_pool_LLR.append((pvalue_pool_LLR.ravel()))
    p_values_cpoolintopop_LLR.append((pvalue_cpoolintopop_LLR.ravel()))
        
# histogram showing standard deviations across all 8 timestamps of the individual
for m in range(len(multiplier)):
    # L1
    # plt.hist(p_values_pop_L1[m], bins=50, label=f"noise multiplier number {m}")
    # plt.xlabel("p values population L1")
    # plt.ylabel("count of deviations across 50 different range values")
    # plt.legend(loc="upper right")
    # plt.show()

    # plt.hist(p_values_pool_L1[m], bins=50, label=f"noise multiplier number {m}")
    # plt.xlabel("p values pool L1")
    # plt.ylabel("count of deviations across 50 different range values")
    # plt.legend(loc="upper right")
    # plt.show()

    # plt.hist(p_values_cpoolintopop_L1[m], bins=50, label=f"noise multiplier number {m}")
    # plt.xlabel("p values cpool moved to pop L1")
    # plt.ylabel("count of deviations across 50 different range values")
    # plt.legend(loc="upper right")
    # plt.show()

    plt.hist((p_values_pop_L1[m], p_values_pool_L1[m], p_values_cpoolintopop_L1[m]), bins=20, density=True, 
             label=["population", "pool", "transferred pool"])
    plt.xlabel("L1-distances p values for population, pool and transferred pool")
    plt.ylabel("normalised count of 20 different range values")
    plt.legend(loc="upper center")
    plt.show()

    # LLR
    # plt.hist(p_values_pop_LLR[m], bins=300, label=f"noise multiplier number {m}")
    # plt.xlabel("p values population LLR")
    # plt.ylabel("count of deviations across 300 different range values")
    # plt.legend(loc="upper right")
    # plt.show()

    # plt.hist(p_values_pool_LLR[m], bins=100, label=f"noise multiplier number {m}")
    # plt.xlabel("p values pool LLR")
    # plt.ylabel("count of deviations across 100 different range values")
    # plt.legend(loc="upper right")
    # plt.show()

    # plt.hist(p_values_cpoolintopop_LLR[m], bins=100, label=f"noise multiplier number {m}")
    # plt.xlabel("p values cpool moved to pop LLR")
    # plt.ylabel("count of deviations across 100 different range values")
    # plt.legend(loc="upper right")
    # plt.show()

    a = (p_values_pop_LLR[m], p_values_pool_LLR[m], p_values_cpoolintopop_LLR[m])
    b = tuple(np.clip(i, -1000, 1000) for i in a)
    plt.hist(b, bins=25, density=True, label=["population", "pool", "transferred pool"])
    plt.xlabel("Likelihood ratio test statistics for population, pool and transferred pool")
    plt.ylabel("normalised count of 25 different range values")
    plt.legend(loc="upper left")
    plt.show()
