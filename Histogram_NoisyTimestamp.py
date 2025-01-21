import numpy as np
import matplotlib.pyplot as plt
from utils_datasets import load_timestamp_dataset, drop_timestamp_index
from utils import auc_scores, Gaussian_noise

# load dataset
ti_pop, ti_pool, ti_sample = load_timestamp_dataset()
ti_pop, ti_pool = drop_timestamp_index(ti_pop, ti_pool)

pop = ti_pop[0] # make pop configurable
pool = ti_pool[0] # make pool configurable
sigma_j = np.std(pop, axis=0) #this is doing it over all the columns (miRNAs)

# fractions of standard deviation applied to the dataset
# multiplier = np.arange(0, 8, 0.04)
# multiplier = np.arange(0, 20, 0.1)
multiplier = [0, 1, 2, 3] # fractions of standard deviation applied to the dataset

p_values_pop_L1 = []
p_values_pool_L1 = []
p_values_pop_LLR = []
p_values_pool_LLR = []

# the 'noise' increases throughout each of the multipliers
for m in multiplier:
    deviation = m # not tailored variance to each miRNA; otherwise replace m with m * sigma_j
    noisy_pop, noisy_pool = Gaussian_noise(pop, pool, 0, deviation)
    
    roc_L1, pvalue_pop_L1, pvalue_pool_L1 = auc_scores(noisy_pop, noisy_pool, pop, pool)
    roc_LLR, pvalue_pop_LLR, pvalue_pool_LLR = auc_scores(noisy_pop, noisy_pool, pop, pool, LR=True)

    p_values_pop_L1.append(pvalue_pop_L1)
    p_values_pool_L1.append(pvalue_pool_L1)
    p_values_pop_LLR.append((pvalue_pop_LLR.ravel()))
    p_values_pool_LLR.append((pvalue_pool_LLR.ravel()))
        
# histogram showing standard deviations across all 8 timestamps of the individual
for m in range(len(multiplier)):
    # L1
    plt.hist(p_values_pop_L1[m], bins=40, label=f"timestamp {m}")
    plt.xlabel("p values population L1")
    plt.ylabel("count of deviations across 40 different range values")
    plt.legend(loc="upper right")
    plt.show()

    plt.hist(p_values_pool_L1[m], bins=40, label=f"timestamp {m}")
    plt.xlabel("p values pool L1")
    plt.ylabel("count of deviations across 40 different range values")
    plt.legend(loc="upper right")
    plt.show()

    plt.hist((p_values_pop_L1[m], p_values_pool_L1[m]), bins=40, label=f"timestamp {m}")
    plt.xlabel("p values pop & pool L1")
    plt.ylabel("count of deviations across 40 different range values")
    plt.legend(loc="upper right")
    plt.show()

    # LLR
    plt.hist(p_values_pop_LLR[m], bins=40, label=f"timestamp {m}")
    plt.xlabel("p values population LLR")
    plt.ylabel("count of deviations across 40 different range values")
    plt.legend(loc="upper right")
    plt.show()

    plt.hist(p_values_pool_LLR[m], bins=40, label=f"timestamp {m}")
    plt.xlabel("p values pool LLR")
    plt.ylabel("count of deviations across 40 different range values")
    plt.legend(loc="upper right")
    plt.show()

    plt.hist((p_values_pop_LLR[m], p_values_pool_LLR[m]), bins=40, label=f"timestamp {m}")
    plt.xlabel("p values pop & pool LLR")
    plt.ylabel("count of deviations across 40 different range values")
    plt.legend(loc="upper right")
    plt.show()
