import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from utils import load_dataset, LLR, L1, L1_ttest, L1_threshold, LLR_threshold, ground_truth, D3, split_pool

# load dataset
pop_rpool, pop_cpool, rpool, cpool = load_dataset(case_sample=D3)

pop_rpool = pop_rpool.drop(columns="diseases")
pop_cpool = pop_cpool.drop(columns="diseases")
pop = pop_cpool # make pop configurable

rpool = rpool.drop(columns="diseases")
cpool = cpool.drop(columns="diseases")
pool = cpool # make pool configurable

split_pop, split_cpool, cpool_into_pop = split_pool(pop, pool)
pop = split_pop
pool = split_cpool

sigma_j = np.std(pop, axis=0) # this is doing it over all the columns (miRNAs)

# fractions of standard deviation applied to the dataset
multiplier = [0, 0.25, 0.5, 0.75, 1] # fractions of standard deviation applied to the dataset
# multiplier = [0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# multiplier = np.logspace(0, 15, base=2) # used for standard noise values
# multiplier = np.arange(0, 8, 0.04) # sufficient for m * sigma_j

aucs_L1 = []
aucs_LLR = []
p_values_pop_L1 = []
p_values_pool_L1 = []
p_values_cpoolintopop_L1 = []
p_values_pop_LLR = []
p_values_pool_LLR = []
p_values_cpoolintopop_LLR = []

for m in multiplier:
    
    # pop_noise = np.random.normal(0, m, pop.shape) #make this A LOT bigger, then plot on np.logspace scale
    # pool_noise = np.random.normal(0, m, pool.shape)
    # cpoolintopop_noise = np.random.normal(0, m, cpool_into_pop.shape)
    pop_noise = np.random.normal(0, m * sigma_j, pop.shape)
    pool_noise = np.random.normal(0, m * sigma_j, pool.shape)
    cpoolintopop_noise = np.random.normal(0, m * sigma_j, cpool_into_pop.shape)

    noised_pop = pop + pop_noise
    nonneg_pop_noise = np.clip(noised_pop, 0, None)

    noised_pool = pool + pool_noise
    nonneg_pool_noise = np.clip(noised_pool, 0, None)

    noised_cpoolintopop = cpool_into_pop + cpoolintopop_noise
    nonneg_cpoolintopop_noise = np.clip(noised_cpoolintopop, 0, None)


    pvalue_pop_L1 = L1_ttest(nonneg_pop_noise, pop, pool)
    pvalue_pool_L1 = L1_ttest(nonneg_pool_noise, pop, pool)
    pvalue_cpoolintopop_L1 = L1_ttest(nonneg_cpoolintopop_noise, pop, pool)

    p_values_pop_L1.append(pvalue_pop_L1)
    p_values_pool_L1.append(pvalue_pool_L1)
    p_values_cpoolintopop_L1.append(pvalue_cpoolintopop_L1)

    pvalue_pop_LLR = LLR(nonneg_pop_noise, pop, pool)
    pvalue_pool_LLR = LLR(nonneg_pool_noise, pop, pool)
    pvalue_cpoolintopop_LLR = LLR(nonneg_cpoolintopop_noise, pop, pool)

    pvalue_pop_LLR_ravel = pvalue_pop_LLR.ravel()
    p_values_pop_LLR.append(pvalue_pop_LLR_ravel)
    pvalue_pool_LLR_ravel = pvalue_pool_LLR.ravel()
    p_values_pool_LLR.append(pvalue_pool_LLR_ravel)
    pvalue_cpoolintopop_LLR_ravel = pvalue_cpoolintopop_LLR.ravel()
    p_values_cpoolintopop_LLR.append(pvalue_cpoolintopop_LLR_ravel)


    y_true_L1 = np.concatenate((np.zeros(len(pvalue_pop_L1)), np.ones(len(pvalue_pool_L1))))
    y_score_L1 = np.concatenate((pvalue_pop_L1, pvalue_pool_L1))
    roc_L1 = roc_auc_score(y_true_L1, y_score_L1)
    aucs_L1.append(roc_L1)

    y_true_LLR = np.concatenate((np.zeros(len(pvalue_pop_LLR)), np.ones(len(pvalue_pool_LLR))))
    y_score_LLR = np.concatenate((pvalue_pop_LLR, pvalue_pool_LLR))
    roc_LLR = roc_auc_score(y_true_LLR, y_score_LLR)
    aucs_LLR.append(roc_LLR)
        
# histogram showing standard deviations across all 8 timestamps of the individual
for m in range(len(multiplier)):
    # plt.hist(p_values_pop_L1[m], bins=50, label=f"noise multiplier number {m}")
    # plt.xlabel("p values population L1")
    # plt.ylabel("count of deviations across 50 different range values")
    # plt.legend(loc="upper right")
    # plt.show()

    # # histogram showing standard deviations across all 8 timestamps of the individual
    # plt.hist(p_values_pool_L1[m], bins=50, label=f"noise multiplier number {m}")
    # plt.xlabel("p values pool L1")
    # plt.ylabel("count of deviations across 50 different range values")
    # plt.legend(loc="upper right")
    # plt.show()

    # histogram showing standard deviations across all 8 timestamps of the individual
    plt.hist((p_values_pop_L1[m], p_values_pool_L1[m]), bins=50, label=f"noise multiplier number {m}")
    plt.xlabel("p values pop & pool L1")
    plt.ylabel("count of deviations across 50 different range values")
    plt.legend(loc="upper right")
    plt.show()

    # histogram showing standard deviations across all 8 timestamps of the individual
    plt.hist(p_values_cpoolintopop_L1[m], bins=50, label=f"noise multiplier number {m}")
    plt.xlabel("p values cpool moved to pop L1")
    plt.ylabel("count of deviations across 50 different range values")
    plt.legend(loc="upper right")
    plt.show()

    # # histogram showing standard deviations across all 8 timestamps of the individual
    # plt.hist(p_values_pop_LLR[m], bins=300, label=f"noise multiplier number {m}")
    # plt.xlabel("p values population LLR")
    # plt.ylabel("count of deviations across 300 different range values")
    # plt.legend(loc="upper right")
    # plt.show()

    # # histogram showing standard deviations across all 8 timestamps of the individual
    # plt.hist(p_values_pool_LLR[m], bins=100, label=f"noise multiplier number {m}")
    # plt.xlabel("p values pool LLR")
    # plt.ylabel("count of deviations across 100 different range values")
    # plt.legend(loc="upper right")
    # plt.show()

    # histogram showing standard deviations across all 8 timestamps of the individual
    plt.hist((p_values_pop_LLR[m], p_values_pool_LLR[m]), bins=300, label=f"noise multiplier number {m}")
    plt.xlabel("p values pop & pool LLR")
    plt.ylabel("count of deviations across 300 different range values")
    plt.legend(loc="upper right")
    plt.show()

    # histogram showing standard deviations across all 8 timestamps of the individual
    plt.hist(p_values_cpoolintopop_LLR[m], bins=100, label=f"noise multiplier number {m}")
    plt.xlabel("p values cpool moved to pop L1")
    plt.ylabel("count of deviations across 100 different range values")
    plt.legend(loc="upper right")
    plt.show()

# increase no bins, also show split pool separetely
# can we do better with knowing distribution is gaussian?