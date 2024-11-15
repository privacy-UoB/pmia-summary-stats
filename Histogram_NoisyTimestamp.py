import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.metrics import roc_auc_score
from utils import load_timestamp_dataset, LLR, L1, L1_ttest, L1_threshold, LLR_threshold, ground_truth, D3, split_pool

# load dataset
ti_pop, ti_pool, ti_sample = load_timestamp_dataset()

for x, y in zip(ti_pop, ti_pool):
    x.drop(["disease", "timepoint", "patient_id"], axis=1, inplace=True)
    y.drop(["disease", "timepoint", "patient_id"], axis=1, inplace=True)

pop = ti_pop[0] # make pop configurable
pool = ti_pool[0] # make pool configurable
sigma_j = np.std(pop, axis=0) #this is doing it over all the columns (miRNAs)

# fractions of standard deviation applied to the dataset
# multiplier = np.arange(0, 8, 0.04)
# multiplier = np.arange(0, 20, 0.1)
multiplier = [0, 1, 2, 3] # fractions of standard deviation applied to the dataset

aucs_L1 = []
aucs_LLR = []
p_values_pop_L1 = []
p_values_pool_L1 = []
p_values_pop_LLR = []
p_values_pool_LLR = []

for m in multiplier:
    
    pop_noise = np.random.normal(0, m, pop.shape) # not tailored variance to each miRNA
    # pop_noise = np.random.normal(0, m * sigma_j, pop.shape)
    noised_pop = pop + pop_noise

    pool_noise = np.random.normal(0, m, pool.shape) # not tailored variance to each miRNA
    # pool_noise = np.random.normal(0, m * sigma_j, pool.shape)
    noised_pool = pool + pool_noise
    
    # the 'noise' increases throughout each of the later timepoints the data is collected from
    local_noised_pop = noised_pop
    local_noised_pool = noised_pool
    local_pop = pop
    local_pool = pool


    pvalue_pop_L1 = L1_ttest(local_noised_pop, local_pop, local_pool)
    pvalue_pool_L1 = L1_ttest(local_noised_pool, local_pop, local_pool)

    p_values_pop_L1.append(pvalue_pop_L1)
    p_values_pool_L1.append(pvalue_pool_L1)

    pvalue_pop_LLR = LLR(local_noised_pop, local_pop, local_pool)
    pvalue_pool_LLR = LLR(local_noised_pool, local_pop, local_pool)

    pvalue_pop_LLR_ravel = pvalue_pop_LLR.ravel()
    p_values_pop_LLR.append(pvalue_pop_LLR_ravel)
    pvalue_pool_LLR_ravel = pvalue_pool_LLR.ravel()
    p_values_pool_LLR.append(pvalue_pool_LLR_ravel)


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
    # histogram showing standard deviations across all 8 timestamps of the individual
    # plt.hist(p_values_pop_L1[m], bins=40, label=f"timestamp {m}")
    # plt.xlabel("p values population L1")
    # plt.ylabel("count of deviations across 40 different range values")
    # plt.legend(loc="upper right")
    # plt.show()

    # # histogram showing standard deviations across all 8 timestamps of the individual
    # plt.hist(p_values_pool_L1[m], bins=40, label=f"timestamp {m}")
    # plt.xlabel("p values pool L1")
    # plt.ylabel("count of deviations across 40 different range values")
    # plt.legend(loc="upper right")
    # plt.show()

    # histogram showing standard deviations across all 8 timestamps of the individual
    plt.hist((p_values_pop_L1[m], p_values_pool_L1[m]), bins=40, label=f"timestamp {m}")
    plt.xlabel("p values pop & pool L1")
    plt.ylabel("count of deviations across 40 different range values")
    plt.legend(loc="upper right")
    plt.show()

    # histogram showing standard deviations across all 8 timestamps of the individual
    plt.hist(p_values_pop_L1[m], bins=40, label=f"timestamp {m}")
    plt.xlabel("p values population LLR")
    plt.ylabel("count of deviations across 40 different range values")
    plt.legend(loc="upper right")
    plt.show()

    # histogram showing standard deviations across all 8 timestamps of the individual
    plt.hist(p_values_pool_LLR[m], bins=40, label=f"timestamp {m}")
    plt.xlabel("p values pool LLR")
    plt.ylabel("count of deviations across 40 different range values")
    plt.legend(loc="upper right")
    plt.show()

    # histogram showing standard deviations across all 8 timestamps of the individual
    plt.hist((p_values_pop_LLR[m], p_values_pool_LLR[m]), bins=40, label=f"timestamp {m}")
    plt.xlabel("p values pop & pool LLR")
    plt.ylabel("count of deviations across 40 different range values")
    plt.legend(loc="upper right")
    plt.show()
