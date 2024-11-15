import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
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

split_pop, split_cpool = split_pool(pop, pool)
pop = split_pop
pool = split_cpool


num_miRNAs = []
miRNAs = list(pop.keys()) # get the list of miRNAs ["miRNA_1234", "miRNA_1235", ...]
current_miRNA_list = list(miRNAs)
random.shuffle(current_miRNA_list)

aucs_L1 = []
aucs_LLR = []
p_values_pop_L1 = []
p_values_pool_L1 = []
p_values_pop_LLR = []
p_values_pool_LLR = []

for i in range(2,len(miRNAs),2): # MiRNAs range from 1 to 466 in paper
    num_miRNAs.append(i)
    selected_miRNAs = current_miRNA_list[:i]

    local_pop = pop[selected_miRNAs]
    local_pool = pool[selected_miRNAs]


    pvalue_pop_L1 = L1_ttest(local_pop, local_pop, local_pool)
    pvalue_pool_L1 = L1_ttest(local_pool, local_pop, local_pool)

    p_values_pop_L1.append(pvalue_pop_L1)
    p_values_pool_L1.append(pvalue_pool_L1)

    pvalue_pop_LLR = LLR(local_pop, local_pop, local_pool)
    pvalue_pool_LLR = LLR(local_pool, local_pop, local_pool)

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
# plt.hist(p_values_pop_L1, bins=40)
# plt.xlabel("p values population L1")
# plt.ylabel("count of deviations across 40 different range values")
# plt.show()

# # histogram showing standard deviations across all 8 timestamps of the individual
# plt.hist(p_values_pool_L1, bins=40)
# plt.xlabel("p values pool L1")
# plt.ylabel("count of deviations across 40 different range values")
# plt.show()

# histogram showing standard deviations across all 8 timestamps of the individual
plt.hist((p_values_pop_L1, p_values_pool_L1), bins=40)
plt.xlabel("p values pop & pool L1")
plt.ylabel("count of deviations across 40 different range values")
plt.show()

# histogram showing standard deviations across all 8 timestamps of the individual
plt.hist(p_values_pop_L1, bins=40)
plt.xlabel("p values population LLR")
plt.ylabel("count of deviations across 40 different range values")
plt.show()

# histogram showing standard deviations across all 8 timestamps of the individual
plt.hist(p_values_pool_LLR, bins=40)
plt.xlabel("p values pool LLR")
plt.ylabel("count of deviations across 40 different range values")
plt.show()

# histogram showing standard deviations across all 8 timestamps of the individual
plt.hist((p_values_pop_LLR, p_values_pool_LLR), bins=40)
plt.xlabel("p values pop & pool LLR")
plt.ylabel("count of deviations across 40 different range values")
plt.show()
