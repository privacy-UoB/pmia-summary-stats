import numpy as np
import matplotlib.pyplot as plt
import random
from utils_datasets import load_dataset, drop_dataset_index, D3
from utils import auc_scores, fpr_power, LLR, L1, L1_threshold

# paper: the demonstrated graphs showing roc curves
    # 1st: 50 subsets of n/1049 different individuals (n = 35, 65, 124)
    # 2nd: 6 case groups D19, D17, D10, D7, D3, D1

# load dataset
pop_rpool, pop_cpool, rpool, cpool = load_dataset(case_sample=D3)
pop_rpool, pop_cpool, rpool, cpool = drop_dataset_index(pop_rpool, pop_cpool, rpool, cpool)

pop = pop_cpool # make pop configurable
pool = cpool # make pool configurable

auc_L1 = []
auc_LLR = []
num_miRNAs = []
miRNAs = list(pop.keys()) # get the list of miRNAs ["miRNA_1234", "miRNA_1235", ...]
num_orders = 50 # number of different samples of MiRNAs

shuffled_lists = []
for j in range (num_orders):
    current_miRNA_list = list(miRNAs)
    random.shuffle(current_miRNA_list)
    shuffled_lists.append(current_miRNA_list)

for i in range(2,len(miRNAs),2): # MiRNAs range from 1 to 466 in paper
    aucs_L1 = []
    aucs_LLR = []
    num_miRNAs.append(i)

    for j in range (num_orders):
        current_shuffled_list = shuffled_lists[j]
        selected_miRNAs = current_shuffled_list[:i]

        local_pop = pop[selected_miRNAs]
        local_pool = pool[selected_miRNAs]
        
        # print(L1(victim, local_pop, local_pool).sum())
        # print(L1_threshold(local_pop, local_pool))

        # print(LLR(local_pop, local_pop, local_pool))
        # print(L1(local_pop, local_pop, local_pool))

        # Query: should these actually be local_pop, local_pool, pop, pool?
        roc_L1, pvalue_pop_L1, pvalue_pool_L1 = auc_scores(local_pop, local_pool, local_pop, local_pool)
        roc_LLR, pvalue_pop_LLR, pvalue_pool_LLR = auc_scores(local_pop, local_pool, local_pop, local_pool, LR=True)

        # fpr_L1, power_L1 = fpr_power(local_pop, local_pool, pvalue_pop_L1, pvalue_pool_L1)
        # fpr_LLR, power_LLR = fpr_power(local_pop, local_pool, pvalue_pop_LLR, pvalue_pool_LLR, LR=True)

        aucs_L1.append(roc_L1)        
        aucs_LLR.append(roc_LLR)

    if len(aucs_L1) >0:
        auc_L1.append(np.average(aucs_L1))

    if len(aucs_LLR) >0:
        auc_LLR.append(np.average(aucs_LLR))

# plots!
# fig, ax = plt.subplots()
# ax.set_xscale("log")
# ax.plot(fpr_L1, power_L1, linewidth=2.0)
# ax.plot(fpr_L1, power_LLR, linewidth=2.0)
# plt.xlabel("fpr")
# plt.ylabel("power")
# plt.show()

# print(f'AUC score:{auc_L1}')
# print(f'AUC score:{auc_LLR}')

# plots!
fig, ax = plt.subplots()
ax.plot(num_miRNAs, auc_L1, "-b", linewidth=2.0, label="L1")
ax.plot(num_miRNAs, auc_LLR, "-r", linewidth=2.0, label="LLR")
ax.invert_xaxis()
ax.set_ylim([0.5,1]) # enables comparable auc scores between L1 and LLR
plt.xlabel("number MiRNAs")
plt.ylabel("AUC scores")
plt.legend(loc="upper right")
plt.show() 
