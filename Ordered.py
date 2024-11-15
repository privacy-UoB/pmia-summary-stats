import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import roc_auc_score
from utils import load_dataset, LLR, L1, L1_ttest, L1_threshold, LLR_threshold, ground_truth, D3

# paper: the demonstrated graphs showing roc curves
    # 1st: 50 subsets of n/1049 different individuals (n = 35, 65, 124)
    # 2nd: 6 case groups D19, D17, D10, D7, D3, D1

# load dataset
pop_rpool, pop_cpool, rpool, cpool = load_dataset(case_sample=D3)

pop_rpool = pop_rpool.drop(columns="diseases")
pop_cpool = pop_cpool.drop(columns="diseases")
pop = pop_cpool # make pop configurable

rpool = rpool.drop(columns="diseases")
cpool = cpool.drop(columns="diseases")
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

        pvalue_pop_L1 = L1_ttest(local_pop, local_pop, local_pool)
        pvalue_pool_L1 = L1_ttest(local_pool, local_pop, local_pool)

        pvalue_pop_LLR = LLR(local_pop, local_pop, local_pool)
        pvalue_pool_LLR = LLR(local_pool, local_pop, local_pool)


        # power_L1 = []
        # fpr_L1 = []
        # for t in L1_threshold(local_pop, local_pool):
        #     p, f = ground_truth(pvalue_pop_L1, pvalue_pool_L1, t)
        #     power_L1.append(p)
        #     fpr_L1.append(f)
        # fpr_L1 = np.array(fpr_L1)
        # power_L1 = np.array(power_L1)

        # order_L1 = np.argsort(fpr_L1)
        # fpr_L1 = fpr_L1[order_L1]
        # power_L1 = power_L1[order_L1]

        # power_LLR = []
        # fpr_LLR = []
        # for t in LLR_threshold(local_pop, local_pool):
        #     p, f = ground_truth(pvalue_pop_LLR, pvalue_pool_LLR, t)
        #     power_LLR.append(p)
        #     fpr_LLR.append(f)
        # fpr_LLR = np.array(fpr_LLR)
        # power_LLR = np.array(power_LLR)

        # order_LLR = np.argsort(fpr_LLR)
        # fpr_LLR = fpr_LLR[order_LLR]
        # power_LLR = power_LLR[order_LLR]


        y_true_L1 = np.concatenate((np.zeros(len(pvalue_pop_L1)), np.ones(len(pvalue_pool_L1))))
        y_score_L1 = np.concatenate((pvalue_pop_L1, pvalue_pool_L1))
        roc_L1 = roc_auc_score(y_true_L1, y_score_L1)

        aucs_L1.append(roc_L1)

        y_true_LLR = np.concatenate((np.zeros(len(pvalue_pop_LLR)), np.ones(len(pvalue_pool_LLR))))
        y_score_LLR = np.concatenate((pvalue_pop_LLR, pvalue_pool_LLR))
        roc_LLR = roc_auc_score(y_true_LLR, y_score_LLR)
        
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
plt.xlabel("number MiRNAs")
plt.ylabel("AUC scores")
plt.legend(loc="upper right")
plt.show() 
