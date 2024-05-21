import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import roc_auc_score
from utils import load_dataset, L1, L1_ttest, L1_threshold, ground_truth, D19

# TODO:
# run for all diseases - check names as these don't match!
# run for llr
# change plot to match paper (invert xaxis)
# check the code to spot the error (esp. num mirna in debugger)
# commit file!
# mon 1pm

# Notes:
# 1st: 50 subsets of n/1049 diff. individuals (35, 65, 124)
# 4 attacks (L1, LLR with pool&pop stats, LLR without variance stats, LLR without variance stats and using theoretical relations)
# 2nd: case groups: D19, D17, D10, D7, D3, D1


# load dataset
pop, rpool, cpool = load_dataset(case_sample=D19)
pop = pop.drop(columns="diseases")
rpool = rpool.drop(columns="diseases")
cpool = cpool.drop(columns="diseases")

auc = []
num_miRNAs = []
miRNAs = list(pop.keys()) # get the list of miRNAs ["miRNA_1234", "miRNA_1235", ...]
num_orders = 50 # number of diff. samples of MiRNAs

shuffled_lists = []
for j in range (num_orders):
    current_miRNA_list = list(miRNAs)
    random.shuffle(current_miRNA_list)
    shuffled_lists.append(current_miRNA_list)

for i in range(1,len(miRNAs),5): # MiRNAs range from 1 to 466 in paper
    aucs = []
    num_miRNAs.append(i)

    for j in range (num_orders):
        current_shuffled_list = shuffled_lists[j]
        selected_miRNAs = current_shuffled_list[:i]

        local_pop = pop[selected_miRNAs]
        local_rpool = rpool[selected_miRNAs]
        local_cpool = cpool[selected_miRNAs]
        
        # print (L1(victim, local_pop, local_cpool).sum())
        # test = L1_threshold(local_pop, local_cpool)

        power = []
        fpr = []
        pvalue_pop = L1_ttest(local_pop, local_pop, local_cpool)[1]
        pvalue_cpool = L1_ttest(local_cpool, local_pop, local_cpool)[1]

        for t in L1_threshold(local_pop, local_cpool):
            p, f = ground_truth(pvalue_pop, pvalue_cpool, t)
            power.append(p)
            fpr.append(f)
        fpr = np.array(fpr)
        power = np.array(power)

        order = np.argsort(fpr)
        fpr = fpr[order]
        power = power[order]

        y_true = np.concatenate((np.zeros(len(pvalue_pop)), np.ones(len(pvalue_cpool))))
        y_score = np.concatenate((pvalue_pop, pvalue_cpool))
        roc = roc_auc_score(y_true, y_score)

        aucs.append(roc)
    if len(aucs) >0:
        auc.append(np.average(aucs))

# plots!
# fig, ax = plt.subplots()
# ax.set_xscale("log")

# ax.plot(fpr, power, linewidth=2.0)
# plt.xlabel("fpr")
# plt.ylabel("power")

# plt.show()

print(f'AUC score:{auc}')

# plots!
fig, ax = plt.subplots()
ax.set_xscale("log")

ax.plot(num_miRNAs, auc, linewidth=2.0)
plt.xlabel("number MiRNAs")
plt.ylabel("ROC scores")
plt.show()
