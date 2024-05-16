import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from utils import load_dataset, L1, L1_ttest, L1_threshold, ground_truth

# check paper and make sure mirna order is the same for comparison
# 1st: 50 subsets of n/1049 diff. individuals (35, 65, 124)
# 4 attacks (L1, LLR with pool&pop stats, LLR without variance stats, LLR without variance stats and using theoretical relations)
# 2nd: case groups: D19, D17, D10, D7, D3, D1

auc = []
num_miRNAs = []

for i in range(10,1000,10):
    aucs = []
    num_miRNAs.append(i)
    # i=1, num_miRNA = 10
    for j in range (50):
        ret = load_dataset(i, 35, D19)
        if ret is None:
            del num_miRNAs[-1]
            break
        pop, rpool, cpool = ret

        pop = pop.drop(columns="diseases")
        rpool = rpool.drop(columns="diseases")
        cpool = cpool.drop(columns="diseases")
        victim = cpool.iloc[99]

        print (L1(victim, pop, cpool).sum())
        

        test = L1_threshold(pop, cpool)
        print (test)

        power = []
        fpr = []
        pvalue_pop = L1_ttest(pop, pop, cpool)[1]
        pvalue_cpool = L1_ttest(cpool, pop, cpool)[1]

        for t in L1_threshold(pop, cpool):
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
