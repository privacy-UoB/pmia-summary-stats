import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_1samp
from utils import load_dataset, LLR, LLR_ttest, LLR_threshold, ground_truth

# Likelihood-Ratio Test
# LLR = Sum_j=1^m [(x_j^v - mu_j)^2 / 2sigma_j^2 - (x_j^v - mu-hat_j)^2 / 2sigma-hat_j^2 + log sigma_j/sigma-hat_j]
# x_j^v is the value of miRNA j for the individual victim in column v
# mu_j & sigma_j are the average & standard deviation miRNA j in the population
# mu-hat_j & sigma-hat_j are the average & standard deviation miRNA j in the pool

# def varying_MiRNA ():
#     pop = []
#     rpool = []
#     cpool = []
#     num_miRNAs = []
#     for i in range(10,1000,10):
#         ret = load_dataset(i)
#         if ret is None:
#             break
#         a,b,c = ret
#         pop.append(a)
#         rpool.append(b)
#         cpool.append(c)
#         num_miRNAs.append(i)
#     return pop, rpool, cpool, num_miRNAs
# pop, rpool, cpool, num_miRNAs = varying_MiRNA()

auc = []
num_miRNAs = []

for i in range(10,1000,10):
    aucs = []
    num_miRNAs.append(i)
    # i=1, num_miRNA = 10
    for j in range (5):
        ret = load_dataset(i)
        if ret is None:
            del num_miRNAs[-1]
            break
        pop, rpool, cpool = ret
    # pop = [pop(50), pop(100), ..., pop(1000)]
    # rpool = [rpool(50), rpool(100), ..., rpool(1000)]
    # cpool = [cpool(50), cpool(100), ..., cpool(1000)]
    # -> zip[(pop(50),rpool(50),cpool(50)), (pop(100), rpool(100), cpool(100)), ..., (pop(1000), rpool(1000), cpool(1000))]

        pop = pop.drop(columns="diseases")
        rpool = rpool.drop(columns="diseases")
        cpool = cpool.drop(columns="diseases")
        victim = cpool.iloc[10]

        print (LLR(victim, pop, cpool).sum())

        test = LLR_threshold(pop, cpool)
        print (test)

        power = []
        fpr = []
        pvalue_pop = LLR_ttest(pop, pop, cpool)[1]
        pvalue_cpool = LLR_ttest(cpool, pop, cpool)[1]

        for t in LLR_threshold(pop, cpool):
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
# plt.show()

print(f'AUC score:{auc}')

# plots!
fig, ax = plt.subplots()
ax.set_xscale("log")

ax.plot(num_miRNAs, auc, linewidth=2.0)
plt.xlabel("number MiRNAs")
plt.ylabel("ROC scores")
plt.show()
