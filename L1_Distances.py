import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from utils import L1, ground_truth, load_dataset, threshold, ttest

# L1 Distances Difference
# D(x_j^v) = |x_j^v - mu_j| - |x_j^v - mu-hat_j|
# x_j^v is the value of miRNA j for the individual victim in column v
# mu_j is the average miRNA j in the population
# mu-hat_j is the average miRNA j in the pool



def varying_MiRNA ():
    pop = []
    rpool = []
    cpool = []
    num_miRNAs = []
    for i in range(10,1000,10):
        ret = load_dataset(i)
        if ret is None:
            break
        a,b,c = ret
        pop.append(a)
        rpool.append(b)
        cpool.append(c)
        num_miRNAs.append(i)
    return pop, rpool, cpool, num_miRNAs
pop, rpool, cpool, num_miRNAs = varying_MiRNA()

auc = []

for pop, rpool, cpool in zip(pop, rpool, cpool):
    # pop = [pop(50), pop(100), ..., pop(1000)]
    # rpool = [rpool(50), rpool(100), ..., rpool(1000)]
    # cpool = [cpool(50), cpool(100), ..., cpool(1000)]
    # -> zip[(pop(50),rpool(50),cpool(50)), (pop(100), rpool(100), cpool(100)), ..., (pop(1000), rpool(1000), cpool(1000))]

    pop = pop.drop(columns="diseases")
    rpool = rpool.drop(columns="diseases")
    cpool = cpool.drop(columns="diseases")
    victim = cpool.iloc[99]

    print (L1(victim, pop, cpool).sum())


    
    test = threshold(pop, cpool)
    print (test)

    power = []
    fpr = []
    pvalue_pop = ttest(pop, pop, cpool)[1]
    pvalue_cpool = ttest(cpool, pop, cpool)[1]

    for t in threshold(pop, cpool):
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

    auc.append(roc)

# plots!
# fig, ax = plt.subplots()
# ax.set_xscale("log")

# ax.plot(fpr, power, linewidth=2.0)
# plt.xlabel("fpr")
# plt.ylabel("power")

# plt.show()

# def auc():
#     for i in range(num_miRNA):
#         auc = []

#         y_true = np.concatenate((np.zeros(len(pvalue_pop[i])), np.ones(len(pvalue_cpool[i]))))
#         y_score = np.concatenate((pvalue_pop[i], pvalue_cpool[i]))
#         roc = roc_auc_score(y_true, y_score)

#         auc.append(roc)
#     return auc

print(f'AUC score:{auc}')

# plots!
fig, ax = plt.subplots()
ax.set_xscale("log")

ax.plot(num_miRNAs, auc, linewidth=2.0)
plt.xlabel("number MiRNAs")
plt.ylabel("ROC scores")
plt.show()

# Todo: average the plots over multiple runs (5) so the curve is smoother

# ipython3 in terminal - use when zsh error
# may 15th network

# t test
# note different threshold based on victims - check the results and pass/fail rate for t
# check over all victims, reproduce graphs in paper


# git remote add origin git@github.com:privacy-UoB/pmia-summary-stats.git
# git branch -M main
# git push -u origin main

# friday 1pm
# For next time, implement this into the LLR
# Plot the area under curve for the ROC vs the variances of the number of MiRNAs (so no longer >49 only)
# Closer to 1, the better the performance of the attack
# This will be like adding noise

# Tuesday 1-4pm in Bham! Meet at Pascal's office