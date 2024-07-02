import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from utils import load_dataset, LLR, LLR_threshold, ground_truth

auc = []
num_miRNAs = []

for i in range(10,470,10):
    aucs = []
    num_miRNAs.append(i)
    for j in range (5): # 5 is arbitrary number of runs to average the auc curve
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
        pool = cpool # make pool configurable
        victim = pool.iloc[10]

        # print(LLR(victim, pop, pool).sum()) # only works if LLR sum axis=0??
        # print(LLR_threshold(pop, pool))

        pvalue_pop = LLR(pop, pop, pool)
        pvalue_pool = LLR(pool, pop, pool)

        # power = []
        # fpr = []
        # for t in LLR_threshold(pop, pool):
        #     p, f = ground_truth(pvalue_pop, pvalue_pool, t)
        #     power.append(p)
        #     fpr.append(f)
        # fpr = np.array(fpr)
        # power = np.array(power)

        # order = np.argsort(fpr)
        # fpr = fpr[order]
        # power = power[order]

        y_true = np.concatenate((np.zeros(len(pvalue_pop)), np.ones(len(pvalue_pool))))
        y_score = np.concatenate((pvalue_pop, pvalue_pool))
        roc = roc_auc_score(y_true, y_score)

        aucs.append(roc)
    if len(aucs) >0:
        auc.append(np.average(aucs))

# plots!
# fig, ax = plt.subplots()
# ax.set_xscale("log")

# ax.plot(fpr, power, linewidth=2.0)
# plt.show()


# print(f'AUC score:{auc}')

# plots!
fig, ax = plt.subplots()
# ax.set_xscale("log")

ax.plot(num_miRNAs, auc, linewidth=2.0)
plt.xlabel("number MiRNAs")
plt.ylabel("ROC scores")
plt.show()
