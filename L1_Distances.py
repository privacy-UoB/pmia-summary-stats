import numpy as np
import matplotlib.pyplot as plt
from utils_datasets import load_dataset, drop_dataset_index
from utils import auc_scores, L1, L1_threshold, fpr_power

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

        rpop, cpop, rpool, cpool = ret
        # pop = [pop(50), pop(100), ..., pop(1000)]
        # rpool = [rpool(50), rpool(100), ..., rpool(1000)]
        # cpool = [cpool(50), cpool(100), ..., cpool(1000)]
        # -> zip[(pop(50),rpool(50),cpool(50)), (pop(100), rpool(100), cpool(100)), ..., (pop(1000), rpool(1000), cpool(1000))]

        pop_rpool, pop_cpool, rpool, cpool = drop_dataset_index(rpop, cpop, rpool, cpool)
        pop = pop_cpool # make pop configurable
        pool = cpool # make pool configurable

        # victim = pool.iloc[99]
        # print(L1(victim, pop, pool).sum())
        # print(L1_threshold(pop, pool))

        roc, pvalue_pop, pvalue_pool = auc_scores(pop, pool, pop, pool)
        # fpr, power = fpr_power(pop, pool, pvalue_pop, pvalue_pool)
        aucs.append(roc) # auc = average rocs

    if len(aucs) >0:
        auc.append(np.average(aucs))

# plots!
# fig, ax = plt.subplots()
# ax.set_xscale("log")        
# ax.plot(fpr, power, linewidth=2.0)
# plt.xlabel("fpr")
# plt.ylabel("power")
# plt.show()

# print(f'AUC score:{auc}')

# plots!
fig, ax = plt.subplots()
# ax.set_xscale("log")
ax.plot(num_miRNAs, auc, linewidth=2.0)
ax.invert_xaxis()
plt.xlabel("number MiRNAs")
plt.ylabel("ROC scores")
plt.show()
