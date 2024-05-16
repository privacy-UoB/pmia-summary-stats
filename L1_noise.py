import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from utils import ground_truth, load_dataset, L1_threshold, L1_ttest

noise_scales = [0.1, 1, 2, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300, 400, 500, 750]

pop, rpool, cpool = load_dataset()
pop = pop.drop(columns="diseases")
rpool = rpool.drop(columns="diseases")
cpool = cpool.drop(columns="diseases")

# Make pool configurable
pool = cpool
auc = []

for scale in noise_scales:
    aucs = []
    for j in range (5):
        pool_noise = np.random.normal(0, scale, pool.shape)
        pop_noise = np.random.normal(0, scale, pop.shape)

        noised_pool = pool + pool_noise
        noised_pop = pop + pop_noise
        
        test = L1_threshold(pop, pool, noised_pop, noised_pool)
        print (test)

        power = []
        fpr = []
        pvalue_pop = L1_ttest(noised_pop, pop, pool)[1]
        pvalue_cpool = L1_ttest(noised_pool, pop, pool)[1]

        for t in L1_threshold(pop, pool, noised_pop, noised_pool):
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

print(f'AUC score:{auc}')

# plots!
fig, ax = plt.subplots()
ax.set_xscale("log")

ax.plot(noise_scales, auc, linewidth=2.0)
plt.xlabel("noise scale")
plt.ylabel("ROC scores")
plt.show()
