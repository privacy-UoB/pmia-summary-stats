import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from utils import ground_truth, load_dataset, L1_threshold, L1_ttest, LLR, D3

# load dataset
pop_rpool, pop_cpool, rpool, cpool = load_dataset(case_sample=D3)

pop_rpool = pop_rpool.drop(columns="diseases")
pop_cpool = pop_cpool.drop(columns="diseases")
pop = pop_cpool # make pop configurable

rpool = rpool.drop(columns="diseases")
cpool = cpool.drop(columns="diseases")
pool = cpool # make pool configurable

mu = np.average(pop)
mu_j = np.average(pop, axis=0)
mu_hat = np.average(pool)
mu_hat_j = np.average(pool, axis=0)
print(mu, mu_j, mu_hat, mu_hat_j)

sigma = np.std(pop)
sigma_j = np.std(pop, axis=0) #this is doing it over all the columns (miRNAs)
sigma_hat = np.std(pool)
sigma_hat_j = np.std(pool, axis=0)
print(sigma, sigma_j, sigma_hat, sigma_hat_j)

x = pop.sample(20, axis=1)
print("pop: max", np.max(x, axis=0), 
      "min", np.min(x, axis=0), 
      "mean", np.average(x, axis=0), 
      "deviation", np.std(x, axis=0))

y = pool.sample(20, axis=1)
print("pool: max", np.max(y, axis=0), 
      "min", np.min(y, axis=0), 
      "mean", np.average(y, axis=0), 
      "deviation", np.std(y, axis=0))

plt.hist(x, bins=40)
plt.xlabel("sample of 20 miRNAs from population")
plt.ylabel("count of miRNAs within 40 different range values")
plt.show()

plt.hist(y, bins=40)
plt.xlabel("sample of 20 miRNAs from pool")
plt.ylabel("count of miRNAs within 40 different range values")
plt.show()

# hist over all sigma j for all mirna
plt.hist(sigma_j, bins=100)
plt.xlabel("standard deviation of miRNAs")
plt.ylabel("number of the 466 miRNAs in each bar")
plt.show()


auc_L1 = []
auc_LLR = []
# fractions of standard deviation applied to the dataset
# multiplier = [0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
multiplier = np.logspace(0, 15, base=2) # used for standard noise values
# multiplier = np.arange(0, 8, 0.04) # sufficient for m * sigma_j

for m in multiplier:
    aucs_L1 = []
    aucs_LLR = []
    for j in range (30):
        pop_noise = np.random.normal(0, m, pop.shape) #make this A LOT bigger, then plot on np.logspace scale
        pool_noise = np.random.normal(0, m, pool.shape)
        # pop_noise = np.random.normal(0, m * sigma_j, pop.shape)
        # pool_noise = np.random.normal(0, m * sigma_j, pool.shape)
        # still need to double check random.normal when inputting array of std dev

        noised_pop = pop + pop_noise
        nonneg_pop_noise = np.clip(noised_pop, 0, None)

        noised_pool = pool + pool_noise
        nonneg_pool_noise = np.clip(noised_pool, 0, None)

        pvalue_pop_L1 = L1_ttest(nonneg_pop_noise, pop, pool)
        pvalue_pool_L1 = L1_ttest(nonneg_pool_noise, pop, pool)

        pvalue_pop_LLR = LLR(nonneg_pop_noise, pop, pool)
        pvalue_pool_LLR = LLR(nonneg_pool_noise, pop, pool)

        # print(L1_threshold(pop, pool, nonneg_pop_noise, nonneg_pool_noise))

        # power = []
        # fpr = []
        # for t in L1_threshold(pop, pool, nonneg_pop_noise, nonneg_pool_noise):
        #     p, f = ground_truth(pvalue_pop, pvalue_cpool, t)
        #     power.append(p)
        #     fpr.append(f)
        # fpr = np.array(fpr)
        # power = np.array(power)

        # order = np.argsort(fpr)
        # fpr = fpr[order]
        # power = power[order]

        y_true_L1 = np.concatenate((np.zeros(len(pvalue_pop_L1)), np.ones(len(pvalue_pool_L1))))
        y_score_L1 = np.concatenate((pvalue_pop_L1, pvalue_pool_L1))
        roc = roc_auc_score(y_true_L1, y_score_L1)

        aucs_L1.append(roc)

        y_true_LLR = np.concatenate((np.zeros(len(pvalue_pop_LLR)), np.ones(len(pvalue_pool_LLR))))
        y_score_LLR = np.concatenate((pvalue_pop_LLR, pvalue_pool_LLR))
        roc_LLR = roc_auc_score(y_true_LLR, y_score_LLR)
        
        aucs_LLR.append(roc_LLR)

    if len(aucs_L1) >0:
        auc_L1.append(np.average(aucs_L1))

    if len(aucs_LLR) >0:
        auc_LLR.append(np.average(aucs_LLR))


# print(f'AUC score:{auc_L1}')
# print(f'AUC score:{auc_LLR}')

# plots!
fig, ax = plt.subplots()
# ax.set_xscale("log")

# ax.plot(noise_scales_pop, auc_L1, "-b", linewidth=2.0, label="L1")
# ax.plot(noise_scales_pool, auc_LLR, "-r", linewidth=2.0, label="LLR")
ax.plot(multiplier, auc_L1, "-b", linewidth=2.0, label="L1")
ax.plot(multiplier, auc_LLR, "-r", linewidth=2.0, label="LLR")
plt.xlabel("noise scale")
plt.ylabel("AUC scores")
plt.legend(loc="upper right")
plt.show()
