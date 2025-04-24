import numpy as np
import matplotlib.pyplot as plt
from utils_datasets import load_dataset, D3, drop_dataset_index
from utils import auc_scores, Gaussian_noise, fpr_power, L1_threshold

# load dataset
pop_rpool, pop_cpool, rpool, cpool = load_dataset(case_sample=D3)
pop_rpool, pop_cpool, rpool, cpool = drop_dataset_index(pop_rpool, pop_cpool, rpool, cpool)

pop = pop_rpool # make pop configurable
pool = rpool # make pool configurable

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

    for j in range (2000):
        # make deviation A LOT bigger, then plot on np.logspace scale
        deviation = m # not tailored variance to each miRNA; otherwise replace m with m * sigma_j
        nonneg_noisy_pop, nonneg_noisy_pool = Gaussian_noise(pop, pool, 0, deviation, clip=True)

        roc_L1, pvalue_pop_L1, pvalue_pool_L1 = auc_scores(nonneg_noisy_pop, nonneg_noisy_pool, pop, pool)
        roc_LLR, pvalue_pop_LLR, pvalue_pool_LLR = auc_scores(nonneg_noisy_pop, nonneg_noisy_pool, pop, pool, LR=True)

        # fpr, power = fpr_power(pop, pool, pvalue_pop_L1, pvalue_pool_L1, victim_pop=nonneg_noisy_pop, victim_pool=nonneg_noisy_pool)
        # print(L1_threshold(pop, pool, nonneg_noisy_pop, nonneg_noisy_pool))

        aucs_L1.append(roc_L1)
        aucs_LLR.append(roc_LLR)

    if len(aucs_L1) >0:
        auc_L1.append(np.average(aucs_L1))

    if len(aucs_LLR) >0:
        auc_LLR.append(np.average(aucs_LLR))

# print(f'AUC score:{auc_L1}')
# print(f'AUC score:{auc_LLR}')

# plots!
fig, ax = plt.subplots()
ax.set_xscale("log")
# ax.plot(noise_scales_pop, auc_L1, "-b", linewidth=2.0, label="L1")
# ax.plot(noise_scales_pool, auc_LLR, "-r", linewidth=2.0, label="LLR")
ax.plot(multiplier, auc_L1, "-b", linewidth=2.0, label="L1")
ax.plot(multiplier, auc_LLR, "-r", linewidth=2.0, label="LLR")
ax.set_ylim([0.45,1]) # enables comparable auc scores between L1 and LLR
plt.xlabel("noise scale")
plt.ylabel("AUC scores")
plt.legend(loc="upper right")
plt.show()
