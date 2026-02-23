import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from utils_datasets import load_dataset
from utils import auc_scores, Gaussian_noise

include_synthetic_noise = True

num_orders = 2000 # number of iterations to average over
auc_L1 = []
auc_LLR = []

if include_synthetic_noise:
    auc_syntheticL1 = []
    auc_syntheticLLR = []

# for loop for numorder lots of train/test, then average at end
for j in range (num_orders):
    print("Run Number", j)

    aucs_L1 = []
    aucs_LLR = []

    if include_synthetic_noise:
        aucs_syntheticL1 = []
        aucs_syntheticLLR = []

    # load new partitioned dataset each time we call num_orders
    pop_year_i, pool_year_i = load_dataset(electricity=True)

    # configuring the reference pop & pool to match the dataframe of a particular timepoint
    pop = pop_year_i[0]
    pool = pool_year_i[0]

    # the 'noise' increases throughout each of the later timepoints the data is collected from
    for yr_pop, yr_pool in zip(pop_year_i, pool_year_i):

        # get performance/accuracy for L1 & LLR statistics over the noisy stat inputs compared to the 'original' pop & pool
        roc_L1, pvalue_pop_L1, pvalue_pool_L1 = auc_scores(yr_pop, yr_pool, pop, pool)
        roc_LLR, pvalue_pop_LLR, pvalue_pool_LLR = auc_scores(yr_pop, yr_pool, pop, pool, LR=True)

        aucs_L1.append(roc_L1)
        aucs_LLR.append(roc_LLR)

        if include_synthetic_noise:
            # here we are adding noise to yrpop[0] that is normally distributed by the timestamp deviation
            # this should enable the same dataset split and also check for errors
            local_noised_pop = np.array(yr_pop)
            local_noised_pool = np.array(yr_pool)
            local_pop = np.array(pop)
            local_pool = np.array(pool)

            pop_diff = np.ravel(np.subtract(local_pop, local_noised_pop))
            pool_diff = np.ravel(np.subtract(local_pool, local_noised_pool))
            c = np.concatenate((pop_diff, pool_diff))
            m = np.std(c)

            # fixed Gaussian
            a1 = np.mean(local_noised_pop, axis=0) # mu_j of miRNAs for each pop timestamp
            a2 = np.mean(local_noised_pool, axis=0) # mu_j of miRNAs for each pool timestamp
            m1 = np.std(local_noised_pop, axis=0) # sigma_j of miRNAs for each pop timestamp
            m2 = np.std(local_noised_pool, axis=0) # sigma_j of miRNAs for each pool timestamp
            skew1 = stats.skew(local_noised_pop, axis=0) # skew_j of miRNAs for each pop timestamp
            skew2 = stats.skew(local_noised_pool, axis=0) # skew_j of miRNAs for each pool timestamp

            noisy_pop, noisy_pool = Gaussian_noise(local_pop, local_pool, 0, m1, clip=True, mean2=0, deviation2=m2)

            skew_test = stats.skewtest(c, axis=0)
            print(skew_test, skew1, skew2)

            # fig, ax = plt.subplots()
            # ax.plot(range(len(yr_pop)), yr_pop, "-b", linewidth=2.0, label="yr_pop")
            # plt.show()
            # fig, ax = plt.subplots()
            # ax.plot(range(len(noisy_pop)), noisy_pop, "-r", linewidth=2.0, label="synth_pop")
            # plt.show()
            # fig, ax = plt.subplots()
            # ax.plot(range(len(yr_pool)), yr_pool, "-b", linewidth=2.0, label="yr_pool")
            # plt.show()
            # fig, ax = plt.subplots()
            # ax.plot(range(len(noisy_pool)), noisy_pool, "-r", linewidth=2.0, label="synth_pool")
            # plt.show()

            # get performance/accuracy for L1 & LLR statistics over the noisy stat inputs compared to the 'original' pop & pool
            roc_synthL1, pvalue_synthpop_L1, pvalue_synthpool_L1 = auc_scores(noisy_pop, noisy_pool, pop, pool)
            roc_synthLLR, pvalue_synthpop_LLR, pvalue_synthpool_LLR = auc_scores(noisy_pop, noisy_pool, pop, pool, LR=True)

            aucs_syntheticL1.append(roc_synthL1)
            aucs_syntheticLLR.append(roc_synthLLR)

    # num_order rows of datasets, columns are each timestamp
    if len(aucs_L1) >0:
        auc_L1.append(aucs_L1)

    if len(aucs_LLR) >0:
        auc_LLR.append(aucs_LLR)

    if include_synthetic_noise:
        # num_order rows of datasets, columns are each timestamp
        if len(aucs_syntheticL1) >0:
            auc_syntheticL1.append(aucs_syntheticL1)

        if len(aucs_syntheticLLR) >0:
            auc_syntheticLLR.append(aucs_syntheticLLR)

# averaging the results from num_order iterations
auc_L1 = np.average(auc_L1, axis=0)
auc_LLR = np.average(auc_LLR, axis=0)

if include_synthetic_noise:
    # averaging the results from num_order iterations
    auc_syntheticL1 = np.average(auc_syntheticL1, axis=0)
    auc_syntheticLLR = np.average(auc_syntheticLLR, axis=0)

# plotting the performance of the inference for each of the 8 timestamps
fig, ax = plt.subplots()
ax.plot(range(len(pop_year_i)), auc_L1, "-b", linewidth=2.0, label="L1")
ax.plot(range(len(pool_year_i)), auc_LLR, "-r", linewidth=2.0, label="LLR")
if include_synthetic_noise:
    ax.plot(range(len(auc_syntheticL1)), auc_syntheticL1, "-g", linewidth=2.0, label="L1 synth")
    ax.plot(range(len(auc_syntheticLLR)), auc_syntheticLLR, "-y", linewidth=2.0, label="LLR synth")
ax.set_ylim([0.2,1.1]) # enables comparable auc scores between L1 and LLR

plt.xlabel("timestamp")
plt.ylabel("AUC scores")
plt.legend(loc="upper right")
plt.show()
