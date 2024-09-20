import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_auc_score
from utils import load_FitBit_dataset, LLR, L1, L1_ttest

# pop, pool = load_FitBit_dataset()
# print("fitbit pop", pop[1], "fitbit pool", pool[3])
# NOTE: the dataset will not work because the dates are included in the L1/LLR calculations!
# Check that the increasing dates do actually correspond to the same person ID? And should we do them based on how far the dates are apart?!

num_orders = 50 # number of averages
auc_L1 = []
auc_LLR = []

# for loop for numorder lots of train/test, then average at end
for j in range (num_orders):

    aucs_L1 = []
    aucs_LLR = []

    # load dataset
    population, random_pool = load_FitBit_dataset()

    pop = population[0]
    pool = random_pool[0]

    iterations = max(len(pop), len(pool))


    for i in range(iterations):

        local_noised_pop = population[i]
        local_noised_pool = random_pool[i]
        local_pop = pop
        local_pool = pool


        pvalue_pop_L1 = L1_ttest(local_noised_pop, local_pop, local_pool)
        pvalue_pool_L1 = L1_ttest(local_noised_pool, local_pop, local_pool)
    
        pvalue_pop_LLR = LLR(local_noised_pop, local_pop, local_pool)
        pvalue_pool_LLR = LLR(local_noised_pool, local_pop, local_pool)


        y_true_L1 = np.concatenate((np.zeros(len(pvalue_pop_L1)), np.ones(len(pvalue_pool_L1))))
        y_score_L1 = np.concatenate((pvalue_pop_L1, pvalue_pool_L1))
        roc_L1 = roc_auc_score(y_true_L1, y_score_L1)
        aucs_L1.append(roc_L1)

        y_true_LLR = np.concatenate((np.zeros(len(pvalue_pop_LLR)), np.ones(len(pvalue_pool_LLR))))
        y_score_LLR = np.concatenate((pvalue_pop_LLR, pvalue_pool_LLR))
        roc_LLR = roc_auc_score(y_true_LLR, y_score_LLR)
        aucs_LLR.append(roc_LLR)


    if len(aucs_L1) >0:
        auc_L1.append(aucs_L1)

    if len(aucs_LLR) >0:
        auc_LLR.append(aucs_LLR)

# num_order rows of datasets, columns are each timestamp
auc_L1 = np.average(auc_L1, axis=0)
auc_LLR = np.average(auc_LLR, axis=0)

# print(f'AUC score:{auc_L1}')
# print(f'AUC score:{auc_LLR}')


# plots!
fig, ax = plt.subplots()
ax.plot([range(iterations)], auc_L1, "-b", linewidth=2.0, label="L1")
ax.plot([range(iterations)], auc_LLR, "-r", linewidth=2.0, label="LLR")
ax.set_ylim([0,1]) # enables comparable auc scores between L1 and LLR

plt.xlabel("timestamp")
plt.ylabel("AUC scores")
plt.legend(loc="upper right")
plt.show()
