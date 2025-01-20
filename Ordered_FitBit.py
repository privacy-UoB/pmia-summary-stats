import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from utils_datasets import load_FitBit_dataset
from utils import auc_scores

# pop, pool = load_FitBit_dataset()
# print("fitbit pop", pop[1], "fitbit pool", pool[3])
# NOTE: the dataset will not work because the dates are included in the L1/LLR calculations!
# Check that the increasing dates do actually correspond to the same person ID? And should we do them based on how far the dates are apart?!

num_orders = 50 # number of averages
auc_L1 = []
auc_LLR = []
# roc_curve_L1 = []
# roc_curve_LLR = []

# for loop for numorder lots of train/test, then average at end
for j in range (num_orders):

    aucs_L1 = []
    aucs_LLR = []
    # roc_curves_L1 = []
    # roc_curves_LLR = []

    # load dataset
    population, random_pool = load_FitBit_dataset()

    pop = population[0]
    pool = random_pool[0]

    iterations = max(len(pop), len(pool))
    for i in range(iterations):

        local_noised_pop = population[i]
        local_noised_pool = random_pool[i]        

        roc_L1, pvalue_pop_L1, pvalue_pool_L1 = auc_scores(local_noised_pop, local_noised_pool, pop, pool)
        roc_LLR, pvalue_pop_LLR, pvalue_pool_LLR = auc_scores(local_noised_pop, local_noised_pool, pop, pool, LR=True)

        aucs_L1.append(roc_L1)
        # y_true_L1 = np.concatenate((np.zeros(len(pvalue_pop_L1)), np.ones(len(pvalue_pool_L1))))
        # y_score_L1 = np.concatenate((pvalue_pop_L1, pvalue_pool_L1))
        # curve_L1 = roc_curve(y_true_L1, y_score_L1)
        # roc_curves_L1.append(curve_L1)

        aucs_LLR.append(roc_LLR)
        # y_true_LLR = np.concatenate((np.zeros(len(pvalue_pop_LLR)), np.ones(len(pvalue_pool_LLR))))
        # y_score_LLR = np.concatenate((pvalue_pop_LLR, pvalue_pool_LLR))
        # curve_LLR = roc_curve(y_true_LLR, y_score_LLR)
        # roc_curves_LLR.append(curve_LLR)

    if len(aucs_L1) >0:
        auc_L1.append(aucs_L1)
    # if len(roc_curves_L1) >0:
    #     roc_curve_L1.append(roc_curves_L1)

    if len(aucs_LLR) >0:
        auc_LLR.append(aucs_LLR)
    # if len(roc_curves_LLR) >0:
    #     roc_curve_LLR.append(roc_curves_LLR)

# num_order rows of datasets, columns are each timestamp
auc_L1 = np.average(auc_L1, axis=0)
auc_LLR = np.average(auc_LLR, axis=0)
# roc_curve_L1 = np.average(roc_curve_L1, axis=0)
# roc_curve_LLR = np.average(roc_curve_LLR, axis=0)


# plots!
fig, ax = plt.subplots()
ax.plot([range(iterations)], auc_L1, "-b", linewidth=2.0, label="L1")
ax.plot([range(iterations)], auc_LLR, "-r", linewidth=2.0, label="LLR")
ax.set_ylim([0,1]) # enables comparable auc scores between L1 and LLR

plt.xlabel("timestamp")
plt.ylabel("AUC scores")
plt.legend(loc="upper right")
plt.show()

# # plots!
# fig, ax = plt.subplots()
# ax.plot([range(iterations)], roc_curve_L1, "-b", linewidth=2.0, label="L1")
# ax.plot([range(iterations)], roc_curve_LLR, "-r", linewidth=2.0, label="LLR")
# ax.set_ylim([0,1]) # enables comparable auc scores between L1 and LLR

# plt.xlabel("timestamp")
# plt.ylabel("ROC scores")
# plt.legend(loc="upper right")
# plt.show()
