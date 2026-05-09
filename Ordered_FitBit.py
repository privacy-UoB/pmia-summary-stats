import sys
import numpy as np
import matplotlib
if len(sys.argv) >= 2:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_score, confusion_matrix
from utils_datasets import _prepare_FitBit_per_id, _split_FitBit
from utils import auc_scores

# CLI: python Ordered_FitBit.py [output.pdf]
OUTPUT_FILE = sys.argv[1] if len(sys.argv) >= 2 else None

# Hoist FitBit CSV parse + per-id grouping out of the outer loop;
# only the pop/pool ShuffleSplit + per-timestamp aggregation re-runs per iteration.
unique_ids_data, column_names = _prepare_FitBit_per_id()

# pop, pool = load_FitBit_dataset()
# print("fitbit pop", pop[1], "fitbit pool", pool[3])
# Check that the increasing dates do actually correspond to the same person ID? And should we do them based on how far the dates are apart?!

error_bands = False
fixed_FPR = True
num_orders = 5000 # number of averages
auc_L1 = []
auc_LLR = []
if fixed_FPR == True:
    target_fpr = 1e-2

    tpr_at_fpr_L1 = []
    tpr_at_fpr_LLR = []
# roc_curve_L1 = []
# roc_curve_LLR = []

# for loop for numorder lots of train/test, then average at end
for j in range (num_orders):

    aucs_L1 = []
    aucs_LLR = []
    if fixed_FPR == True:
        tpr_at_fprs_L1 = []
        tpr_at_fprs_LLR = []
    # roc_curves_L1 = []
    # roc_curves_LLR = []

    # randomise the pop/pool split per iteration; CSV/grouping is reused from the prepare step above
    population, random_pool = _split_FitBit(unique_ids_data, column_names)

    pop = population[0]
    pool = random_pool[0]

    # iterations = max(len(pop), len(pool))
    iterations = 8 # not everyone recorded more than 8 timestamps of data
    for i in range(iterations):

        local_noised_pop = population[i]
        local_noised_pool = random_pool[i]

        roc_L1, pvalue_pop_L1, pvalue_pool_L1 = auc_scores(local_noised_pop, local_noised_pool, pop, pool)
        # check first if feature are mostly 0 entries then delete
            # col "LoggedActivitiesDistance", 433 of 457 values are 0.
            # col "SedentaryActiveDistance", 419 of 457 values are 0.
        # otherwise try the above exception to split with 0 var
        # try:
            # something_that_fails
                # except ValueError:
            # continue
        try:
            roc_LLR, pvalue_pop_LLR, pvalue_pool_LLR = auc_scores(local_noised_pop, local_noised_pool, pop, pool, LR=True)
        except ValueError:
            continue

        aucs_L1.append(roc_L1)
        aucs_LLR.append(roc_LLR)

        if fixed_FPR == True:
            fpr_L1, tpr_L1, thresholds_L1 = auc_scores(local_noised_pop, local_noised_pool, pop, pool, FPR=True)
            fpr_LLR, tpr_LLR, thresholds_LLR = auc_scores(local_noised_pop, local_noised_pool, pop, pool, LR=True, FPR=True)

            # TPR at a fixed FPR (e.g., 0.01 = 1%)
            tpr_at_fprs_L1.append(np.interp(target_fpr, fpr_L1, tpr_L1))
            tpr_at_fprs_LLR.append(np.interp(target_fpr, fpr_LLR, tpr_LLR))

        y_true_L1 = np.concatenate((np.zeros(len(pvalue_pop_L1)), np.ones(len(pvalue_pool_L1))))
        y_score_L1 = np.concatenate((pvalue_pop_L1, pvalue_pool_L1))

        # tn, fp, fn, tp = confusion_matrix(y_true_L1, y_score_L1)
        # ap = fp+tp
        # fpr_L1 = fp/(tn+fp)
        # tpr = tp/(tp+fn)
        # precision_L1 = tp/(tp+fp)
        # precision_L1 = (ap-fp)/ap
        # precision_L1_score = precision_score(y_true_L1, y_score_L1)
        # curve_L1 = roc_curve(y_true_L1, y_score_L1)
        # roc_curves_L1.append(curve_L1)

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
        
    if fixed_FPR == True:
        if len(tpr_at_fprs_L1) >0:
            tpr_at_fpr_L1.append(tpr_at_fprs_L1)

        if len(tpr_at_fprs_LLR) >0:
            tpr_at_fpr_LLR.append(tpr_at_fprs_LLR)

# num_order rows of datasets, columns are each timestamp
if error_bands == False:
    auc_L1 = np.average(auc_L1, axis=0)
    auc_LLR = np.average(auc_LLR, axis=0)

    if fixed_FPR == True:
        tpr_at_fpr_L1 = np.average(tpr_at_fpr_L1, axis=0)
        tpr_at_fpr_LLR = np.average(tpr_at_fpr_LLR, axis=0)

else:
    auc_L1_error_bands = [np.average(auc_L1, axis=0), 
                          np.min(auc_L1, axis=0), 
                          np.max(auc_L1, axis=0)]
    auc_LLR_error_bands = [np.average(auc_LLR, axis=0), 
                          np.min(auc_LLR, axis=0), 
                          np.max(auc_LLR, axis=0)]
    
    if fixed_FPR == True:
        tpr_L1_error_bands = [np.average(tpr_at_fpr_L1, axis=0), 
                          np.min(tpr_at_fpr_L1, axis=0), 
                          np.max(tpr_at_fpr_L1, axis=0)]
        tpr_LLR_error_bands = [np.average(tpr_at_fpr_LLR, axis=0), 
                          np.min(tpr_at_fpr_LLR, axis=0), 
                          np.max(tpr_at_fpr_LLR, axis=0)]

# plots!
fig, ax1 = plt.subplots()
colours1 = ["cornflowerblue", "gold"]
if fixed_FPR == True:
    ax2 = ax1.twinx()
    colours2 = ["mediumblue", "orange"]

if error_bands == False:
    ax1.plot(range(iterations), auc_L1, colours1[0], linewidth=2.0, label="AUC L1")
    ax1.plot(range(iterations), auc_LLR, colours1[1], linewidth=2.0, label="AUC LLR")

    if fixed_FPR == True:
        ax2.plot(range(iterations), tpr_at_fpr_L1, colours2[0], linewidth=2.0, label="fpr L1")
        ax2.plot(range(iterations), tpr_at_fpr_LLR, colours2[1], linewidth=2.0, label="fpr LLR")

else:
    ax1.plot(range(iterations), auc_L1_error_bands[0], colours1[0], linewidth=2.0, label="L1")
    ax1.fill_between(range(iterations), auc_L1_error_bands[1], auc_L1_error_bands[2], alpha=0.2)
    ax1.plot(range(iterations), auc_LLR_error_bands[0], colours1[1], linewidth=2.0, label="LLR")
    ax1.fill_between(range(iterations), auc_LLR_error_bands[1], auc_LLR_error_bands[2], alpha=0.2)

    if fixed_FPR == True:
        ax2.plot(range(iterations), tpr_L1_error_bands[0], colours2[0], linewidth=2.0, label="L1")
        ax2.fill_between(range(iterations), tpr_L1_error_bands[1], tpr_L1_error_bands[2], alpha=0.2)
        ax2.plot(range(iterations), tpr_LLR_error_bands[0], colours2[1], linewidth=2.0, label="LLR")
        ax2.fill_between(range(iterations), tpr_LLR_error_bands[1], tpr_LLR_error_bands[2], alpha=0.2)

ax1.legend(loc='upper right')
if fixed_FPR == True:
    # Merge handles and labels
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    # Add combined legend to one axis
    ax1.legend(h1 + h2, l1 + l2, loc='upper right')
    ax2.set_ylabel("TPR at 0.01 FPR")

ax1.set_xlabel("timestamp")
ax1.set_ylabel("AUC scores")
ax1.set_ylim([0.2,1.1]) # enables comparable auc scores between L1 and LLR
ax1.grid(True)

if OUTPUT_FILE:
    plt.savefig(OUTPUT_FILE)
    print(f"Saved to {OUTPUT_FILE}")
else:
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
