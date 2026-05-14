import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from plot_style import line_kwargs
from utils_datasets import load_timestamp_dataset, drop_timestamp_index
from utils import auc_scores

# load partitioned dataset
ti_pop, ti_pool, ti_sample = load_timestamp_dataset()
ti_pop, ti_pool = drop_timestamp_index(ti_pop, ti_pool)

# configuring the reference pop & pool to match the dataframe of a particular timepoint
pop = ti_pop[0]
pool = ti_pool[0]

# comparing L1 & LL1 for timepoint 1 to timepoint 0
roc_L1, pvalue_pop_L1, pvalue_pool_L1 = auc_scores(ti_pop[1], ti_pool[1], pop, pool)
roc_LLR, pvalue_pop_LLR, pvalue_pool_LLR = auc_scores(ti_pop[1], ti_pool[1], pop, pool, LR=True)

# determine the performance of the L1 attack comparing the accuracy of inference to the real data
y_true_L1 = np.concatenate((np.zeros(len(pvalue_pop_L1)), np.ones(len(pvalue_pool_L1))))
y_score_L1 = np.concatenate((pvalue_pop_L1, pvalue_pool_L1))
fpr_L1, tpr_L1, thresholds_L1 = roc_curve(y_true_L1, y_score_L1)

# determine the performance of the LLR attack comparing the accuracy of inference to the real data
y_true_LLR = np.concatenate((np.zeros(len(pvalue_pop_LLR)), np.ones(len(pvalue_pool_LLR))))
y_score_LLR = np.concatenate((pvalue_pop_LLR, pvalue_pool_LLR))
fpr_LLR, tpr_LLR, thresholds_LLR = roc_curve(y_true_LLR, y_score_LLR)


# plotting the ROC performance of the inference
fig, ax = plt.subplots()
ax.plot(fpr_L1, tpr_L1, label="L1 time 0-1",
        **line_kwargs("L1", marker=None, linewidth=2.0))
ax.plot(fpr_LLR, tpr_LLR, label="LLR time 0-1",
        **line_kwargs("LLR", marker=None, linewidth=2.0))
ax.set_ylim([0,1]) # enables comparable auc scores between L1 and LLR

plt.xlabel("fpr")
plt.ylabel("tpr")
plt.legend(loc="upper right")
plt.show()

# using the metrics ROC Display to plot
roc_auc_L1 = auc(fpr_L1, tpr_L1)
roc_auc_LLR = auc(fpr_LLR, tpr_LLR)

roc_curve_L1 = RocCurveDisplay(fpr=fpr_L1, tpr=tpr_L1, roc_auc=roc_auc_L1, estimator_name="L1 t0-t1 estimator")
roc_curve_L1.plot()
plt.show()

roc_curve_LLR = RocCurveDisplay(fpr=fpr_LLR, tpr=tpr_LLR, roc_auc=roc_auc_LLR, estimator_name="LLR t0-t1 estimator")
roc_curve_LLR.plot()
plt.show()
