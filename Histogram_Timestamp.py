import matplotlib.pyplot as plt
from utils_datasets import load_timestamp_dataset, drop_timestamp_index
from utils import auc_scores

# load dataset
ti_pop, ti_pool, ti_sample = load_timestamp_dataset()
ti_pop, ti_pool = drop_timestamp_index(ti_pop, ti_pool)

pop = ti_pop[0] # make pop configurable
pool = ti_pool[0] # make pool configurable

p_values_pop_L1 = []
p_values_pool_L1 = []
p_values_pop_LLR = []
p_values_pool_LLR = []

# the 'noise' increases throughout each of the later timepoints the data is collected from
for t_pop, t_pool in zip(ti_pop, ti_pool):
    roc_L1, pvalue_pop_L1, pvalue_pool_L1 = auc_scores(t_pop, t_pool, pop, pool)
    roc_LLR, pvalue_pop_LLR, pvalue_pool_LLR = auc_scores(t_pop, t_pool, pop, pool, LR=True)

    p_values_pop_L1.append(pvalue_pop_L1)
    p_values_pool_L1.append(pvalue_pool_L1)
    p_values_pop_LLR.append((pvalue_pop_LLR.ravel()))
    p_values_pool_LLR.append((pvalue_pool_LLR.ravel()))

# histogram showing standard deviations across all 8 timestamps of the individual
for m in range(len(ti_pop)):
    # L1
    plt.hist(p_values_pop_L1[m], bins=40, label=f"timestamp {m}")
    plt.xlabel("p values population L1")
    plt.ylabel("count of deviations across 40 different range values")
    plt.legend(loc="upper right")
    plt.show()

    plt.hist(p_values_pool_L1[m], bins=40, label=f"timestamp {m}")
    plt.xlabel("p values pool L1")
    plt.ylabel("count of deviations across 40 different range values")
    plt.legend(loc="upper right")
    plt.show()

    plt.hist((p_values_pop_L1[m], p_values_pool_L1[m]), bins=40, label=f"timestamp {m}")
    plt.xlabel("p values pop & pool L1")
    plt.ylabel("count of deviations across 40 different range values")
    plt.legend(loc="upper right")
    plt.show()

    # LLR
    plt.hist(p_values_pop_LLR[m], bins=40, label=f"timestamp {m}")
    plt.xlabel("p values population LLR")
    plt.ylabel("count of deviations across 40 different range values")
    plt.legend(loc="upper right")
    plt.show()

    plt.hist(p_values_pool_LLR[m], bins=40, label=f"timestamp {m}")
    plt.xlabel("p values pool LLR")
    plt.ylabel("count of deviations across 40 different range values")
    plt.legend(loc="upper right")
    plt.show()

    plt.hist((p_values_pop_LLR[m], p_values_pool_LLR[m]), bins=40, label=f"timestamp {m}")
    plt.xlabel("p values pop & pool LLR")
    plt.ylabel("count of deviations across 40 different range values")
    plt.legend(loc="upper right")
    plt.show()
