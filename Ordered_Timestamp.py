import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from utils import load_timestamp_dataset, LLR, L1, L1_ttest
from scipy.spatial import distance

# load dataset
(pop_timestamps_graph, pool_timestamps_graph, sample_timestamps_graph) = load_timestamp_dataset()

for i in sample_timestamps_graph:
    i.drop(["disease", "timepoint", "patient_id"], axis=1, inplace=True)

# cosine similarity better than euclidean distance
cosine_distances = []
for u in sample_timestamps_graph:
    timepoint_comparison = u

    cosine_distance = []
    for t in sample_timestamps_graph:
        if (t.shape == (1,1205)) & (timepoint_comparison.shape == (1,1205)):
            d = distance.cosine((timepoint_comparison.to_numpy()).ravel(), (t.to_numpy()).ravel())
            cosine_distance.append(d)
        else:
            cosine_distance.append(0) #some impossible value to fill in the place of an empty value and keep the timestamps aligned
    cosine_distances.append(cosine_distance)

# plot cosine distance
fig, ax = plt.subplots()
for l in range(len(cosine_distances)):
    ax.plot(range(len(cosine_distances)), cosine_distances[l], linewidth=3.0, label=f"timepoint {l}")
plt.xlabel("timestamp")
plt.ylabel("cosine distance")
plt.legend(loc="upper right")
plt.show()

# some statistical plots for one individual of the dataset
sample_timestamps_graph_deviation = []
for t in sample_timestamps_graph:

    sample_graph = t
    mu_j = np.average(sample_graph, axis=0)
    sigma_j = np.std(sample_graph, axis=0)
    sample_timestamps_graph_deviation.append(sigma_j)

    # histogram of 100 miRNAs from the individual
    x = sample_graph.sample(100, axis=1)
    plt.hist(x, bins=10)
    plt.xlabel("sample of 100 miRNAs from population")
    plt.ylabel("count of miRNAs within 10 different range values")
    plt.show()

    # histogram over all sigma j for all mirna
    plt.hist(sigma_j, bins=300)
    plt.xlabel("standard deviation of miRNAs")
    plt.ylabel("number of the 1208 miRNAs in each bar")
    plt.show()

    # histogram over all sigma j / mu j for all mirna
    if mu_j.all != 0:
        plt.hist(np.divide(sigma_j,mu_j), bins=100)
        plt.xlabel("standard deviation of miRNAs divided by mean")
        plt.ylabel("number in each bar")
        plt.show()

# histogram showing standard deviations across all 8 timestamps of the individual
plt.hist(sample_timestamps_graph_deviation, bins=40)
plt.xlabel("8 timestamps of standard deviation of miRNAs from individual")
plt.ylabel("count of deviations across 40 different range values")
plt.show()


num_orders = 50 # number of iterations to average over
auc_L1 = []
auc_LLR = []

# for loop for numorder lots of train/test, then average at end
for j in range (num_orders):

    aucs_L1 = []
    aucs_LLR = []

    # load new partitioned dataset each time we call num_orders
    (ti_pop, ti_pool, ti_sample) = load_timestamp_dataset()

    for x, y in zip(ti_pop, ti_pool):
        x.drop(["disease", "timepoint", "patient_id"], axis=1, inplace=True)
        y.drop(["disease", "timepoint", "patient_id"], axis=1, inplace=True)

    # configuring the reference pop & pool to match the dataframe of a particular timepoint
    pop = ti_pop[0]
    pool = ti_pool[0]

    for t_pop, t_pool in zip(ti_pop, ti_pool):

        # the 'noise' increases throughout each of the later timepoints the data is collected from
        local_noised_pop = t_pop
        local_noised_pool = t_pool
        local_pop = pop
        local_pool = pool

        # get values for L1 & LLR statistics over the noisy stat inputs compared to the 'original' pop & pool
        pvalue_pop_L1 = L1_ttest(local_noised_pop, local_pop, local_pool)
        pvalue_pool_L1 = L1_ttest(local_noised_pool, local_pop, local_pool)
    
        pvalue_pop_LLR = LLR(local_noised_pop, local_pop, local_pool)
        pvalue_pool_LLR = LLR(local_noised_pool, local_pop, local_pool)

        # determine the performance of the attack comparing the accuracy of inference to the real data
        y_true_L1 = np.concatenate((np.zeros(len(pvalue_pop_L1)), np.ones(len(pvalue_pool_L1))))
        y_score_L1 = np.concatenate((pvalue_pop_L1, pvalue_pool_L1))
        roc_L1 = roc_auc_score(y_true_L1, y_score_L1)
        aucs_L1.append(roc_L1)

        y_true_LLR = np.concatenate((np.zeros(len(pvalue_pop_LLR)), np.ones(len(pvalue_pool_LLR))))
        y_score_LLR = np.concatenate((pvalue_pop_LLR, pvalue_pool_LLR))
        roc_LLR = roc_auc_score(y_true_LLR, y_score_LLR)
        aucs_LLR.append(roc_LLR)

    # num_order rows of datasets, columns are each timestamp
    if len(aucs_L1) >0:
        auc_L1.append(aucs_L1)

    if len(aucs_LLR) >0:
        auc_LLR.append(aucs_LLR)

# averaging the results from num_order iterations
auc_L1 = np.average(auc_L1, axis=0)
auc_LLR = np.average(auc_LLR, axis=0)

# plotting the performance of the inference for each of the 8 timestamps
fig, ax = plt.subplots()
ax.plot(range(len(ti_pop)), auc_L1, "-b", linewidth=2.0, label="L1")
ax.plot(range(len(ti_pool)), auc_LLR, "-r", linewidth=2.0, label="LLR")
ax.set_ylim([0,1]) # enables comparable auc scores between L1 and LLR

plt.xlabel("timestamp")
plt.ylabel("AUC scores")
plt.legend(loc="upper right")
plt.show()
