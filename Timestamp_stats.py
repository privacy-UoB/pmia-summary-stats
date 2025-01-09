import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from utils import load_timestamp_dataset, LLR, L1, L1_ttest
from scipy.spatial import distance
from itertools import chain

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
            cosine_distance.append(0) # some impossible value to fill in the place of an empty value and keep the timestamps aligned
    cosine_distances.append(cosine_distance)

# plot cosine distance
fig, ax = plt.subplots()
for l in range(len(cosine_distances)):
    ax.plot(range(len(cosine_distances)), cosine_distances[l], linewidth=3.0, label=f"timepoint {l}")
plt.xlabel("timestamp")
plt.ylabel("cosine distance")
plt.legend(loc="upper right")
plt.show()

# create dataset of all 26 diseased individuals
full_timestamps_graph = []
for x, y in zip(pop_timestamps_graph, pool_timestamps_graph):
    x.drop(["disease", "timepoint", "patient_id"], axis=1, inplace=True)
    y.drop(["disease", "timepoint", "patient_id"], axis=1, inplace=True)
    # z = list(chain(x,y))
    # full_timestamps_graph.append(z)


# Commented bit doesn't seem to work
    
# # cosine similarity amongst all individuals by timepoint
# timepoint_distances = []
# for timepoint_pop, timepoint_pool in zip(pop_timestamps_graph, pool_timestamps_graph):

#     # For each row[i] = individual of timepoint, do cosine distance
#     timepoint_distance = []
#     for row in range(len(timepoint_pop)):
#         if (timepoint_pop.iloc[row] is not None):
#             d = distance.cosine(((timepoint_pop.iloc[0]).to_numpy()).ravel(), ((timepoint_pop.iloc[row]).to_numpy()).ravel())
#             timepoint_distance.append(d)
#         else:
#             timepoint_distance.append(0) # some impossible value to fill in the place of an empty value and keep the timestamps aligned
#     timepoint_distances.append(timepoint_distance)

#     timepoint_distance = []
#     for row in range(len(timepoint_pool)):
#         if (timepoint_pool.iloc[row] is not None):
#             d = distance.cosine(((timepoint_pop.iloc[0]).to_numpy()).ravel(), ((timepoint_pool.iloc[row]).to_numpy()).ravel())
#             timepoint_distance.append(d)
#         else:
#             timepoint_distance.append(0) # some impossible value to fill in the place of an empty value and keep the timestamps aligned
#     timepoint_distances.append(timepoint_distance)

# # plot cosine distance
# fig, ax = plt.subplots()
# for l in range(len(timepoint_distances)):
#     ax.plot(range(len(timepoint_distances)), timepoint_distances[l], linewidth=3.0, label=f"timepoint {l}")
# plt.xlabel("timestamp")
# plt.ylabel("cosine distance")
# plt.legend(loc="upper right")
# plt.show()


# cosine similarity amongst all individuals for timepoint 0
for i in range(len(pop_timestamps_graph)):
    # For each row[i] = individual of timepoint, do cosine distance
    timepoint_distance = []

    # make comparison timepoint-configurable
    pop_timestamp_comparison = pop_timestamps_graph[i]

    for row in range(len(pop_timestamp_comparison)):
        if ((pop_timestamp_comparison).iloc[row] is not None):
            d = distance.cosine((((pop_timestamp_comparison).iloc[0]).to_numpy()).ravel(), 
                                (((pop_timestamp_comparison).iloc[row]).to_numpy()).ravel())
            timepoint_distance.append(d)
        else:
            timepoint_distance.append(0) #some impossible value to fill in the place of an empty value and keep the timestamps aligned

    # make comparison timepoint-configurable
    pool_timestamp_comparison = pool_timestamps_graph[i]

    for row in range(len(pool_timestamp_comparison)):
        if ((pool_timestamp_comparison).iloc[row] is not None):
            d = distance.cosine((((pool_timestamp_comparison).iloc[0]).to_numpy()).ravel(), 
                                (((pool_timestamp_comparison).iloc[row]).to_numpy()).ravel())
            timepoint_distance.append(d)
        else:
            timepoint_distance.append(0) #some impossible value to fill in the place of an empty value and keep the timestamps aligned

    # plot cosine distance
    fig, ax = plt.subplots()
    ax.plot(range(len(timepoint_distance)), timepoint_distance, linewidth=3.0, label=f"timepoint {i}")
    plt.xlabel("individual")
    plt.ylabel("cosine distance")
    plt.legend(loc="upper right")
    plt.show()


# some statistical plots for one individual of the dataset
sample_timestamps_graph_deviation = []
one_miRNA = []
for index, t in enumerate(sample_timestamps_graph):

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

    # histogram showing t[t] - t[t-1]
    if index == 0:
        continue
    else:
        previous_graph = sample_timestamps_graph[index-1]
        difference = np.ravel(sample_graph) - np.ravel(previous_graph)
        difference_no_nan = np.nan_to_num(difference) # maybe not needed now...
        print(np.std(difference_no_nan))
        plt.hist(difference_no_nan, bins=100)
        plt.xlabel("t[n]-t[n-1] for n in no_timestamp")
        plt.ylabel("number in each bar")
        plt.show()

        # do the 1 miRNA compare over all 8 timepoints here
        # range = np.random.randint([0, 401, 802], [400, 801, 1205])
        range = np.random.randint(0, 1205)
        one_difference = np.ravel(sample_graph)[range] - np.ravel(previous_graph)[range]
        one_miRNA.append(one_difference)

# histogram showing standard deviations across all 8 timestamps of the individual
plt.hist(sample_timestamps_graph_deviation, bins=40)
plt.xlabel("8 timestamps of standard deviation of miRNAs from individual")
plt.ylabel("count of deviations across 40 different range values")
plt.show()

# histogram showing difference for one miRNA for one individual across 8 timestamps
plt.hist(one_miRNA, bins=100)
plt.xlabel("for one miRNA, t[n]-t[n-1] for all n in no_timestamp")
plt.ylabel("number in each bar")
plt.show()
