import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils_datasets import load_timestamp_dataset, drop_timestamp_index
from scipy.spatial import distance

# df = df.dropna() # this drops all NaN values from a dataframe?

# load dataset (ensure it's the NaN, i.e. t = new population)
pop_timestamps_graph, pool_timestamps_graph, sample_timestamps_graph = load_timestamp_dataset(withNaN=True)
# pop_timestamps_graph is a list of length 8 timestamps, each containing <=18 individuals
# pool_timestamps_graph is a list of length 8 timestamps, each containing <=8 individuals

# ensure patient order over timestamps and remove columns from dataset
for x, y in zip(pop_timestamps_graph, pool_timestamps_graph):
    x.sort_values("patient_id")
    y.sort_values("patient_id")

pop_labels = list(pop_timestamps_graph[0]["patient_id"])
pool_labels = list(pool_timestamps_graph[0]["patient_id"])
patient_labels = pop_labels + pool_labels # in order as they appear in for loop below

pop_timestamps_graph, pool_timestamps_graph = drop_timestamp_index(pop_timestamps_graph, pool_timestamps_graph)

timepoint_distance = []
patient_0_time_0 = (pop_timestamps_graph[0]).iloc[0]

# cosine similarity amongst all individuals for timepoint 0
for i in range(len(pop_timestamps_graph)):
    # For each row[i] = individual of timepoint, do cosine distance
    timepoint_i_distance = []

    # make comparison timepoint-configurable
    pop_timestamp_comparison = pop_timestamps_graph[i]

    for row in range(len(pop_timestamp_comparison)):
        if ((pop_timestamp_comparison).iloc[row] is not None):
            d = distance.cosine(((patient_0_time_0).to_numpy()).ravel(), 
                                (((pop_timestamp_comparison).iloc[row]).to_numpy()).ravel())
            timepoint_i_distance.append(d)
        else:
            timepoint_i_distance.append(0) # some impossible value to fill in the place of an empty value and keep the timestamps aligned

    # make comparison timepoint-configurable
    pool_timestamp_comparison = pool_timestamps_graph[i]

    for row in range(len(pool_timestamp_comparison)):
        if ((pool_timestamp_comparison).iloc[row] is not None):
            d = distance.cosine(((patient_0_time_0).to_numpy()).ravel(), 
                                (((pool_timestamp_comparison).iloc[row]).to_numpy()).ravel())
            timepoint_i_distance.append(d)
        else:
            timepoint_i_distance.append(0) # some impossible value to fill in the place of an empty value and keep the timestamps aligned
    
    # create list of 8 timepoint rows vs 26 individual cosine comparison columns
    timepoint_distance.append(timepoint_i_distance)

    print(len(timepoint_i_distance))
print(len(timepoint_distance))

# plot cosine distance
fig, ax = plt.subplots()
ax.plot(range(len(timepoint_distance)), timepoint_distance, linewidth=3.0)
plt.xlabel("timestamp per individual")
plt.ylabel("cosine distance")
plt.legend(loc="upper right")
plt.show() # something is going wrong or just messy??


# https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
timepoint_distance = np.array(timepoint_distance)

fig, ax = plt.subplots()
im = ax.imshow(timepoint_distance)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(patient_labels)), labels=(patient_labels))
ax.set_yticks(np.arange(len(pop_timestamps_graph)), labels=(range(len(pop_timestamps_graph))))

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
# CANNOT INCLUDE BECAUSE NUMBERS OBSCURE HEATMAP
# for i in range(len(pop_timestamps_graph)):
#     for j in range(len(patient_labels)):
#         text = ax.text(j, i, timepoint_distance[i, j], ha="center", va="center", color="w")

ax.set_title("Cosine Distances between one individual at timepoint 0 to all datapoints")
fig.tight_layout()
plt.show()
