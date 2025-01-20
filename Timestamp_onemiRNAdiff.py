import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from scipy import stats 
from utils_datasets import load_timestamp_dataset, drop_timestamp_index

testing_distribution_fit = True

# load dataset
pop_timestamps, pool_timestamps, sample_timestamps = load_timestamp_dataset()
pop_timestamps, pool_timestamps = drop_timestamp_index(pop_timestamps, pool_timestamps)

# miRNA = np.random.randint(0, 1205)

for miRNA in range(1206):

    all_timepoints_one_miRNA_difference = []
    # difference for one miRNA amongst all individuals for all timepoints
    for index, (t, u) in enumerate(zip(pop_timestamps, pool_timestamps)):
        one_miRNA_difference = []
        if index == 0:
            continue
        else:
            if not testing_distribution_fit:
                current_pop = t
                current_pool = u
            if testing_distribution_fit:
                current_pop = pop_timestamps[0]
                current_pool = pool_timestamps[0]

            previous_pop = pop_timestamps[index-1]
            previous_pool = pool_timestamps[index-1]


            if not testing_distribution_fit:
                for individual in range(len(current_pop)):
                    d = np.ravel((current_pop.iloc[individual]))[miRNA] - np.ravel((previous_pop.iloc[individual]))[miRNA]
                    one_miRNA_difference.append(d)

                for individual in range(len(current_pool)):
                    d = np.ravel((current_pool.iloc[individual]))[miRNA] - np.ravel((previous_pool.iloc[individual]))[miRNA]
                    one_miRNA_difference.append(d)


            if testing_distribution_fit:
                if len(current_pop) == len(previous_pop):
                    for individual in range(len(current_pop)):
                        d = np.ravel((current_pop.iloc[individual]))[miRNA] - np.ravel((previous_pop.iloc[individual]))[miRNA]
                        one_miRNA_difference.append(d)
                else:
                    continue

                if len(current_pool) == len(previous_pool):
                    for individual in range(len(current_pool)):
                        d = np.ravel((current_pool.iloc[individual]))[miRNA] - np.ravel((previous_pool.iloc[individual]))[miRNA]
                        one_miRNA_difference.append(d)
                else:
                    continue

            all_timepoints_one_miRNA_difference.append(one_miRNA_difference)

    # print(len(all_timepoints_one_miRNA_difference), len(all_timepoints_one_miRNA_difference[0]), 
            # all_timepoints_one_miRNA_difference)
    
    if not testing_distribution_fit:
        flattening_one_miRNA_data = []  
        for l in range(len(all_timepoints_one_miRNA_difference)):
            flattening_one_miRNA_data.extend(np.ravel(all_timepoints_one_miRNA_difference[l]))
        # print(flattening_one_miRNA_data)

        plt.hist(flattening_one_miRNA_data, bins=100, label=[f"timepoint"])
        plt.xlabel("for one miRNA, t[n]-t[n-1] for all n in no_timestamp")
        plt.ylabel("number in each bar")
        plt.show()

    if testing_distribution_fit:
        flattening_one_miRNA_data = []  
        for l in range(len(all_timepoints_one_miRNA_difference)):
            flattening_one_miRNA_data.extend(np.ravel(all_timepoints_one_miRNA_difference[l]))
        # print(flattening_one_miRNA_data)

        plt.hist(flattening_one_miRNA_data, bins=100, label=[f"timepoint"])
        plt.xlabel("for one miRNA, t[n]-t[n-1] for all n in no_timestamp")
        plt.ylabel("number in each bar")
        plt.show()

