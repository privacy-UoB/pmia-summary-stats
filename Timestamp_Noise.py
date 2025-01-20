import numpy as np
import matplotlib.pyplot as plt
from utils_datasets import load_timestamp_dataset, drop_timestamp_index
from utils import auc_scores

include_all_timestamps = False # if we wish to create one graph with all 8 timestamps

# load one partitioned dataset (moved from num_orders loop)
ti_pop, ti_pool, ti_sample = load_timestamp_dataset()
ti_pop, ti_pool = drop_timestamp_index(ti_pop, ti_pool)

# for x, y in zip(ti_pop, ti_pool):
    # for row in range(len(x)):
    #     (x.iloc[row]).dropna(inplace=True) # remove NaN rows from the dataframe
    # print(x)
    # for row in range(len(y)):
    #     (y.iloc[row]).dropna() # remove NaN rows from the dataframe

# for i in range(8):
#     for row in range(len(ti_pop[i])):
#         (ti_pop[i].iloc[row]).dropna(inplace=True) # remove NaN rows from the dataframe
#         print(ti_pop[i])
#     for row in range(len(ti_pool[i])):
#         (ti_pool[i].iloc[row]).dropna() # remove NaN rows from the dataframe
    
# This isn't working because it's 'a value is trying to be set on a copy of a slice from a DataFrame'?????
    # so why is the above for x, y in zip() working?!?!??


# fractions of standard deviation applied to the dataset
multiplier = np.arange(0, 8, 0.04)
# multiplier = np.arange(0, 20, 0.1)
num_orders = 30 # number of iterations to average over

if include_all_timestamps:
    noisy_timestamps_L1 = []
    noisy_timestamps_LLR = []

    # include below for loop if include_all_timestamps == True
    # for t_pop, t_pool in zip(ti_pop, ti_pool):

# configuring the reference pop & pool to match the dataframe of a particular timepoint
pop = ti_pop[0]
pool = ti_pool[0]
sigma_j = np.std(pop, axis=0) #this is doing it over all the columns (miRNAs)

auc_L1 = []
auc_LLR = []

for m in multiplier:
    aucs_L1 = []
    aucs_LLR = []

    # for loop for numorder lots of train/test, then average at end
    for j in range (num_orders):

        # the 'noise' increases throughout each of the later timepoints the data is collected from
        pop_noise = np.random.normal(0, m, pop.shape) # changed from m*sigmaj so it's not tailored variance to each miRNA
        noised_pop = pop + pop_noise

        pool_noise = np.random.normal(0, m, pool.shape)
        noised_pool = pool + pool_noise

        # get performance/accuracy for L1 & LLR statistics over the noisy stat inputs compared to the 'original' pop & pool
        roc_L1 = auc_scores(noised_pop, noised_pool, pop, pool, p_values=False)
        roc_LLR = auc_scores(noised_pop, noised_pool, pop, pool, LR=True, p_values=False)
        
        aucs_L1.append(roc_L1)
        aucs_LLR.append(roc_LLR)

    # num_order rows of datasets, columns are each timestamp
    if len(aucs_L1) >0:
        auc_L1.append(np.average(aucs_L1))

    if len(aucs_LLR) >0:
        auc_LLR.append(np.average(aucs_LLR))
    
    if include_all_timestamps:
        noisy_timestamps_L1.append(auc_L1)
        noisy_timestamps_LLR.append(auc_LLR)

# averaging the results from num_order iterations
# auc_L1 = np.average(auc_L1, axis=0) (replaced by np.average)
# auc_LLR = np.average(auc_LLR, axis=0) (replaced by np.average)

# plots!
fig, ax = plt.subplots()
if not include_all_timestamps:
    # plotting the performance of the inference for one timestamps over noise
    ax.plot(multiplier, auc_L1, "-b", linewidth=2.0, label="L1")
    ax.plot(multiplier, auc_LLR, "-r", linewidth=2.0, label="LLR")

if include_all_timestamps:
    # plotting the performance of the inference for each of the 8 timestamps over noise
    for l in range(len(ti_pop)):
        ax.plot(multiplier, noisy_timestamps_L1[l], linewidth=2.0, label=f"L1 {l}")
        # ax.plot(multiplier, noisy_timestamps_LLR[l], linewidth=2.0, label=f"LLR {l}")

ax.set_ylim([0.3, 1]) # enables comparable auc scores between L1 and LLR
plt.xlabel("noise scale")
plt.ylabel("AUC scores")
plt.legend(loc="upper right")
plt.show()
