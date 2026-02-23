import numpy as np
import matplotlib.pyplot as plt
import random
from utils_datasets import load_dataset
from utils import auc_scores, Gaussian_noise

# load dataset
pop_year_i, pool_year_i = load_dataset(electricity=True)

sigma_j = np.std(pop_year_i[0], axis=0) # this is doing it over all the columns (miRNAs)
multiplier = [0, 0.25, 0.5, 0.75, 1] # fractions of standard deviation applied to the dataset

hours = list(pop_year_i.keys()) # get the list of hours
num_orders = 2000 # number of different samples of hours

shuffled_lists = []
nonneg_noised_pop_matrix = []
nonneg_noised_pool_matrix = []

for j in range (num_orders):
    print("Run Number", j)

    current_hour_list = list(hours)
    random.shuffle(current_hour_list)
    shuffled_lists.append(current_hour_list)

    nonneg_noised_pop_row = []
    nonneg_noised_pool_row = []

    # create a noisy matrix of m columns with num_order rows 
    for m in multiplier:
        # broadcast the deviation over all 8766 hours to the [370 x 8766] pop & pool shape
        nonneg_noisy_pop, nonneg_noisy_pool = Gaussian_noise(pop_year_i[0], pool_year_i[0], 0, (m * sigma_j), clip=True)
        
        nonneg_noised_pop_row.append(nonneg_noisy_pop)
        nonneg_noised_pool_row.append(nonneg_noisy_pool)

    nonneg_noised_pop_matrix.append(nonneg_noised_pop_row)
    nonneg_noised_pool_matrix.append(nonneg_noised_pool_row)


noise_fraction_L1 = []
noise_fraction_LLR = []

for count, m in enumerate(multiplier):
    auc_L1 = []
    auc_LLR = []
    num_miRNAs = []
   
    for i in range(2,len(hours),24): # to show missing info per day - NO, shuffled by hours not days, rethink this
        aucs_L1 = []
        aucs_LLR = []
        num_miRNAs.append(i)

        for j in range (num_orders):
            current_shuffled_list = shuffled_lists[j]
            selected_hours = current_shuffled_list[:i]

            selected_nonneg_noised_pop = nonneg_noised_pop_matrix[j][count]
            local_noised_pop = selected_nonneg_noised_pop[selected_hours]

            selected_nonneg_pool_noise = nonneg_noised_pool_matrix[j][count]
            local_noised_pool = selected_nonneg_pool_noise[selected_hours]

            local_pop = pop_year_i[0][selected_hours]
            local_pool = pool_year_i[0][selected_hours]

            # roc_L1 = auc_scores(local_noised_pop, local_noised_pool, local_pop, local_pool, p_values=False)
            roc_LLR = auc_scores(local_noised_pop, local_noised_pool, local_pop, local_pool, LR=True, p_values=False)

            # aucs_L1.append(roc_L1)            
            aucs_LLR.append(roc_LLR)

        # if len(aucs_L1) >0:
        #     auc_L1.append(np.average(aucs_L1))

        if len(aucs_LLR) >0:
            auc_LLR.append(np.average(aucs_LLR))

    # noise_fraction_L1.append(auc_L1)
    noise_fraction_LLR.append(auc_LLR)

# print(f'AUC score:{auc_L1}')
# print(f'AUC score:{auc_LLR}')

# plots!
fig, ax = plt.subplots()

for index, noise in enumerate(multiplier):
    # ax.plot(num_miRNAs, noise_fraction_L1[index], linewidth=2.0, label=f"Std. dev. * {noise}")
    ax.plot(num_miRNAs, noise_fraction_LLR[index], linewidth=2.0, label=f"Std. dev. * {noise}")
ax.invert_xaxis()
ax.set_ylim([0.5,1]) # enables comparable auc scores between L1 and LLR
plt.xlabel("number MiRNAs")
plt.ylabel("AUC scores")
plt.legend(loc="upper right")
plt.show() 
