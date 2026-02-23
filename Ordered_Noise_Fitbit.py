import numpy as np
import matplotlib.pyplot as plt
import random
from utils_datasets import load_dataset
from utils import auc_scores, Gaussian_noise

L1_or_LLR = "LLR"
error_bands = True

# load dataset
population, random_pool = load_dataset(FitBit=True)

pop = population[0] # make pop configurable
pool = random_pool[0] # make pool configurable

sigma_j = np.std(pop, axis=0) # this is doing it over all the columns (miRNAs)
multiplier = [0, 0.25, 0.5, 0.75, 1] # fractions of standard deviation applied to the dataset

features = list(pop.keys()) # get the list of fitness activities
num_orders = 5000 # number of different samples of activities

shuffled_lists = []
nonneg_noised_pop_matrix = []
nonneg_noised_pool_matrix = []

for j in range (num_orders):
    current_feature_list = list(features)
    random.shuffle(current_feature_list)
    shuffled_lists.append(current_feature_list)

    nonneg_noised_pop_row = []
    nonneg_noised_pool_row = []

    # create a noisy matrix of m columns with num_order rows 
    for m in multiplier:
        # broadcast the deviation over all 11 activities to the pop & pool shape
        nonneg_noisy_pop, nonneg_noisy_pool = Gaussian_noise(pop, pool, 0, (m * sigma_j), clip=True)
        
        nonneg_noised_pop_row.append(nonneg_noisy_pop)
        nonneg_noised_pool_row.append(nonneg_noisy_pool)

    nonneg_noised_pop_matrix.append(nonneg_noised_pop_row)
    nonneg_noised_pool_matrix.append(nonneg_noised_pool_row)


noise_fraction_L1 = []
noise_fraction_LLR = []
L1_error_bands = []
LLR_error_bands = []

for count, m in enumerate(multiplier):
    auc_L1 = []
    auc_LLR = []
    auc_L1_lower_bands, auc_L1_upper_bands = [[],[]]
    auc_LLR_lower_bands, auc_LLR_upper_bands = [[],[]]
    num_features = []
   
    for i in range(2,len(features)):
        aucs_L1 = []
        aucs_LLR = []
        
        num_features.append(i)

        for j in range (num_orders):
            current_shuffled_list = shuffled_lists[j]
            selected_features = current_shuffled_list[:i]

            selected_nonneg_noised_pop = nonneg_noised_pop_matrix[j][count]
            local_noised_pop = selected_nonneg_noised_pop[selected_features]

            selected_nonneg_pool_noise = nonneg_noised_pool_matrix[j][count]
            local_noised_pool = selected_nonneg_pool_noise[selected_features]

            local_pop = pop[selected_features]
            local_pool = pool[selected_features]

            if L1_or_LLR == "L1":
                roc_L1 = auc_scores(local_noised_pop, local_noised_pool, local_pop, local_pool, p_values=False)
                aucs_L1.append(roc_L1)  
            else:
                try:
                    roc_LLR = auc_scores(local_noised_pop, local_noised_pool, local_pop, local_pool, LR=True, p_values=False)
                except ValueError:
                    continue         
                aucs_LLR.append(roc_LLR)

        if L1_or_LLR == "L1":
            if len(aucs_L1) >0:
                auc_L1.append(np.average(aucs_L1))
                if error_bands == True:
                    auc_L1_lower_bands.append(np.min(aucs_L1, axis=0))
                    auc_L1_upper_bands.append(np.max(aucs_L1, axis=0))
        else:
            if len(aucs_LLR) >0:
                auc_LLR.append(np.average(aucs_LLR))
                if error_bands == True:
                    auc_LLR_lower_bands.append(np.min(aucs_LLR, axis=0))
                    auc_LLR_upper_bands.append(np.max(aucs_LLR, axis=0))

    if L1_or_LLR == "L1":
        noise_fraction_L1.append(auc_L1)
        if error_bands == True:
            L1_error_bands.append([auc_L1_lower_bands, auc_L1_upper_bands])
    else:
        noise_fraction_LLR.append(auc_LLR)
        if error_bands == True:
            LLR_error_bands.append([auc_LLR_lower_bands, auc_LLR_upper_bands])

# print(f'AUC score:{auc_L1}')
# print(f'AUC score:{auc_LLR}')

# plots!
fig, ax = plt.subplots()

for index, noise in enumerate(multiplier):
    if L1_or_LLR == "L1":
        ax.plot(num_features, noise_fraction_L1[index], linewidth=2.0, label=f"Std. dev. * {noise}")
        if error_bands == True:
            ax.fill_between(num_features, L1_error_bands[index][0], L1_error_bands[index][1], alpha=0.2)
            print(L1_error_bands[index][0], L1_error_bands[index][1])
    else:
        ax.plot(num_features, noise_fraction_LLR[index], linewidth=2.0, label=f"Std. dev. * {noise}")
        if error_bands == True:
            ax.fill_between(num_features, LLR_error_bands[index][0], LLR_error_bands[index][1], alpha=0.2)
            print(LLR_error_bands[index][0], LLR_error_bands[index][1])
ax.invert_xaxis()
ax.set_ylim([0.1,1.1]) # enables comparable auc scores between L1 and LLR
plt.xlabel("number features")
plt.ylabel("AUC scores")
plt.legend(loc="upper right")
plt.show() 
