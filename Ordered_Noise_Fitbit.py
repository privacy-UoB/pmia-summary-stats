import numpy as np
import matplotlib.pyplot as plt
import random
from utils_datasets import load_dataset
from utils import auc_scores, Gaussian_noise

L1_or_LLR = "L1"
error_bands = True
fixed_FPR = True

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
if fixed_FPR == True:
    target_fpr = 1e-2

    noise_fraction_tpr_at_fpr_L1 = []
    noise_fraction_tpr_at_fpr_LLR = []
    tpr_L1_error_bands = []
    tpr_LLR_error_bands = []

for count, m in enumerate(multiplier):
    num_features = []
    auc_L1 = []
    auc_LLR = []
    auc_L1_lower_bands, auc_L1_upper_bands = [[],[]]
    auc_LLR_lower_bands, auc_LLR_upper_bands = [[],[]]
    if fixed_FPR == True:
        tpr_at_fpr_L1 = []
        tpr_at_fpr_LLR = []
        tpr_L1_lower_bands, tpr_L1_upper_bands = [[],[]]
        tpr_LLR_lower_bands, tpr_LLR_upper_bands = [[],[]]
   
    for i in range(2,len(features)):
        aucs_L1 = []
        aucs_LLR = []
        if fixed_FPR == True:
            tpr_at_fprs_L1 = []
            tpr_at_fprs_LLR = []
        
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

                if fixed_FPR == True:
                    fpr_L1, tpr_L1, thresholds_L1 = auc_scores(local_noised_pop, local_noised_pool, local_pop, local_pool, FPR=True)
                    # TPR at a fixed FPR (e.g., 0.01 = 1%)
                    tpr_at_fprs_L1.append(np.interp(target_fpr, fpr_L1, tpr_L1))

            else:
                try:
                    roc_LLR = auc_scores(local_noised_pop, local_noised_pool, local_pop, local_pool, LR=True, p_values=False)
                except ValueError:
                    continue         
                aucs_LLR.append(roc_LLR)

                if fixed_FPR == True:
                    fpr_LLR, tpr_LLR, thresholds_LLR = auc_scores(local_noised_pop, local_noised_pool, local_pop, local_pool, LR=True, FPR=True)
                    # TPR at a fixed FPR (e.g., 0.01 = 1%)
                    tpr_at_fprs_LLR.append(np.interp(target_fpr, fpr_LLR, tpr_LLR))

        if L1_or_LLR == "L1":
            if len(aucs_L1) >0:
                auc_L1.append(np.average(aucs_L1))

                if fixed_FPR == True:
                    if len(tpr_at_fprs_L1) >0:
                        tpr_at_fpr_L1.append(np.average(tpr_at_fprs_L1))

                if error_bands == True:
                    auc_L1_lower_bands.append(np.min(aucs_L1, axis=0))
                    auc_L1_upper_bands.append(np.max(aucs_L1, axis=0))

                    if fixed_FPR == True:
                        tpr_L1_lower_bands.append(np.min(tpr_at_fprs_L1, axis=0))
                        tpr_L1_upper_bands.append(np.max(tpr_at_fprs_L1, axis=0))

        else:
            if len(aucs_LLR) >0:
                auc_LLR.append(np.average(aucs_LLR))

                if fixed_FPR == True:
                    if len(tpr_at_fprs_LLR) >0:
                        tpr_at_fpr_L1.append(np.average(tpr_at_fprs_LLR))

                if error_bands == True:
                    auc_LLR_lower_bands.append(np.min(aucs_LLR, axis=0))
                    auc_LLR_upper_bands.append(np.max(aucs_LLR, axis=0))

                    if fixed_FPR == True:
                        tpr_L1_lower_bands.append(np.min(tpr_at_fprs_LLR, axis=0))
                        tpr_L1_upper_bands.append(np.max(tpr_at_fprs_LLR, axis=0))

    if L1_or_LLR == "L1":
        noise_fraction_L1.append(auc_L1)

        if fixed_FPR == True:
            noise_fraction_tpr_at_fpr_L1.append(tpr_at_fpr_L1)

        if error_bands == True:
            L1_error_bands.append([auc_L1_lower_bands, auc_L1_upper_bands])

            if fixed_FPR == True:
                tpr_L1_error_bands.append([tpr_L1_lower_bands, tpr_L1_upper_bands])

    else:
        noise_fraction_LLR.append(auc_LLR)

        if fixed_FPR == True:
            noise_fraction_tpr_at_fpr_LLR.append(tpr_at_fpr_LLR)

        if error_bands == True:
            LLR_error_bands.append([auc_LLR_lower_bands, auc_LLR_upper_bands])

            if fixed_FPR == True:
                tpr_LLR_error_bands.append([tpr_LLR_lower_bands, tpr_LLR_upper_bands])

# plots!
fig, ax1 = plt.subplots()
colours1 = ["cornflowerblue", "gold", "springgreen", "red", "mediumpurple"]
if fixed_FPR == True:
    ax2 = ax1.twinx()
    colours2 = ["mediumblue", "orange", "green", "brown", "purple"]

for index, noise in enumerate(multiplier):
    if L1_or_LLR == "L1":
        ax1.plot(num_features, noise_fraction_L1[index], colours1[index], linewidth=2.0, label=f"AUC Std. dev. * {noise}")

        if fixed_FPR == True:
            ax2.plot(num_features, noise_fraction_tpr_at_fpr_L1[index], colours2[index], linewidth=2.0, label=f"fpr Std. dev. * {noise}")

        if error_bands == True:
            ax1.fill_between(num_features, L1_error_bands[index][0], L1_error_bands[index][1], alpha=0.2)
            # print(L1_error_bands[index][0], L1_error_bands[index][1])

            if fixed_FPR == True:
                ax2.fill_between(num_features, tpr_L1_error_bands[index][0], tpr_L1_error_bands[index][1], alpha=0.2)

    else:
        ax1.plot(num_features, noise_fraction_LLR[index], colours1[index], linewidth=2.0, label=f"AUC Std. dev. * {noise}")

        if fixed_FPR == True:
            ax2.plot(num_features, noise_fraction_tpr_at_fpr_LLR[index], colours2[index], linewidth=2.0, label=f"fpr Std. dev. * {noise}")

        if error_bands == True:
            ax1.fill_between(num_features, LLR_error_bands[index][0], LLR_error_bands[index][1], alpha=0.2)
            # print(LLR_error_bands[index][0], LLR_error_bands[index][1])

            if fixed_FPR == True:
                ax2.fill_between(num_features, tpr_LLR_error_bands[index][0], tpr_LLR_error_bands[index][1], alpha=0.2)

ax1.legend(loc='upper right')
if fixed_FPR == True:
    # Merge handles and labels
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    # Add combined legend to one axis
    ax1.legend(h1 + h2, l1 + l2, loc='upper right')
    ax2.set_ylabel("TPR at 0.01 FPR")

ax1.invert_xaxis()
ax1.set_xlabel("number features")
ax1.set_ylabel("AUC scores")
ax1.set_ylim([0.1,1.1]) # enables comparable auc scores between L1 and LLR
ax1.grid(True)

plt.show() 
