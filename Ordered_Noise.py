import numpy as np
import matplotlib.pyplot as plt
import random
from utils_datasets import load_dataset, separate_diseased_miRNAs, D17
from utils import auc_scores, Gaussian_noise

# paper: the demonstrated graphs showing roc curves
    # 1st: 50 subsets of n/1049 different individuals (n = 35, 65, 124)
    # 2nd: 6 case groups D19, D17, D10, D7, D3, D1

L1_or_LLR = "L1"
stratifying = False # Not enough pool miRNAs in True case

if stratifying == False:
    # load dataset
    population, pool = load_dataset(miRNA=True, disease_case_sample=D17)

    # 0 = random, 1 = case
    pop = population[1] # make pop configurable
    pool = pool[1] # make pool configurable

else:
    # diseased case sample pop/pool only
    only_diseased_miRNAs_pop, without_diseased_miRNAs_pop, only_diseased_miRNAs_pool, without_diseased_miRNAs_pool = separate_diseased_miRNAs(D17, "miRNA")
    pop = without_diseased_miRNAs_pop
    pool = without_diseased_miRNAs_pool

sigma_j = np.std(pop, axis=0) # this is doing it over all the columns (miRNAs)
sigma_j_pool = np.std(pool, axis=0)
ranges = [[0, 0.25, 0.5, 0.75, 1], # 0, fractions of standard deviation applied to the dataset
              [0, 100, 200, 300, 400], # 1, static values produce similar noise to one observed case
              np.concatenate(([0], np.logspace(1, 4, num=4)))] # 2
multiplier = ranges[2]

# for m in multiplier:
#     x = sigma_j * m

miRNAs = list(pop.keys()) # get the list of miRNAs ["miRNA_1234", "miRNA_1235", ...]
num_orders = 2000 # number of different samples of MiRNAs

shuffled_lists = []
nonneg_noised_pop_matrix = []
nonneg_noised_pool_matrix = []

for j in range (num_orders):
    current_miRNA_list = list(miRNAs)
    random.shuffle(current_miRNA_list)
    shuffled_lists.append(current_miRNA_list)

    nonneg_noised_pop_row = []
    nonneg_noised_pool_row = []

    # create a noisy matrix of m columns with num_order rows 
    for m in multiplier:
        # broadcast the deviation over all 466 miRNAs to the [1049 x 466] pop & pool shape
        nonneg_noisy_pop, nonneg_noisy_pool = Gaussian_noise(pop, pool, 0, m, clip=True) # note: replaced m * sigma_j with m
        
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
   
    for i in range(2, len(miRNAs), 2): # MiRNAs range from 1 to 466 in paper
        aucs_L1 = []
        aucs_LLR = []
        num_miRNAs.append(i)

        print("miRNA", i, "on iteration", count)

        for j in range (num_orders):
            current_shuffled_list = shuffled_lists[j]
            selected_miRNAs = current_shuffled_list[:i]

            selected_nonneg_noised_pop = nonneg_noised_pop_matrix[j][count]
            local_noised_pop = selected_nonneg_noised_pop[selected_miRNAs]

            selected_nonneg_pool_noise = nonneg_noised_pool_matrix[j][count]
            local_noised_pool = selected_nonneg_pool_noise[selected_miRNAs]

            local_pop = pop[selected_miRNAs]
            local_pool = pool[selected_miRNAs]

            # local_pop & local_pool instead of pop & pool due to attacker's lacking access to full information
            if L1_or_LLR == "L1":
                roc_L1 = auc_scores(local_noised_pop, local_noised_pool, local_pop, local_pool, p_values=False)
                aucs_L1.append(roc_L1)   
            elif L1_or_LLR == "LLR":
                roc_LLR = auc_scores(local_noised_pop, local_noised_pool, local_pop, local_pool, LR=True, p_values=False)       
                aucs_LLR.append(roc_LLR)

        if L1_or_LLR == "L1":
            if len(aucs_L1) >0:
                auc_L1.append(np.average(aucs_L1))
        elif L1_or_LLR == "LLR":
            if len(aucs_LLR) >0:
                auc_LLR.append(np.average(aucs_LLR))

    noise_fraction_L1.append(auc_L1) if L1_or_LLR == "L1" else noise_fraction_LLR.append(auc_LLR)

# plots!
fig, ax = plt.subplots()

for index, noise in enumerate(multiplier):
    if L1_or_LLR == "L1":
        ax.plot(num_miRNAs, noise_fraction_L1[index], linewidth=2.0, label=f"Std. dev. = {noise}")
    elif L1_or_LLR == "LLR":
        ax.plot(num_miRNAs, noise_fraction_LLR[index], linewidth=2.0, label=f"Std. dev. = {noise}")
ax.invert_xaxis()
ax.set_ylim([0.3,1]) # enables comparable auc scores between L1 and LLR
plt.xlabel("number MiRNAs")
plt.ylabel("AUC scores")
plt.legend(loc="upper right")
plt.show() 
