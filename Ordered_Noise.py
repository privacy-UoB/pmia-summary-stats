import numpy as np
import matplotlib.pyplot as plt
import random
from utils_datasets import load_dataset, D3, drop_dataset_index
from utils import auc_scores, Gaussian_noise

# paper: the demonstrated graphs showing roc curves
    # 1st: 50 subsets of n/1049 different individuals (n = 35, 65, 124)
    # 2nd: 6 case groups D19, D17, D10, D7, D3, D1

# load dataset
pop_rpool, pop_cpool, rpool, cpool = load_dataset(case_sample=D3)
pop_rpool, pop_cpool, rpool, cpool = drop_dataset_index(pop_rpool, pop_cpool, rpool, cpool)

pop = pop_rpool # make pop configurable
pool = rpool # make pool configurable

sigma_j = np.std(pop, axis=0) # this is doing it over all the columns (miRNAs)
multiplier = [0, 0.25, 0.5, 0.75, 1] # fractions of standard deviation applied to the dataset

miRNAs = list(pop.keys()) # get the list of miRNAs ["miRNA_1234", "miRNA_1235", ...]
num_orders = 40 # number of different samples of MiRNAs

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
        nonneg_noisy_pop, nonneg_noisy_pool = Gaussian_noise(pop, pool, 0, (m * sigma_j), clip=True)
        
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
   
    for i in range(2,len(miRNAs),2): # MiRNAs range from 1 to 466 in paper
        aucs_L1 = []
        aucs_LLR = []
        num_miRNAs.append(i)

        for j in range (num_orders):
            current_shuffled_list = shuffled_lists[j]
            selected_miRNAs = current_shuffled_list[:i]

            selected_nonneg_noised_pop = nonneg_noised_pop_matrix[j][count]
            local_noised_pop = selected_nonneg_noised_pop[selected_miRNAs]

            selected_nonneg_pool_noise = nonneg_noised_pool_matrix[j][count]
            local_noised_pool = selected_nonneg_pool_noise[selected_miRNAs]

            local_pop = pop[selected_miRNAs]
            local_pool = pool[selected_miRNAs]

            # Query: should these actually be local_noised_pop, local_noised_pool, pop, pool?
            roc_L1 = auc_scores(local_noised_pop, local_noised_pool, local_pop, local_pool, p_values=False)
            roc_LLR = auc_scores(local_noised_pop, local_noised_pool, local_pop, local_pool, LR=True, p_values=False)

            aucs_L1.append(roc_L1)            
            aucs_LLR.append(roc_LLR)

        if len(aucs_L1) >0:
            auc_L1.append(np.average(aucs_L1))

        if len(aucs_LLR) >0:
            auc_LLR.append(np.average(aucs_LLR))

    noise_fraction_L1.append(auc_L1)
    noise_fraction_LLR.append(auc_LLR)

# print(f'AUC score:{auc_L1}')
# print(f'AUC score:{auc_LLR}')

# plots!
fig, ax = plt.subplots()

for l in range(len(multiplier)):
    # ax.plot(num_miRNAs, noise_fraction_L1[l], linewidth=2.0, label=f"L1 {l}")
    ax.plot(num_miRNAs, noise_fraction_LLR[l], linewidth=2.0, label=f"LLR {l}")
ax.invert_xaxis()
ax.set_ylim([0.5,1]) # enables comparable auc scores between L1 and LLR
plt.xlabel("number MiRNAs")
plt.ylabel("AUC scores")
plt.legend(loc="upper right")
plt.show() 
