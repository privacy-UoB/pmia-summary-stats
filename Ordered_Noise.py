import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import roc_auc_score
from utils import load_dataset, LLR, L1, L1_ttest, L1_threshold, LLR_threshold, ground_truth, D3


# paper: the demonstrated graphs showing roc curves
    # 1st: 50 subsets of n/1049 different individuals (n = 35, 65, 124)
    # 2nd: 6 case groups D19, D17, D10, D7, D3, D1

# load dataset
pop, rpool, cpool = load_dataset(case_sample=D3)
pop = pop.drop(columns="diseases")
rpool = rpool.drop(columns="diseases")
cpool = cpool.drop(columns="diseases")
pool = cpool # make pool configurable

sigma_j = np.std(pop, axis=0) # this is doing it over all the columns (miRNAs)
multiplier = [0, 0.25, 0.5, 0.75, 1] # fractions of standard deviation applied to the dataset

miRNAs = list(pop.keys()) # get the list of miRNAs ["miRNA_1234", "miRNA_1235", ...]
num_orders = 15 # number of different samples of MiRNAs

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
        # broadcast the deviation over all 466 miRNAs to the [1049 x 466] pop shape
        pop_noise = np.random.normal(0, m * sigma_j, pop.shape)
        noised_pop = pop_noise + pop
        nonneg_noised_pop = np.clip(noised_pop, 0, None)
        nonneg_noised_pop_row.append(nonneg_noised_pop)

        # broadcast the deviation over all 466 miRNAs to the [n x 466] pool shape
        pool_noise = np.random.normal(0, m * sigma_j, pool.shape)
        noised_pool = pool_noise + pool
        nonneg_noised_pool = np.clip(noised_pool, 0, None)
        nonneg_noised_pool_row.append(nonneg_noised_pool)

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


            pvalue_pop_L1 = L1_ttest(local_noised_pop, local_pop, local_pool)
            pvalue_pool_L1 = L1_ttest(local_noised_pool, local_pop, local_pool)

            pvalue_pop_LLR = LLR(local_noised_pop, local_pop, local_pool)
            pvalue_pool_LLR = LLR(local_noised_pool, local_pop, local_pool)


            y_true_L1 = np.concatenate((np.zeros(len(pvalue_pop_L1)), np.ones(len(pvalue_pool_L1))))
            y_score_L1 = np.concatenate((pvalue_pop_L1, pvalue_pool_L1))
            roc_L1 = roc_auc_score(y_true_L1, y_score_L1)

            aucs_L1.append(roc_L1)

            y_true_LLR = np.concatenate((np.zeros(len(pvalue_pop_LLR)), np.ones(len(pvalue_pool_LLR))))
            y_score_LLR = np.concatenate((pvalue_pop_LLR, pvalue_pool_LLR))
            roc_LLR = roc_auc_score(y_true_LLR, y_score_LLR)
            
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
    ax.plot(num_miRNAs, noise_fraction_L1[l], linewidth=2.0, label=f"L1 {l}")
    # ax.plot(num_miRNAs, noise_fraction_LLR[l], linewidth=2.0, label=f"LLR {l}")
ax.invert_xaxis()
ax.set_ylim([0.5,1]) # enables comparable auc scores between L1 and LLR
plt.xlabel("number MiRNAs")
plt.ylabel("AUC scores")
plt.legend(loc="upper right")
plt.show() 

# TODO 3rd July
# generate the noisy population - a list per num orders (save doing it over i)
# check inference of patients when miRNA samples are taken a year apart
# TODO 11th July
# more optimal to generate numorder noisy lists here not seeds
# either just one lot of noise or add it to numorder lots of the pop and pool
# another for loop for the m lots, diff. distributions (matrix of m rows numorder cols)
# do extra todo task 2
# write meeting summary for Mark
