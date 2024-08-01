import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import roc_auc_score
from utils import load_timestamp_dataset, LLR, L1, L1_ttest, L1_threshold, LLR_threshold, ground_truth

# load dataset
pop, case_pool, t1, t2, t3, t4, t5, t6, t7, t8 = load_timestamp_dataset()
pop = pop.drop(columns=["disease", "timepoint"])
pool = case_pool.drop(columns=["disease", "timepoint"])
t1 = t1.drop(columns=["disease", "timepoint"])
t2 = t2.drop(columns=["disease", "timepoint"])
t3 = t3.drop(columns=["disease", "timepoint"])
t4 = t4.drop(columns=["disease", "timepoint"])
t5 = t5.drop(columns=["disease", "timepoint"])
t6 = t6.drop(columns=["disease", "timepoint"])
t7 = t7.drop(columns=["disease", "timepoint"])
t8 = t8.drop(columns=["disease", "timepoint"])
timestamps = [t1, t2, t3, t4, t5, t6, t7, t8] # samples of miRNAs over different weeks

miRNAs = list(pop.keys()) # get the list of miRNAs ["miRNA_1234", "miRNA_1235", ...]
num_orders = 15 # number of different samples of MiRNAs

shuffled_lists = []
for j in range (num_orders):
    current_miRNA_list = list(miRNAs)
    random.shuffle(current_miRNA_list)
    shuffled_lists.append(current_miRNA_list)


noise_fraction_L1 = []
noise_fraction_LLR = []

for t in timestamps:
    auc_L1 = []
    auc_LLR = []
    num_miRNAs = []
   
# remove the mirna filter to make it just noisy, so graph is auc over time v auc
    for i in range(2,len(miRNAs),2): # miRNAs range from 1 to 1207
        aucs_L1 = []
        aucs_LLR = []
        num_miRNAs.append(i)

        for j in range (num_orders):
            current_shuffled_list = shuffled_lists[j]
            selected_miRNAs = current_shuffled_list[:i]

            # local_noised_pop = t[selected_miRNAs]
            local_noised_pool = t[:-1][selected_miRNAs]

            local_pop = pop[selected_miRNAs]
            local_pool = pool[selected_miRNAs]


            # pvalue_pop_L1 = L1_ttest(local_noised_pop, local_pop, local_pool)
            pvalue_pool_L1 = L1_ttest(local_noised_pool, local_pop, local_pool)

            # pvalue_pop_LLR = LLR(local_noised_pop, local_pop, local_pool)
            pvalue_pool_LLR = LLR(local_noised_pool, local_pop, local_pool)


            y_true_L1 = np.ones(len(pvalue_pool_L1))
            y_score_L1 = pvalue_pool_L1
            roc_L1 = roc_auc_score(y_true_L1, y_score_L1)

            aucs_L1.append(roc_L1)

            y_true_LLR = np.ones(len(pvalue_pool_LLR))
            y_score_LLR = pvalue_pool_LLR
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

for l in range(len(timestamps)):
    # ax.plot(num_miRNAs, noise_fraction_L1[l], linewidth=2.0, label=f"L1 {l}")
    ax.plot(num_miRNAs, noise_fraction_LLR[l], linewidth=2.0, label=f"LLR {l}")
ax.invert_xaxis()
ax.set_ylim([0,1]) # enables comparable auc scores between L1 and LLR
plt.xlabel("number MiRNAs")
plt.ylabel("AUC scores")
plt.legend(loc="upper right")
plt.show() 

# TODO
# check inference of patients when miRNA samples are taken a year apart
# GSE68951 dataset takes miRNA samples for 0-18 months at 3 month intervals and 2-weeks after 0 months, 26 unhealthy individuals
# !series_matrix_table_begin
# GSE61741 dataset takes one lot of miRNA samples, 94 individuals here are healthy

# return tp 2-8 to use as the noisy datasets for attack
# look up papers using summ stats, use datasets and apply it. say, smartwatch
# after we look for data that has a temporal aspect
# start writing this up
# GSR!!!!
# graph x axis timepoints (later on for the paper make the 2 week plot closer)

# Schedule group meeting with Mark & Pascal
# 1st, omitcontrol from timestamp
# 2nd, pop is only control vs pop is both control and t1
# email results to Pascal and Mark
