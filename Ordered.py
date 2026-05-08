import numpy as np
import matplotlib.pyplot as plt
import random
from utils_datasets import load_dataset, separate_diseased_miRNAs, D17
from utils import auc_scores, fpr_power, LLR, L1, L1_threshold

# paper: the demonstrated graphs showing roc curves
    # 1st: 50 subsets of n/1049 different individuals (n = 35, 65, 124)
    # 2nd: 6 case groups D19, D17, D10, D7, D3, D1

stratifying = True
fixed_FPR = True
D = D17

if stratifying == False:
    # load dataset
    population, pool = load_dataset(miRNA=True, disease_case_sample=D, random_sample_size=20)

    # 0 = random, 1 = case
    pop = population[0] # make pop configurable
    pool = pool[0] # make pool configurable

else:
    # diseased case sample pop/pool only
    only_diseased_miRNAs_pop, without_diseased_miRNAs_pop, only_diseased_miRNAs_pool, without_diseased_miRNAs_pool = separate_diseased_miRNAs(D, "miRNA")
    selectpop = [only_diseased_miRNAs_pop, without_diseased_miRNAs_pop]
    selectpool = [only_diseased_miRNAs_pool, without_diseased_miRNAs_pool]

# when not stratifying, comment out/unindent from here
both_L1 = []
both_LLR = []
if fixed_FPR == True:
    both_tpr_L1 = []
    both_tpr_LLR = []
for k in range(len(selectpop)):
    pop = selectpop[k]
    pool = selectpool[k]
    # to here

    auc_L1 = []
    auc_LLR = []
    if fixed_FPR == True:
        tpr_at_fpr_L1 = []
        tpr_at_fpr_LLR = []
    num_miRNAs = []
    miRNAs = list(pop.keys()) # get the list of miRNAs ["miRNA_1234", "miRNA_1235", ...]
    num_orders = 5000 # number of different samples of MiRNAs

    shuffled_lists = []
    for j in range (num_orders):
        current_miRNA_list = list(miRNAs)
        if stratifying == False:
            random.shuffle(current_miRNA_list) 
        else:
           current_miRNA_list = random.sample(current_miRNA_list, len(only_diseased_miRNAs_pool.columns))
        shuffled_lists.append(current_miRNA_list)

    for i in range(2, len(current_miRNA_list)+1, 1): # MiRNAs range from 1 to 466 in paper
        aucs_L1 = []
        aucs_LLR = []
        if fixed_FPR == True:
            tpr_at_fprs_L1 = []
            tpr_at_fprs_LLR = []
        num_miRNAs.append(i)

        for j in range (num_orders):
            current_shuffled_list = shuffled_lists[j]
            selected_miRNAs = current_shuffled_list[:i]

            local_pop = pop[selected_miRNAs]
            local_pool = pool[selected_miRNAs]
            
            # print(L1(victim, local_pop, local_pool).sum())
            # print(L1_threshold(local_pop, local_pool))

            # print(LLR(local_pop, local_pop, local_pool))
            # print(L1(local_pop, local_pop, local_pool))

            # Query: should these actually be local_pop, local_pool, pop, pool?
            roc_L1, pvalue_pop_L1, pvalue_pool_L1 = auc_scores(local_pop, local_pool, local_pop, local_pool)
            roc_LLR, pvalue_pop_LLR, pvalue_pool_LLR = auc_scores(local_pop, local_pool, local_pop, local_pool, LR=True)

            aucs_L1.append(roc_L1)        
            aucs_LLR.append(roc_LLR)

            if fixed_FPR == True:
                fpr_L1, tpr_L1, thresholds_L1 = auc_scores(local_pop, local_pool, local_pop, local_pool, FPR=True)
                fpr_LLR, tpr_LLR, thresholds_LLR = auc_scores(local_pop, local_pool, local_pop, local_pool, LR=True, FPR=True)

                # TPR at a fixed FPR (e.g., 0.01 = 1%)
                target_fpr = 1e-2
                tpr_at_fprs_L1.append(np.interp(target_fpr, fpr_L1, tpr_L1))
                tpr_at_fprs_LLR.append(np.interp(target_fpr, fpr_LLR, tpr_LLR))

            # fpr_L1, power_L1 = fpr_power(local_pop, local_pool, pvalue_pop_L1, pvalue_pool_L1)
            # fpr_LLR, power_LLR = fpr_power(local_pop, local_pool, pvalue_pop_LLR, pvalue_pool_LLR, LR=True)

        if len(aucs_L1) >0:
            auc_L1.append(np.average(aucs_L1))

        if len(aucs_LLR) >0:
            auc_LLR.append(np.average(aucs_LLR))

        if fixed_FPR == True:
            if len(tpr_at_fprs_L1) >0:
                tpr_at_fpr_L1.append(np.average(tpr_at_fprs_L1))

            if len(tpr_at_fprs_LLR) >0:
                tpr_at_fpr_LLR.append(np.average(tpr_at_fprs_LLR))


    both_L1.append(auc_L1)
    both_LLR.append(auc_LLR)
    if fixed_FPR == True:
        both_tpr_L1.append(tpr_at_fpr_L1)
        both_tpr_LLR.append(tpr_at_fpr_LLR)

# plots!
# fig, ax = plt.subplots()
# ax.set_xscale("log")
# ax.plot(fpr_L1, power_L1, linewidth=2.0)
# ax.plot(fpr_L1, power_LLR, linewidth=2.0)
# plt.xlabel("fpr")
# plt.ylabel("power")
# plt.show()

# print(f'AUC score:{auc_L1}')
# print(f'AUC score:{auc_LLR}')

# plots!
fig, ax1 = plt.subplots()
colours1 = ["cornflowerblue", "gold", "springgreen", "red"]

# Left-hand x axis for AUC scores
ax1.plot(num_miRNAs, both_L1[0], colours1[0], linewidth=2.0, label="L1 AUC diseased miRNAs")
ax1.plot(num_miRNAs, both_LLR[0], colours1[1], linewidth=2.0, label="LLR AUC diseased miRNAs")
ax1.plot(num_miRNAs, both_L1[1], colours1[2], linewidth=2.0, label="L1 AUC normal miRNAs")
ax1.plot(num_miRNAs, both_LLR[1], colours1[3], linewidth=2.0, label="LLR AUC normal miRNAs")

# Right hand x axis for TPR at fixed FPR
if fixed_FPR == True:
    ax2 = ax1.twinx()
    colours2 = ["mediumblue", "orange", "green", "brown"]

    ax2.plot(num_miRNAs, both_tpr_L1[0], colours2[0], linewidth=2.0, label="L1 tpr diseased miRNAs")
    ax2.plot(num_miRNAs, both_tpr_LLR[0], colours2[1], linewidth=2.0, label="LLR tpr diseased miRNAs")
    ax2.plot(num_miRNAs, both_tpr_L1[1], colours2[2], linewidth=2.0, label="L1 tpr normal miRNAs")
    ax2.plot(num_miRNAs, both_tpr_LLR[1], colours2[3], linewidth=2.0, label="LLR tpr normal miRNAs")

ax1.legend(loc='upper right')
if fixed_FPR == True:
    # Merge handles and labels
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    # Add combined legend to one axis
    ax1.legend(h1 + h2, l1 + l2, loc='upper right')
    ax2.set_ylabel("TPR at 0.01 FPR")

ax1.invert_xaxis()
ax1.set_xlabel("number miRNAs")
ax1.set_ylabel("AUC scores")
ax1.set_ylim([0.5,1]) # enables comparable auc scores between L1 and LLR
ax1.grid(True)

plt.show() 
