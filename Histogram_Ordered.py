import matplotlib.pyplot as plt
import random
import itertools
from utils_datasets import load_dataset, D3, split_pool, drop_dataset_index
from utils import auc_scores

# load dataset
pop_rpool, pop_cpool, rpool, cpool = load_dataset(case_sample=D3)
pop_rpool, pop_cpool, rpool, cpool = drop_dataset_index(pop_rpool, pop_cpool, rpool, cpool)

pop = pop_cpool # make pop configurable
pool = cpool # make pool configurable

split_pop, split_cpool, cpool_into_pop = split_pool(pop, pool)
pop = split_pop
pool = split_cpool


num_miRNAs = []
miRNAs = list(pop.keys()) # get the list of miRNAs ["miRNA_1234", "miRNA_1235", ...]
current_miRNA_list = list(miRNAs)
random.shuffle(current_miRNA_list)

aucs_L1 = []
aucs_LLR = []
p_values_pop_L1 = []
p_values_pool_L1 = []
p_values_pop_LLR = []
p_values_pool_LLR = []

for i in range(2,len(miRNAs),2): # MiRNAs range from 1 to 466 in paper
    num_miRNAs.append(i)
    selected_miRNAs = current_miRNA_list[:i]

    local_pop = pop[selected_miRNAs]
    local_pool = pool[selected_miRNAs]

    # Query: should these actually be local_pop, local_pool, pop, pool?
    roc_L1, pvalue_pop_L1, pvalue_pool_L1 = auc_scores(local_pop, local_pool, local_pop, local_pool)
    roc_LLR, pvalue_pop_LLR, pvalue_pool_LLR = auc_scores(local_pop, local_pool, local_pop, local_pool, LR=True)

    p_values_pop_L1.append(pvalue_pop_L1)
    p_values_pool_L1.append(pvalue_pool_L1)
    p_values_pop_LLR.append((pvalue_pop_LLR.ravel()))
    p_values_pool_LLR.append((pvalue_pool_LLR.ravel()))

    aucs_L1.append(roc_L1)
    aucs_LLR.append(roc_LLR)
        
flat_p_values_pop_L1 = list(itertools.chain.from_iterable(p_values_pop_L1))
flat_p_values_pool_L1 = list(itertools.chain.from_iterable(p_values_pool_L1))
flat_p_values_pop_LLR = list(itertools.chain.from_iterable(p_values_pop_LLR))
flat_p_values_pool_LLR = list(itertools.chain.from_iterable(p_values_pool_LLR))

# histogram showing standard deviations across all 8 timestamps of the individual
# L1
plt.hist(p_values_pop_L1, bins=40)
plt.xlabel("p values population L1")
plt.ylabel("count of deviations across 40 different range values")
plt.show()

plt.hist(p_values_pool_L1, bins=40)
plt.xlabel("p values pool L1")
plt.ylabel("count of deviations across 40 different range values")
plt.show()

plt.hist((flat_p_values_pop_L1, flat_p_values_pool_L1), bins=40)
plt.xlabel("p values pop & pool L1")
plt.ylabel("count of deviations across 40 different range values")
plt.show()

# LLR
plt.hist(p_values_pop_LLR, bins=40)
plt.xlabel("p values population LLR")
plt.ylabel("count of deviations across 40 different range values")
plt.show()

plt.hist(p_values_pool_LLR, bins=40)
plt.xlabel("p values pool LLR")
plt.ylabel("count of deviations across 40 different range values")
plt.show()

plt.hist((flat_p_values_pop_LLR, flat_p_values_pool_LLR), bins=40)
plt.xlabel("p values pop & pool LLR")
plt.ylabel("count of deviations across 40 different range values")
plt.show()
