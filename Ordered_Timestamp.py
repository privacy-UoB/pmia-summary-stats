import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import stats
from sklearn.preprocessing import normalize
from tabulate import tabulate
from utils_datasets import load_dataset, separate_diseased_miRNAs, independent, D2
from utils import auc_scores, normalise, Gaussian_noise, L1

include_synthetic_noise = True
include_pvalue_histogram = False
include_tabulate = False
stratifying = False
fixed_FPR = True
selected_distribution = 7
# 0 = fixed Gaussian
# 1 = shifted Gaussian
# 2 = skewed normal - function call doesn't work
# 3 = skewed Cauchy
# 4 = sanity check 1 (samples from real dataset)
# 5 = sanity check 2 (shuffling the distance from time i to time i+1), not working yet
# 6 = sanity check 3 (shuffling the vector from time i to time i+1)
# 7 = sanity check 4 (shuffling the normalised vector from time i to time i+1)

num_orders = 2000 # number of iterations to average over
auc_L1 = []
auc_LLR = []

if fixed_FPR:
    target_fpr = 1e-2

    tpr_at_fpr_L1 = []
    tpr_at_fpr_LLR = []

if include_synthetic_noise:
    auc_syntheticL1 = []
    auc_syntheticLLR = []

    if fixed_FPR:
        fpr_syntheticL1 = []
        fpr_syntheticLLR = []

if include_pvalue_histogram:
    p_values_realpop_L1 = []
    p_values_realpool_L1 = []
    p_values_realpop_LLR = []
    p_values_realpool_LLR = []
    p_values_synthpop_L1 = []
    p_values_synthpool_L1 = []
    p_values_synthpop_LLR = []
    p_values_synthpool_LLR = []

if include_tabulate:
    table_counter = []
    stats_counter = []
    total_counter = []
    num_orders = 1
    table_of_pop = True
    individual = 0

    def counter(person, population=True):
        if population==True:
            result = len([n for n in person if n<=0])
        else:
            result = len([n for n in person if n>0])
        return result
        

# for loop for numorder lots of train/test, then average at end
for j in range (num_orders):

    aucs_L1 = []
    aucs_LLR = []

    if fixed_FPR == True:
        tpr_at_fprs_L1 = []
        tpr_at_fprs_LLR = []

    if include_synthetic_noise:
        aucs_syntheticL1 = []
        aucs_syntheticLLR = []

        if fixed_FPR:
            fprs_syntheticL1 = []
            fprs_syntheticLLR = []

    # load new partitioned dataset each time we call num_orders
    if stratifying == False:
        ti_pop, ti_pool = load_dataset(timestamp=True, with_independent_features=True)
    # ti_pop, ti_pool, statistics, independent_columns = independent(ti_pop, ti_pool, correlation=0.8)
    # ti_pop, ti_pool = independent(ti_pop, ti_pool, correlation=0.9)

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

    # normalise over the miRNA values
    # ti_pop, ti_pool = normalise(ti_pop, ti_pool)
    # currently isn't working... think need to drop column names for normalise then readd
    # for x, y in zip(ti_pop, ti_pool):
    #     x, y = normalise(x, y)

    else:
        # lung disease miRNAs only/excluded in longitudinal pop & pool
        only_pops, without_pops, only_pools, without_pools = separate_diseased_miRNAs(D2, "timestamp", with_independent_features=True)
        ti_pop = without_pops
        ti_pool = without_pools

        miRNAs = list(ti_pop[0].keys())
        current_miRNA_list = random.sample(miRNAs, len(only_pops[0].columns))

        for index, (t_pop, t_pool) in enumerate(zip(ti_pop, ti_pool)):
            t_pop = t_pop[t_pop.columns.intersection(current_miRNA_list)]
            t_pool = t_pool[t_pool.columns.intersection(current_miRNA_list)]

            ti_pop[index] = t_pop
            ti_pool[index] = t_pool

    # configuring the reference pop & pool to match the dataframe of a particular timepoint
    pop = ti_pop[0]
    pool = ti_pool[0]

    # the 'noise' increases throughout each of the later timepoints the data is collected from
    for t_pop, t_pool in zip(ti_pop, ti_pool):

        # get performance/accuracy for L1 & LLR statistics over the noisy stat inputs compared to the 'original' pop & pool
        roc_L1, pvalue_pop_L1, pvalue_pool_L1 = auc_scores(t_pop, t_pool, pop, pool)
        roc_LLR, pvalue_pop_LLR, pvalue_pool_LLR = auc_scores(t_pop, t_pool, pop, pool, LR=True)

        aucs_L1.append(roc_L1)
        aucs_LLR.append(roc_LLR)

        if fixed_FPR == True:
            fpr_L1, tpr_L1, thresholds_L1 = auc_scores(t_pop, t_pool, pop, pool, FPR=True)
            fpr_LLR, tpr_LLR, thresholds_LLR = auc_scores(t_pop, t_pool, pop, pool, LR=True, FPR=True)

            # TPR at a fixed FPR (e.g., 0.01 = 1%)
            target_fpr = 1e-2
            tpr_at_fprs_L1.append(np.interp(target_fpr, fpr_L1, tpr_L1))
            tpr_at_fprs_LLR.append(np.interp(target_fpr, fpr_LLR, tpr_LLR))

        if include_tabulate:
            # create table for original data
            data = []
            data.extend([np.average(pop, axis=0)] + [np.average(pool, axis=0)])
            orig_pop, orig_pop_mu, orig_pop_muhat = L1(t_pop, pop, pool, table=True)
            orig_pool, orig_pool_mu, orig_pool_muhat = L1(t_pool, pop, pool, table=True)
            if table_of_pop==True:
                data.extend([np.array(t_pop.iloc[individual])] + [orig_pop_mu[individual]] + [orig_pop_muhat[individual]])
                original_table_count = counter(orig_pop[individual])
            else:
                data.extend([np.array(t_pool.iloc[individual])] + [orig_pool_mu[individual]] + [orig_pool_muhat[individual]])
                original_table_count = counter(orig_pool[individual], population=False)

            # create counter for all individuals for per timestamp stats table
            sum_orig = [[],[],[]]
            closer = True
            for a, b in zip(orig_pop_mu, orig_pop_muhat):
                sum_orig[0].append(np.sum(a))
                sum_orig[1].append(np.sum(b))
            if closer==True:
                for c, d in zip(orig_pool_mu, orig_pool_muhat):
                    sum_orig[1].append(np.sum(c))
                    sum_orig[0].append(np.sum(d))
            else:
                for c, d in zip(orig_pool_mu, orig_pool_muhat):
                    sum_orig[0].append(np.sum(c))
                    sum_orig[1].append(np.sum(d))
            sum_orig[2] = np.subtract(sum_orig[0], sum_orig[1])

            # create counter for all individuals for original data
            total_opop_count = []
            total_opool_count = []
            for pop_ind in orig_pop:
                pop_count = counter(pop_ind)
                total_opop_count.append(pop_count)
            for pool_ind in orig_pool:
                pool_count = counter(pool_ind, population=False)
                total_opool_count.append(pool_count)

        if include_pvalue_histogram:
            p_values_realpop_L1.append(pvalue_pop_L1)
            p_values_realpool_L1.append(pvalue_pool_L1)
            p_values_realpop_LLR.append((pvalue_pop_LLR.ravel()))
            p_values_realpool_LLR.append((pvalue_pool_LLR.ravel()))

        # set p-value and run attack without it being calculated
        # above list will then automatically accept or reject based on this value ???

        # print("standard deviations.", np.std(local_noised_pop), np.std(local_noised_pool), 
        #       np.std(local_pop), np.std(local_pool))

        if include_synthetic_noise:
            # think about the t_pop noise = np.normal(0, m) that Pascal drew
            # here we are trying to add the noise to tpop[0] that is normally distributed by the timestamp deviation
            # this should enable the same dataset split and also check for errors
            # add this to another set of aucs and plot on the same graph
            local_noised_pop = np.array(t_pop)
            local_noised_pool = np.array(t_pool)
            local_pop = np.array(pop)
            local_pool = np.array(pool)

            # local_pop = local_pop[:len(local_noised_pop), :]
            # local_pool = local_pool[:len(local_noised_pool), :]

            if len(local_pop) != len(local_noised_pop) or len(local_pool) != len(local_noised_pool):
                continue

            pop_diff = np.ravel(np.subtract(local_pop, local_noised_pop))
            pool_diff = np.ravel(np.subtract(local_pool, local_noised_pool))
            c = np.concatenate((pop_diff, pool_diff))
            m = np.std(c)

            if selected_distribution == 0: # fixed Gaussian
                a1 = np.mean(local_noised_pop, axis=0) # mu_j of miRNAs for each pop timestamp
                a2 = np.mean(local_noised_pool, axis=0) # mu_j of miRNAs for each pool timestamp
                m1 = np.std(local_noised_pop, axis=0) # sigma_j of miRNAs for each pop timestamp
                m2 = np.std(local_noised_pool, axis=0) # sigma_j of miRNAs for each pool timestamp
                skew1 = stats.skew(local_noised_pop, axis=0) # skew_j of miRNAs for each pop timestamp
                skew2 = stats.skew(local_noised_pool, axis=0) # skew_j of miRNAs for each pool timestamp

                noisy_pop, noisy_pool = Gaussian_noise(local_pop, local_pool, a1, m1, mean2=a2, deviation2=m2)

                skew_test = stats.skewtest(c, axis=0)
                print(skew_test, skew1, skew2)

            if selected_distribution == 1: # shifted Gaussian
                # when creating local_noised_pop by adding to local_pop, you must ensure the distribution is shifted
                # ensure mean/variance are tailored to each miRNA for all 26 patients
                shifted_a1 = np.mean(pop_diff, axis=0)
                shifted_a2 = np.mean(pool_diff, axis=0)
                shifted_m1 = np.std(pop_diff, axis=0)
                shifted_m2 = np.std(pool_diff, axis=0)

                noisy_pop, noisy_pool = Gaussian_noise(local_pop, local_pool, shifted_a1, shifted_m1, mean2=shifted_a2, deviation2=shifted_m2)

            if selected_distribution == 2: # skewed normal
                # https://stackoverflow.com/questions/5884768/skew-normal-distribution-in-scipy?rq=3
                # X = np.linspace(min(your_data), max(your_data))
                # plt.plot(X, skewnorm.pdf(X, *skewnorm.fit(your_data)))
                print(pop_diff.shape, pool_diff.shape)
                a1, loc1, scale1 = stats.skewnorm.fit(pop_diff)
                a2, loc2, scale2 = stats.skewnorm.fit(pool_diff)
                x1 = np.linspace(min(pop_diff), max(pop_diff))
                x2 = np.linspace(min(pool_diff), max(pool_diff))

                unshaped_pop_noise = stats.skewnorm.pdf(x1, args=(a1, loc1, scale1))
                pop_noise = np.reshape(unshaped_pop_noise, (len(local_pop), 1205))
                noisy_pop = local_pop + pop_noise

                unshaped_pool_noise = stats.skewnorm.pdf(x2, args=(a2, loc2, scale2))
                pool_noise = np.reshape(unshaped_pool_noise, (len(local_pool), 1205))
                noisy_pool = local_pool + pool_noise
                
            if selected_distribution == 3: # skewed Cauchy
                # from a brief search this seems to be the best to model a big 'spike' at 0
                a1, loc1, scale1 = stats.skewcauchy.fit(pop_diff)
                a2, loc2, scale2 = stats.skewcauchy.fit(pool_diff)

                unshaped_pop_noise = stats.skewcauchy(a=a1, loc=loc1, scale=scale1).rvs(size=(len(local_pop)*1205))
                # unshaped_pop_noise = stats.skewcauchy.rvs(a=a1, loc=loc1, scale=scale1, size=(len(local_pop)*1205))
                pop_minimum = (np.std(unshaped_pop_noise))*(np.min(local_pop))
                pop_maximum = (np.std(unshaped_pop_noise))*(np.max(local_pop))
                clipped_pop_noise = np.clip(unshaped_pop_noise, pop_minimum, pop_maximum)
                pop_noise = np.reshape(clipped_pop_noise, (len(local_pop), 1205))
                noisy_pop = local_pop + pop_noise

                unshaped_pool_noise = (stats.skewcauchy(a=a2, loc=loc2, scale=scale2).rvs(size=(len(local_pool)*1205)))
                # unshaped_pool_noise = (stats.skewcauchy.rvs(a=a2, loc=loc2, scale=scale2, size=(len(local_pool)*1205)))
                pool_minimum = (np.std(unshaped_pool_noise))*(np.min(local_pool))
                pool_maximum = (np.std(unshaped_pool_noise))*(np.max(local_pool))
                clipped_pool_noise = np.clip(unshaped_pool_noise, pool_minimum, pool_maximum)
                pool_noise = np.reshape(clipped_pool_noise, (len(local_pool), 1205))
                noisy_pool = local_pool + pool_noise

            if selected_distribution == 4: # sanity check 1 (samples from real dataset)
                # Instead of synthetic distribution, sample differences from real timepoints and run - sanity check
                shuffled_pop_diff = list(pop_diff)
                random.shuffle(shuffled_pop_diff)
                pop_noise = np.reshape(shuffled_pop_diff, local_pop.shape)
                noisy_pop = local_pop + pop_noise

                shuffled_pool_diff = list(pool_diff)
                random.shuffle(shuffled_pool_diff)
                pool_noise = np.reshape(shuffled_pool_diff, local_pool.shape)
                noisy_pool = local_pool + pool_noise

            if selected_distribution == 5: # sanity check 2 (shuffling the distance from time i to time i+1)
                # approach where compare the distance between timepoints and the mean
                # then ensure that the distance is added in the right direction (and away from the other mean?)
                
                if index == 0:
                    # |t0 - t0mean|
                    noise0, noise_pop0, noise_pop_alt0 = L1(local_noised_pop, local_pop, local_pool)
                    noise0, noise_pool0, noise_pool_alt0 = L1(local_noised_pool, local_pop, local_pool)

                    noisy_pop = local_noised_pop
                    noisy_pool = local_noised_pool
                else:
                    # time - t0 (vector movement from t0)
                    t0_vector = np.subtract(local_noised_pop, local_pop)
                    t0_vector = np.subtract(local_noised_pool, local_pool)
                    # |time - t0mean|
                    noise, noise_pop, noise_pop_alt = L1(local_noised_pop, local_pop, local_pool)
                    noise, noise_pool, noise_pool_alt = L1(local_noised_pool, local_pop, local_pool)

                    if (np.subtract(noise_pop, noise_pop0)>0):
                        pop_vector = local_noised_pop + noise_pop0
                    else:
                        pop_vector = local_noised_pop - noise_pop0

                    if (np.subtract(noise_pool, noise_pool)>0):
                        pool_vector = local_noised_pool + noise_pool0
                    else:
                        pool_vector = local_noised_pool - noise_pool0
                        # shuffle
                        # as selected_distribution == 4

            if selected_distribution == 6: # sanity check 3 (shuffling the vector from time i to time i+1)
                # if True then vector is taken via pop mean; otherwise via pool mean
                from_pop_mean = True

                # gather vectors from timepoint to corresponding pop/pool mean to timepoint 0
                if from_pop_mean == True:
                    pop_vector = (np.subtract(local_noised_pop, np.average(local_pop, axis=0)) + 
                                    np.subtract(np.average(local_pop, axis=0), local_pop))
                    pool_vector = (np.subtract(local_noised_pool, np.average(local_pop, axis=0)) + 
                                    np.subtract(np.average(local_pop, axis=0), local_pool))
                else:
                    pop_vector = (np.subtract(local_noised_pop, np.average(local_pool, axis=0)) + 
                                    np.subtract(np.average(local_pool, axis=0), local_pop))
                    pool_vector = (np.subtract(local_noised_pool, np.average(local_pool, axis=0)) + 
                                    np.subtract(np.average(local_pool, axis=0), local_pool))
                
                # shuffle vectors
                shuffled_pop_vector = list(pop_vector)
                random.shuffle(shuffled_pop_vector)
                pop_noise = np.reshape(shuffled_pop_vector, local_pop.shape)
                noisy_pop = local_pop + pop_noise

                shuffled_pool_vector = list(pool_vector)
                random.shuffle(shuffled_pool_vector)
                pool_noise = np.reshape(shuffled_pool_vector, local_pool.shape)
                noisy_pool = local_pool + pool_noise

            if selected_distribution == 7: # sanity check 4 (shuffling the normalised vector from time i to time i+1)
                # if True then vector is taken via pop mean; otherwise via pool mean
                from_pop_mean = True

                # gather vectors from timepoint to corresponding pop/pool mean to timepoint 0
                if from_pop_mean == True:
                    pop_vector = (np.subtract(local_noised_pop, np.average(local_pop, axis=0)) + 
                                    np.subtract(np.average(local_pop, axis=0), local_pop))
                    pool_vector = (np.subtract(local_noised_pool, np.average(local_pop, axis=0)) + 
                                    np.subtract(np.average(local_pop, axis=0), local_pool))
                else:
                    pop_vector = (np.subtract(local_noised_pop, np.average(local_pool, axis=0)) + 
                                    np.subtract(np.average(local_pool, axis=0), local_pop))
                    pool_vector = (np.subtract(local_noised_pool, np.average(local_pool, axis=0)) + 
                                    np.subtract(np.average(local_pool, axis=0), local_pool))
                
                # shuffle vectors
                shuffled_pop_vector = list(pop_vector)
                random.shuffle(shuffled_pop_vector)
                shuffled_pop_vector = np.divide(shuffled_pop_vector, local_noised_pop)
                pop_noise = np.reshape(shuffled_pop_vector, local_pop.shape)
                noisy_pop = local_pop + pop_noise

                shuffled_pool_vector = list(pool_vector)
                random.shuffle(shuffled_pool_vector)
                shuffled_pool_vector = np.divide(shuffled_pool_vector, local_noised_pool)
                pool_noise = np.reshape(shuffled_pool_vector, local_pool.shape)
                noisy_pool = local_pool + pool_noise


            # # plot histogram of the timestamp differences vs the modelled synthetic noise added to timestamp0
            # plt.hist((c, np.concatenate((np.ravel(pop_noise), np.ravel(pool_noise)))), bins=50, 
            #         label=(f"differences of timepoints", "selected distribution"))
            # plt.xlabel("difference")
            # plt.ylabel("count")
            # plt.legend(loc="upper right")
            # plt.show()

            # get performance/accuracy for L1 & LLR statistics over the noisy stat inputs compared to the 'original' pop & pool
            roc_synthL1, pvalue_synthpop_L1, pvalue_synthpool_L1 = auc_scores(noisy_pop, noisy_pool, pop, pool)
            roc_synthLLR, pvalue_synthpop_LLR, pvalue_synthpool_LLR = auc_scores(noisy_pop, noisy_pool, pop, pool, LR=True)

            aucs_syntheticL1.append(roc_synthL1)
            aucs_syntheticLLR.append(roc_synthLLR)

            if fixed_FPR == True:
                fpr_synthL1, tpr_synthL1, thresholds_synthL1 = auc_scores(t_pop, t_pool, pop, pool, FPR=True)
                fpr_synthLLR, tpr_synthLLR, thresholds_synthLLR = auc_scores(t_pop, t_pool, pop, pool, LR=True, FPR=True)

                # TPR at a fixed FPR (e.g., 0.01 = 1%)
                target_fpr = 1e-2
                fprs_syntheticL1.append(np.interp(target_fpr, fpr_synthL1, tpr_synthL1))
                fprs_syntheticLLR.append(np.interp(target_fpr, fpr_synthLLR, tpr_synthLLR))

            if include_tabulate:
                # create table for synthetic data
                synth_pop, synth_pop_mu, synth_pop_muhat = L1(noisy_pop, pop, pool, table=True)
                synth_pool, synth_pool_mu, synth_pool_muhat = L1(noisy_pool, pop, pool, table=True)
                if table_of_pop==True:
                    data.extend([noisy_pop[individual]] + [synth_pop_mu[individual]] + [synth_pop_muhat[individual]])
                    synthetic_table_count = counter(synth_pop[individual])
                else:
                    data.extend([noisy_pool[individual]] + [synth_pool_mu[individual]] + [synth_pool_muhat[individual]])
                    synthetic_table_count = counter(synth_pool[individual], population=False)
                table_counter.append([original_table_count, synthetic_table_count])

                # create counter for all individuals for per timestamp stats table
                sum_synth = [[],[],[]]
                for a, b in zip(synth_pop_mu, synth_pop_muhat):
                    sum_synth[0].append(np.sum(a))
                    sum_synth[1].append(np.sum(b))
                if closer==True:
                    for c, d in zip(synth_pool_mu, synth_pool_muhat):
                        sum_synth[1].append(np.sum(c))
                        sum_synth[0].append(np.sum(d))
                else:
                    for c, d in zip(synth_pool_mu, synth_pool_muhat):
                        sum_synth[0].append(np.sum(c))
                        sum_synth[1].append(np.sum(d))
                sum_synth[2] = np.subtract(sum_synth[0], sum_synth[1])
                stats_counter.append([counter(sum_orig[2])] + [counter(sum_synth[2])])

                # create counter for all individuals for synthetic data
                total_spop_count = []
                total_spool_count = []
                for pop_ind in synth_pop:
                    pop_count = counter(pop_ind)
                    total_spop_count.append(pop_count)
                for pool_ind in synth_pool:
                    pool_count = counter(pool_ind, population=False)
                    total_spool_count.append(pool_count)
                total_counter.append([total_opop_count, total_opool_count, total_spop_count, total_spool_count])

                # create table (https://pypi.org/project/tabulate/) of L1 miRNA stats for one person per timestamp
                zipped_data = zip(*data)
                transposed_data = [list(sublist) for sublist in zipped_data]

                if table_of_pop==True:
                    print(tabulate(transposed_data[:40], headers=["mu", "mu-hat", "orig x:", "|x - mu|", "|x - mu-hat|", 
                                                             "synth x:", "|x - mu|", "|x - mu-hat|"], tablefmt="presto"), 
                                                             "\n original and synthetic number miRNAs closer to mu than muhat:", 
                                                             original_table_count, synthetic_table_count)
                else:
                    print(tabulate(transposed_data[:40], headers=["mu", "mu-hat", "orig x:", "|x - mu|", "|x - mu-hat|", 
                                                             "synth x:", "|x - mu|", "|x - mu-hat|"], tablefmt="presto"), 
                                                             "\n original and synthetic number miRNAs closer to muhat than mu:", 
                                                             original_table_count, synthetic_table_count)
                
                # create table of L1 stats per individual per timestamp to the correct pop/pool mean for data
                zipped_stats = zip(*(sum_orig + sum_synth))
                transposed_stats = [list(sublist) for sublist in zipped_stats]

                if closer==True:
                    print(tabulate(transposed_stats, headers=["sum orig right", "sum orig wrong", "sum orig diff", 
                                                            "sum synth right", "sum synth wrong", "sum synth diff"], 
                                                            showindex="always", tablefmt="presto"))
                else:
                    print(tabulate(transposed_stats, headers=["sum o |x - mu|", "sum o |x - mu-hat|", "sum o L1", 
                                                            "sum s |x - mu|", "sum s |x - mu-hat|", "sum s L1"], 
                                                            showindex="always", tablefmt="presto"))
                    
            if include_pvalue_histogram:
                p_values_synthpop_L1.append(pvalue_synthpop_L1)
                p_values_synthpool_L1.append(pvalue_synthpool_L1)
                p_values_synthpop_LLR.append((pvalue_synthpop_LLR.ravel()))
                p_values_synthpool_LLR.append((pvalue_synthpool_LLR.ravel()))

    # num_order rows of datasets, columns are each timestamp
    if len(aucs_L1) >0:
        auc_L1.append(aucs_L1)

    if len(aucs_LLR) >0:
        auc_LLR.append(aucs_LLR)

    if fixed_FPR:
        if len(tpr_at_fprs_L1) >0:
            tpr_at_fpr_L1.append(tpr_at_fprs_L1)

        if len(tpr_at_fprs_LLR) >0:
            tpr_at_fpr_LLR.append(tpr_at_fprs_LLR)

    if include_synthetic_noise:
        # num_order rows of datasets, columns are each timestamp
        if len(aucs_syntheticL1) >0:
            auc_syntheticL1.append(aucs_syntheticL1)

        if len(aucs_syntheticLLR) >0:
            auc_syntheticLLR.append(aucs_syntheticLLR)

        if fixed_FPR:
            if len(fprs_syntheticL1) >0:
                fpr_syntheticL1.append(fprs_syntheticL1)

            if len(fprs_syntheticLLR) >0:
                fpr_syntheticLLR.append(fprs_syntheticLLR)

    if include_tabulate:
        print("original vs synthetic counter per timestamp in table for indiv", table_counter)

        ordered_by_timestamp = []
        diff_count = [[],[]]
        for index, timestamp in enumerate(total_counter):
            # print(f"original pop/pool vs synthetic pop/pool counter for timestamp {index}", timestamp)

            # create table to compare counters over all original and synthetic individuals
            orig = timestamp[0] + timestamp[1]
            synt = timestamp[2] + timestamp[3]
            diff = np.subtract(orig, synt)
            diff_count[0].append(counter(diff, population=False))
            diff_count[1].append(counter(diff))

            ordered_by_timestamp.append(orig)
            ordered_by_timestamp.append(synt)

        zipped = zip(*ordered_by_timestamp)
        transposed = [list(sublist) for sublist in zipped]
        print(tabulate(transposed, headers=["orig 0", "synt 0", "orig 1", "synt 1", "orig 2", "synt 2", "orig 3", "synt 3", 
                                            "orig 4", "synt 4", "orig 5", "synt 5"], showindex="always", tablefmt="presto"),
                                            "\n number of people with more miRNAs closer to correct value in original data per time:", 
                                            diff_count[0], 
                                            "\n number of people with more miRNAs closer to correct value in synthetic data per time:", 
                                            diff_count[1],
                                            "\n number of people with distance closer to correct mean in original/synthetic per time",
                                            stats_counter)

# averaging the results from num_order iterations
auc_L1 = np.average(auc_L1, axis=0)
auc_LLR = np.average(auc_LLR, axis=0)

if fixed_FPR:
    tpr_at_fpr_L1 = np.average(tpr_at_fpr_L1, axis=0)
    tpr_at_fpr_LLR = np.average(tpr_at_fpr_LLR, axis=0)

if include_synthetic_noise:
    # averaging the results from num_order iterations
    auc_syntheticL1 = np.average(auc_syntheticL1, axis=0)
    auc_syntheticLLR = np.average(auc_syntheticLLR, axis=0)

    if fixed_FPR:
        fpr_syntheticL1 = np.average(fpr_syntheticL1, axis=0)
        fpr_syntheticLLR = np.average(fpr_syntheticLLR, axis=0)

# plotting the performance of the inference for each of the 8 timestamps
fig, ax1 = plt.subplots()
colours1 = ["cornflowerblue", "gold", "springgreen", "red"]
if fixed_FPR == True:
    ax2 = ax1.twinx()
    colours2 = ["mediumblue", "orange", "green", "brown"]

if not include_synthetic_noise:
    ax1.plot(range(len(ti_pop)), auc_L1, colours1[0], linewidth=2.0, label="AUC L1")
    ax1.plot(range(len(ti_pool)), auc_LLR, colours1[1], linewidth=2.0, label="AUC LLR")

    if fixed_FPR:
        ax2.plot(range(len(ti_pop)), tpr_at_fpr_L1, colours2[0], linewidth=2.0, label="fpr L1")
        ax2.plot(range(len(ti_pool)), tpr_at_fpr_LLR, colours2[1], linewidth=2.0, label="fpr LLR")

if include_synthetic_noise:
    ax1.plot(range(len(auc_syntheticL1)), auc_L1[:6], colours1[0], linewidth=2.0, label="AUC L1 real")
    ax1.plot(range(len(auc_syntheticLLR)), auc_LLR[:6], colours1[1], linewidth=2.0, label="AUC LLR real")
    ax1.plot(range(len(auc_syntheticL1)), auc_syntheticL1, colours1[2], linewidth=2.0, label="AUC L1 synth")
    ax1.plot(range(len(auc_syntheticLLR)), auc_syntheticLLR, colours1[3], linewidth=2.0, label="AUC LLR synth")

    if fixed_FPR:
        ax2.plot(range(len(fpr_syntheticL1)), fpr_L1[:6], colours2[0], linewidth=2.0, label="fpr L1 real")
        ax2.plot(range(len(fpr_syntheticLLR)), fpr_LLR[:6], colours2[1], linewidth=2.0, label="fpr LLR real")
        ax2.plot(range(len(fpr_syntheticL1)), fpr_syntheticL1, colours2[2], linewidth=2.0, label="fpr L1 synth")
        ax2.plot(range(len(fpr_syntheticLLR)), fpr_syntheticLLR, colours2[3], linewidth=2.0, label="fpr LLR synth")

ax1.legend(loc='upper right')
if fixed_FPR == True:
    # Merge handles and labels
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    # Add combined legend to one axis
    ax1.legend(h1 + h2, l1 + l2, loc='upper right')
    ax2.set_ylabel("TPR at 0.01 FPR")

ax1.set_xlabel("timestamp")
ax1.set_ylabel("AUC scores")
ax1.set_ylim([0.2,1.1]) # enables comparable auc scores between L1 and LLR
ax1.grid(True)

plt.show()

if include_pvalue_histogram:
    for m in range(num_orders):
        plt.hist((p_values_realpop_L1[m], p_values_realpool_L1[m], p_values_synthpop_L1[m], p_values_synthpool_L1[m]), 
                 bins=50, color=["blue", "red", "green", "gold"], label=f"script run {m}")
        plt.xlabel("p values pop & pool L1")
        plt.ylabel("count of deviations across 50 different range values")
        plt.legend(loc="upper right")
        plt.show()

        plt.hist((p_values_realpop_LLR[m], p_values_realpool_LLR[m], p_values_synthpop_LLR[m], p_values_synthpool_LLR[m]), 
                 bins=300, color=["blue", "red", "green", "gold"], label=f"script run {m}")
        plt.xlabel("p values pop & pool LLR")
        plt.ylabel("count of deviations across 300 different range values")
        plt.legend(loc="upper right")
        plt.show()
