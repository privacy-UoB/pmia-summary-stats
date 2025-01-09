import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import stats
from sklearn.metrics import roc_auc_score
from utils import load_timestamp_dataset, LLR, L1, L1_ttest

include_synthetic_noise = True
include_pvalue_histogram = True
selected_distribution = 4
# 0 = fixed Gaussian
# 1 = shifted Gaussian
# 2 = skewed normal - function call doesn't work
# 3 = skewed Cauchy
# 4 = sanity check (samples from real dataset)

num_orders = 20 # number of iterations to average over
auc_L1 = []
auc_LLR = []

if include_synthetic_noise:
    auc_syntheticL1 = []
    auc_syntheticLLR = []

if include_pvalue_histogram:
    p_values_realpop_L1 = []
    p_values_realpool_L1 = []
    p_values_realpop_LLR = []
    p_values_realpool_LLR = []
    p_values_synthpop_L1 = []
    p_values_synthpool_L1 = []
    p_values_synthpop_LLR = []
    p_values_synthpool_LLR = []

# for loop for numorder lots of train/test, then average at end
for j in range (num_orders):

    aucs_L1 = []
    aucs_LLR = []

    if include_synthetic_noise:
        aucs_syntheticL1 = []
        aucs_syntheticLLR = []

    # load new partitioned dataset each time we call num_orders
    (ti_pop, ti_pool, ti_sample) = load_timestamp_dataset()

    for x, y in zip(ti_pop, ti_pool):
        x.drop(["disease", "timepoint", "patient_id"], axis=1, inplace=True)
        # for row in range(len(x)):
        #     (x.iloc[row]).dropna(inplace=True) # remove NaN rows from the dataframe
        # print(x)

        y.drop(["disease", "timepoint", "patient_id"], axis=1, inplace=True)
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

    # configuring the reference pop & pool to match the dataframe of a particular timepoint
    pop = ti_pop[0]
    pool = ti_pool[0]

    for t_pop, t_pool in zip(ti_pop, ti_pool):

        # the 'noise' increases throughout each of the later timepoints the data is collected from
        # local_noised_pop = t_pop
        # local_noised_pool = t_pool
        # local_pop = pop
        # local_pool = pool

        # get values for L1 & LLR statistics over the noisy stat inputs compared to the 'original' pop & pool
        pvalue_pop_L1 = L1_ttest(t_pop, pop, pool)
        pvalue_pool_L1 = L1_ttest(t_pool, pop, pool)

        if include_pvalue_histogram:
            p_values_realpop_L1.append(pvalue_pop_L1)
            p_values_realpool_L1.append(pvalue_pool_L1)
    
        pvalue_pop_LLR = LLR(t_pop, pop, pool)
        pvalue_pool_LLR = LLR(t_pool, pop, pool)

        if include_pvalue_histogram:
            p_values_realpop_LLR.append((pvalue_pop_LLR.ravel()))
            p_values_realpool_LLR.append((pvalue_pool_LLR.ravel()))

        # set p-value and run attack without it being calculated
        # above list will then automatically accept or reject based on this value ???

        # print("standard deviations.", np.std(local_noised_pop), np.std(local_noised_pool), 
        #       np.std(local_pop), np.std(local_pool))

        # determine the performance of the attack comparing the accuracy of inference to the real data
        y_true_L1 = np.concatenate((np.zeros(len(pvalue_pop_L1)), np.ones(len(pvalue_pool_L1))))
        y_score_L1 = np.concatenate((pvalue_pop_L1, pvalue_pool_L1))
        roc_L1 = roc_auc_score(y_true_L1, y_score_L1)
        aucs_L1.append(roc_L1)

        y_true_LLR = np.concatenate((np.zeros(len(pvalue_pop_LLR)), np.ones(len(pvalue_pool_LLR))))
        y_score_LLR = np.concatenate((pvalue_pop_LLR, pvalue_pool_LLR))
        roc_LLR = roc_auc_score(y_true_LLR, y_score_LLR)
        aucs_LLR.append(roc_LLR)

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

                pop_noise = np.random.normal(a1, m1, local_pop.shape)
                noised_pop = local_pop + pop_noise
                pool_noise = np.random.normal(a2, m2, local_pool.shape)
                noised_pool = local_pool + pool_noise

                skew_test = stats.skewtest(c, axis=0)
                print(skew_test, skew1, skew2)

            if selected_distribution == 1: # shifted Gaussian
                # when creating local_noised_pop by adding to local_pop, you must ensure the distribution is shifted
                # ensure mean/variance are tailored to each miRNA for all 26 patients
                shifted_mean_pop = np.mean(pop_diff, axis=0)
                shifted_mean_pool = np.mean(pool_diff, axis=0)
                shifted_variance_pop = np.std(pop_diff, axis=0)
                shifted_variance_pool = np.std(pool_diff, axis=0)

                pop_noise = np.random.normal(shifted_mean_pop, shifted_variance_pop, local_pop.shape)
                noised_pop = local_pop + pop_noise
                pool_noise = np.random.normal(shifted_mean_pool, shifted_variance_pool, local_pool.shape)
                noised_pool = local_pool + pool_noise

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
                noised_pop = local_pop + pop_noise

                unshaped_pool_noise = stats.skewnorm.pdf(x2, args=(a2, loc2, scale2))
                pool_noise = np.reshape(unshaped_pool_noise, (len(local_pool), 1205))
                noised_pool = local_pool + pool_noise
                
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
                noised_pop = local_pop + pop_noise

                unshaped_pool_noise = (stats.skewcauchy(a=a2, loc=loc2, scale=scale2).rvs(size=(len(local_pool)*1205)))
                # unshaped_pool_noise = (stats.skewcauchy.rvs(a=a2, loc=loc2, scale=scale2, size=(len(local_pool)*1205)))
                pool_minimum = (np.std(unshaped_pool_noise))*(np.min(local_pool))
                pool_maximum = (np.std(unshaped_pool_noise))*(np.max(local_pool))
                clipped_pool_noise = np.clip(unshaped_pool_noise, pool_minimum, pool_maximum)
                pool_noise = np.reshape(clipped_pool_noise, (len(local_pool), 1205))
                noised_pool = local_pool + pool_noise

            if selected_distribution == 4: # sanity check (samples from real dataset)
                # Instead of synthetic distribution, sample differences from real timepoints and run - sanity check
                shuffled_pop_diff = list(pop_diff)
                random.shuffle(shuffled_pop_diff)
                pop_noise = np.reshape(shuffled_pop_diff, local_pop.shape)
                noised_pop = local_pop + pop_noise

                shuffled_pool_diff = list(pool_diff)
                random.shuffle(shuffled_pool_diff)
                pool_noise = np.reshape(shuffled_pool_diff, local_pool.shape)
                noised_pool = local_pool + pool_noise


            # # plot histogram of the timestamp differences vs the modelled synthetic noise added to timestamp0
            plt.hist((c, np.concatenate((np.ravel(pop_noise), np.ravel(pool_noise)))), bins=50, 
                    label=(f"differences of timepoints", "selected distribution"))
            plt.xlabel("difference")
            plt.ylabel("count")
            plt.legend(loc="upper right")
            plt.show()


            # get values for L1 & LLR statistics over the noisy stat inputs compared to the 'original' pop & pool
            pvalue_syntheticpop_L1 = L1_ttest(noised_pop, pop, pool)
            pvalue_syntheticpool_L1 = L1_ttest(noised_pool, pop, pool)

            if include_pvalue_histogram:
                p_values_synthpop_L1.append(pvalue_syntheticpop_L1)
                p_values_synthpool_L1.append(pvalue_syntheticpool_L1)
        
            pvalue_syntheticpop_LLR = LLR(noised_pop, pop, pool)
            pvalue_syntheticpool_LLR = LLR(noised_pool, pop, pool)

            if include_pvalue_histogram:
                p_values_synthpop_LLR.append((pvalue_syntheticpop_LLR.ravel()))
                p_values_synthpool_LLR.append((pvalue_syntheticpool_LLR.ravel()))

            # determine the performance of the attack comparing the accuracy of inference to the real data
            y_true_syntheticL1 = np.concatenate((np.zeros(len(pvalue_syntheticpop_L1)), np.ones(len(pvalue_syntheticpool_L1))))
            y_score_syntheticL1 = np.concatenate((pvalue_syntheticpop_L1, pvalue_syntheticpool_L1))
            roc_syntheticL1 = roc_auc_score(y_true_syntheticL1, y_score_syntheticL1)
            aucs_syntheticL1.append(roc_syntheticL1)

            y_true_syntheticLLR = np.concatenate((np.zeros(len(pvalue_syntheticpop_LLR)), np.ones(len(pvalue_syntheticpool_LLR))))
            y_score_syntheticLLR = np.concatenate((pvalue_syntheticpop_LLR, pvalue_syntheticpool_LLR))
            roc_syntheticLLR = roc_auc_score(y_true_syntheticLLR, y_score_syntheticLLR)
            aucs_syntheticLLR.append(roc_syntheticLLR)

    # num_order rows of datasets, columns are each timestamp
    if len(aucs_L1) >0:
        auc_L1.append(aucs_L1)

    if len(aucs_LLR) >0:
        auc_LLR.append(aucs_LLR)

    if include_synthetic_noise:
        # num_order rows of datasets, columns are each timestamp
        if len(aucs_syntheticL1) >0:
            auc_syntheticL1.append(aucs_syntheticL1)

        if len(aucs_syntheticLLR) >0:
            auc_syntheticLLR.append(aucs_syntheticLLR)

# averaging the results from num_order iterations
auc_L1 = np.average(auc_L1, axis=0)
auc_LLR = np.average(auc_LLR, axis=0)

if include_synthetic_noise:
    # averaging the results from num_order iterations
    auc_syntheticL1 = np.average(auc_syntheticL1, axis=0)
    auc_syntheticLLR = np.average(auc_syntheticLLR, axis=0)

# plotting the performance of the inference for each of the 8 timestamps
fig, ax = plt.subplots()
ax.plot(range(len(ti_pop)), auc_L1, "-b", linewidth=2.0, label="L1")
ax.plot(range(len(ti_pool)), auc_LLR, "-r", linewidth=2.0, label="LLR")
if include_synthetic_noise:
    ax.plot(range(len(auc_syntheticL1)), auc_syntheticL1, "-g", linewidth=2.0, label="L1")
    ax.plot(range(len(auc_syntheticLLR)), auc_syntheticLLR, "-y", linewidth=2.0, label="LLR")
ax.set_ylim([0.3,1]) # enables comparable auc scores between L1 and LLR

plt.xlabel("timestamp")
plt.ylabel("AUC scores")
plt.legend(loc="upper right")
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
