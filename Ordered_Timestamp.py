import sys
import numpy as np
import matplotlib

from experiment_io import parse_flags, seed_all, save_figdata, load_figdata, resolve_output_path

_flags = parse_flags(sys.argv)
seed_all(_flags["seed"])

if len(sys.argv) >= 3 or _flags["replot"]:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random
from scipy import stats
from sklearn.preprocessing import normalize
from plot_style import line_kwargs, stacked_auc_tpr
from utils_datasets import load_dataset, separate_diseased_miRNAs, independent, D2, _prepare_timestamp_data, _split_timestamp, drop_timestamp_index
from utils import auc_scores, normalise, Gaussian_noise, L1


def make_figure(data: dict, output_path: str | None) -> None:
    fixed_FPR = bool(np.asarray(data["_fixed_FPR"]).item())
    include_synthetic_noise = bool(np.asarray(data["_include_synthetic_noise"]).item())

    auc_L1 = np.asarray(data["auc_L1"])
    auc_LLR = np.asarray(data["auc_LLR"])
    if fixed_FPR:
        tpr_at_fpr_L1 = np.asarray(data["tpr_at_fpr_L1"])
        tpr_at_fpr_LLR = np.asarray(data["tpr_at_fpr_LLR"])
    if include_synthetic_noise:
        auc_syntheticL1 = np.asarray(data["auc_syntheticL1"])
        auc_syntheticLLR = np.asarray(data["auc_syntheticLLR"])
        if fixed_FPR:
            fpr_syntheticL1 = np.asarray(data["fpr_syntheticL1"])
            fpr_syntheticLLR = np.asarray(data["fpr_syntheticLLR"])

    if fixed_FPR:
        fig, ax_auc, ax_tpr = stacked_auc_tpr()
    else:
        fig, ax_auc = plt.subplots()
        ax_tpr = None

    # real curves are solid; synthetic-noise overlay (if present) is dashed.
    if not include_synthetic_noise:
        ax_auc.plot(range(len(auc_L1)), auc_L1, label="L1",
                    **line_kwargs("L1", marker=None, linewidth=2.0))
        ax_auc.plot(range(len(auc_LLR)), auc_LLR, label="LLR",
                    **line_kwargs("LLR", marker=None, linewidth=2.0))
        if fixed_FPR:
            ax_tpr.plot(range(len(auc_L1)), tpr_at_fpr_L1, label="L1",
                        **line_kwargs("L1", marker=None, linewidth=2.0))
            ax_tpr.plot(range(len(auc_LLR)), tpr_at_fpr_LLR, label="LLR",
                        **line_kwargs("LLR", marker=None, linewidth=2.0))
    else:
        ax_auc.plot(range(len(auc_syntheticL1)), auc_L1[:6], label="L1 real",
                    **line_kwargs("L1", marker=None, linewidth=2.0))
        ax_auc.plot(range(len(auc_syntheticLLR)), auc_LLR[:6], label="LLR real",
                    **line_kwargs("LLR", marker=None, linewidth=2.0))
        ax_auc.plot(range(len(auc_syntheticL1)), auc_syntheticL1, label="L1 synth",
                    **line_kwargs("L1", marker=None, linewidth=2.0, linestyle="--"))
        ax_auc.plot(range(len(auc_syntheticLLR)), auc_syntheticLLR, label="LLR synth",
                    **line_kwargs("LLR", marker=None, linewidth=2.0, linestyle="--"))
        if fixed_FPR:
            ax_tpr.plot(range(len(fpr_syntheticL1)), tpr_at_fpr_L1[:6], label="L1 real",
                        **line_kwargs("L1", marker=None, linewidth=2.0))
            ax_tpr.plot(range(len(fpr_syntheticLLR)), tpr_at_fpr_LLR[:6], label="LLR real",
                        **line_kwargs("LLR", marker=None, linewidth=2.0))
            ax_tpr.plot(range(len(fpr_syntheticL1)), fpr_syntheticL1, label="L1 synth",
                        **line_kwargs("L1", marker=None, linewidth=2.0, linestyle="--"))
            ax_tpr.plot(range(len(fpr_syntheticLLR)), fpr_syntheticLLR, label="LLR synth",
                        **line_kwargs("LLR", marker=None, linewidth=2.0, linestyle="--"))

    ax_auc.legend(loc='upper right')
    ax_auc.set_ylabel("AUC")
    ax_auc.set_ylim([0.5, 1])
    ax_auc.grid(True)
    if fixed_FPR:
        ax_tpr.legend(loc='upper right')
        ax_tpr.set_xlabel("timestamp")
        ax_tpr.set_ylabel("TPR @ 1% FPR")
        ax_tpr.set_ylim([0, 1])
        ax_tpr.grid(True)
    else:
        ax_auc.set_xlabel("timestamp")

    if output_path:
        plt.savefig(output_path)
        print(f"Saved to {output_path}")
    else:
        plt.show()


# CLI: python Ordered_Timestamp.py [selected_distribution] [output.pdf]
selected_distribution = int(sys.argv[1]) if len(sys.argv) >= 2 else 0
OUTPUT_FILE = resolve_output_path(sys.argv[2] if len(sys.argv) >= 3 else None)

if _flags["replot"]:
    data, _meta = load_figdata(_flags["replot"])
    make_figure(data, OUTPUT_FILE)
    sys.exit(0)

include_synthetic_noise = True
include_pvalue_histogram = False
include_tabulate = False
stratifying = False
fixed_FPR = True
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
    from tabulate import tabulate
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


# Hoist timestamp CSV parse + independent-feature filter out of the outer loop;
# only the patient-id ShuffleSplit re-runs per iteration.
if stratifying == False:
    _prepared_timestamp = _prepare_timestamp_data(with_independent_miRNAs=True)


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

    # re-randomise pop/pool split each iteration; CSV parse is reused from above
    if stratifying == False:
        ti_pop, ti_pool, _sample_t, _healthy_t = _split_timestamp(_prepared_timestamp)
        ti_pop, ti_pool = drop_timestamp_index(ti_pop, ti_pool)

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

        if include_synthetic_noise:
            local_noised_pop = np.array(t_pop)
            local_noised_pool = np.array(t_pool)
            local_pop = np.array(pop)
            local_pool = np.array(pool)

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
                shifted_a1 = np.mean(pop_diff, axis=0)
                shifted_a2 = np.mean(pool_diff, axis=0)
                shifted_m1 = np.std(pop_diff, axis=0)
                shifted_m2 = np.std(pool_diff, axis=0)

                noisy_pop, noisy_pool = Gaussian_noise(local_pop, local_pool, shifted_a1, shifted_m1, mean2=shifted_a2, deviation2=shifted_m2)

            if selected_distribution == 2: # skewed normal
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
                a1, loc1, scale1 = stats.skewcauchy.fit(pop_diff)
                a2, loc2, scale2 = stats.skewcauchy.fit(pool_diff)

                unshaped_pop_noise = stats.skewcauchy(a=a1, loc=loc1, scale=scale1).rvs(size=(len(local_pop)*1205))
                pop_minimum = (np.std(unshaped_pop_noise))*(np.min(local_pop))
                pop_maximum = (np.std(unshaped_pop_noise))*(np.max(local_pop))
                clipped_pop_noise = np.clip(unshaped_pop_noise, pop_minimum, pop_maximum)
                pop_noise = np.reshape(clipped_pop_noise, (len(local_pop), 1205))
                noisy_pop = local_pop + pop_noise

                unshaped_pool_noise = (stats.skewcauchy(a=a2, loc=loc2, scale=scale2).rvs(size=(len(local_pool)*1205)))
                pool_minimum = (np.std(unshaped_pool_noise))*(np.min(local_pool))
                pool_maximum = (np.std(unshaped_pool_noise))*(np.max(local_pool))
                clipped_pool_noise = np.clip(unshaped_pool_noise, pool_minimum, pool_maximum)
                pool_noise = np.reshape(clipped_pool_noise, (len(local_pool), 1205))
                noisy_pool = local_pool + pool_noise

            if selected_distribution == 4: # sanity check 1 (samples from real dataset)
                shuffled_pop_diff = list(pop_diff)
                random.shuffle(shuffled_pop_diff)
                pop_noise = np.reshape(shuffled_pop_diff, local_pop.shape)
                noisy_pop = local_pop + pop_noise

                shuffled_pool_diff = list(pool_diff)
                random.shuffle(shuffled_pool_diff)
                pool_noise = np.reshape(shuffled_pool_diff, local_pool.shape)
                noisy_pool = local_pool + pool_noise

            if selected_distribution == 5: # sanity check 2 (shuffling the distance from time i to time i+1)
                if index == 0:
                    noise0, noise_pop0, noise_pop_alt0 = L1(local_noised_pop, local_pop, local_pool)
                    noise0, noise_pool0, noise_pool_alt0 = L1(local_noised_pool, local_pop, local_pool)

                    noisy_pop = local_noised_pop
                    noisy_pool = local_noised_pool
                else:
                    t0_vector = np.subtract(local_noised_pop, local_pop)
                    t0_vector = np.subtract(local_noised_pool, local_pool)
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

            if selected_distribution == 6: # sanity check 3 (shuffling the vector from time i to time i+1)
                from_pop_mean = True

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

                shuffled_pop_vector = list(pop_vector)
                random.shuffle(shuffled_pop_vector)
                pop_noise = np.reshape(shuffled_pop_vector, local_pop.shape)
                noisy_pop = local_pop + pop_noise

                shuffled_pool_vector = list(pool_vector)
                random.shuffle(shuffled_pool_vector)
                pool_noise = np.reshape(shuffled_pool_vector, local_pool.shape)
                noisy_pool = local_pool + pool_noise

            if selected_distribution == 7: # sanity check 4 (shuffling the normalised vector from time i to time i+1)
                from_pop_mean = True

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


            # get performance/accuracy for L1 & LLR statistics over the noisy stat inputs compared to the 'original' pop & pool
            roc_synthL1, pvalue_synthpop_L1, pvalue_synthpool_L1 = auc_scores(noisy_pop, noisy_pool, pop, pool)
            roc_synthLLR, pvalue_synthpop_LLR, pvalue_synthpool_LLR = auc_scores(noisy_pop, noisy_pool, pop, pool, LR=True)

            aucs_syntheticL1.append(roc_synthL1)
            aucs_syntheticLLR.append(roc_synthLLR)

            if fixed_FPR == True:
                fpr_synthL1, tpr_synthL1, thresholds_synthL1 = auc_scores(noisy_pop, noisy_pool, pop, pool, FPR=True)
                fpr_synthLLR, tpr_synthLLR, thresholds_synthLLR = auc_scores(noisy_pop, noisy_pool, pop, pool, LR=True, FPR=True)

                # TPR at a fixed FPR (e.g., 0.01 = 1%)
                target_fpr = 1e-2
                fprs_syntheticL1.append(np.interp(target_fpr, fpr_synthL1, tpr_synthL1))
                fprs_syntheticLLR.append(np.interp(target_fpr, fpr_synthLLR, tpr_synthLLR))

            if include_tabulate:
                synth_pop, synth_pop_mu, synth_pop_muhat = L1(noisy_pop, pop, pool, table=True)
                synth_pool, synth_pool_mu, synth_pool_muhat = L1(noisy_pool, pop, pool, table=True)
                if table_of_pop==True:
                    data.extend([noisy_pop[individual]] + [synth_pop_mu[individual]] + [synth_pop_muhat[individual]])
                    synthetic_table_count = counter(synth_pop[individual])
                else:
                    data.extend([noisy_pool[individual]] + [synth_pool_mu[individual]] + [synth_pool_muhat[individual]])
                    synthetic_table_count = counter(synth_pool[individual], population=False)
                table_counter.append([original_table_count, synthetic_table_count])

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

                total_spop_count = []
                total_spool_count = []
                for pop_ind in synth_pop:
                    pop_count = counter(pop_ind)
                    total_spop_count.append(pop_count)
                for pool_ind in synth_pool:
                    pool_count = counter(pool_ind, population=False)
                    total_spool_count.append(pool_count)
                total_counter.append([total_opop_count, total_opool_count, total_spop_count, total_spool_count])

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

# Build the figure-data dict and save before plotting.
fig_data = {
    "auc_L1": np.asarray(auc_L1),
    "auc_LLR": np.asarray(auc_LLR),
    "_fixed_FPR": fixed_FPR,
    "_include_synthetic_noise": include_synthetic_noise,
}
if fixed_FPR:
    fig_data["tpr_at_fpr_L1"] = np.asarray(tpr_at_fpr_L1)
    fig_data["tpr_at_fpr_LLR"] = np.asarray(tpr_at_fpr_LLR)
if include_synthetic_noise:
    fig_data["auc_syntheticL1"] = np.asarray(auc_syntheticL1)
    fig_data["auc_syntheticLLR"] = np.asarray(auc_syntheticLLR)
    if fixed_FPR:
        fig_data["fpr_syntheticL1"] = np.asarray(fpr_syntheticL1)
        fig_data["fpr_syntheticLLR"] = np.asarray(fpr_syntheticLLR)

meta = {
    "seed": _flags["seed"],
    "selected_distribution": selected_distribution,
    "num_orders": num_orders,
    "include_synthetic_noise": include_synthetic_noise,
    "fixed_FPR": fixed_FPR,
    "stratifying": stratifying,
}

if OUTPUT_FILE:
    save_figdata(OUTPUT_FILE, fig_data, meta)

make_figure(fig_data, OUTPUT_FILE)

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
