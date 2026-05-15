import sys
import numpy as np
import matplotlib

from experiment_io import parse_flags, seed_all, save_figdata, load_figdata, resolve_output_path

_flags = parse_flags(sys.argv)
seed_all(_flags["seed"])

if len(sys.argv) >= 2 or _flags["replot"]:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_score, confusion_matrix
from plot_style import line_kwargs, stacked_auc_tpr, METRIC_COLOR
from utils_datasets import _prepare_FitBit_per_id, _split_FitBit
from utils import auc_scores


def make_figure(data: dict, output_path: str | None) -> None:
    fixed_FPR = bool(np.asarray(data["_fixed_FPR"]).item())
    error_bands = bool(np.asarray(data["_error_bands"]).item())
    iterations = int(np.asarray(data["_iterations"]).item())

    # auc_L1/LLR are stored as the raw (num_orders, iterations) arrays so we
    # can recompute mean / min / max here without rerunning the experiment.
    auc_L1_raw = np.asarray(data["auc_L1"], dtype=float)
    auc_LLR_raw = np.asarray(data["auc_LLR"], dtype=float)
    if fixed_FPR:
        tpr_L1_raw = np.asarray(data["tpr_at_fpr_L1"], dtype=float)
        tpr_LLR_raw = np.asarray(data["tpr_at_fpr_LLR"], dtype=float)

    if fixed_FPR:
        fig, ax_auc, ax_tpr = stacked_auc_tpr()
    else:
        fig, ax_auc = plt.subplots()
        ax_tpr = None

    if not error_bands:
        auc_L1 = np.average(auc_L1_raw, axis=0)
        auc_LLR = np.average(auc_LLR_raw, axis=0)
        ax_auc.plot(range(iterations), auc_L1, label="AUC L1",
                    **line_kwargs("L1", marker=None, linewidth=2.0))
        ax_auc.plot(range(iterations), auc_LLR, label="AUC LLR",
                    **line_kwargs("LLR", marker=None, linewidth=2.0))
        if fixed_FPR:
            tpr_at_fpr_L1 = np.average(tpr_L1_raw, axis=0)
            tpr_at_fpr_LLR = np.average(tpr_LLR_raw, axis=0)
            ax_tpr.plot(range(iterations), tpr_at_fpr_L1, label="TPR L1",
                        **line_kwargs("L1", marker=None, linewidth=2.0))
            ax_tpr.plot(range(iterations), tpr_at_fpr_LLR, label="TPR LLR",
                        **line_kwargs("LLR", marker=None, linewidth=2.0))
    else:
        auc_L1_eb = [np.average(auc_L1_raw, axis=0), np.min(auc_L1_raw, axis=0), np.max(auc_L1_raw, axis=0)]
        auc_LLR_eb = [np.average(auc_LLR_raw, axis=0), np.min(auc_LLR_raw, axis=0), np.max(auc_LLR_raw, axis=0)]
        ax_auc.plot(range(iterations), auc_L1_eb[0], label="L1",
                    **line_kwargs("L1", marker=None, linewidth=2.0))
        ax_auc.fill_between(range(iterations), auc_L1_eb[1], auc_L1_eb[2],
                            alpha=0.2, color=METRIC_COLOR["L1"])
        ax_auc.plot(range(iterations), auc_LLR_eb[0], label="LLR",
                    **line_kwargs("LLR", marker=None, linewidth=2.0))
        ax_auc.fill_between(range(iterations), auc_LLR_eb[1], auc_LLR_eb[2],
                            alpha=0.2, color=METRIC_COLOR["LLR"])
        if fixed_FPR:
            tpr_L1_eb = [np.average(tpr_L1_raw, axis=0), np.min(tpr_L1_raw, axis=0), np.max(tpr_L1_raw, axis=0)]
            tpr_LLR_eb = [np.average(tpr_LLR_raw, axis=0), np.min(tpr_LLR_raw, axis=0), np.max(tpr_LLR_raw, axis=0)]
            ax_tpr.plot(range(iterations), tpr_L1_eb[0], label="L1",
                        **line_kwargs("L1", marker=None, linewidth=2.0))
            ax_tpr.fill_between(range(iterations), tpr_L1_eb[1], tpr_L1_eb[2],
                                alpha=0.2, color=METRIC_COLOR["L1"])
            ax_tpr.plot(range(iterations), tpr_LLR_eb[0], label="LLR",
                        **line_kwargs("LLR", marker=None, linewidth=2.0))
            ax_tpr.fill_between(range(iterations), tpr_LLR_eb[1], tpr_LLR_eb[2],
                                alpha=0.2, color=METRIC_COLOR["LLR"])

    ax_auc.legend(loc='upper right')
    ax_auc.set_ylabel("AUC")
    ax_auc.set_ylim([0.5, 1])
    ax_auc.grid(True)
    if fixed_FPR:
        ax_tpr.legend(loc='upper right')
        ax_tpr.set_xlabel("timestamp")
        ax_tpr.set_ylabel("TPR at 0.01 FPR")
        ax_tpr.set_ylim([0, 1])
        ax_tpr.grid(True)
    else:
        ax_auc.set_xlabel("timestamp")

    if output_path:
        plt.savefig(output_path)
        print(f"Saved to {output_path}")
    else:
        plt.show()


# CLI: python Ordered_FitBit.py [output.pdf]
OUTPUT_FILE = resolve_output_path(sys.argv[1] if len(sys.argv) >= 2 else None)

if _flags["replot"]:
    data, _meta = load_figdata(_flags["replot"])
    make_figure(data, OUTPUT_FILE)
    sys.exit(0)

# Hoist FitBit CSV parse + per-id grouping out of the outer loop;
# only the pop/pool ShuffleSplit + per-timestamp aggregation re-runs per iteration.
unique_ids_data, column_names = _prepare_FitBit_per_id()

error_bands = False
fixed_FPR = True
num_orders = 5000 # number of averages
auc_L1 = []
auc_LLR = []
if fixed_FPR == True:
    target_fpr = 1e-2

    tpr_at_fpr_L1 = []
    tpr_at_fpr_LLR = []

# for loop for numorder lots of train/test, then average at end
for j in range (num_orders):

    aucs_L1 = []
    aucs_LLR = []
    if fixed_FPR == True:
        tpr_at_fprs_L1 = []
        tpr_at_fprs_LLR = []

    # randomise the pop/pool split per iteration; CSV/grouping is reused from the prepare step above
    population, random_pool = _split_FitBit(unique_ids_data, column_names)

    pop = population[0]
    pool = random_pool[0]

    # iterations = max(len(pop), len(pool))
    iterations = 8 # not everyone recorded more than 8 timestamps of data
    for i in range(iterations):

        local_noised_pop = population[i]
        local_noised_pool = random_pool[i]

        roc_L1, pvalue_pop_L1, pvalue_pool_L1 = auc_scores(local_noised_pop, local_noised_pool, pop, pool)
        # check first if feature are mostly 0 entries then delete
            # col "LoggedActivitiesDistance", 433 of 457 values are 0.
            # col "SedentaryActiveDistance", 419 of 457 values are 0.
        try:
            roc_LLR, pvalue_pop_LLR, pvalue_pool_LLR = auc_scores(local_noised_pop, local_noised_pool, pop, pool, LR=True)
        except ValueError:
            continue

        aucs_L1.append(roc_L1)
        aucs_LLR.append(roc_LLR)

        if fixed_FPR == True:
            fpr_L1, tpr_L1, thresholds_L1 = auc_scores(local_noised_pop, local_noised_pool, pop, pool, FPR=True)
            fpr_LLR, tpr_LLR, thresholds_LLR = auc_scores(local_noised_pop, local_noised_pool, pop, pool, LR=True, FPR=True)

            # TPR at a fixed FPR (e.g., 0.01 = 1%)
            tpr_at_fprs_L1.append(np.interp(target_fpr, fpr_L1, tpr_L1))
            tpr_at_fprs_LLR.append(np.interp(target_fpr, fpr_LLR, tpr_LLR))

        y_true_L1 = np.concatenate((np.zeros(len(pvalue_pop_L1)), np.ones(len(pvalue_pool_L1))))
        y_score_L1 = np.concatenate((pvalue_pop_L1, pvalue_pool_L1))

    if len(aucs_L1) >0:
        auc_L1.append(aucs_L1)

    if len(aucs_LLR) >0:
        auc_LLR.append(aucs_LLR)

    if fixed_FPR == True:
        if len(tpr_at_fprs_L1) >0:
            tpr_at_fpr_L1.append(tpr_at_fprs_L1)

        if len(tpr_at_fprs_LLR) >0:
            tpr_at_fpr_LLR.append(tpr_at_fprs_LLR)

# Build the figure-data dict (raw per-iteration arrays, so make_figure can
# recompute mean / error bands without rerunning).
data = {
    "auc_L1": np.asarray(auc_L1),
    "auc_LLR": np.asarray(auc_LLR),
    "_fixed_FPR": fixed_FPR,
    "_error_bands": error_bands,
    "_iterations": iterations,
}
if fixed_FPR:
    data["tpr_at_fpr_L1"] = np.asarray(tpr_at_fpr_L1)
    data["tpr_at_fpr_LLR"] = np.asarray(tpr_at_fpr_LLR)

meta = {
    "seed": _flags["seed"],
    "iterations": iterations,
    "num_orders": num_orders,
    "error_bands": error_bands,
    "fixed_FPR": fixed_FPR,
}

if OUTPUT_FILE:
    save_figdata(OUTPUT_FILE, data, meta)

make_figure(data, OUTPUT_FILE)
