import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils_datasets import load_timestamp_dataset

output_file = sys.argv[1] if len(sys.argv) > 1 else "fig_timepoint_drift.pdf"

# Load dataset — returns lists of 8 DataFrames (one per timepoint)
pop, pool, _sample, _healthy = load_timestamp_dataset()

# Combine pop + pool for each timepoint (we're characterising the dataset, not running an attack)
timepoints = []
for p, q in zip(pop, pool):
    combined = pd.concat([
        p.set_index("patient_id").drop(columns=["disease", "timepoint"]),
        q.set_index("patient_id").drop(columns=["disease", "timepoint"]),
    ])
    timepoints.append(combined)

# Compute |Δx| for each consecutive transition
labels = []
all_abs_diffs = []
for k in range(7):
    t_k, t_next = timepoints[k], timepoints[k + 1]
    # Inner join: only patients present in both timepoints
    common = t_k.index.intersection(t_next.index)
    diff = t_next.loc[common].values - t_k.loc[common].values
    abs_diff = np.abs(diff).ravel()
    abs_diff = abs_diff[~np.isnan(abs_diff)]
    all_abs_diffs.append(abs_diff)
    labels.append(f"{k+1}\u2192{k+2}")

# Print summary stats
print(f"{'Transition':<12} {'N':>8} {'Median':>10} {'Mean':>10} {'Std':>10}")
for label, diffs in zip(labels, all_abs_diffs):
    print(f"{label:<12} {len(diffs):>8} {np.median(diffs):>10.2f} {np.mean(diffs):>10.2f} {np.std(diffs):>10.2f}")

# Violin plot
fig, ax = plt.subplots(figsize=(8, 5))
parts = ax.violinplot(all_abs_diffs, positions=range(1, 8), showmedians=True)
ax.set_xticks(range(1, 8))
ax.set_xticklabels(labels)
ax.set_xlabel("Consecutive timepoint transition")
ax.set_ylabel("|Δx| (absolute per-feature change)")
ax.set_yscale("log")
ax.set_title("Per-feature drift between consecutive timepoints (GSE68951)")
fig.tight_layout()
fig.savefig(output_file)
print(f"\nSaved to {output_file}")
