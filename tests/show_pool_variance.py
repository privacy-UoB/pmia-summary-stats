"""Quick diagnostic: dump per-pool AUC peaks vs the mean curve from a saved
Ordered_Noise.py / Noise.py run.

Usage: uv run python tests/show_pool_variance.py <path.npz>
"""

import os
import sys

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def show(npz_path: str) -> None:
    d = np.load(npz_path, allow_pickle=True)
    keys = list(d.keys())
    print(f"keys: {keys}")
    # Find the per-pool key
    pp_key = next((k for k in keys if k.endswith("_per_pool") and "tpr" not in k), None)
    mean_key = pp_key.removesuffix("_per_pool") if pp_key else None
    print(f"per-pool key: {pp_key}    mean key: {mean_key}")
    pp = np.asarray(d[pp_key])
    mean = np.asarray(d[mean_key])
    mult = np.asarray(d["multiplier"])

    print(f"\nshapes: per-pool={pp.shape}   mean={mean.shape}")
    print(f"K pools = {pp.shape[0]}\n")

    # peaks per pool per multiplier
    if pp.ndim == 3:           # Ordered_Noise.py: (K, n_mults, n_cnts)
        peaks = pp.max(axis=-1)            # (K, n_mults)
        mean_peaks = mean.max(axis=-1)
    else:                       # Noise.py: (K, n_mults)
        peaks = pp
        mean_peaks = mean

    n_mults = peaks.shape[1]
    print(f"{'multiplier':>12} {'mean':>8} {'min':>8} {'max':>8} {'std':>8} {'SEM':>8}")
    for i in range(n_mults):
        col = peaks[:, i]
        sem = col.std(ddof=1) / np.sqrt(len(col)) if len(col) > 1 else 0.0
        print(f"{mult[i]:12.4g} {mean_peaks[i]:8.4f} {col.min():8.4f} {col.max():8.4f} {col.std():8.4f} {sem:8.4f}")

    print(f"\npool seeds used: {list(map(int, d.get('pool_seeds', [])))[:5]}{'...' if pp.shape[0] > 5 else ''}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: show_pool_variance.py <path.npz>", file=sys.stderr)
        sys.exit(2)
    show(sys.argv[1])
