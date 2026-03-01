"""Compute Table 1 noise thresholds (α_{0.6} and α*) programmatically.

Sweeps noise levels at full feature count for 6 configurations,
finding α_{0.6} (AUC drops below 0.6) and α* (L1 overtakes LLR).

All sweep_noise calls are run in parallel via ProcessPoolExecutor.
"""
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils_datasets import load_dataset, load_timestamp_dataset, drop_timestamp_index, D3, D17

NUM_REPS = 2000
EPS = 1e-30  # guard against zero-variance features in LLR


def compute_auc(scores_pop, scores_pool):
    """AUC via Mann-Whitney."""
    gt = scores_pool[np.newaxis, :] > scores_pop[:, np.newaxis]
    eq = scores_pool[np.newaxis, :] == scores_pop[:, np.newaxis]
    return (gt.sum() + 0.5 * eq.sum()) / (len(scores_pop) * len(scores_pool))


def sweep_noise(pop_np, pool_np, alphas, clip_zero=True, seed=None):
    """Sweep noise levels, return (alphas, auc_l1, auc_llr).

    Noise model matches Noise.py: sigma_j = std(pop) used for BOTH groups.
    LLR reference stats still use per-group variance.
    """
    rng = np.random.default_rng(seed)
    n_features = pop_np.shape[1]

    # Reference statistics (computed once from original data)
    mu = np.mean(pop_np, axis=0)
    mu_hat = np.mean(pool_np, axis=0)
    sigma_pop = np.std(pop_np, axis=0, ddof=0)
    sigma_pool = np.std(pool_np, axis=0, ddof=0)
    var_pop = np.maximum(sigma_pop ** 2, EPS)
    var_pool = np.maximum(sigma_pool ** 2, EPS)
    log_ratio = np.log(np.maximum(sigma_pop, EPS) / np.maximum(sigma_pool, EPS))

    # Noise sigma: use pop's std for both groups (matching Noise.py line 113)
    sigma_j = np.std(pop_np, axis=0, ddof=0)

    auc_l1 = np.zeros(len(alphas))
    auc_llr = np.zeros(len(alphas))

    for rep in range(NUM_REPS):
        for ai, alpha in enumerate(alphas):
            noised_pop = pop_np + rng.normal(0, alpha * sigma_j, pop_np.shape)
            noised_pool = pool_np + rng.normal(0, alpha * sigma_j, pool_np.shape)
            if clip_zero:
                noised_pop = np.clip(noised_pop, 0, None)
                noised_pool = np.clip(noised_pool, 0, None)

            # L1 t-statistic
            D_pop = np.abs(noised_pop - mu) - np.abs(noised_pop - mu_hat)
            D_pool = np.abs(noised_pool - mu) - np.abs(noised_pool - mu_hat)
            m_p = D_pop.mean(axis=1)
            v_p = np.maximum(np.mean(D_pop ** 2, axis=1) - m_p ** 2, 0)
            se_p = np.sqrt(v_p / (n_features - 1))
            m_l = D_pool.mean(axis=1)
            v_l = np.maximum(np.mean(D_pool ** 2, axis=1) - m_l ** 2, 0)
            se_l = np.sqrt(v_l / (n_features - 1))
            auc_l1[ai] += compute_auc(np.where(se_p > 0, m_p / se_p, 0.0),
                                      np.where(se_l > 0, m_l / se_l, 0.0))

            # LLR sum (per-group variance for reference stats, matching utils.py LLR)
            contrib_pop = (np.square(noised_pop - mu) / (2 * var_pop)
                           - np.square(noised_pop - mu_hat) / (2 * var_pool)
                           + log_ratio)
            contrib_pool = (np.square(noised_pool - mu) / (2 * var_pop)
                            - np.square(noised_pool - mu_hat) / (2 * var_pool)
                            + log_ratio)
            auc_llr[ai] += compute_auc(contrib_pop.sum(axis=1),
                                       contrib_pool.sum(axis=1))

    return alphas, auc_l1 / NUM_REPS, auc_llr / NUM_REPS


def _sweep_worker(args):
    """Worker function for ProcessPoolExecutor."""
    pop_np, pool_np, alphas, clip_zero, seed = args
    return sweep_noise(pop_np, pool_np, alphas, clip_zero, seed)


def interpolate_crossing(alphas, values, target, direction="below"):
    """Find α where values cross target via linear interpolation."""
    for i in range(1, len(alphas)):
        if direction == "below" and values[i] < target <= values[i - 1]:
            frac = (target - values[i - 1]) / (values[i] - values[i - 1])
            return alphas[i - 1] + frac * (alphas[i] - alphas[i - 1])
        if direction == "above" and values[i] >= target > values[i - 1]:
            frac = (target - values[i - 1]) / (values[i] - values[i - 1])
            return alphas[i - 1] + frac * (alphas[i] - alphas[i - 1])
    return None


if __name__ == "__main__":
    # ── Alpha range (log-spaced, matching Noise.py deviation_range) ───────
    ALPHAS = np.logspace(-2, 2, 50)  # 0.01 to 100, matching Noise.py

    # ── Load datasets ──────────────────────────────────────────────────────
    # Each config: (name, pop_list, pool_list, clip)
    # pop_list/pool_list are lists of arrays — length 1 for cross-sectional,
    # length 8 for longitudinal (one per timepoint/submission).
    configs = []

    print("Loading D3 (prostate cancer)...")
    pop_d3, pool_d3 = load_dataset(miRNA=True, disease_case_sample=D3)
    configs.append(("D3 case",
                    [np.asarray(pop_d3[1], dtype=np.float64)],
                    [np.asarray(pool_d3[1], dtype=np.float64)], True))
    configs.append(("D3 random",
                    [np.asarray(pop_d3[0], dtype=np.float64)],
                    [np.asarray(pool_d3[0], dtype=np.float64)], True))

    print("Loading D17 (renal cancer)...")
    pop_d17, pool_d17 = load_dataset(miRNA=True, disease_case_sample=D17, random_sample_size=20)
    configs.append(("D17 case",
                    [np.asarray(pop_d17[1], dtype=np.float64)],
                    [np.asarray(pool_d17[1], dtype=np.float64)], True))
    configs.append(("D17 random",
                    [np.asarray(pop_d17[0], dtype=np.float64)],
                    [np.asarray(pool_d17[0], dtype=np.float64)], True))

    print("Loading Timestamp (longitudinal miRNA)...")
    ti_pop, ti_pool, _, _ = load_timestamp_dataset()
    ti_pop, ti_pool = drop_timestamp_index(ti_pop, ti_pool)
    configs.append(("Timestamp",
                    [np.asarray(ti_pop[i], dtype=np.float64) for i in range(8)],
                    [np.asarray(ti_pool[i], dtype=np.float64) for i in range(8)], False))

    print("Loading FitBit...")
    fb_pop, fb_pool = load_dataset(FitBit=True)
    configs.append(("FitBit",
                    [np.asarray(fb_pop[i], dtype=np.float64) for i in range(8)],
                    [np.asarray(fb_pool[i], dtype=np.float64) for i in range(8)], False))

    # ── Build flat job list ───────────────────────────────────────────────
    # Each job: (config_name, timepoint_index, n_timepoints, args_tuple)
    jobs = []
    ss = np.random.SeedSequence()
    total_calls = sum(len(pop_list) for _, pop_list, _, _ in configs)
    seeds = ss.spawn(total_calls)

    seed_idx = 0
    for name, pop_list, pool_list, clip in configs:
        n_tp = len(pop_list)
        for tp in range(n_tp):
            jobs.append((name, tp, n_tp,
                         (pop_list[tp], pool_list[tp], ALPHAS, clip, seeds[seed_idx])))
            seed_idx += 1

    print(f"\nSubmitting {len(jobs)} sweep jobs to process pool...")

    # ── Run all sweeps in parallel ────────────────────────────────────────
    # Collect results keyed by (config_name, timepoint)
    sweep_results = {}
    with ProcessPoolExecutor() as executor:
        future_to_key = {}
        for name, tp, n_tp, args in jobs:
            future = executor.submit(_sweep_worker, args)
            future_to_key[future] = (name, tp, n_tp)

        completed = 0
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            name, tp, n_tp = key
            alphas, auc_l1, auc_llr = future.result()
            sweep_results[(name, tp)] = (auc_l1, auc_llr)
            completed += 1
            print(f"  [{completed}/{len(jobs)}] {name} tp={tp} done")

    # ── Aggregate results per config ──────────────────────────────────────
    results = []
    curve_rows = []

    for name, pop_list, pool_list, clip in configs:
        n_tp = len(pop_list)
        auc_l1_all = np.zeros(len(ALPHAS))
        auc_llr_all = np.zeros(len(ALPHAS))
        for tp in range(n_tp):
            auc_l1_tp, auc_llr_tp = sweep_results[(name, tp)]
            auc_l1_all += auc_l1_tp
            auc_llr_all += auc_llr_tp
        auc_l1 = auc_l1_all / n_tp
        auc_llr = auc_llr_all / n_tp

        a06_llr = interpolate_crossing(ALPHAS, auc_llr, 0.6, "below")
        a06_l1 = interpolate_crossing(ALPHAS, auc_l1, 0.6, "below")
        a_star = interpolate_crossing(ALPHAS, auc_l1 - auc_llr, 0, "above")

        results.append({"Config": name, "n_pop": pop_list[0].shape[0], "n_pool": pool_list[0].shape[0],
                         "m": pop_list[0].shape[1], "a06_LLR": a06_llr, "a06_L1": a06_l1, "a_star": a_star})
        for i, a in enumerate(ALPHAS):
            curve_rows.append({"config": name, "alpha": a, "auc_l1": auc_l1[i], "auc_llr": auc_llr[i]})

        fmt = lambda v: f"{v:.2f}" if v is not None else ">max"
        print(f"\n{name}: α_{{0.6}} LLR={fmt(a06_llr)}, L1={fmt(a06_l1)}, α*={fmt(a_star)}")

    # ── Output ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Table 1: Noise thresholds\n")
    print(pd.DataFrame(results).to_string(index=False))

    # LaTeX
    print("\n% LaTeX table")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{l c c c c c c}")
    print("\\hline")
    print("Config & $n$ & $n'$ & $m$ & $\\alpha_{0.6}$ LLR & $\\alpha_{0.6}$ L1 & $\\alpha^*$ \\\\")
    print("\\hline")
    for r in results:
        fmt = lambda v: f"{v:.2f}" if v is not None else "$>\\alpha_\\mathrm{max}$"
        print(f"{r['Config']} & {r['n_pop']} & {r['n_pool']} & {r['m']} "
              f"& {fmt(r['a06_LLR'])} & {fmt(r['a06_L1'])} & {fmt(r['a_star'])} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")

    # CSV
    pd.DataFrame(curve_rows).to_csv("threshold_results.csv", index=False)
    print("\nSaved threshold_results.csv")
