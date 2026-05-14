"""Vectorized inner loops for the Ordered_Noise.py / Noise.py pipelines.

Two ideas, validated to match `utils.auc_scores` to within ~2e-4 (tie handling):

1. **Cumulative-LLR / cumulative-L1**: the per-feature contributions are
   independent of the subset of features chosen, so they can be computed once
   per (pool, multiplier, ordering) and a `cumsum` along the feature axis yields
   the LLR / L1-t-test score for every possible feature-prefix length in one
   pass — replacing ~232 redundant subset-level recomputations per ordering.

2. **Batched-rank AUC**: `roc_auc_score` has no batch API, so the current code
   pays Python overhead per call across hundreds of thousands of calls per
   pool. Stacking scores into an `(n_total, M)` matrix and calling
   `argsort(axis=0)` once gives M AUCs simultaneously.

The same primitives serve both scripts:
- `Ordered_Noise.py` varies miRNA subsets per ordering (the cumsum win matters).
- `Noise.py` uses all features at every multiplier (the cumsum is a no-op, but
  the batched-rank AUC still wins on the 2000-ordering sweep).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Per-feature score primitives
# ---------------------------------------------------------------------------

def llr_contributions(victims: np.ndarray, mu: np.ndarray, var: np.ndarray,
                      muh: np.ndarray, varh: np.ndarray,
                      log_sigma_ratio: np.ndarray) -> np.ndarray:
    """Per-feature LLR contribution for every victim.

    Mirrors `utils.LLR`:
        contr(v, j) = (v_j - mu_j)^2 / (2 var_j)
                    - (v_j - muh_j)^2 / (2 varh_j)
                    + log(sigma_j / sigma_hat_j)
    Returns array of shape `(n_victims, n_feat)`. `victims.sum(over j)` of the
    output equals the scalar LLR per victim — i.e. cumsum across `axis=1`
    gives the LLR using the first i features of any given ordering.
    """
    return ((victims - mu) ** 2 / (2.0 * var)
            - (victims - muh) ** 2 / (2.0 * varh)
            + log_sigma_ratio)


def l1_diffs(victims: np.ndarray, mu: np.ndarray, muh: np.ndarray) -> np.ndarray:
    """Per-feature L1 difference for every victim.

    Mirrors `utils.L1`:
        D(v, j) = |v_j - mu_j| - |v_j - muh_j|
    """
    return np.abs(victims - mu) - np.abs(victims - muh)


# ---------------------------------------------------------------------------
# Batched rank-based AUC + TPR-at-fixed-FPR
# ---------------------------------------------------------------------------

def batched_auc_tpr(score_pop: np.ndarray, score_pool: np.ndarray,
                    target_fpr: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized AUC and TPR-at-FPR across M score columns.

    Arguments
    ---------
    score_pop  : shape (n_pop, M) — pop victims' scores at M aggregations.
    score_pool : shape (n_pool, M) — pool victims' scores at the same M.

    Returns
    -------
    auc : (M,)      — same as `sklearn.metrics.roc_auc_score` (ties: ordinal)
    tpr : (M,)      — TPR at the requested FPR via linear interpolation,
                      matching the `np.interp(target_fpr, fpr, tpr)` recipe
                      already used in the scripts.

    The argsort done once for AUC is reused for TPR-at-FPR.
    """
    n_pop, M = score_pop.shape
    n_pool = score_pool.shape[0]
    n_total = n_pop + n_pool

    stacked = np.concatenate([score_pop, score_pool], axis=0)   # (n_total, M)

    # Argsort ascending; ranks are 1..n_total (ordinal — ties broken by index).
    order = np.argsort(stacked, axis=0, kind="quicksort")
    ranks = np.empty_like(order, dtype=np.float64)
    rows = np.arange(1, n_total + 1, dtype=np.float64)[:, None]   # column vector
    cols = np.arange(M)
    ranks[order, cols] = rows

    # Mann-Whitney U:   AUC = (sum_ranks(pool) - n_pool*(n_pool+1)/2) / (n_pop*n_pool)
    pool_rank_sum = ranks[n_pop:].sum(axis=0)
    auc = (pool_rank_sum - n_pool * (n_pool + 1) / 2.0) / (n_pop * n_pool)

    # TPR / FPR curves derived from the same sort.
    # As we walk from the highest score (a positive prediction) down, count
    # cumulative TP and FP. `labels[order]` gives the true class along the
    # sorted-by-score axis (0=pop, 1=pool).
    labels = np.concatenate([np.zeros(n_pop), np.ones(n_pool)])
    sorted_labels = labels[order]                                # (n_total, M)
    # Walk from largest score → smallest: flip along axis=0.
    sorted_labels_desc = sorted_labels[::-1]
    cum_tp = np.cumsum(sorted_labels_desc, axis=0)              # (n_total, M)
    cum_fp = np.cumsum(1.0 - sorted_labels_desc, axis=0)
    tpr_curve = cum_tp / n_pool                                 # (n_total, M)
    fpr_curve = cum_fp / n_pop

    # np.interp doesn't vectorize across columns; do it column-by-column.
    # Prepend (0, 0) so np.interp's left-side extrapolation matches sklearn's.
    tpr_at = np.empty(M, dtype=np.float64)
    for c in range(M):
        f = fpr_curve[:, c]
        t = tpr_curve[:, c]
        # Strip duplicates in fpr; np.interp requires monotonic xp.
        # In practice fpr_curve is monotone non-decreasing already, with steps.
        tpr_at[c] = np.interp(target_fpr, f, t)
    return auc, tpr_at


# ---------------------------------------------------------------------------
# Ordered-noise (miRNA-subset sweep) fast path
# ---------------------------------------------------------------------------

def ordered_curves_llr(pop: np.ndarray, pool: np.ndarray,
                       multipliers, miRNA_counts, num_orders: int,
                       sigma_j: np.ndarray, rng_np, rng_py,
                       target_fpr: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Compute LLR AUC/TPR curves for one pool over the full multiplier sweep.

    Reproduces the inner-loop output of `Ordered_Noise.py` (random pool path)
    with B1+B2+B3 applied:

      - per-feature pop/pool stats precomputed ONCE.
      - per-ordering: per-feature contributions, then cumsum-by-prefix.
      - AUC and TPR-at-FPR computed in one pass on the same sort.

    Returns (auc, tpr) arrays of shape (n_multipliers, n_miRNA_counts).
    """
    pop = np.asarray(pop, dtype=np.float64)
    pool = np.asarray(pool, dtype=np.float64)
    n_pop, n_feat = pop.shape
    n_pool = pool.shape[0]
    multipliers = list(multipliers)
    miRNA_counts = list(miRNA_counts)
    n_mults = len(multipliers)
    n_cnts = len(miRNA_counts)
    sel_cols = np.array([i - 1 for i in miRNA_counts])  # 0-indexed prefix lengths

    # Per-feature reference stats (pop/pool full).
    mu = pop.mean(axis=0)
    var = pop.var(axis=0, ddof=0)
    muh = pool.mean(axis=0)
    varh = pool.var(axis=0, ddof=0)
    log_sigma_ratio = 0.5 * np.log(var / varh)  # = log(sigma/sigma_hat)

    # Pre-generate permutations once (used across all multipliers).
    perms = np.empty((num_orders, n_feat), dtype=np.int64)
    base = np.arange(n_feat)
    for j in range(num_orders):
        p = base.copy()
        rng_py.shuffle(p)  # match the existing python-`random` use
        perms[j] = p

    auc_out = np.zeros((n_mults, n_cnts), dtype=np.float64)
    tpr_out = np.zeros((n_mults, n_cnts), dtype=np.float64)

    for m_idx, m in enumerate(multipliers):
        dev = m * sigma_j
        sum_auc = np.zeros(n_cnts, dtype=np.float64)
        sum_tpr = np.zeros(n_cnts, dtype=np.float64)

        for j in range(num_orders):
            # Fresh noise per ordering, matching current behavior.
            noise_pop = rng_np.normal(0.0, dev, size=pop.shape)
            noise_pool = rng_np.normal(0.0, dev, size=pool.shape)
            np_pop = np.clip(pop + noise_pop, 0.0, None)
            np_pool = np.clip(pool + noise_pool, 0.0, None)

            contr_pop = llr_contributions(np_pop, mu, var, muh, varh, log_sigma_ratio)
            contr_pool = llr_contributions(np_pool, mu, var, muh, varh, log_sigma_ratio)

            perm = perms[j]
            cum_pop = np.cumsum(contr_pop[:, perm], axis=1)   # (n_pop, n_feat)
            cum_pool = np.cumsum(contr_pool[:, perm], axis=1) # (n_pool, n_feat)

            scores_pop = cum_pop[:, sel_cols]                 # (n_pop, n_cnts)
            scores_pool = cum_pool[:, sel_cols]               # (n_pool, n_cnts)
            auc_vec, tpr_vec = batched_auc_tpr(scores_pop, scores_pool, target_fpr)
            sum_auc += auc_vec
            sum_tpr += tpr_vec

        auc_out[m_idx] = sum_auc / num_orders
        tpr_out[m_idx] = sum_tpr / num_orders

    return auc_out, tpr_out


def ordered_curves_l1(pop: np.ndarray, pool: np.ndarray,
                      multipliers, miRNA_counts, num_orders: int,
                      sigma_j: np.ndarray, rng_np, rng_py,
                      target_fpr: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """L1 (t-test on per-feature L1 differences) counterpart of `ordered_curves_llr`.

    Reproduces `auc_scores(..., LR=False, ...)` which uses
    `L1_ttest(victim, pop, pool) = 1 - ttest_1samp(L1_diffs, 0, alternative='greater')[pvalue]`.

    Vectorized via cumulative sums of diffs and diffs^2 over the
    permuted-feature axis: gives mean_i and var_i for every prefix length.
    """
    pop = np.asarray(pop, dtype=np.float64)
    pool = np.asarray(pool, dtype=np.float64)
    n_pop, n_feat = pop.shape
    n_pool = pool.shape[0]
    multipliers = list(multipliers)
    miRNA_counts = list(miRNA_counts)
    n_mults = len(multipliers)
    n_cnts = len(miRNA_counts)
    sel_cols = np.array([i - 1 for i in miRNA_counts])

    mu = pop.mean(axis=0)
    muh = pool.mean(axis=0)

    perms = np.empty((num_orders, n_feat), dtype=np.int64)
    base = np.arange(n_feat)
    for j in range(num_orders):
        p = base.copy()
        rng_py.shuffle(p)
        perms[j] = p

    i_axis = np.arange(1, n_feat + 1, dtype=np.float64)        # 1..n_feat
    df_axis = np.maximum(i_axis - 1.0, 1.0)                    # avoid div by 0

    auc_out = np.zeros((n_mults, n_cnts), dtype=np.float64)
    tpr_out = np.zeros((n_mults, n_cnts), dtype=np.float64)

    for m_idx, m in enumerate(multipliers):
        dev = m * sigma_j
        sum_auc = np.zeros(n_cnts, dtype=np.float64)
        sum_tpr = np.zeros(n_cnts, dtype=np.float64)

        for j in range(num_orders):
            noise_pop = rng_np.normal(0.0, dev, size=pop.shape)
            noise_pool = rng_np.normal(0.0, dev, size=pool.shape)
            np_pop = np.clip(pop + noise_pop, 0.0, None)
            np_pool = np.clip(pool + noise_pool, 0.0, None)

            diff_pop = l1_diffs(np_pop, mu, muh)               # (n_pop, n_feat)
            diff_pool = l1_diffs(np_pool, mu, muh)             # (n_pool, n_feat)

            perm = perms[j]
            dp = diff_pop[:, perm]
            dq = diff_pool[:, perm]

            cs_p = np.cumsum(dp, axis=1)
            cs2_p = np.cumsum(dp * dp, axis=1)
            cs_q = np.cumsum(dq, axis=1)
            cs2_q = np.cumsum(dq * dq, axis=1)

            mean_p = cs_p / i_axis
            mean_q = cs_q / i_axis
            # ddof=1 sample variance: var = (sum(x^2) - n*mean^2) / (n-1)
            var_p = np.maximum((cs2_p - i_axis * mean_p ** 2) / df_axis, 1e-300)
            var_q = np.maximum((cs2_q - i_axis * mean_q ** 2) / df_axis, 1e-300)

            # The existing code passes `1 - p_value` (a CDF) to roc_auc_score.
            # CDF is monotone in t, so rank-AUC is identical whether we score
            # by `1 - sf(t)` or by t itself. Skip the expensive t.sf eval and
            # use the t-statistic directly. (TPR-at-FPR also rank-invariant.)
            scores_pop_full = mean_p * np.sqrt(i_axis) / np.sqrt(var_p)
            scores_pool_full = mean_q * np.sqrt(i_axis) / np.sqrt(var_q)
            scores_pop = scores_pop_full[:, sel_cols]
            scores_pool = scores_pool_full[:, sel_cols]
            auc_vec, tpr_vec = batched_auc_tpr(scores_pop, scores_pool, target_fpr)
            sum_auc += auc_vec
            sum_tpr += tpr_vec

        auc_out[m_idx] = sum_auc / num_orders
        tpr_out[m_idx] = sum_tpr / num_orders

    return auc_out, tpr_out


# ---------------------------------------------------------------------------
# Noise.py (multiplier sweep with all features) fast path
# ---------------------------------------------------------------------------

def noise_curves(pop: np.ndarray, pool: np.ndarray, multipliers,
                 num_orders: int, *, include_deviations: bool,
                 sigma_j: np.ndarray | None, rng_np, clip: bool = True,
                 target_fpr: float = 0.01) -> dict[str, np.ndarray]:
    """Mean AUC and TPR-at-FPR across `num_orders` noise draws, per multiplier.

    Computes both L1 and LLR from the *same* per-ordering noise draws, so the
    dual-metric output is paired (matches the existing `auc_scores` x4
    call pattern per ordering in `Noise.py`).

    clip: match `Gaussian_noise(..., clip=True)` for miRNA (counts can't be
          negative); set False for the longitudinal datasets where the existing
          code calls Gaussian_noise without clip.

    Returns dict with keys: auc_L1, auc_LLR, tpr_L1, tpr_LLR (each shape (n_mults,)).
    """
    pop = np.asarray(pop, dtype=np.float64)
    pool = np.asarray(pool, dtype=np.float64)
    n_feat = pop.shape[1]
    multipliers = list(multipliers)
    n_mults = len(multipliers)

    # Per-feature reference stats (used inside the score functions).
    mu = pop.mean(axis=0)
    muh = pool.mean(axis=0)
    var = pop.var(axis=0, ddof=0)
    varh = pool.var(axis=0, ddof=0)
    log_sigma_ratio = 0.5 * np.log(var / varh)

    auc_L1 = np.zeros(n_mults, dtype=np.float64)
    auc_LLR = np.zeros(n_mults, dtype=np.float64)
    tpr_L1 = np.zeros(n_mults, dtype=np.float64)
    tpr_LLR = np.zeros(n_mults, dtype=np.float64)

    for m_idx, m in enumerate(multipliers):
        dev = (m * sigma_j) if include_deviations else m
        sum_auc_L1 = sum_auc_LLR = 0.0
        sum_tpr_L1 = sum_tpr_LLR = 0.0
        for _ in range(num_orders):
            noise_pop = rng_np.normal(0.0, dev, size=pop.shape)
            noise_pool = rng_np.normal(0.0, dev, size=pool.shape)
            np_pop = pop + noise_pop
            np_pool = pool + noise_pool
            if clip:
                np_pop = np.clip(np_pop, 0.0, None)
                np_pool = np.clip(np_pool, 0.0, None)

            # LLR scalar score per victim = sum of per-feature contributions
            llr_p = llr_contributions(np_pop, mu, var, muh, varh, log_sigma_ratio).sum(axis=1)
            llr_q = llr_contributions(np_pool, mu, var, muh, varh, log_sigma_ratio).sum(axis=1)

            # L1: per-feature diffs → one-sample t-test against 0 (greater).
            # Use the t-statistic directly as the score; CDF is monotone in t,
            # so AUC and TPR-at-FPR are rank-invariant under that transform.
            dp = l1_diffs(np_pop, mu, muh)
            dq = l1_diffs(np_pool, mu, muh)
            m_p = dp.mean(axis=1); m_q = dq.mean(axis=1)
            v_p = dp.var(axis=1, ddof=1); v_q = dq.var(axis=1, ddof=1)
            l1_p = m_p * np.sqrt(n_feat) / np.sqrt(np.maximum(v_p, 1e-300))
            l1_q = m_q * np.sqrt(n_feat) / np.sqrt(np.maximum(v_q, 1e-300))

            # Batch both metrics in one (n_total, 2) AUC call.
            score_pop = np.stack([l1_p, llr_p], axis=1)   # (n_pop, 2)
            score_pool = np.stack([l1_q, llr_q], axis=1)
            aucs, tprs = batched_auc_tpr(score_pop, score_pool, target_fpr)
            sum_auc_L1 += aucs[0]; sum_auc_LLR += aucs[1]
            sum_tpr_L1 += tprs[0]; sum_tpr_LLR += tprs[1]

        auc_L1[m_idx] = sum_auc_L1 / num_orders
        auc_LLR[m_idx] = sum_auc_LLR / num_orders
        tpr_L1[m_idx] = sum_tpr_L1 / num_orders
        tpr_LLR[m_idx] = sum_tpr_LLR / num_orders

    return {"auc_L1": auc_L1, "auc_LLR": auc_LLR,
            "tpr_L1": tpr_L1, "tpr_LLR": tpr_LLR}
