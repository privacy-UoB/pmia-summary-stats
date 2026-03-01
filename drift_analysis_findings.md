# Drift analysis findings

## Background

The LLR membership inference attack achieves AUC ~0.997 at t=0 on the timestamp (GSE68951) dataset. When real biological drift occurs (t>0), AUC crashes to ~0.56. The question is: **what property of real drift causes this crash?**

## Three prior observations

1. **Real biological drift** (t=0 to t>0) crashes LLR AUC from 0.997 to ~0.56.
2. **Gaussian noise** of the same total magnitude does NOT crash it — LLR stays ~0.98.
3. **Shuffled (correlation-preserving) drift** — applying one individual's real delta vector to a different individual's baseline — does NOT crash it either (LLR stays ~0.87–0.90).

These are from the `Ordered_Noise.py` and `fig_correlation_preserving.py` experiments.

## diagnose2 hypothesis

The `diagnose2` diagnostic (in `fig_correlation_preserving.py`) identified a candidate mechanism:

- The LLR concentrates discriminating power in low-variance "super-features" (top 10% of features carry ~33% of total 1/(2σ²) weight).
- Real drift has a **negative baseline-drift correlation** (regression to the mean, Pearson r ≈ -0.15): individuals with extreme baseline values tend to drift back toward the population mean.
- This systematically disrupts the super-features that the LLR relies on.
- Capping variance weights at the 50th percentile nearly eliminates the real-vs-shuffled AUC gap (+0.285 to +0.076).

## Drift decomposition experiment

`fig_drift_decomposition.py` tests whether we can reproduce the AUC crash with simple synthetic drift models, using a factorial design over three properties: total magnitude, per-feature profile, and baseline-drift correlation.

### Four models tested

| Model | Total magnitude | Per-feature profile | Baseline-drift corr |
|-------|:-:|:-:|:-:|
| 1. Real drift | yes | yes | yes |
| 2. Gauss (pop-std scaled) | yes | no | no |
| 3. Gauss (drift-profile matched) | yes | yes | no |
| 4. RTM emulation | yes | yes | yes (linear) |

- **Model 2** — `δ_kj ~ N(0, c·σ_j^pop)` with c calibrated to match total drift energy.
- **Model 3** — `δ_kj ~ N(0, σ_j^drift)` using the empirical per-feature drift std. Matches the real magnitude distribution but draws are iid (no baseline correlation).
- **Model 4** — `δ_kj = β_avg·x_kj(0) + α_avg + N(0, σ_ε_j)` where β and α are fit by OLS per individual then averaged, and σ_ε is the per-feature residual std. This preserves both per-feature magnitude and a linear baseline-drift correlation.

### Results (500 iterations, mean AUC)

#### LLR AUC

| Timepoint | Real | Gauss (pop-std) | Gauss (drift) | RTM |
|:---------:|:----:|:---------------:|:--------------:|:---:|
| 0 | 0.997 | 0.997 | 0.997 | 0.997 |
| 1 | **0.565** | 0.985 | 0.985 | 0.982 |
| 2 | **0.556** | 0.986 | 0.987 | 0.984 |
| 3 | **0.589** | 0.987 | 0.988 | 0.986 |
| 4 | **0.539** | 0.985 | 0.987 | 0.983 |
| 5 | **0.558** | 0.990 | 0.988 | 0.986 |
| 6 | **0.521** | — | — | — |
| 7 | **0.518** | — | — | — |

#### L1 AUC

| Timepoint | Real | Gauss (pop-std) | Gauss (drift) | RTM |
|:---------:|:----:|:---------------:|:--------------:|:---:|
| 0 | 0.897 | 0.897 | 0.897 | 0.897 |
| 1 | **0.580** | 0.903 | 0.900 | 0.904 |
| 2 | **0.576** | 0.905 | 0.901 | 0.902 |
| 3 | **0.586** | 0.907 | 0.900 | 0.900 |
| 4 | **0.565** | 0.903 | 0.904 | 0.904 |
| 5 | **0.580** | 0.906 | 0.900 | 0.901 |

t=6 and t=7 are missing for synthetic models due to individual count mismatches at those timepoints.

### Interpretation

This is the **"4 ≠ 1" outcome** from the factorial:

- All three synthetic models (2, 3, 4) stay at AUC ~0.98 for LLR and ~0.90 for L1.
- None of them reproduce the crash to ~0.56.
- Models 2, 3, and 4 are nearly indistinguishable from each other.

**Conclusion: a simple linear regression-to-mean model does not capture the destructive property of real biological drift.** The fact that model 4 (which preserves linear baseline-drift correlation) performs identically to model 3 (which has no correlation at all) means the linear component of regression-to-mean is not the mechanism.

### What this rules out

- Per-feature magnitude profile alone (model 3 vs model 2): no difference, so the per-feature structure of drift magnitudes doesn't matter either.
- Linear baseline-drift correlation (model 4 vs model 3): no difference, so a simple `δ = β·x₀ + α + ε` relationship is insufficient.

### What remains

The real drift must have **nonlinear or higher-order structure** that these models miss. Candidates include:

- **Nonlinear baseline-drift relationships** (e.g., the drift is largest for features at the extreme tails, not captured by a linear fit).
- **Inter-feature correlation structure** in the drift vectors — real drift preserves biological covariance patterns that synthetic iid draws destroy. The shuffled-drift experiment (correlation-preserving, AUC ~0.87) partially supports this, since it preserves inter-feature structure and partially degrades AUC.
- **Individual-specific drift patterns** — each individual's drift vector may be coherent in a way that simple models with shared β/α cannot capture.
- **Heteroscedastic or fat-tailed residuals** — real drift residuals may not be Gaussian, with occasional large shifts on specific features.

## Correlation-preserving experiment (reference)

From `fig_correlation_preserving.py` (2000 iterations):

| Timepoint | Real LLR | Shuffled LLR | Real L1 | Shuffled L1 |
|:---------:|:--------:|:------------:|:-------:|:-----------:|
| 0 | 0.997 | 0.997 | 0.893 | 0.893 |
| 1 | 0.561 | 0.875 | 0.574 | 0.541 |
| 4 | 0.541 | 0.879 | 0.561 | 0.546 |

Shuffled drift (same individual's full delta vector applied to a random individual's baseline) keeps LLR at ~0.87 — higher than real (~0.56) but lower than Gaussian models (~0.98). This confirms that the **identity mapping** between baseline and drift (which individual gets which drift) is critical, and that preserving the inter-feature covariance within drift vectors partially but not fully explains the gap.

## Student's sanity check experiments (Ordered_Timestamp.py)

Four "sanity check" experiments attempted to reproduce the AUC crash using shuffled versions of the real drift. These preceded the more targeted experiments below.

### Case 1: scalar-shuffled drift (selected_distribution == 4)

**Intent:** shuffle sample differences from real timepoints, add to baseline.

**Bug:** `pop_diff = np.ravel(np.subtract(local_pop, local_noised_pop))` flattens the (n × d) drift matrix into a 1D vector of scalar values. `random.shuffle` then permutes individual scalars, not drift vectors. This destroys **all** structure — both which individual gets which drift AND the inter-feature correlations within each person's drift vector. The result is essentially iid noise drawn from the empirical distribution of drift scalars, which is more destructive than a proper row-shuffle.

Additionally, the sign is flipped (`x_0 - x_t` instead of `x_t - x_0`), though after ravel-shuffle this is inconsequential since positive and negative values get mixed.

**What Case 1 actually tests:** whether noise with the correct marginal distribution of individual scalar drift values (but no individual or feature structure) crashes AUC. It should not — and indeed no simple iid noise model crashes AUC.

### Case 2: distance-based shuffle (selected_distribution == 5)

**Status:** incomplete. The student noted this wasn't working. The code has issues: it references an undefined `index` variable and has incomplete conditional logic.

### Case 3: vector-shuffled drift (selected_distribution == 6)

**Intent:** decompose drift through the population mean, shuffle, add to baseline.

**Algebra:** `pop_vector = (x_t - μ_0) + (μ_0 - x_0) = x_t - x_0 = drift`. The decomposition through the mean cancels out — this is algebraically identical to the raw drift. Since `list()` on a 2D array produces a list of row vectors, `random.shuffle` permutes which individual gets which complete drift vector.

**What Case 3 actually tests:** this is equivalent to our correlation-preserving experiment (`fig_correlation_preserving.py`). It shuffles which individual gets which drift vector while preserving the inter-feature covariance within each vector. Expected LLR AUC: ~0.87.

### Case 4: normalised vector-shuffled drift (selected_distribution == 7)

**Intent:** same as Case 3, but divide by the current timepoint values.

**What it does:** after shuffling drift vectors (same as Case 3), divides element-wise by `local_noised_pop` (the current timepoint data): `shuffled_vector / x_t`. This creates a feature-magnitude-dependent scaling:

- High-value features (large x_t): noise shrinks → relatively unaffected
- Low-value features (small x_t): noise amplifies → disproportionately disrupted

### Student's observation about metric swapping

The student noted that with 0.9 independence filtering:
- **Case 3** (vector shuffle): LLR suddenly performs like the independent-only L1
- **Case 4** (normalised vector): L1 suddenly performs like the independent-only LLR

This is likely a real effect of the normalisation in Case 4, not a coincidence. The two metrics have different sensitivity profiles across features:

- **LLR** weights features by 1/(2σ²) — it concentrates discriminating power on low-variance features
- **L1** uses |x - μ| — it's sensitive to all features roughly in proportion to their absolute magnitude

Case 4's division by x_t amplifies noise on low-value features (which tend to be low-variance and therefore LLR-important) while shrinking noise on high-value features (which tend to dominate L1's absolute distances). This inverts which metric is more disrupted, causing the apparent "swap". The 0.9 independence filter sharpens this effect by removing correlated features and changing the relative weight distribution across the remaining features.

### Relationship to later experiments

Cases 1 and 3 are special cases of the manipulations explored more systematically in later experiments:
- Case 1 (scalar shuffle) ≈ a more extreme version of iid noise (destroys all structure)
- Case 3 (row shuffle) = our correlation-preserving experiment (destroys individual assignment, preserves feature structure)

Neither reproduces the AUC crash. The later NN and linear/residual experiments pinpoint exactly why: the destructive signal is in the individual-specific nonlinear residual, which even the closest neighbor's drift doesn't share.

## Nearest-neighbor drift experiment

`fig_nn_drift.py` (500 iterations): instead of random shuffle, assign each individual's drift vector from one of their k nearest neighbors in baseline feature space. Sweep k from 1 (closest neighbor) to 50.

### Results (LLR AUC, mean over 500 iterations)

| k | t=1 LLR | t=4 LLR |
|:-:|:-------:|:-------:|
| 0 (self = real) | **0.564** | **0.543** |
| 1 | 0.808 | 0.854 |
| 2 | 0.810 | 0.852 |
| 5 | 0.834 | 0.839 |
| 10 | 0.875 | 0.868 |
| 20 | 0.896 | 0.897 |
| 50 | 0.896 | 0.894 |
| perm (random shuffle) | 0.874 | 0.873 |

Note: pool has only n=18, so k>17 is capped at 17 (all neighbors) for pool. Pop has n≈157, so all listed k values are meaningful for pop.

### Interpretation

**The coupling is truly individual-specific, not about local neighborhood structure.** Even the single closest neighbor's drift (k=1) jumps AUC from ~0.56 to ~0.81 — recovering most of the 0.31 gap to shuffled. The entire gap hierarchy:

- **Self → k=1**: +0.25 AUC (individual-specific component, ~80% of total gap)
- **k=1 → shuffle**: +0.06 AUC (small local-neighborhood effect, ~20% of total gap)
- **k=1 → k=50**: gradual increase, curve is nearly flat

The small k=1-vs-shuffle difference (~0.06) suggests nearest neighbors do have slightly more similar drift patterns (expected from shared biology), but this accounts for a minor fraction of the crash. The dominant effect is that each individual's own drift vector is uniquely destructive when applied to their own baseline.

### What this tells us

The destructive mechanism requires the **exact pairing** of an individual's baseline with their own drift. Even a biologically similar neighbor's drift doesn't reproduce the crash. This rules out explanations based on:
- Local feature-space structure (nearby individuals sharing drift modes)
- Coarse population substructure (disease subtypes with shared drift patterns)

The mechanism must involve fine-grained individual-specific coupling — something about how each person's own features change over time that is unique to that person's starting point, in a way that isn't captured by even the closest neighbor's drift pattern.

## Summary of all experiments

| Drift type | LLR AUC (t>0) | Key property |
|------------|:-:|---|
| No drift (t=0) | ~0.997 | Baseline |
| Gaussian (any flavour) | ~0.985 | Independent noise, no structure |
| RTM emulation | ~0.983 | Linear baseline-drift correlation |
| Shuffled real drift | ~0.875 | Real inter-feature covariance, wrong individual |
| k=1 NN drift | ~0.81 | Closest neighbor's drift, almost as benign as random |
| Real drift | ~0.56 | Everything: correct individual + correct drift |

The gap decomposes roughly as:

- 0.997 → 0.985: any noise at all (−0.01)
- 0.985 → 0.875: inter-feature covariance structure (−0.11)
- 0.875 → 0.81: local neighborhood similarity (−0.06)
- 0.81 → 0.56: **individual-specific baseline-drift coupling** (−0.25)

The largest single factor (~0.25 AUC) is the coupling between an individual's exact baseline and their own drift vector. This is not captured by a linear RTM model, and cannot be approximated even by the closest neighbor in feature space. The mechanism is truly individual-specific.

## Linear/residual drift factorial experiment

`fig_linear_residual.py` (500 iterations): decomposes each individual's drift into a per-individual linear component and a residual, then swaps each independently using the k=1 nearest neighbor.

### Method

For each individual k, fit per-individual OLS across features:
- `δ_kj = β_k · x_kj(0) + α_k + ε_kj`
- Linear prediction: `δ̂_kj = β_k · x_kj(0) + α_k`
- Residual: `ε_kj = δ_kj - δ̂_kj`

Note: the previous RTM experiment (Model 4) **averaged** β and α across individuals, losing individual-specific slopes. This experiment keeps per-individual β_k, α_k and swaps components via the nearest neighbor m(k). When swapping β_m, it is always applied to **k's own baseline** x_k, isolating the effect of the regression parameters.

### Results (LLR AUC, mean over 500 iterations)

| Condition | Linear (β, α) | Residual (ε) | t=1 LLR | t=4 LLR |
|:---------:|:-:|:-:|:-------:|:-------:|
| self_self | self | self | **0.554** | **0.543** |
| self_nn | self | NN | 0.800 | 0.853 |
| nn_self | NN | self | **0.545** | **0.537** |
| nn_nn | NN | NN | 0.799 | 0.856 |
| full_nn | — (whole drift from NN) | — | 0.809 | 0.863 |
| perm | — (random shuffle) | — | 0.874 | 0.877 |

### Sanity checks

- **self_self ≈ 0.554**: matches real drift (~0.56). Decomposition is lossless.
- **full_nn ≈ 0.809**: matches known k=1 NN result (~0.81). NN logic is correct.
- **nn_nn ≈ 0.799**: close to full_nn, confirming that decompose-then-reassemble with NN components ≈ direct NN swap.
- **perm ≈ 0.874**: matches known shuffle reference (~0.87).

### Interpretation

The result is unambiguous — **the destructive signal lives entirely in the nonlinear residual (ε), not the linear slope (β, α):**

- **nn_self ≈ 0.545**: swapping the NN's linear parameters while keeping self residual has essentially **no effect** on the crash (0.554 → 0.545). The per-individual regression slope is irrelevant.
- **self_nn ≈ 0.800**: swapping just the NN's residual while keeping self linear parameters **eliminates the crash** (0.554 → 0.800). Nearly all destructive information was in the residual.
- **nn_nn ≈ 0.799**: swapping both components gives the same result as self_nn, confirming the linear parameters contribute nothing beyond what the residual already explains.

The effect is clean and one-sided. The 2×2 factorial shows zero interaction — the linear component has no effect regardless of residual source, and the residual component fully determines the outcome regardless of linear source.

### What this tells us

The destructive baseline-drift coupling is **not** regression-to-mean in any conventional sense. A per-individual linear slope β_k — even one fit specifically to that individual's data — contributes nothing to the AUC crash. What matters is the **feature-specific deviation from that linear trend**: the pattern of which features overshoot and undershoot their predicted drift.

This residual ε_kj is:
1. **Individual-specific** — even the nearest neighbor's residual doesn't reproduce the crash
2. **Nonlinear** — not captured by any first-order baseline-drift relationship
3. **Feature-specific** — encodes which particular features change more or less than a linear model predicts

## Summary of all experiments

| Drift type | LLR AUC (t>0) | Key property |
|------------|:-:|---|
| No drift (t=0) | ~0.997 | Baseline |
| Gaussian (any flavour) | ~0.985 | Independent noise, no structure |
| RTM emulation (averaged β) | ~0.983 | Linear baseline-drift correlation, shared slope |
| NN-linear + self-residual | ~0.545 | NN's β_k applied to self baseline + self ε → still crashes |
| Shuffled real drift | ~0.875 | Real inter-feature covariance, wrong individual |
| k=1 NN drift (full) | ~0.81 | Closest neighbor's entire drift vector |
| Self-linear + NN-residual | ~0.80 | Own β_k + neighbor's ε → crash eliminated |
| Real drift | ~0.56 | Everything: correct individual + correct drift |

The gap decomposes roughly as:

- 0.997 → 0.985: any noise at all (−0.01)
- 0.985 → 0.875: inter-feature covariance structure (−0.11)
- 0.875 → 0.81: local neighborhood similarity (−0.06)
- 0.81 → 0.56: **individual-specific nonlinear residual** (−0.25)

The largest single factor (~0.25 AUC, ~60% of total gap) is the nonlinear residual ε_kj — the feature-specific pattern of how each individual's drift deviates from their own linear trend.

## Conditional Gaussian experiment and overfitting test

`fig_nn_drift.py` (500 iterations): tests whether a population-level conditional Gaussian model — multivariate regression of drift on baseline — can reproduce the AUC crash.

### Models tested

| Model | Description |
|-------|-------------|
| Real drift | Ground truth |
| Cond. mean (memorising) | `μ_diff + B(x - μ_base)`, fit on all n individuals |
| LOO cond. mean | Same regression, but individual i held out when fitting |
| NN k=1 (Euclidean) | Nearest neighbor's real drift vector |
| Independent Gaussian | `N(0, σ_j²)` per feature, variance matched to real drift |

Three distance metrics (Euclidean, cosine, correlation) were also tested for the NN condition — all produce identical results, confirming the metric is irrelevant.

### Results (500 iterations, mean AUC)

| Model | LLR | L1 |
|-------|:---:|:--:|
| Real drift | 0.563 | 0.572 |
| Cond. mean (memorising) | 0.563 | 0.572 |
| **LOO cond. mean** | **0.827** | **0.597** |
| NN k=1 | 0.812 | 0.625 |
| Independent Gaussian | 0.998 | 0.910 |

### The overfitting trap

The non-LOO conditional mean appeared to perfectly reproduce the AUC crash (matching real drift at 0.563). However, with p >> n (~1205 features, ~18 individuals), the least-squares regression `B = X⁺Y` uses the pseudoinverse. Since `X @ X⁺ = Iₙ` when n < d:

```
predicted = X @ B = X @ X⁺ @ Y = Y
```

The conditional mean is literally returning each individual's original drift vector — it memorises rather than generalises. This is confirmed by the LOO test: when individual i is held out during fitting, the model predicts their drift from the remaining n−1 individuals, and AUC jumps to 0.827 — essentially identical to NN k=1 (0.812).

### Interpretation

The population-level linear baseline-drift coupling does **not** generalise to held-out individuals. The LOO conditional mean performs no better than simply using the nearest neighbor's drift. This means:

1. The AUC crash is **not** driven by a shared linear relationship between baseline and drift
2. The individual-specific structure responsible for the crash cannot be captured by any population-level linear model
3. The earlier finding (drift decomposition) that the linear component (β, α) is inert is consistent — both the per-individual OLS and the population-level regression fail to capture the destructive mechanism

This reinforces the conclusion from the linear/residual factorial: the destructive component is in the nonlinear, individual-specific residual that no simple model — linear regression, nearest neighbor, or independent noise — can reproduce.

## Can we emulate real drift?

The chain of experiments answers this clearly: **no simple parametric model can reproduce the AUC crash.**

### What we've tried and what failed

Every synthetic model stays at AUC ≥ 0.80:

| Model | LLR AUC | What it captures |
|-------|:---:|------------------|
| Independent Gaussian (per-feature σ) | ~0.998 | Per-feature variance, no structure |
| Gaussian (iid, pop-std scaled) | ~0.985 | Magnitude only |
| Skewed Cauchy | ~0.98* | Better distributional fit, still iid |
| RTM (averaged β, α) | ~0.983 | + population-level linear baseline-drift correlation |
| Row-shuffled real drift | ~0.875 | Real drift vectors, wrong individual |
| LOO conditional mean | ~0.827 | Population-level linear regression, held-out prediction |
| k=1 NN drift | ~0.81 | Real drift vector from biologically closest neighbor |
| RTM (per-individual β_k, α_k) | ~0.80** | + individual-specific linear slopes |

†From student's Ordered_Timestamp.py experiments.
*Estimated — scalar shuffle destroys all structure, equivalent to iid noise.
**Inferred from nn_self ≈ self_self (linear component is inert) and self_nn ≈ 0.80 (swapping residual to NN already reaches 0.80).

### Why it's hard

The destructive component is the nonlinear residual ε_kj — the part of each individual's drift that **cannot** be predicted from their baseline by any linear model. To synthesise drift that reproduces the crash, you would need to generate residuals that are:

1. **Individually coherent** — each person's residual vector must be internally consistent in a way that is specific to that person (even the nearest neighbor's doesn't work)
2. **Nonlinearly coupled to baseline** — the residuals encode a mapping from baseline features to drift deviations that no first-order model captures
3. **Feature-resolved** — the pattern is about *which specific features* overshoot or undershoot, not aggregate statistics

This is essentially asking for a model of individual-level biological temporal variation — the full complexity of how each person's miRNA profile changes over time. No simple noise model (Gaussian, RTM, shuffled, or neighbor-based) captures this.

### Implications

**For privacy defense:** The AUC crash from real drift cannot be synthetically reproduced with simple additive noise. An attacker who wanted to simulate "realistic" drift to test robustness would underestimate the crash — their synthetic drift would leave AUC at ~0.80–0.98, far above the real ~0.56. Real biological variation provides a level of natural privacy protection that is surprisingly hard to emulate.

**For the attack:** The LLR attack's vulnerability to temporal drift is intrinsic and robust. It cannot be "patched" by accounting for linear regression-to-mean or per-feature drift profiles — the vulnerability comes from complex individual-specific biology that operates below the resolution of any simple statistical correction.

**For understanding:** The finding that the destructive mechanism is in the *residual* (what we can't model) rather than the *signal* (what we can model) suggests it may arise from nonlinear gene-regulatory dynamics, individual-specific environmental responses, or stochastic biological processes that are fundamentally hard to parameterise. The fact that even the biologically nearest neighbor's residual doesn't reproduce the crash implies the relevant variation is at the individual level, not the subpopulation level.
