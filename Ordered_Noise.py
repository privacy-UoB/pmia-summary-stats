import sys
import numpy as np
import matplotlib
if len(sys.argv) >= 4:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from utils_datasets import load_dataset, separate_diseased_miRNAs, D3, D17

# CLI: python Ordered_Noise.py <disease> <metric> <pool_idx> [random_sample_size] [output.pdf]
# Falls back to interactive defaults when no args given.
DISEASES = {"D3": D3, "D17": D17}
if len(sys.argv) >= 4:
    DISEASE = DISEASES[sys.argv[1]]
    L1_or_LLR = sys.argv[2]
    POOL_IDX = int(sys.argv[3])
    RANDOM_SAMPLE_SIZE = int(sys.argv[4]) if len(sys.argv) >= 5 and sys.argv[4] != "_" else None
    OUTPUT_FILE = sys.argv[5] if len(sys.argv) >= 6 else None
else:
    DISEASE = D17
    L1_or_LLR = "L1"
    POOL_IDX = 1
    RANDOM_SAMPLE_SIZE = None
    OUTPUT_FILE = None

# paper: the demonstrated graphs showing roc curves
    # 1st: 50 subsets of n/1049 different individuals (n = 35, 65, 124)
    # 2nd: 6 case groups D19, D17, D10, D7, D3, D1

stratifying = False # Not enough pool miRNAs in True case

if stratifying == False:
    # load dataset
    population, pool = load_dataset(miRNA=True, disease_case_sample=DISEASE, random_sample_size=RANDOM_SAMPLE_SIZE)

    # 0 = random, 1 = case
    pop = population[POOL_IDX]
    pool = pool[POOL_IDX]

else:
    # diseased case sample pop/pool only
    only_diseased_miRNAs_pop, without_diseased_miRNAs_pop, only_diseased_miRNAs_pool, without_diseased_miRNAs_pool = separate_diseased_miRNAs(DISEASE, "miRNA")
    pop = without_diseased_miRNAs_pop
    pool = without_diseased_miRNAs_pool

# --- Convert to numpy once ---
pop_np = np.asarray(pop, dtype=np.float64)      # (n_pop, n_miRNAs)
pool_np = np.asarray(pool, dtype=np.float64)     # (n_pool, n_miRNAs)
n_pop, n_miRNAs = pop_np.shape
n_pool = pool_np.shape[0]

# Noise standard deviations per miRNA
sigma_j_np = np.std(pop_np, axis=0)        # (n_miRNAs,)
sigma_j_pool_np = np.std(pool_np, axis=0)  # (n_miRNAs,)

# --- Pre-compute reference statistics (once) ---
mu = np.mean(pop_np, axis=0)        # population mean per miRNA
mu_hat = np.mean(pool_np, axis=0)   # pool mean per miRNA

if L1_or_LLR == "LLR":
    var_pop = np.var(pop_np, axis=0, ddof=0)
    var_pool = np.var(pool_np, axis=0, ddof=0)
    sigma_pop = np.std(pop_np, axis=0, ddof=0)
    sigma_pool_ref = np.std(pool_np, axis=0, ddof=0)
    log_ratio = np.log(sigma_pop / sigma_pool_ref)

# miRNA step sizes: [2, 4, 6, ..., max_even <= n_miRNAs]
step_counts = np.arange(2, n_miRNAs + 1, 2)
n_steps = len(step_counts)
step_indices = step_counts - 1  # 0-based indices into cumsum arrays

# Pre-generate shuffled orderings as integer index arrays
num_orders = 2000
miRNA_indices = np.arange(n_miRNAs)
shuffled_indices = []
for j in range(num_orders):
    idx = miRNA_indices.copy()
    np.random.shuffle(idx)
    shuffled_indices.append(idx)

multiplier = [0, 0.25, 0.5, 0.75, 1]
noise_results = []

for count, m in enumerate(multiplier):
    # Accumulate AUC across orders: shape (n_steps,)
    auc_sum = np.zeros(n_steps)

    for j in range(num_orders):
        idx = shuffled_indices[j]

        # Generate noise for this order (skip when m == 0)
        if m == 0:
            noised_pop = pop_np
            noised_pool = pool_np
        else:
            pop_noise = np.random.normal(0, m * sigma_j_np, pop_np.shape)
            pool_noise = np.random.normal(0, m * sigma_j_np, pool_np.shape)
            noised_pop = np.clip(pop_np + pop_noise, 0, None)
            noised_pool = np.clip(pool_np + pool_noise, 0, None)

        if L1_or_LLR == "L1":
            # L1 differences: |x - mu| - |x - mu_hat| for each individual × miRNA
            D_pop = np.abs(noised_pop - mu) - np.abs(noised_pop - mu_hat)   # (n_pop, n_miRNAs)
            D_pool = np.abs(noised_pool - mu) - np.abs(noised_pool - mu_hat) # (n_pool, n_miRNAs)

            # Shuffle columns and compute cumulative sums
            D_pop_s = D_pop[:, idx]
            D_pool_s = D_pool[:, idx]
            S1_pop = np.cumsum(D_pop_s, axis=1)[:, step_indices]
            S2_pop = np.cumsum(D_pop_s ** 2, axis=1)[:, step_indices]
            S1_pool = np.cumsum(D_pool_s, axis=1)[:, step_indices]
            S2_pool = np.cumsum(D_pool_s ** 2, axis=1)[:, step_indices]

            # Vectorized t-statistic (raw t-stats used as scores — AUC is invariant
            # under the monotone CDF transform stdtr, so we skip it for speed)
            counts = step_counts[np.newaxis, :]  # (1, n_steps) for broadcasting
            # Pop scores
            mean_pop = S1_pop / counts
            var0_pop = np.maximum(S2_pop / counts - mean_pop ** 2, 0)
            se_pop = np.sqrt(var0_pop / (counts - 1))
            scores_pop = np.where(se_pop > 0, mean_pop / se_pop, 0.0)  # (n_pop, n_steps)
            # Pool scores
            mean_pool_v = S1_pool / counts
            var0_pool = np.maximum(S2_pool / counts - mean_pool_v ** 2, 0)
            se_pool = np.sqrt(var0_pool / (counts - 1))
            scores_pool = np.where(se_pool > 0, mean_pool_v / se_pool, 0.0)  # (n_pool, n_steps)

        elif L1_or_LLR == "LLR":
            # LLR per-miRNA contributions
            contrib_pop = (np.square(noised_pop - mu) / (2 * var_pop)
                          - np.square(noised_pop - mu_hat) / (2 * var_pool)
                          + log_ratio)   # (n_pop, n_miRNAs)
            contrib_pool = (np.square(noised_pool - mu) / (2 * var_pop)
                           - np.square(noised_pool - mu_hat) / (2 * var_pool)
                           + log_ratio)  # (n_pool, n_miRNAs)

            # Shuffle columns, cumsum, extract at step positions
            scores_pop = np.cumsum(contrib_pop[:, idx], axis=1)[:, step_indices]   # (n_pop, n_steps)
            scores_pool = np.cumsum(contrib_pool[:, idx], axis=1)[:, step_indices] # (n_pool, n_steps)

        # Vectorized AUC for all steps at once via Mann-Whitney comparison
        # AUC = P(pool_score > pop_score) + 0.5 * P(pool_score == pop_score)
        gt = scores_pool[np.newaxis, :, :] > scores_pop[:, np.newaxis, :]   # (n_pop, n_pool, n_steps)
        eq = scores_pool[np.newaxis, :, :] == scores_pop[:, np.newaxis, :]
        aucs = (gt.sum(axis=(0, 1)) + 0.5 * eq.sum(axis=(0, 1))) / (n_pop * n_pool)  # (n_steps,)

        auc_sum += aucs

        if j % 200 == 0:
            print(f"Multiplier {m} ({count + 1}/{len(multiplier)}): order {j}/{num_orders}")

    noise_results.append(auc_sum / num_orders)

# Save results to CSV
num_miRNAs = step_counts.tolist()
csv_path = OUTPUT_FILE.replace('.pdf', '.csv') if OUTPUT_FILE else 'ordered_noise_results.csv'
df = pd.DataFrame({'num_miRNAs': num_miRNAs})
for idx, noise in enumerate(multiplier):
    df[f'std_dev_{noise}'] = noise_results[idx]
df.to_csv(csv_path, index=False)
print(f"Saved CSV to {csv_path}")

# plots!
fig, ax = plt.subplots()

for index, noise in enumerate(multiplier):
    ax.plot(num_miRNAs, noise_results[index], linewidth=2.0, label=f"Std. dev. * {noise}")
ax.invert_xaxis()
ax.set_ylim([0.3,1]) # enables comparable auc scores between L1 and LLR
plt.xlabel("number MiRNAs")
plt.ylabel("AUC scores")
plt.legend(loc="upper right")
if OUTPUT_FILE:
    plt.savefig(OUTPUT_FILE)
    print(f"Saved to {OUTPUT_FILE}")
else:
    plt.show()
