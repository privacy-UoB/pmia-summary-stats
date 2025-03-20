import pandas as pd
from scipy import stats
from utils_datasets import load_dataset, D19, drop_dataset_index

# load dataset
pop_rpool, pop_cpool, rpool, cpool = load_dataset(case_sample=D19)
pop_rpool, pop_cpool, rpool, cpool = drop_dataset_index(pop_rpool, pop_cpool, rpool, cpool)

pop = pop_cpool # make pop configurable
pool = cpool # make pool configurable
df = pd.concat([pop, pool], ignore_index=True)


# Function to apply normality test to each column
def test_normality(df):
    results = {}
    
    for column in df.columns:
        # Shapiro-Wilk test
        stat, p_value = stats.shapiro(df[column])
        results[column] = p_value
    
    return results

# Apply the normality test
normality_results = test_normality(pool)

# Print results
normally_distributed = 0
for column, p_value in normality_results.items():
    if p_value < 0.05:
        print(f"Column '{column}' is not normally distributed (p-value: {p_value:.4f})")
    else:
        print(f"Column '{column}' is normally distributed (p-value: {p_value:.4f})")
        normally_distributed += 1
print(normally_distributed, 'out of', len(normality_results), 'columns are normally distributed')

# Results:
# D1 22/465
# D2 115/465
# D3 85/465
# D4 87/465
# D5 114/465
# D6 75/465
# D7 78/465
# D8 94/465
# D9 104/465
# D10 168/465
# D11 306/465
# D12 228/465
# D13 307/465
# D14 234/465
# D15 342/465
# D16 277/465
# D17 310/465
# D18 243/465
# D19 358/465
# All pops 0/465
# pop & pool 0/465
