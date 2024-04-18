import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from scipy.stats import ttest_1samp

# L1 Distances Difference
# D(x_j^v) = |x_j^v - mu_j| - |x_j^v - mu-hat_j|
# x_j^v is the value of miRNA j for the individual victim in column v
# mu_j is the average miRNA j in the population
# mu-hat_j is the average miRNA j in the pool

def old_load_dataset():
    df = pd.read_csv(
        "GSE61741_series_matrix.csv"
    )
    # skiprows up to 52

    median_df = np.median(df, axis=1)
    filter_population = np.where(median_df > 49)
    filter_population = df[filter_population]
    # filter_population = np.where(median_df > 49, median_df, np.delete(median_df))
    # Delete all non-expressed MiRNAs: median MiRNA <50 over all cols in df

    rng = np.random.randint(len(filter_population), size=65)
    random_pool = filter_population[rng]
    # Randomly select the rows that will be deleted

    # random_pool = np.delete(filter_population, rng, axis=0)
    # Delete random rows (check: does rng need to be ordered to work?)

    D1 = "disease: Wilms Tumor"
    D2 = "disease: lung cancer"
    D3 = "disease prostate cancer"
    D4 = "disease: myocardial_infarction"
    D5 = "disease: chronic obstructive pulmonary disease (COPD)"
    D6 = "disease sarcoidosis"
    D7 = "disease ductal adenocarcinoma"
    D8 = "disease psoriasis"
    D9 = "disease: pancreatitis"
    D10 = "disease benign prostate hyperplasia"
    D11 = "disease melanoma"
    D12 = "disease: non-ischaemic systolic heart failure"
    D13 = "disease colon cancer"
    D14 = "disease: ovarian cancer"
    D15 = "disease: multiple sclerosis"
    D16 = "disease: glioma"
    D17 = "disease renal cancer"
    D18 = "disease periodontitis"
    D19 = "disease stomach tumor"
    # check 6, 7, 8, 10, 15, 17, 18, 19
    # long-lived individual, normal, any others for the healthy people? Can I print these after the list has been sorted?

    case_pool = np.sort(filter_population, order = [D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D14, D15, D16, D17, D18, D19])
    
    return filter_population, random_pool, case_pool

# need few as pool from the csv file, random then not random (see paper)

def load_dataset():
    df = pd.read_csv('GSE61741_series_matrix.csv', skiprows=52, skipfooter=1, sep='\t', index_col=0)

    df_median = df.median(axis=1)

    filter_population = df[df_median > 49]
    filter_population = filter_population.transpose()

    # Getting diseases
    with open('GSE61741_series_matrix.csv', 'rt') as f:
        lines = f.readlines()

    start = '!Sample_characteristics_ch1'
        
    for line in lines:
        if line.startswith(start):
            diseases = line.strip()

    diseases = diseases.split('\t')
    diseases = diseases[1:]
    diseases = [disease.strip('"') for disease in diseases]


    filter_population.insert(0, 'diseases', diseases)
    filter_population = filter_population.sort_values('diseases')

    random_pool = filter_population.sample(65)

    D1 = "disease: Wilms Tumor"
    D2 = "disease: lung cancer"
    D3 = "disease prostate cancer"
    D4 = "disease: myocardial_infarction"
    D5 = "disease: chronic obstructive pulmonary disease (COPD)"
    D6 = "disease sarcoidosis"
    D7 = "disease ductal adenocarcinoma"
    D8 = "disease psoriasis"
    D9 = "disease: pancreatitis"
    D10 = "disease benign prostate hyperplasia"
    D11 = "disease melanoma"
    D12 = "disease: non-ischaemic systolic heart failure"
    D13 = "disease colon cancer"
    D14 = "disease: ovarian cancer"
    D15 = "disease: multiple sclerosis"
    D16 = "disease: glioma"
    D17 = "disease renal cancer"
    D18 = "disease periodontitis"
    D19 = "disease stomach tumor"

    # case_pool = filter_population.mask("diseases" == D1)
    case_pool = filter_population[filter_population["diseases"] == D1]

    return filter_population, random_pool, case_pool

def L1(
        X_victim: ArrayLike, population, pool
):

    mu = np.average(population, axis=0)
    mu_hat = np.average(pool, axis=0)

    population_difference = np.abs(X_victim - mu)
    pool_difference = np.abs(X_victim - mu_hat)

    return population_difference - pool_difference

pop, rpool, cpool = load_dataset()
# better to do local instead of global
# would later call the function L1(X_v, row_j, a, b) passing arguments a, b for df1 & df2

pop = pop.drop(columns="diseases")
rpool = rpool.drop(columns="diseases")
cpool = cpool.drop(columns="diseases")
victim = cpool.iloc[99]

print (L1(victim, pop, cpool).sum())


def ttest(victim, pop, pool):
    ttest = ttest_1samp(L1(victim, pop, pool), 0)
    return ttest

# ttest = ttest(victim, pop, cpool)
# ttest_1samp(L1(victim, pop, cpool), 0)
# print(ttest)

def threshold():
    pvalue_pop = ttest(pop, pop, cpool)[1]
    pvalue_cpool = ttest(cpool, pop, cpool)[1]
    a = np.concatenate((pvalue_pop, pvalue_cpool))

    threshold = a.max()+.1
    newa = np.append(a, threshold)
    newa = np.unique(newa)
    return newa
    
test = threshold()
print (test) #this isn't returning the right threshold, shouldn't be using percentile

def ground_truth(pop, pool, threshold):
    TP = np.sum(pool >= threshold) #all values where pool >= threshold = accept
    FP = np.sum(pop >= threshold) #all values where pop >= threshold = accept
    FN = np.sum(pool < threshold) #all values where pool < threshold = reject
    TN = np.sum(pop < threshold) #all values where pop < threshold = reject

    power =  TP / (TP+FN) #(sensitivity)
    fpr =  FP / (FP+TN)
    return power, fpr

power = []
fpr = []
pvalue_pop = ttest(pop, pop, cpool)[1]
pvalue_cpool = ttest(cpool, pop, cpool)[1]

for t in threshold():
    p, f = ground_truth(pvalue_pop, pvalue_cpool, t)
    power.append(p)
    fpr.append(f)
fpr = np.array(fpr)
power = np.array(power)

order = np.argsort(fpr)
fpr = fpr[order]
power = power[order]

# plots!
fig, ax = plt.subplots()
ax.set_xscale("log")

ax.plot(fpr, power, linewidth=2.0)
plt.show()

# def ttest():
#     mu = np.average(pop, axis=0) #true average
#     m0 = np.average(rpool, axis=0) #average over all victims j
#     sigma_hat = np.std(rpool, axis=0)

#     T = 
#     t = (mu - m0) / (sigma_hat / np.sqrt(len(rpool)))

#     if T > t:
#         print ("victim is in pool")
#     else: 
#         print ("victim is not in pool")
# ipython3 in terminal - use when zsh error
# may 15th network

# t test
# note different threshold based on victims - check the results and pass/fail rate for t
# check over all victims, reproduce graphs in paper