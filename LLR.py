import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from scipy.stats import ttest_1samp

# Likelihood-Ratio Test
# LLR = Sum_j=1^m [(x_j^v - mu_j)^2 / 2sigma_j^2 - (x_j^v - mu-hat_j)^2 / 2sigma-hat_j^2 + log sigma_j/sigma-hat_j]
# x_j^v is the value of miRNA j for the individual victim in column v
# mu_j & sigma_j are the average & standard deviation miRNA j in the population
# mu-hat_j & sigma-hat_j are the average & standard deviation miRNA j in the pool

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


def LLR(
        X_victim: ArrayLike, population, pool
):
    
    mu = np.average(population, axis=0)
    mu_hat = np.average(pool, axis=0)

    sigma = np.std(population, axis=0)
    sigma_hat = np.std(pool, axis=0)

    population_difference = np.square(X_victim - mu) / 2*sigma**2
    pool_difference = np.square(X_victim - mu_hat) / 2*sigma_hat**2

    simplified_expression = population_difference - pool_difference + np.log((sigma/sigma_hat))

    return np.sum(simplified_expression)
# shouldn't need to sum it if you use the axis correct

pop, rpool, cpool = load_dataset()

pop = pop.drop(columns="diseases")
rpool = rpool.drop(columns="diseases")
cpool = cpool.drop(columns="diseases")
victim = cpool.iloc[10]

print (LLR(victim, pop, cpool))
# make result from number so decision

def ttest(victim, pop, pool):
    ttest = ttest_1samp(LLR(victim, pop, pool), 0)
    return ttest

def threshold():
    pvalue_pop = ttest(pop, pop, cpool)[1]
    pvalue_cpool = ttest(cpool, pop, cpool)[1]
    a = np.concatenate((pvalue_pop, pvalue_cpool))

    threshold = a.max()+.1
    newa = np.append(a, threshold)
    newa = np.unique(newa)
    return newa

test = threshold()
print (test)

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
