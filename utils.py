import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import ttest_1samp

def load_dataset(MiRNA_filter=None, random_sample=None, case_sample=None):
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

    if MiRNA_filter is not None:
        rows, columns = filter_population.shape
        if MiRNA_filter > columns:
            return None

        filter_population = filter_population.sample(MiRNA_filter, axis=1)
    filter_population.insert(0, 'diseases', diseases)
    filter_population = filter_population.sort_values('diseases')

    random_pool = filter_population.sample(65 if random_sample is None else random_sample)

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

    D1 = (D1 if case_sample is None else case_sample.strip('"'))
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

def L1_ttest(victim, pop, pool):
    ttest = ttest_1samp(L1(victim, pop, pool), 0)
    return ttest
    
def L1_threshold(pop, pool, victim_pop=None, victim_pool=None):
    pvalue_pop = L1_ttest(pop if victim_pop is None else victim_pop, pop, pool)[1]
    pvalue_cpool = L1_ttest(pool if victim_pool is None else victim_pool, pop, pool)[1]
    a = np.concatenate((pvalue_pop, pvalue_cpool))

    threshold = a.max()+.1
    newa = np.append(a, threshold)
    newa = np.unique(newa)
    return newa

def LLR_ttest(victim, pop, pool):
    ttest = ttest_1samp(L1(victim, pop, pool), 0)
    return ttest

def LLR_threshold(pop, pool, victim_pop=None, victim_pool=None):
    pvalue_pop = LLR_ttest(pop if victim_pop is None else victim_pop, pop, pool)[1]
    pvalue_cpool = LLR_ttest(pool if victim_pool is None else victim_pool, pop, pool)[1]
    a = np.concatenate((pvalue_pop, pvalue_cpool))

    threshold = a.max()+.1
    newa = np.append(a, threshold)
    newa = np.unique(newa)
    return newa

def ground_truth(pop, pool, threshold):
    TP = np.sum(pool >= threshold) #all values where pool >= threshold = accept
    FP = np.sum(pop >= threshold) #all values where pop >= threshold = accept
    FN = np.sum(pool < threshold) #all values where pool < threshold = reject
    TN = np.sum(pop < threshold) #all values where pop < threshold = reject

    power =  TP / (TP+FN) #(sensitivity)
    fpr =  FP / (FP+TN)
    return power, fpr
