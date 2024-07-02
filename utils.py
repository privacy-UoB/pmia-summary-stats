import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import ttest_1samp

D1 = "disease: Wilms Tumor"
D2 = "disease: lung cancer"
D3 = "disease: prostate cancer"
D4 = "disease: myocardial_infarction"
D5 = "disease: chronic obstructive pulmonary disease (COPD)"
D6 = "disease: sarcoidosis"
D7 = "disease: ductal adenocarcinoma"
D8 = "disease: psoriasis"
D9 = "disease: pancreatitis"
D10 = "disease: benign prostate hyperplasia"
D11 = "disease: melanoma"
D12 = "disease: non-ischaemic systolic heart failure"
D13 = "disease: colon cancer"
D14 = "disease: ovarian cancer"
D15 = "disease: multiple sclerosis"
D16 = "disease: glioma"
D17 = "disease: Renal cancer"
D18 = "disease: Periodontitis"
D19 = "disease: tumor of stomach"

def load_dataset(MiRNA_filter=None, random_sample=None, case_sample=None):
    df = pd.read_csv('GSE61741_series_matrix.csv', skiprows=52, skipfooter=1, sep='\t', index_col=0)
    # columns are individuals, rows are miRNAs

    df_median = df.median(axis=1)
    filter_population = df[df_median >= 50]
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
        # print(columns)
        if MiRNA_filter > columns:
            return None

        filter_population = filter_population.sample(MiRNA_filter, axis=1)
    filter_population.insert(0, 'diseases', diseases)
    filter_population = filter_population.sort_values('diseases')

    a = 65 if random_sample is None else random_sample
    random_pool = filter_population.sample(a)
    assert random_pool.shape == (a, 466 if MiRNA_filter is None else (MiRNA_filter+1))
    # miRNA_Filter+1 because varying miRNA functions drop column: "disease" after calling function

    D = (D1 if case_sample is None else case_sample)
    case_pool = filter_population[filter_population["diseases"] == D]

    print("pop shape", filter_population.shape, "rpool shape", random_pool.shape, "cpool shape", case_pool.shape)

    return filter_population, random_pool, case_pool

def L1(
        X_victim: ArrayLike, population, pool
):
    """
    L1 Distances Difference:
    D(x_j^v) = |x_j^v - mu_j| - |x_j^v - mu-hat_j|, 
    where x_j^v is the value of miRNA j for the individual victim in column v, 
    mu_j is the average miRNA j in the population, 
    and mu-hat_j is the average miRNA j in the pool.
    """

    mu = np.average(population, axis=0)
    mu_hat = np.average(pool, axis=0)

    population_difference = np.abs(X_victim - mu)
    pool_difference = np.abs(X_victim - mu_hat)

    return population_difference - pool_difference

# LLR taken over all individuals i for each miRNA j
def LLR(
        X_victim: ArrayLike, population, pool
):
    """
    Likelihood-Ratio Test: 
    LLR = Sum_j=1^m [(x_j^v - mu_j)^2 / 2sigma_j^2 - (x_j^v - mu-hat_j)^2 / 2sigma-hat_j^2 + log sigma_j/sigma-hat_j], 
    where x_j^v is the value of miRNA j for the individual victim in column v, 
    mu_j & sigma_j are the average & standard deviation miRNA j in the population, 
    and mu-hat_j & sigma-hat_j are the average & standard deviation miRNA j in the pool.
    """
    
# axis0 means function applied over all values in a column (x_1 = y_11, y_21, y_n1, x_2 = y_12, y_22, y_n2)
# axis1 means function applied to rows (x_1 = y_11, y_12, y_1n, x_2 = y_21, y_22, y_2n)
    
    mu = np.average(population, axis=0)
    mu_hat = np.average(pool, axis=0)

    var = np.var(population, axis=0)
    var_hat = np.var(pool, axis=0)

    sigma = np.std(population, axis=0)
    sigma_hat = np.std(pool, axis=0)

    population_difference = np.divide(np.square(X_victim - mu),2*var)
    pool_difference = np.divide(np.square(X_victim - mu_hat),2*var_hat)

    simplified_expression = population_difference - pool_difference + np.log(np.divide(sigma,sigma_hat))
    # this should be a matrix over all victims for each of their miRNAs j

    s = np.sum(simplified_expression, axis=1)
    return np.transpose([s])
    # this is a n x 1 vector of LLRs summed over all miRNAs for each victim v

# scipy ttest: null hypothesis that sample mean = popmean
# paper: null hypothesis victim's x_j not in pool, D(x_j^v) -> 0
    # alternate hypothesis victim's x_j in pool, D(x_j^v) > 0
    # sum over miRNA js then D(x^v) -> normal distribution
    # test > threshold then victim is in pool
def L1_ttest(victim, pop, pool):
    s = L1(victim, pop, pool)
    ttest = ttest_1samp(s, 0, axis=1, alternative="greater")[1]
    return 1-ttest
    
def L1_threshold(pop, pool, victim_pop=None, victim_pool=None):
    pvalue_pop = np.ravel(L1_ttest(pop if victim_pop is None else victim_pop, pop, pool)[1])
    pvalue_pool = np.ravel(L1_ttest(pool if victim_pool is None else victim_pool, pop, pool)[1])
    a = np.concatenate((pvalue_pop, pvalue_pool))

    threshold = a.max()+.1
    newa = np.append(a, threshold)
    newa = np.unique(newa)
    return newa

def LLR_threshold(pop, pool, victim_pop=None, victim_pool=None):
    pvalue_pop = LLR(pop if victim_pop is None else victim_pop, pop, pool)
    pvalue_pool = LLR(pool if victim_pool is None else victim_pool, pop, pool)
    a = np.concatenate((pvalue_pop, pvalue_pool))

    threshold = a.max()+.1
    newa = np.append(a, threshold)
    newa = np.unique(newa)
    return newa

def ground_truth(pop, pool, threshold):
    TP = np.sum(pool >= threshold) # all values where pool >= threshold = accept
    FP = np.sum(pop >= threshold) # all values where pop >= threshold = accept
    FN = np.sum(pool < threshold) # all values where pool < threshold = reject
    TN = np.sum(pop < threshold) # all values where pop < threshold = reject

    power =  TP / (TP+FN) # sensitivity
    fpr =  FP / (FP+TN)
    return power, fpr
