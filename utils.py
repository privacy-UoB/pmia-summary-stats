import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import ttest_1samp
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score

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

    population = np.array(population)
    pool = np.array(pool)
    X_victim = np.array(X_victim)

    mu = np.average(population, axis=0)
    mu_hat = np.average(pool, axis=0)

    population_difference = np.abs(X_victim - mu)
    pool_difference = np.abs(X_victim - mu_hat)

    print("pop mean", mu, "pool mean", mu_hat, "pop diff", population_difference, "pool diff", pool_difference)

    # mu = np.average(population, axis=0).reshape(1, population.shape[1])
    # mu_hat = np.average(pool, axis=0).reshape(1, pool.shape[1])

    # population_difference = np.abs(np.subtract(X_victim, mu))
    # pool_difference = np.abs(np.subtract(X_victim, mu_hat))

    return np.subtract(population_difference, pool_difference)

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
    
    population = np.array(population)
    pool = np.array(pool)
    X_victim = np.array(X_victim)

    mu = np.average(population, axis=0)
    mu_hat = np.average(pool, axis=0)

    var = np.var(population, axis=0)
    var_hat = np.var(pool, axis=0)

    sigma = np.std(population, axis=0)
    sigma_hat = np.std(pool, axis=0)

    print(X_victim.shape, mu.shape, var.shape)
    
    p = np.square(X_victim - mu)
    print(p.shape)
    population_difference = np.divide(np.square(X_victim - mu),2*var)
    pool_difference = np.divide(np.square(X_victim - mu_hat),2*var_hat)

    simplified_expression = population_difference - pool_difference + np.log(np.divide(sigma,sigma_hat))
    # this should be a matrix over all victims for each of their miRNAs j

    # population = np.array(population)
    # pool = np.array(pool)
    # X_victim = np.array(X_victim)

    # mu = np.average(population, axis=0).reshape(1, population.shape[1])
    # mu_hat = np.average(pool, axis=0).reshape(1, pool.shape[1])

    # var = np.var(population, axis=0).reshape(1, population.shape[1])
    # var_hat = np.var(pool, axis=0).reshape(1, pool.shape[1])

    # sigma = np.std(population, axis=0).reshape(1, population.shape[1])
    # sigma_hat = np.std(pool, axis=0).reshape(1, pool.shape[1])

    # population_difference = np.divide(np.square(np.subtract(X_victim, mu)),2*var)
    # pool_difference = np.divide(np.square(np.subtract(X_victim, mu_hat)),2*var_hat)

    # simplified_expression = np.subtract(population_difference, pool_difference) + np.log(np.divide(sigma,sigma_hat))

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

def normalise(pop, pool, sample=None):
    normalisedpop = []
    normalisedpool = []

    for x, y in zip(pop, pool):
        normalisedx = normalize(x, norm="max", axis=0)
        normalisedy = normalize(y, norm="max", axis=0)
        normalisedpop.append(normalisedx)
        normalisedpool.append(normalisedy)
    result = (normalisedpop, normalisedpool)

    if sample is not None:
        normalisedsample = []

        for i in sample:
            normalisedi = normalize(i, norm="max", axis=0)
            normalisedsample.append(normalisedi)
        result = (normalisedpop, normalisedpool, normalisedsample)

    return result

def auc_scores(victim_pop, victim_pool, pop, pool, LR=False, p_values=True):

    # collect p_values for pop and pool
    pvalue_pop = (L1_ttest(victim_pop, pop, pool) if LR==False else LLR(victim_pop, pop, pool))
    pvalue_pool = (L1_ttest(victim_pool, pop, pool) if LR==False else LLR(victim_pool, pop, pool))
    
    # compare the true pop/pool placement to the predicted pop/pool placement
    y_true = np.concatenate((np.zeros(len(pvalue_pop)), np.ones(len(pvalue_pool))))
    y_score = np.concatenate((pvalue_pop, pvalue_pool))
    aucs = roc_auc_score(y_true, y_score)

    scores = ((aucs, pvalue_pop, pvalue_pool) if p_values==True else aucs)
    return scores

def Gaussian_noise(pop, pool, mean, deviation, clip=False, mean2=None, deviation2=None):
    pop_noise = np.random.normal(mean, deviation, pop.shape)
    pool_noise = (np.random.normal(mean, deviation, pool.shape)) if deviation2 is None else (np.random.normal(mean2, deviation2, pool.shape))

    noisy_pop = pop + pop_noise
    noisy_pool = pool + pool_noise

    nonneg_noisy_pop = np.clip(noisy_pop, 0, None)
    nonneg_noisy_pool = np.clip(noisy_pool, 0, None)

    result = ((noisy_pop, noisy_pool) if clip==False else (nonneg_noisy_pop, nonneg_noisy_pool))
    return result

# everything below is the fallback for roc_auc_score
def L1_threshold(pop, pool, victim_pop=None, victim_pool=None):
    pvalue_pop = np.ravel(L1_ttest(pop if victim_pop is None else victim_pop, pop, pool))
    pvalue_pool = np.ravel(L1_ttest(pool if victim_pool is None else victim_pool, pop, pool))
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

def fpr_power(pop, pool, pvalue_pop, pvalue_pool, LR=False, victim_pop=None, victim_pool=None):
    fpr = []
    power = []

    threshold = (L1_threshold(pop, pool, victim_pop, victim_pool) if LR==False else 
                 LLR_threshold(pop, pool, victim_pop, victim_pool))
    for t in threshold:
        p, f = ground_truth(pvalue_pop, pvalue_pool, t)
        fpr.append(f)
        power.append(p)
    fpr = np.array(fpr)
    power = np.array(power)

    order = np.argsort(fpr)
    fpr = fpr[order]
    power = power[order]
    return fpr, power
