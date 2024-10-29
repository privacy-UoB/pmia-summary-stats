import numpy as np
import pandas as pd
import random
from numpy.typing import ArrayLike
from scipy.stats import ttest_1samp
from sklearn.model_selection import ShuffleSplit

# listing all 19 case pools for GSE61741 dataset
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

    # columns are individuals, rows are miRNAs
    df = pd.read_csv('GSE61741_series_matrix.csv', skiprows=52, skipfooter=1, sep='\t', index_col=0)

    # common to filter out miRNAs with median below 50
    df_median = df.median(axis=1)
    filter_population = df[df_median >= 50]
    filter_population = filter_population.transpose()

    # getting diseases
    with open('GSE61741_series_matrix.csv', 'rt') as f:
        lines = f.readlines()

    start = '!Sample_characteristics_ch1'
        
    for line in lines:
        if line.startswith(start):
            diseases = line.strip()

    diseases = diseases.split('\t')
    diseases = diseases[1:]
    diseases = [disease.strip('"') for disease in diseases]

    # reducing number of columns (miRNAs) according to the passed argument 'MiRNA_filer'
    if MiRNA_filter is not None:
        rows, columns = filter_population.shape

        if MiRNA_filter > columns:
            return None

        filter_population = filter_population.sample(MiRNA_filter, axis=1)

    # insert column label for diseases
    filter_population.insert(0, 'diseases', diseases)
    filter_population = filter_population.sort_values('diseases')

    # reduce number of rows (individuals) according to the passed argument 'random_sample'
    a = 65 if random_sample is None else random_sample

    randomshuffle = ShuffleSplit(n_splits=1, test_size=a)
    (train, test) = next(randomshuffle.split(filter_population))
    filter_population_rpool = filter_population.iloc[train] # population all individuals not in random pool
    random_pool = filter_population.iloc[test] # random pool of a individuals
    assert random_pool.shape == (a, 466 if MiRNA_filter is None else (MiRNA_filter+1)) # miRNA_Filter+1 because varying miRNA functions drop column: "disease" after calling function

    # reduce number of rows (individuals) according to the passed argument 'case_sample'
    D = (D1 if case_sample is None else case_sample)
    case_pool = filter_population[filter_population["diseases"] == D]
    filter_population_cpool = filter_population[filter_population["diseases"] != D]

    print("pop shape minus rpool", filter_population_rpool.shape,
          "pop shape minus cpool", filter_population_cpool.shape,
          "rpool shape", random_pool.shape,
          "cpool shape", case_pool.shape)

    return filter_population_rpool, filter_population_cpool, random_pool, case_pool

def load_timestamp_dataset(MiRNA_filter=None, withNaN=None):

    # columns are 215 individuals, rows are 1026 (1205?) miRNAs
    df = pd.read_csv('GSE68951_series_matrix.txt', skiprows=58, skipfooter=1, sep='\t', index_col=0)

    population = df.transpose()

    with open('GSE68951_series_matrix.txt', 'rt') as f:
        lines = f.readlines()

    # getting 'disease: lung cancer' & 'disease: non-cancerous lung disease (control)'
    disease = lines[35]
    disease = disease.split("\t")
    disease = disease[1:]
    disease = [diseases.strip('\n, ,"') for diseases in disease]

    # getting patients
    patient_id = lines[36]
    patient_id = patient_id.split("\t")
    patient_id = patient_id[1:]
    patient_id = [patient_ids.strip('\n, ,"') for patient_ids in patient_id]

    # list all patient ids included
    unique_patient_id = list(set(patient_id))
    unique_patient_id_nocontrol = [patient for patient in unique_patient_id if patient != "patient id: ZZ_control"]
    sample_patient = random.sample(unique_patient_id_nocontrol, 1)
    unique_patient_id_nocontrol = np.array(unique_patient_id_nocontrol)

    # getting timepoints (diseases have timepoints 1-8, control has timepoints 1-12)
    timepoint = lines[37]
    timepoint = timepoint.split("\t")
    timepoint = timepoint[1:]
    timepoint = [timepoints.strip('\n, ,"') for timepoints in timepoint]

    # insert column label for timepoints
    population.insert(0, 'timepoint', timepoint)

    # reducing number of columns (miRNAs) according to the passed argument 'MiRNA_filer'
    if MiRNA_filter is not None:
        rows, columns = population.shape

        if MiRNA_filter > columns:
            return None
        population = population.sample(MiRNA_filter, axis=1)

    # insert column label for diseases
    population.insert(0, 'disease', disease)

    # insert column label for patient id
    population.insert(0, 'patient_id', patient_id)

    # add new row of NaN for all the missing timestamp entries for diseased individuals
    # alternative method: convert dataframes into a list then convert back after
    new_population = pd.DataFrame(columns=population.columns, index=[0])
    for patient in unique_patient_id_nocontrol:
        new_patient_df = pd.DataFrame(columns=population.columns, index=[0])
        p = population[population["patient_id"] == f"{patient}"]
        for i in range(8):
            t = (p["timepoint"] == f"timepoint: {(i+1)}")
            if (len(t)>i):
                if (t[i]): # if i in t should replace if (len(t)>i): but IndexError
                    new_patient_df = pd.concat([new_patient_df, p[t]], ignore_index=True)
            else:
                rowNaN = pd.Series([np.nan for i in range(len(p.columns))], index=p.columns)
                rowNaN.iloc[0] = p.iat[0,0]
                rowNaN.iloc[1] = p.iat[0,1]
                rowNaN.iloc[2] = f"timepoint: {(i+1)}"
                new_patient_df.loc[i+1] = rowNaN
        new_patient_df = new_patient_df.iloc[1:] # is there a way to stop row0 being NaN from setup??
        new_population = pd.concat([new_population, new_patient_df], ignore_index=True)
    new_population = new_population.iloc[1:]

    # filtering data into the 8 timepoints for diseases only
    """
    timepoint 1 (or any other chosen timepoint) is similar to the original pool
    timepoints 2-8 simulate increasing levels of noise as the miRNA readings differ over time
    """
    timepoint_i = []
    for i in range(8):
        t = population[(population["timepoint"] == f"timepoint: {(i+1)}") & (population["disease"] == "disease: lung cancer")]
        if withNaN:
            t = new_population[(new_population["timepoint"] == f"timepoint: {(i+1)}") & 
                               (new_population["disease"] == "disease: lung cancer")]
        timepoint_i.append(t)
        print(t.shape)

    # filter dataset via patient ids into distinct population and pool
    randomshuffle = ShuffleSplit(n_splits=1, test_size=18) # (test, train) = function -> matches ML train/test splitting order
    (pool_patients, pop_patients) = next(randomshuffle.split(unique_patient_id_nocontrol))
    pop_patients = unique_patient_id_nocontrol[pop_patients]
    pool_patients = unique_patient_id_nocontrol[pool_patients]

    pop_timepoint_i = []
    pool_timepoint_i = []
    sample_timepoint_i = []

    for i in timepoint_i:
        pop_t = i[i["patient_id"].isin(pop_patients)]
        pop_timepoint_i.append(pop_t)

        pool_t = i[i["patient_id"].isin(pool_patients)]
        pool_timepoint_i.append(pool_t)

        sample_t = i[i["patient_id"].isin(sample_patient)]
        sample_timepoint_i.append(sample_t)

    print("pop shape", pop_timepoint_i[0].shape,
          "rpool shape", pool_timepoint_i[0].shape,
          "sample shape", sample_timepoint_i[0].shape)

    return (pop_timepoint_i, pool_timepoint_i, sample_timepoint_i)

def load_FitBit_dataset(pool_size=None):

    # 15 columns are activities, 457 rows are people's IDs
    df = pd.read_csv('dailyActivity_merged.csv', sep=',')

    population = df

    # create list of only ids
    column_names = df.columns
    ids = df["Id"]
    ids = [int(x) for x in ids] # remove "'" from each id
    unique_id = list(set(ids)) # list all 35 unique ids included

    # get dataframe for each unique_id
    unique_ids_data = []
    for person in unique_id:
        x = population[(population["Id"] == person)]
        unique_ids_data.append(x)

# TODO replace activitydate by range of length for each unique id. This will be the new timestamp
        
    # create random sample of ids for pop and pool
    a = 30 if pool_size is None else pool_size
    randomshuffle = ShuffleSplit(n_splits=1, test_size=a)
    (pool, pop) = next(randomshuffle.split(unique_ids_data))
    
    # create list of 30 dataframes for each pop id
    pop_data = []
    for pop_index in pop:
        x = unique_ids_data[pop_index]
        pop_data.append(x)

    # find maximum number of data submissions in pop
    max_pop_entries = 0
    for i in pop_data:
        size = len(i)
        if max_pop_entries < size:
            max_pop_entries = size

# TODO filter by timestamp date based on range
    pop_timestamp = []
    for i in range(max_pop_entries):
        pop_timestamp_i = pd.DataFrame(columns=column_names, index=[0])

        for j in pop_data:
            if i < len(j):
                timestamp_i = j.iloc[i]
                pop_timestamp_i = pd.concat([pop_timestamp_i, timestamp_i.to_frame().T], ignore_index=True)
        pop_timestamp_i = pop_timestamp_i.iloc[1:]
        pop_timestamp_i = pop_timestamp_i.drop(columns=["ActivityDate","Calories\n"]) # quick fix
        # TODO drop id too
        pop_timestamp.append(pop_timestamp_i)


    # create list of 5 dataframes for each pool id
    pool_data = []
    for pool_index in pool:
        x = unique_ids_data[pool_index]
        pool_data.append(x)

    # find maximum number of data submissions in pool
    max_pool_entries = 0
    for i in pool_data:
        size = len(i)
        if max_pool_entries < size:
            max_pool_entries = size

    pool_timestamp = []
    for i in range(max_pool_entries):
        pool_timestamp_i = pd.DataFrame(columns=column_names, index=[0])

        for j in pool_data:
            if i < len(j):
                timestamp_i = j.iloc[i]
                pool_timestamp_i = pd.concat([pool_timestamp_i, timestamp_i.to_frame().T], ignore_index=True)
        pool_timestamp_i = pool_timestamp_i.iloc[1:] # why is this suddenly 16 columns?? seems to be +1 column: "Calories\n"
        pool_timestamp_i = pool_timestamp_i.drop(columns=["ActivityDate","Calories\n"]) # quick fix
        pool_timestamp.append(pool_timestamp_i)

    return pop_timestamp, pool_timestamp

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
