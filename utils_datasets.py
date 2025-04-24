import numpy as np
import pandas as pd
import random
import os
from sklearn.model_selection import ShuffleSplit
from scipy.stats import pearsonr
from itertools import combinations

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

    # reducing number of columns (miRNAs) according to the passed argument 'MiRNA_filter'
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

def drop_dataset_index(pop_rpool, pop_cpool, random_pool, case_pool):
    pop_rpool = pop_rpool.drop(columns="diseases")
    pop_cpool = pop_cpool.drop(columns="diseases")
    random_pool = random_pool.drop(columns="diseases")
    case_pool = case_pool.drop(columns="diseases")
    return pop_rpool, pop_cpool, random_pool, case_pool

def load_timestamp_dataset(with_independent_miRNAs=False, withNaN=False, MiRNA_filter=None, correlation=None):

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

    if with_independent_miRNAs==True:

        # remember to rename the independent_90.csv file if rerun withNaN or different correlation
        if os.path.exists("independent_90.csv"):
            with open("independent_90.csv", "r") as f:
                independent = f.read().splitlines()
                population = population[independent]
        
        else:
            threshold = 0.9 if correlation is None else correlation

            correlations = {}
            columns_list = population.columns.tolist()
            random.shuffle(columns_list)
            independent_miRNAs = columns_list.copy()

            for col1, col2 in combinations(columns_list, 2):
                a, b = pearsonr(population.loc[:, col1], population.loc[:, col2])
                correlations[col1 + '___' + col2] = pearsonr(population.loc[:, col1], population.loc[:, col2])

                if any(col1 == x for x in independent_miRNAs):
                    if (np.abs(a) > threshold):
                        independent_miRNAs.remove(col1)


            result = pd.DataFrame.from_dict(correlations, orient='index')
            result.columns = ['PCC', 'p-value']

            population = population.loc[:, independent_miRNAs]

            with open("independent_90.csv", "w") as f:
                for item in population:
                    f.write("%s\n" % item)

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
        if withNaN==True:
            chosen_population = new_population
        else:
            chosen_population = population 

        t = chosen_population[(chosen_population["timepoint"] == f"timepoint: {(i+1)}") & 
                              (chosen_population["disease"] == "disease: lung cancer")]
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

def drop_timestamp_index(pop, pool, sample=None):
    for x, y in zip(pop, pool):
        x.drop(["disease", "timepoint", "patient_id"], axis=1, inplace=True)
        y.drop(["disease", "timepoint", "patient_id"], axis=1, inplace=True)
    result = (pop, pool)

    if sample is not None:
        for i in sample:
            i.drop(["disease", "timepoint", "patient_id"], axis=1, inplace=True)
        result = (pop, pool, sample)

    return result

# https://stackoverflow.com/questions/33997753/calculating-pairwise-correlation-among-all-columns
def independent(pop_timestamps, pool_timestamps, separate_timestamps=False, print_independent_miRNAs=False, correlation=None):
    
    if os.path.exists("independent_pop.txt"):
        with open("independent_pop.txt", "r") as f:
            independent_pop = f.read().splitlines()
        with open("independent_pool.txt", "r") as f:
            independent_pool = f.read().splitlines()
        return independent_pop, independent_pool
    
    else:
        independent_miRNAs_pertime = []
        threshold = 0.9 if correlation is None else correlation

        if separate_timestamps == False:
            alltimes_combined_df = pd.DataFrame(columns=pop_timestamps[0].columns, index=[0])
            alltimes_combined_df = alltimes_combined_df.iloc[1:]

            for t_pop, t_pool in zip(pop_timestamps, pool_timestamps):
                combined_df = pd.concat([t_pop, t_pool], ignore_index=True)
                alltimes_combined_df = pd.concat([alltimes_combined_df, combined_df], ignore_index=True)
            smaller_combined_df = combined_df.iloc[:,:]

            correlations = {}
            columns = smaller_combined_df.columns.tolist()
            random.shuffle(columns)
            independent_miRNAs = columns.copy()

            for col1, col2 in combinations(columns, 2):
                a, b = pearsonr(smaller_combined_df.loc[:, col1], smaller_combined_df.loc[:, col2])
                correlations[col1 + '___' + col2] = pearsonr(smaller_combined_df.loc[:, col1], smaller_combined_df.loc[:, col2])

                if any(col1 == x for x in independent_miRNAs):
                    if (np.abs(a) > threshold):
                        independent_miRNAs.remove(col1)

            result = pd.DataFrame.from_dict(correlations, orient='index')
            result.columns = ['PCC', 'p-value']

            if print_independent_miRNAs == True:
                print(result.sort_index())
                print(independent_miRNAs, len(independent_miRNAs))
            independent_columns = independent_miRNAs

            pop_dataframe = []
            pool_dataframe = []
            for t_pop, t_pool in zip(pop_timestamps, pool_timestamps):
                independent_pop = t_pop.loc[:, independent_miRNAs]
                independent_pool = t_pool.loc[:, independent_miRNAs]
                pop_dataframe.append(independent_pop)
                pool_dataframe.append(independent_pool)
                    
        else:
            pop_dataframe = []
            pool_dataframe = []
            for t_pop, t_pool in zip(pop_timestamps, pool_timestamps):
                combined_df = pd.concat([t_pop, t_pool], ignore_index=True)
                smaller_combined_df = combined_df.iloc[:,:120]

                correlations = {}
                columns = smaller_combined_df.columns.tolist()
                random.shuffle(columns)
                independent_miRNAs = columns.copy()

                for col1, col2 in combinations(columns, 2):
                    a, b = pearsonr(smaller_combined_df.loc[:, col1], smaller_combined_df.loc[:, col2])
                    correlations[col1 + '___' + col2] = pearsonr(smaller_combined_df.loc[:, col1], smaller_combined_df.loc[:, col2])

                    if any(col1 == x for x in independent_miRNAs):
                        if (np.abs(a) > threshold):
                            independent_miRNAs.remove(col1)
                        # independent_miRNAs.remove(col1) if (np.abs(correlations[0]) > 0.3) else continue

                result = pd.DataFrame.from_dict(correlations, orient='index')
                result.columns = ['PCC', 'p-value']

                independent_miRNAs_pertime.append(independent_miRNAs)
                independent_miRNAs_pertime.append(len(independent_miRNAs))

                independent_pop = t_pop.loc[:, independent_miRNAs]
                independent_pool = t_pool.loc[:, independent_miRNAs]
                pop_dataframe.append(independent_pop)
                pool_dataframe.append(independent_pool)

                if print_independent_miRNAs == True:
                    print(result.sort_index())
            if print_independent_miRNAs == True:
                print (independent_miRNAs_pertime)
            independent_columns = independent_miRNAs_pertime

        with open("independent_pop.txt", "w") as f:
            for item in pop_dataframe:
                f.write("%s\n" % item)
        with open("independent_pool.txt", "w") as f:
            for item in pool_dataframe:
                f.write("%s\n" % item)

    # return (pop_dataframe, pool_dataframe, result, independent_columns)
    return pop_dataframe, pool_dataframe

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

def split_pool(pop, pool, include_transfer = True):
    # filter dataset via cpool into distinct added to population and remaining of split cpool
    randomshuffle = ShuffleSplit(n_splits=1, test_size=(int(len(pool)/4)))
    (pool_patients, pop_patients) = next(randomshuffle.split(pool))

    pop_patients = pool.iloc[pop_patients]
    pool_patients = pool.iloc[pool_patients]

    if include_transfer == True:
        split_pop = pd.concat([pop, pop_patients], ignore_index=True)
    else:
        split_pop = pop
    split_pool = pool_patients

    return split_pop, split_pool, pop_patients
