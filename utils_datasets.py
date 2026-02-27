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
D20 = "disease: normal"

def load_miRNA_dataset(MiRNA_filter=None, random_sample=None, case_sample=None):

    # columns are individuals, rows are miRNAs
    df = pd.read_csv('Datasets/GSE61741_series_matrix.csv', skiprows=52, skipfooter=1, sep='\t', index_col=0)

    # common to filter out miRNAs with median below 50
    df_median = df.median(axis=1)
    filter_population = df[df_median >= 50]
    filter_population = filter_population.transpose()

    # getting diseases
    with open('Datasets/GSE61741_series_matrix.csv', 'rt') as f:
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

def drop_miRNA_index(pop_rpool, pop_cpool, random_pool, case_pool):
    pop_rpool = pop_rpool.drop(columns="diseases")
    pop_cpool = pop_cpool.drop(columns="diseases")
    random_pool = random_pool.drop(columns="diseases")
    case_pool = case_pool.drop(columns="diseases")
    return pop_rpool, pop_cpool, random_pool, case_pool

def diseased_miRNAs(D_label):
    # inputs for D_label are the D variable or "Supplementary:"

    with open('Datasets/Diseased_miRNAs.txt', 'rt') as f:
        lines = f.readlines()

    lines = [line.strip("\n") for line in lines]
    disease_position = lines.index(D_label)

    # removing other diseases from file
    miRNA_list = lines[(disease_position)+1:]
    miRNA_list = miRNA_list[:miRNA_list.index("")]

    return miRNA_list

def separate_diseased_miRNAs(D_label, dataset, with_independent_features=False):
    disease_related_miRNAs = diseased_miRNAs(D_label)
    # if dataset == "timestamp" then D_label must be D2

    def splitting_data(pop, pool):
        only_diseased_miRNAs_pop = pop[pop.columns.intersection(disease_related_miRNAs)]
        only_diseased_miRNAs_pool = pool[pool.columns.intersection(disease_related_miRNAs)]

        common_pop_cols = pop.columns.intersection(disease_related_miRNAs)
        without_diseased_miRNAs_pop = pop.drop(columns=common_pop_cols)

        common_pool_cols = pool.columns.intersection(disease_related_miRNAs)
        without_diseased_miRNAs_pool = pool.drop(columns=common_pool_cols)

        return only_diseased_miRNAs_pop, without_diseased_miRNAs_pop, only_diseased_miRNAs_pool, without_diseased_miRNAs_pool

    if dataset == "miRNA":
        populations, pools = load_dataset(miRNA=True, disease_case_sample=D_label)
        # 0 = random sample, 1 = case sample
        pop = populations[1]
        pool = pools[1]

        o_pop, wo_pop, o_pool, wo_pool = splitting_data(pop, pool)

    if dataset == "timestamp":
        pops, pools = load_dataset(timestamp=True, with_independent_features=with_independent_features)          

        o_pop = []
        wo_pop = []
        o_pool = []
        wo_pool = []

        for pop, pool in zip(pops, pools):
            a, b, c, d = splitting_data(pop, pool)
            o_pop.append(a)
            wo_pop.append(b)
            o_pool.append(c)
            wo_pool.append(d)

    return o_pop, wo_pop, o_pool, wo_pool

def load_timestamp_dataset(with_independent_miRNAs=False, withNaN=False, MiRNA_filter=None, correlation=None):

    # columns are 215 individuals, rows are 1026 (1205?) miRNAs
    df = pd.read_csv('Datasets/GSE68951_series_matrix.txt', skiprows=58, skipfooter=1, sep='\t', index_col=0)

    population = df.transpose()

    with open('Datasets/GSE68951_series_matrix.txt', 'rt') as f:
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
        if os.path.exists("Datasets/independent_90.csv"):
            with open("Datasets/independent_90.csv", "r") as f:
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

            with open("Datasets/independent_90.csv", "w") as f:
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

    healthy_population = population[population["patient_id"]=="patient id: ZZ_control"]

    # add new row of NaN for all the missing timestamp entries for diseased individuals
    # alternative method: convert dataframes into a list then convert back after
    new_population = pd.DataFrame(columns=population.columns, index=[0])
    for patient in unique_patient_id_nocontrol:
        new_patient_df = pd.DataFrame(columns=population.columns, index=[0])
        p = population[population["patient_id"] == f"{patient}"]
        for i in range(8):
            t = (p["timepoint"] == f"timepoint: {(i+1)}")
            if (len(t)>i):
                if (t.iloc[i]): # if i in t should replace if (len(t)>i): but IndexError
                    new_patient_df = pd.concat([new_patient_df, p[t]], ignore_index=True)
            else:
                row_data = {col: np.nan for col in p.columns}
                row_data["patient_id"] = p.iat[0,0]
                row_data["disease"] = p.iat[0,1]
                row_data["timepoint"] = f"timepoint: {(i+1)}"
                rowNaN = pd.Series(row_data)
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
    
    return (pop_timepoint_i, pool_timepoint_i, sample_timepoint_i, healthy_population)

def drop_timestamp_index(pop, pool, sample=None, healthy=None):
    for idx in range(len(pop)):
        pop[idx] = pop[idx].drop(["disease", "timepoint", "patient_id"], axis=1)
        pool[idx] = pool[idx].drop(["disease", "timepoint", "patient_id"], axis=1)
    result = [pop, pool]

    if sample is not None:
        for idx in range(len(sample)):
            sample[idx] = sample[idx].drop(["disease", "timepoint", "patient_id"], axis=1)
        result.append(sample)

    if healthy is not None:
        for idx in range(len(healthy)):
            healthy[idx] = healthy[idx].drop(["disease", "timepoint", "patient_id"], axis=1)
        result.append(healthy)

    return result

# https://stackoverflow.com/questions/33997753/calculating-pairwise-correlation-among-all-columns
def independent(pop_timestamps, pool_timestamps, separate_timestamps=False, print_independent_miRNAs=False, correlation=None):
    
    if os.path.exists("Datasets/independent_pop.txt"):
        with open("Datasets/independent_pop.txt", "r") as f:
            independent_pop = f.read().splitlines()
        with open("Datasets/independent_pool.txt", "r") as f:
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

        with open("Datasets/independent_pop.txt", "w") as f:
            for item in pop_dataframe:
                f.write("%s\n" % item)
        with open("Datasets/independent_pool.txt", "w") as f:
            for item in pool_dataframe:
                f.write("%s\n" % item)

    # return (pop_dataframe, pool_dataframe, result, independent_columns)
    return pop_dataframe, pool_dataframe

def load_FitBit_dataset(pool_size=None):

    # 15 columns are activities, 457 rows are people's IDs
    df = pd.read_csv('Datasets/dailyActivity_merged.csv', sep=',')
    # check last column \n can be removed

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
    a = 27 if pool_size is None else pool_size
    randomshuffle = ShuffleSplit(n_splits=1, test_size=a)
    (pool, pop) = next(randomshuffle.split(unique_ids_data))
    
    # create list of 27 dataframes for each pop id
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
        pop_timestamp_i = pop_timestamp_i.set_index('Id') 
        pop_timestamp_i = pop_timestamp_i.drop(columns=["ActivityDate", "LoggedActivitiesDistance", "SedentaryActiveDistance"]) # quick fix, excluded "Calories\n" from drop
        pop_timestamp_i = pop_timestamp_i.apply(pd.to_numeric)
        pop_timestamp.append(pop_timestamp_i) #pop_timestamp_i.to_numpy()

    # create list of 8 dataframes for each pool id
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
        pool_timestamp_i = pool_timestamp_i.set_index('Id')
        pool_timestamp_i = pool_timestamp_i.drop(columns=["ActivityDate", "LoggedActivitiesDistance", "SedentaryActiveDistance"]) # quick fix, excluded "Calories\n" from drop
        pool_timestamp_i = pool_timestamp_i.apply(pd.to_numeric)
        pool_timestamp.append(pool_timestamp_i) #pool_timestamp_i.to_numpy()

    # pop time 0-7=30; 8=27; 9=25; 10=21; 11=19; 12-13=6; 14=5; 15-18=4; 19-31=2
    # pool time 0-11=5; 12-14=2
    return pop_timestamp, pool_timestamp

def load_electricity_dataset():

    # if os.path.exists("hourly_LD2011_2014.txt"):
    #     with open("hourly_LD2011_2014.txt", "rt") as f:
    #         lines = f.read().splitlines()
    #     hourly_df = pd.read_csv('hourly_LD2011_2014.txt', sep=';', index_col=0)

    # else:
        # columns are 370 individuals, rows are 140256 kWh/4
    df = pd.read_csv('Datasets/LD2011_2014.txt', sep=';', index_col=0, decimal=",")

    # with open('LD2011_2014.txt', 'rt') as f:
    #     lines = f.read().splitlines()

    # rows = []
    # for i in lines:
    #     i = i[:21]
    #     rows.append(i)
    # rows = rows[4::4]

    # look into group_by the hour instead of trying to sum over every set of 4 in the dataset
    # df.rename(index = lambda x : str(x)[:-6]) #bear in mind we have 2011-01-01 00 x3 and 2015-01-01 00 x1

    # create list of all hours in df
    all_hours = [i for i in range(35064)]
    all_hours = np.repeat(all_hours, 4)
    # add hour column to df and sum quarterly kWh
    df.insert(0, "hour", all_hours)
    hourly_df = df.groupby(["hour"]).sum()

    # with open("hourly_LD2011_2014.txt", "w") as f:
    #     for item in hourly_df:
    #         f.write("%s\n" % item)

    # with open('hourly_LD2011_2014.txt', 'rt') as f:
    #     lines = f.read().splitlines()

    # rows = 370 clients, columns = 35064 hours, data = kWh readings
    hourly_df = hourly_df.transpose()

    randomshuffle = ShuffleSplit(n_splits=1, test_size=92)
    (train, test) = next(randomshuffle.split(hourly_df))
    hourly_df_pop = hourly_df.iloc[train] # random population of 278 clients
    hourly_df_pool = hourly_df.iloc[test] # random pool of 92 clients

    # separate into 4 longitudinal dfs split by year, then we can run the attack for 370 clients over the 4 years
    pop_year_i = []
    pool_year_i = []
    for i in range(4):
        pop_yr = hourly_df_pop.iloc[:, i*8766:(i+1)*8766]
        pool_yr = hourly_df_pool.iloc[:, i*8766:(i+1)*8766]
        pop_year_i.append(pop_yr)
        pool_year_i.append(pool_yr)
    # print("pop shape", pop_year_i[0].shape, "pool shape", pool_year_i[0].shape)

    return (pop_year_i, pool_year_i)

def load_psid_dataset():

    labels1 = pd.read_csv('Datasets/J350695_labels.txt', sep='    ', skiprows=3, skipfooter=1)
    labels1 = labels1.iloc[1:5606]
    labels1_map = dict(zip(labels1["Variable"], labels1["Labels"]))

    labels2 = pd.read_csv('Datasets/J350694_labels.txt', sep='    ', skiprows=3, skipfooter=1)
    labels2 = labels2.iloc[1:]
    labels2_map = dict(zip(labels2["Variable"], labels2["Labels"]))


    df1 = pd.read_csv('Datasets/J350695.csv', sep=',') # 2017
    df1 = df1.rename(columns=labels1_map)

    # 9607 rows x 965 columns
    df1.dropna(how='all', inplace=True)
    df1 = df1.drop(columns=(df1.columns[(df1 == 0).all()])) # 40 all-0 columns (50 between 2017/2019)
    df1 = df1.groupby(["1968 FAMILY IDENTIFIER"]).mean() # 2518 rows x 964 columns


    df2 = pd.read_csv('Datasets/J350694.csv', sep=',') # 2019
    df2 = df2.rename(columns=labels2_map)

    # 9569 rows x 942 columns
    df2 = df2.drop(columns=(df2.columns[(df2 == 0).all()])) # 49 all-0 columns (50 between 2017/2019)
    df2 = df2.groupby(["1968 FAMILY IDENTIFIER"]).mean() # 2431 rows x 941 columns

    
    common_cols = df1.columns.intersection(df2.columns)
    df1 = df1[common_cols]
    df2 = df2[common_cols]

    randomshuffle = ShuffleSplit(n_splits=1, test_size=210)
    (pop1, pool1) = next(randomshuffle.split(df1.transpose()))
    (pop2, pool2) = next(randomshuffle.split(df2.transpose()))

    df1_pop = df1.iloc[pop1]
    df1_pool = df1.iloc[pool1]
    df2_pop = df2.iloc[pop2]
    df2_pool = df2.iloc[pool2]

    return df1_pop, df1_pool, df2_pop, df2_pool


def load_setap_dataset():

    # columns are measurements, rows are student groups
    # T1 = 64 rows x 85 columns
    # T2-3 = 74 rows x 85 columns
    # T4 = 63 rows x 85 columns
    # 5-11 = 74 rows x 85 columns
    df = []
    for i in range(11):
        df.append(pd.read_csv(f'Datasets/data+for+software+engineering+teamwork+assessment+in+education+setting/SETAP PROCESS DATA CORRECT AS FIE2016/setapProcessT{i+1}.csv', skiprows=1, sep=','))

    randomshuffle = ShuffleSplit(n_splits=1, test_size=18)
    (train, test) = next(randomshuffle.split(df[1]))
    pop, pool = [], []
    for interval in df:
        pop.append(interval.iloc[train]) # random population of 43-56 groups
        pool.append(interval.iloc[test]) # random pool of 18 groups
    # print("pop shape", pop[0].shape, "pool shape", pool[0].shape)

    return (pop, pool)

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

def load_dataset(miRNA=False, timestamp=False, FitBit=False, electricity=False, psid=False, setap=False, drop_index=True,
                 with_independent_features=False, withNaN=False, sample_individual=False, healthy_timestamp=False, correlation=None, feature_filter=None, random_sample_size=None, disease_case_sample=None):
    
    if miRNA==True:
        pop_rpool, pop_cpool, rpool, cpool = load_miRNA_dataset(MiRNA_filter=feature_filter, random_sample=random_sample_size, case_sample=disease_case_sample)
        if drop_index == True:
            pop_rpool, pop_cpool, rpool, cpool = drop_miRNA_index(pop_rpool, pop_cpool, rpool, cpool)
        population = [pop_rpool, pop_cpool]
        pool = [rpool, cpool]

    elif timestamp==True:
        population, pool, sample, healthy = load_timestamp_dataset(with_independent_miRNAs=with_independent_features, withNaN=withNaN, MiRNA_filter=feature_filter, correlation=correlation)
        if drop_index == True:
            if sample_individual == False and healthy_timestamp == False:
                population, pool = drop_timestamp_index(population, pool)
            elif sample_individual == False and healthy_timestamp == True:
                population, pool, healthy = drop_timestamp_index(population, pool, sample)
                return population, pool, healthy
            elif sample_individual == True and healthy_timestamp == False:
                population, pool, sample = drop_timestamp_index(population, pool, sample)
                return population, pool, sample
            elif sample_individual == True and healthy_timestamp == True:
                population, pool, sample, healthy = drop_timestamp_index(population, pool, sample, healthy)
                return population, pool, sample, healthy
            
    elif FitBit==True:
        population, pool = load_FitBit_dataset(pool_size=random_sample_size)

    elif electricity==True:
        population, pool = load_electricity_dataset()

    elif psid==True:
        df1_pop, df1_pool, df2_pop, df2_pool = load_psid_dataset()
        population = [df1_pop, df2_pop]
        pool = [df1_pool, df2_pool]

    elif setap==True:
        population, pool = load_setap_dataset()

    return population, pool
