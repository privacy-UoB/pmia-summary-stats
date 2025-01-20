import numpy as np
import matplotlib.pyplot as plt
from utils_datasets import load_timestamp_dataset, drop_timestamp_index

# Plot the differences between T1-T0 for each individual mirna
# Do list of timestamp 0 & 1 per patient, then plot one histogram showing diff. Over each mirna.
# Do this for all 26 patients


# load dataset
pop_timestamps, pool_timestamps, sample_timestamps = load_timestamp_dataset()
pop_timestamps, pool_timestamps = drop_timestamp_index(pop_timestamps, pool_timestamps)

# timestamp_0 = []
# timestamp_1 = []

# this for loop makes timestamps 0&1 have two sets of lists in them each, one for pop & pool
# for index, (x, y) in enumerate(zip(pop_timestamps, pool_timestamps)):
#     if index == 0:
#         timestamp_0.append(x)
#         timestamp_0.append(y)
#     elif index == 1:
#         timestamp_1.append(x)
#         timestamp_1.append(y)
#     else: 
#         continue

#     x.drop(["disease", "timepoint", "patient_id"], axis=1, inplace=True)
#     y.drop(["disease", "timepoint", "patient_id"], axis=1, inplace=True)

# currently only looking at timestamp 0 and 1
pop_timestamp0 = pop_timestamps[0]
pop_timestamp1 = pop_timestamps[1]
pool_timestamp0 = pool_timestamps[0]
pool_timestamp1 = pool_timestamps[1]

# create one list of all the sets of timestamps 0 & 1 for each patient
all_patients = []
for x in range(len(pop_timestamp0)):
    one_patient = []
    one_patient.append(pop_timestamp0.iloc[x])
    one_patient.append(pop_timestamp1.iloc[x])
    all_patients.append(one_patient)
for y in range(len(pool_timestamp0)):
    one_patient = []
    one_patient.append(pool_timestamp0.iloc[y])
    one_patient.append(pool_timestamp1.iloc[y])
    all_patients.append(one_patient)

# create a list of the difference between miRNA values for timepoint 0 & 1 for all 26 patients
differences_miRNA = []
for all in all_patients:
    difference = np.subtract(all[1], all[0])
    differences_miRNA.append(difference)

print("standard dev.", np.std(differences_miRNA))
plt.hist(np.ravel(differences_miRNA), bins=100)
plt.xlabel(f"difference over all miRNA values between timepoint 0 & 1")
plt.ylabel("count of miRNAs within 100 different range values")
plt.show()

# 26 histograms showing miRNA difference for timepoints 0 & 1
for index, all in enumerate(differences_miRNA):
    # x = all.sample(100, axis=1) # comment in for sample of 100 miRNAs
    plt.hist(all, bins=100)
    plt.xlabel(f"difference over all miRNA values between timepoint 0 & 1 for patient {(index+1)}")
    plt.ylabel("count of miRNAs within 100 different range values")
    plt.show()


# # create a list of the difference between miRNA values for timepoint 0 & 1 for all 26 patients
# differences_average_miRNA = []
# mu_j = np.average(pop_timestamps[0], axis=0)
# # mu_j0 = np.average(all_patients[0], axis=0) # wanted this over all of the first values in each set
# # mu_j1 = np.average(all_patients[1], axis=0) # wanted this over all of the second values in each set
# for all in all_patients:
#     difference = np.subtract((np.subtract(mu_j, all[1])), (np.subtract(mu_j, all[0])))
#     differences_average_miRNA.append(difference)

# # 26 histograms showing miRNA difference for timepoints (0 - mean) & (1 - mean)
# for index, all in enumerate(differences_average_miRNA):
#     # x = all.sample(100, axis=1) # comment in for sample of 100 miRNAs
#     plt.hist(all, bins=100)
#     plt.xlabel(f"difference over all miRNA values between timepoint (0-mean) & (1-mean) for patient {(index+1)}")
#     plt.ylabel("count of miRNAs within 100 different range values")
#     plt.show()
