import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from utils_datasets import load_dataset, drop_dataset_index, D3, split_pool

# Do the cosine distance comparison on the original dataset for the population, the pool, 
# then the split case pool (some fraction of the case pool added to the population). This will 
# help us to classify the robustness of the membership inference attack (e.g. whether points 
# are being classified as pool vs pop based on inference or based on the 'nearness' to the mean).

# load dataset
pop_rpool, pop_cpool, rpool, cpool = load_dataset(case_sample=D3)
pop_rpool, pop_cpool, rpool, cpool = drop_dataset_index(pop_rpool, pop_cpool, rpool, cpool)

pop = pop_cpool # make pop configurable
pool = cpool # make pool configurable

split_pop, split_cpool, cpool_into_pop = split_pool(pop, pool)

# cosine similarity pop
cosine_distance_pop = []
individual = pop.iloc[0]
for row in range(len(pop)):
    other_individuals = pop.iloc[row]
    d = distance.cosine((individual.to_numpy()).ravel(), (other_individuals.to_numpy()).ravel())
    cosine_distance_pop.append(d)

# cosine similarity pop added split pool
cosine_distance_split = []
individual = pop.iloc[0]
for row in range(len(split_pop)):
    other_individuals = split_pop.iloc[row]
    d = distance.cosine((individual.to_numpy()).ravel(), (other_individuals.to_numpy()).ravel())
    cosine_distance_split.append(d)

# cosine similarity pool
cosine_distance_pool = []
individual = pool.iloc[0]
for row in range(len(pool)):
    other_individuals = pool.iloc[row]
    d = distance.cosine((individual.to_numpy()).ravel(), (other_individuals.to_numpy()).ravel())
    cosine_distance_pool.append(d)

# cosine similarity pop to pool
cosine_distance_pop2pool = []
individual = pop.iloc[0]
for row in range(len(pool)):
    other_individuals = pool.iloc[row]
    d = distance.cosine((individual.to_numpy()).ravel(), (other_individuals.to_numpy()).ravel())
    cosine_distance_pop2pool.append(d)


# plot cosine distance pop
fig, ax = plt.subplots()
for l in range(len(cosine_distance_pop)):
    ax.plot(range(len(cosine_distance_pop)), cosine_distance_pop, linewidth=3.0)
plt.xlabel("individual")
plt.ylabel("cosine distance of one population individual over population")
plt.show()

# plot cosine distance pop added split pool
fig, ax = plt.subplots()
for l in range(len(cosine_distance_split)):
    ax.plot(range(len(cosine_distance_split)), cosine_distance_split, linewidth=3.0)
plt.xlabel("individual")
plt.ylabel("cosine distance of one population individual over population with 1/4 pool added")
plt.show()

# plot cosine distance pool
fig, ax = plt.subplots()
for l in range(len(cosine_distance_pool)):
    ax.plot(range(len(cosine_distance_pool)), cosine_distance_pool, linewidth=3.0)
plt.xlabel("individual")
plt.ylabel("cosine distance one pool individual over pool")
plt.show()

# plot cosine distance pop to pool
fig, ax = plt.subplots()
for l in range(len(cosine_distance_pop2pool)):
    ax.plot(range(len(cosine_distance_pop2pool)), cosine_distance_pop2pool, linewidth=3.0)
plt.xlabel("individual")
plt.ylabel("cosine distance one population individual over pool")
plt.show()
