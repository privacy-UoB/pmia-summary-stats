import numpy as np
import pandas as pd
from utils import LLR, L1, L1_ttest

pop = np.array([[1,2],[2,2],[3,3],[4,5]])
assert pop.shape == (4, 2) # 4 individuals, 2 miRNAs

cpool = np.array([[10,11],[12,5]]) #same figures means standard deviation is 0

print (LLR(pop, pop, cpool))
print (L1(pop, pop, cpool))
print (L1_ttest(pop, pop, cpool))
