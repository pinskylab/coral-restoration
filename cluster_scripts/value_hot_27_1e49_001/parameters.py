#! Load packages
from __future__ import division
import numpy as np
import seaborn as sns
sns.set(style='ticks', palette='Paired')
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import random
from scipy.linalg import circulant
import networkx as nx
from functions import *

#! SET THE NUMBER OF ITERATIONS
iterations = 5 # Number of stochastic iterations

#! GENERAL PARAMETERS
nsp = 2 # Number of species in model
size = 20 # Number of reefs in model 
burnin = 1000 # Length of burn-in period
runtime = 500 # Length of environmental change period
mid = 27. # Mean temperature across all reefs at start of simulation
temp_range = 3. # Range of temperatures across reefs at start of simulation
species_type = np.array([[1,2]]) # Species type ID
species = ["C1","M1"] # Species labels
temp_stoch = 0.3
r_max = np.array([[1.5,1.]])
w = np.array([[1.5,2.]])
beta = np.array([[.1,.1]])
alphas = np.array([[1.,1.2],[1.,1.]]) 
m_const = 0.1 # Value used for constant mortality case
mortality_model = "temp_vary"
alg_mort = 0.15
V = np.array([[0.01,0.01]])
annual_temp_change = 0.011
maxtemp = 32 #for sigmoid temperature increase scenario

#! RESTORATION PARAMETERS
amount = 0.2 #Percentage of reefs to restore (no. of reefs to restore = size * amount)
strategy = 'hot'#Which reefs to restore? Options: hot, cold, random
trait_strategy = 'value' #What traits to send? Options: 'value', 'variance', 'percentile'
value = 27 #For value strategy
scaling_frac = 0.5 #For variance strategy
percentile = 50 #For percentile strategy
source_cover = np.array([1e-49,1e-49]) #Sets the fractional cover at restoration source sites
restoration_years = list(range(runtime))

#! Create multivariate normal covariance matrix for temperature anomalies
mdim = size 
lindec = np.exp(np.linspace(0,-5,num=mdim)) # Decrease the second value to reduce the range of correlation
ma = np.zeros((mdim,mdim))
ma[:,0] = lindec
for i in np.arange(1,mdim):
    ma[i:,i] = lindec[0:-i]
ma_temp = ma + ma.T
np.fill_diagonal(ma_temp,np.diagonal(ma))
ma = ma_temp    
sds = np.array([np.repeat(.2*temp_stoch,size)])
b = np.multiply(sds,sds.T)
spatial_temp = b*ma