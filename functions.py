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

#! Growth
def growth_fun(r_max,T,z,w,species_type):
    if z.shape[0] > 1: # If there is more than one reef
        T = np.repeat(T,z.shape[1]).reshape(z.shape[0],z.shape[1])
    else: # If there is a single reef
        T = np.array([np.repeat(T,z.shape[1])])
        
    r = np.zeros((z.shape[0],z.shape[1])) #Preallocate growth vector
    coral_col = np.where(species_type == 1)[1]
    algae_col = np.where(species_type == 2)[1]
    r[:,coral_col] =( (r_max[:,coral_col]/np.sqrt(2.*np.pi*pow(w[:,coral_col],2.)))
                    *np.exp((-pow((T[:,coral_col]-z[:,coral_col]),2.))/(2*pow(w[:,coral_col],2.))) )
    r[:,algae_col] = 0.49 * r_max[:,algae_col]

    return r
    
#! Mortality
def mortality_fun(r_max,T,z,w,species_type,mpa_status,alg_mort):
    m = np.zeros((z.shape[0],z.shape[1])) # Preallocate mortality vector
    algae_col = np.array([np.where(species_type == 2)[1]]) # Find algae columns
    
    if z.shape[0] > 1: # If there is more than one reef
        T = np.repeat(T,z.shape[1]).reshape(z.shape[0],z.shape[1]) # Reshape T array to correspond with z matrix
        
        m[z<T] = 1 - np.exp(-pow((T-z),2)/pow(w,2))[z<T]
        m[z>=T] = 0
        
        # Indices of mpa reefs (corresponds to rows in N_all)
        is_mpa = np.array([np.where(mpa_status == 1)[1]]) 
        # Indices of non-mpa reefs (corresponds to rows in N_all)
        not_mpa = np.array([np.where(mpa_status != 1)[1]])
        
        # Create arrays of indices that correspond to is_mpa & algae_col and not_mpa & algae_col
        is_mpa_rows = np.array([is_mpa.repeat(algae_col.shape[1])]) 
        not_mpa_rows = np.array([not_mpa.repeat(algae_col.shape[1])])
        algae_col_is_mpa = np.tile(algae_col,is_mpa.shape[1])
        algae_col_not_mpa = np.tile(algae_col,not_mpa.shape[1])

        # Macroalgae calculations for multiple reefs
        m[is_mpa_rows,algae_col_is_mpa] = 0.3
        m[not_mpa_rows,algae_col_not_mpa] = alg_mort[not_mpa_rows]
        
    else: # If there is a single reef
        T = np.array([np.repeat(T,z.shape[1])])
    
        #Coral calculations
        m[z<T] = 1 - np.exp(-pow((T-z),2)/pow(w,2))[z<T]
        m[z>=T] = 0
        
        # Macroalgae calculations for a single reef
        if mpa_status == 1:
             m[0,algae_col] = 0.3
        else:
            m[0,algae_col] = alg_mort

    # Apply a correction such that the minimum amount of mortality experienced is 0.03        
    m[m<0.03] = 0.03
    
    return m
    
#! Fitness
def fitness_fun(r_max,T,z,w,alphas,species_type,mpa_status,N_all,m_const,mortality_model,alg_mort):
    r = growth_fun(r_max,T,z,w,species_type)
    
    # If mortality varies with temperature
    if mortality_model == "temp_vary":
        m = mortality_fun(r_max,T,z,w,species_type,mpa_status,alg_mort)    
    # If mortality is constant
    else: 
        m = m_const
    
    #If there is more than one reef
    if N_all.shape[0] > 1:
        sum_interactions = np.array([np.sum(N_all[index,:] * alphas, axis=1) for index in range(N_all.shape[0])])
    else:
        sum_interactions = np.sum(alphas * N_all, axis=1)

    g = r * (1-sum_interactions) - m
    
    return g
    
#! dGdZ
def dGdZ_fun(r_max,T,z,w,alphas,species_type,mpa_status,N_all,m_const,mortality_model,alg_mort):
    h = 1e-5
    dGdZ = np.zeros(z.shape)
    
    #If there is more than one reef
    if N_all.shape[0] > 1:
        # For each reef
        for i in np.arange(z.shape[0]): 
            # For each species
            for j in np.arange(z.shape[1]):
                h_matrix = np.zeros(z.shape)
                h_matrix[i,j] = h
                # Take the symmetric difference quotient at point z[i,j]
                term1 = fitness_fun(r_max,T,z+h_matrix,w,alphas,species_type,mpa_status,
                                     N_all,m_const,mortality_model,alg_mort)
                term2 = fitness_fun(r_max,T,z-h_matrix,w,alphas,species_type,mpa_status,
                                         N_all,m_const,mortality_model,alg_mort)
                delta = (term1-term2)/(2*h)
                dGdZ[i,j] = delta[i,j]
    else:
        for j in np.arange(z.shape[1]):
            h_array = np.zeros(z.shape)
            h_array[0,j] = h
            # Take the symmetric difference quotient at point z[i,j]
            term1 = fitness_fun(r_max,T,z+h_array,w,alphas,species_type,mpa_status,
                                 N_all,m_const,mortality_model,alg_mort)
            term2 = fitness_fun(r_max,T,z-h_array,w,alphas,species_type,mpa_status,
                                     N_all,m_const,mortality_model,alg_mort)
            delta = (term1-term2)/(2*h)
            dGdZ[0,j] = delta[0,j]
            
    return dGdZ
    
#! d2GdZ2
def dGdZ2_fun(r_max,T,z,w,alphas,species_type,mpa_status,N_all,m_const,mortality_model,alg_mort):
    h = 1e-5
    dGdZ2 = np.zeros(z.shape)
    
    #If there is more than one reef
    if N_all.shape[0] > 1:
        # For each reef
        for i in np.arange(z.shape[0]): 
            # For each species
            for j in np.arange(z.shape[1]):
                h_matrix = np.zeros(z.shape)
                h_matrix[i,j] = h
                # Take the symmetric difference quotient at point z[i,j]
                term1 = fitness_fun(r_max,T,z+h_matrix,w,alphas,species_type,mpa_status,
                                     N_all,m_const,mortality_model,alg_mort)
                term2 = fitness_fun(r_max,T,z-h_matrix,w,alphas,species_type,mpa_status,
                                         N_all,m_const,mortality_model,alg_mort)
                term3 = fitness_fun(r_max,T,z,w,alphas,species_type,mpa_status,
                                         N_all,m_const,mortality_model,alg_mort)
                delta = (term1+term2-2*term3)/pow(h,2)
                dGdZ2[i,j] = delta[i,j]
                
    else:
        for j in np.arange(z.shape[1]):
            h_array = np.zeros(z.shape)
            h_array[0,j] = h
            # Take the symmetric difference quotient at point z[i,j]
            term1 = fitness_fun(r_max,T,z+h_array,w,alphas,species_type,mpa_status,
                                 N_all,m_const,mortality_model,alg_mort)
            term2 = fitness_fun(r_max,T,z-h_array,w,alphas,species_type,mpa_status,
                                     N_all,m_const,mortality_model,alg_mort)
            term3 = fitness_fun(r_max,T,z,w,alphas,species_type,mpa_status,
                                     N_all,m_const,mortality_model,alg_mort)
            delta = (term1+term2-2*term3)/pow(h,2)
            dGdZ2[0,j] = delta[0,j]
            
    return dGdZ2
    
#! dNdt
def dNdt_fun(r_max,T,z,w,alphas,species_type,mpa_status,N_all,m_const,mortality_model,alg_mort,V,D,beta):    
    
    if N_all.shape[0] > 1:
        V = np.tile(V, N_all.shape[0]).reshape(N_all.shape[0],N_all.shape[1])
    
    g = fitness_fun(r_max,T,z,w,alphas,species_type,mpa_status,N_all,m_const,mortality_model,alg_mort)
    dGdZ2 = dGdZ2_fun(r_max,T,z,w,alphas,species_type,mpa_status,N_all,m_const,mortality_model,alg_mort)      
    popdy = np.multiply(N_all,g)
    genload = 0.5 * V * dGdZ2 * N_all
    
    dispersal = beta * np.dot(D,N_all)
    free_space = 1 - N_all.sum(axis=1)
    larval_input = np.array([dispersal[index,:] * x for index, x in enumerate(free_space)])
    
    # Algae don't experience recruitment from other reefs
    algae_ID = np.where(species_type==2)[1] #find algae columns
    larval_input[:,algae_ID] = 0
    
    dNdt = popdy + genload + larval_input
    
    #! Prevent NaN or population values below 1e-6 in output
    if np.isnan(dNdt).any():
        ID = np.where(np.isnan(dNdt))
        dNdt[ID] = 1e-6
    if (dNdt+N_all < 1e-6).any():
        ID = np.where(dNdt+N_all < 1e-6)
        dNdt[ID] = 1e-6

    return dNdt
    
#! q
def q_fun(N_all, N_min=1e-6):
    q = np.maximum(0, 1- N_min/(np.maximum(N_min,2*N_all)))
    return q
    
#! dZdt
def dZdt_fun(r_max,T,z,w,alphas,species_type,mpa_status,N_all,m_const,mortality_model,alg_mort,V,D,beta):
        
    if N_all.shape[0] > 1:
        V = np.tile(V, N_all.shape[0]).reshape(N_all.shape[0],N_all.shape[1])
    
    q = q_fun(N_all)
    dGdZ = dGdZ_fun(r_max,T,z,w,alphas,species_type,mpa_status,N_all,m_const,mortality_model,alg_mort)
    directional_selection = q * V * dGdZ
    gene_flow_term1 = (np.dot(D, N_all * z) / np.dot(D, N_all)) - z
    gene_flow_term2 = (beta * np.dot(D, N_all)) / (beta * np.dot(D, N_all) + N_all)
    free_space = 1 - N_all.sum(axis=1)
    gene_flow = np.array([(gene_flow_term1*gene_flow_term2)[index,:] * x for index, x in enumerate(free_space)])
    
    algae_ID = np.where(species_type==2)[1] #find algae columns
    gene_flow[:,algae_ID] = 0
    
    dZdt = directional_selection + gene_flow
    
    return dZdt
    
#! Generate temps
def generate_temps_fun(size,mid=25,temp_range=2.5,temp_scenario='linear'):
    if temp_scenario == 'uniform':
        temps = np.repeat(mid,size)
    if temp_scenario == 'linear':
        temps = np.linspace(mid-temp_range, mid+temp_range, size)
    elif temp_scenario == 'random':
        a = np.linspace(mid-temp_range, mid+temp_range, size)
        temps = np.random.choice(a, size, replace=False, p=None)
    return temps

#! Generate cosine temps
def generate_temps_cos_fun(size,min_SST=20,max_SST=30):
    range_SST = max_SST - min_SST
    mid = range_SST/2
    mid_SST = (max_SST + min_SST)/2
    x_vals = np.linspace(0,2,size)
    cos_vals = mid * np.cos(np.pi*x_vals) + mid_SST
    return cos_vals

#! Generate initial traits
def generate_traits_fun(nsp,size,temps,mid=25,temp_range=2.5,trait_scenario='perfect_adapt'):
    if trait_scenario == 'u_constant':
        a = np.linspace(mid-(temp_range/4), mid+(temp_range/4), nsp+2)[1:-nsp+2]
        b = np.repeat(a,size)
        traits = np.reshape(b,(nsp,
                               size))
    if trait_scenario == 'same_constant':
        traits = np.full((nsp,size),mid)
    elif trait_scenario == 'perfect_adapt':
        a = np.tile(temps,nsp)
        traits = np.reshape(a,(nsp,size))
    return traits.T
    
#! Generate initial fractional covers
def generate_state_fun(size, nsp, cover=0.01,random=False):
    state = np.full((size,nsp),cover)
    if random:
        state = np.full(nsp*size,np.random.uniform(1e-6,.33,nsp*size)).reshape(size,nsp)
    return state
    
#! Set MPA
def set_MPA_fun(temps,N_all,species_type,size,amount=0.2,strategy='random'):
    mpa = np.zeros(size)
    ID = np.where(species_type==1)[1]
    corals = N_all[:,ID].sum(axis=1) # Get the subset of N_all that corresponds to coral cover per reef
    #The following line applies to the 'portfolio' strategy that I haven't coded here
    ncoral = np.asarray(np.where(species_type==1)).sum() # How many coral species?
    
    if strategy == 'none':
        index = np.asarray([])
    elif strategy=='hot':
        index = (-temps).argsort()[0:np.int(amount*size)]
    elif strategy=='cold':
        index = temps.argsort()[0:np.int(amount*size)]
    elif strategy=='hotcold':
        index = np.r_[0:np.int(amount*size/2),np.int(size-(amount*size/2)):np.int(size)]
    elif strategy=='space':
        index = np.round(np.linspace(0,size-1,np.int(amount*size)))
    elif strategy=='highcoral':
        index = (-corals).argsort()[0:np.int(amount*size)]
    elif strategy=='lowcoral':
        index = corals.argsort()[0:np.int(amount*size)]
    elif strategy=='random':
        index = np.random.choice(np.arange(0,size),np.int(amount*size), replace=False)
    
    mpa[index.astype(int)]=1
    return np.array([mpa])    
    
#! Set restoration strategy
def set_restore_fun(temps,N_all,species_type,size,amount=1.0,strategy='random'):
    restore_sites = np.zeros(size)
    ID = np.where(species_type==1)[1]
    corals = N_all[:,ID].sum(axis=1) # Get the subset of N_all that corresponds to coral cover per reef
    #The following line applies to the 'portfolio' strategy that I haven't coded here
    ncoral = np.asarray(np.where(species_type==1)).sum() # How many coral species?
    
    if strategy == 'none':
        index = np.asarray([])
    elif strategy=='hot':
        index = (-temps).argsort()[0:np.int(amount*size)]
    elif strategy=='cold':
        index = temps.argsort()[0:np.int(amount*size)]
    elif strategy=='hotcold':
        index = np.r_[0:np.int(amount*size/2),np.int(size-(amount*size/2)):np.int(size)]
    elif strategy=='space':
        index = np.round(np.linspace(0,size-1,np.int(amount*size)))
    elif strategy=='highcoral':
        index = (-corals).argsort()[0:np.int(amount*size)]
    elif strategy=='lowcoral':
        index = corals.argsort()[0:np.int(amount*size)]
    elif strategy=='random':
        index = np.random.choice(np.arange(0,size),np.int(amount*size), replace=False)
    
    restore_sites[index.astype(int)]=1
    return np.array([restore_sites])    
    
#! Set restoration trait value
def set_trait_fun(Z_all, V, num_restore, restore_array, strategy='value',value=25, scaling_frac=0.5, percentile=50):
    Z_rest = np.zeros((num_restore,Z_all.shape[1]))
    
    if strategy == 'value':
        Z_rest = Z_rest + value
    elif strategy == 'variance':
        Z_rest = Z_all[restore_array,:] + scaling_frac * np.sqrt(V)
    elif strategy == 'percentile':
        for i in np.arange(0,Z_all.shape[1]):
            Z_rest[i] = np.percentile(Z_all[:-num_restore,i], percentile)

    return Z_rest
    
#! Coral restore function
def coral_restore_fun(param,spp_state,trait_state,temps,anomalies,algaemort_full, temp_change="constant",
                       burnin=True):

    nsp = param['nsp']
    size0 = param['size']
    time_steps = param['time_steps']
    species_type = param['species_type']
    r_max = param['r_max']
    V = param['V']
    D0 = param['D0']
    D1 = param['D1']
    beta = param['beta']
    m_const = param['m_const']
    w = param['w']
    alphas = param['alphas']
    mpa_status = param['mpa_status']
    mortality_model = param['mortality_model']
    maxtemp = param['maxtemp']
    annual_temp_change = param['annual_temp_change']
    timemod = param['timemod']
    restoration_years = param['restoration_years']
    source_cover = param['source_cover']
    trait_strategy = param['trait_strategy']
    value = param['value']
    scaling_frac = param['scaling_frac']
    percentile = param['percentile']
    num_restore = param['num_restore']
    restore_array = param['restore_array']
        
    size = size0 + num_restore #to account for restoration reef
    
    SST_matrix = np.zeros([size,time_steps])
    SST_matrix[:,0] = temps + anomalies[:,0]
    dtemp_array = np.zeros([time_steps])
    
    for i in np.arange(1, time_steps):
        if temp_change == "sigmoid":
            temps =  SST_matrix[:,i-1] 
            dtemp1 = annual_temp_change*temps.mean()
            dtemp2 = 1-(temps.mean()/maxtemp)
            dtemp = np.repeat(dtemp1*dtemp2, size)
            SST_matrix[:,i] = SST_matrix[:,i-1] + dtemp
            
        if temp_change == "constant":
            dtemp = np.repeat(0, size)
            SST_matrix[:,i] = SST_matrix[:,i-1] + dtemp
            
        if temp_change == "linear":
            dtemp = np.repeat(annual_temp_change, size)
            SST_matrix[:,i] = SST_matrix[:,i-1] + dtemp
        
    SST_matrix =  SST_matrix + anomalies
    
    N_ALL = np.zeros((size,nsp,time_steps))
    Z_ALL = np.zeros((size,nsp,time_steps))
    N_ALL[:,:,0] = spp_state
    Z_ALL[:,:,0] = trait_state

    algaemort_sub = algaemort_full[:,timemod:timemod+time_steps]
    tick = 0
    
    if burnin:
        print "BURNIN"
        D = D0 #No restoration, always use baseline connectivity matrix
        # Second-order Runge Kutta solver
        for i in np.arange(0,time_steps-1):
            alg_mort = algaemort_sub[:,i]
            dN1 = dNdt_fun(r_max,SST_matrix[:,tick],Z_ALL[:,:,tick],w,alphas,species_type,mpa_status,
                             N_ALL[:,:,tick],m_const,mortality_model,alg_mort,V,D,beta)
            dZ1 = dZdt_fun(r_max,SST_matrix[:,tick],Z_ALL[:,:,tick],w,alphas,species_type,mpa_status,
                            N_ALL[:,:,tick],m_const,mortality_model,alg_mort,V,D,beta)

            N_ALL_1 = N_ALL[:,:,tick] + dN1*0.5
            Z_ALL_1 = Z_ALL[:,:,tick]  + dZ1*0.5

            dN2 = dNdt_fun(r_max,SST_matrix[:,tick],Z_ALL_1,w,alphas,species_type,mpa_status,
                             N_ALL_1,m_const,mortality_model,alg_mort,V,D,beta)
            dZ2 = dZdt_fun(r_max,SST_matrix[:,tick],Z_ALL_1,w,alphas,species_type,mpa_status,
                            N_ALL_1,m_const,mortality_model,alg_mort,V,D,beta)
 
            N_ALL[:,:,tick+1] = N_ALL[:,:,tick] + (dN1 + dN2)/2
            Z_ALL[:,:,tick+1] = Z_ALL[:,:,tick] + (dZ1 + dZ2)/2
            
            #! CURRENTLY WORKING ON THE FOLLOWING LINES:
            #! Calls the function that sets trait restoration
            trait_restore = set_trait_fun(Z_ALL[:,:,tick+1], V, num_restore, restore_array, trait_strategy, value, scaling_frac, percentile)

            # Keep restoration reef (last reef) at constant N with 100% of all species:
            N_ALL[-num_restore:,:,tick+1] = source_cover
            Z_ALL[-num_restore:,:,tick+1] = trait_restore
            
            tick += 1
            
    else: #For future/restoration scenarios
        print "RUNTIME"
        # Second-order Runge Kutta solver
        for i in np.arange(0,time_steps-1):
            alg_mort = algaemort_sub[:,i]
            
            if tick in restoration_years:
                D=D1
            else:
                D=D0
            
            dN1 = dNdt_fun(r_max,SST_matrix[:,tick],Z_ALL[:,:,tick],w,alphas,species_type,mpa_status,
                             N_ALL[:,:,tick],m_const,mortality_model,alg_mort,V,D,beta)
            dZ1 = dZdt_fun(r_max,SST_matrix[:,tick],Z_ALL[:,:,tick],w,alphas,species_type,mpa_status,
                            N_ALL[:,:,tick],m_const,mortality_model,alg_mort,V,D,beta)

            N_ALL_1 = N_ALL[:,:,tick] + dN1*0.5
            Z_ALL_1 = Z_ALL[:,:,tick]  + dZ1*0.5

            dN2 = dNdt_fun(r_max,SST_matrix[:,tick],Z_ALL_1,w,alphas,species_type,mpa_status,
                             N_ALL_1,m_const,mortality_model,alg_mort,V,D,beta)
            dZ2 = dZdt_fun(r_max,SST_matrix[:,tick],Z_ALL_1,w,alphas,species_type,mpa_status,
                            N_ALL_1,m_const,mortality_model,alg_mort,V,D,beta)

            N_ALL[:,:,tick+1] = N_ALL[:,:,tick] + (dN1 + dN2)/2
            Z_ALL[:,:,tick+1] = Z_ALL[:,:,tick] + (dZ1 + dZ2)/2
            
            #! CURRENTLY WORKING ON THE FOLLOWING LINES:
            #! Calls the function that sets trait restoration
            trait_restore = set_trait_fun(Z_ALL[:,:,tick+1], V, num_restore, restore_array, trait_strategy, value, scaling_frac, percentile)
            # Keep restoration reef (last reef) at constant N with 100% of all species:
            N_ALL[-num_restore:,:,tick+1] = source_cover
            Z_ALL[-num_restore:,:,tick+1] = trait_restore

            tick += 1
    
    return N_ALL, Z_ALL, SST_matrix
    
#! Function that modifies a connectivity matrix to "bias" the diagonal
def bias_diagonal(A, alpha):
    I = np.identity(A.shape[0])
    A_bias = (1-alpha)*A + alpha*I
    return A_bias

#! Functions to generate random and regular matrices
def _distance_matrix(L):
    Dmax = L//2
 
    D  = range(Dmax+1)
    D += D[-2+(L%2):0:-1]
 
    return circulant(D)/Dmax
    
def _pd(d, p0, beta):
    return beta*p0 + (d <= p0)*(1-beta)
    
def watts_strogatz(L, p0, beta, directed=False, rngseed=1):
    rng = np.random.RandomState(rngseed)
 
    d = _distance_matrix(L)
    p = _pd(d, p0, beta)
 
    if directed:
        A = 1*(rng.random_sample(p.shape) < p)
        np.fill_diagonal(A, 0)
    else:
        upper = np.triu_indices(L, 1)
 
        A          = np.zeros_like(p, dtype=int)
        A[upper]   = 1*(rng.rand(len(upper[0])) < p[upper])
        A.T[upper] = A[upper]
 
    return A

def preserve_self_loops(matrix0, edges):
    # matrix0 needs to have 1's in the diagonal for this to work
    matrix1=matrix0
    for i in np.arange(0,matrix0.shape[0]):
        deg_dist = np.count_nonzero(matrix0[i,:])
        if deg_dist > edges:
            for k in np.arange(0,deg_dist-edges):
                non_zeros = (np.where(matrix0[i,:]>0)[0]).tolist()
                non_zeros.remove(i)
                random_index = non_zeros[np.random.randint(0,len(non_zeros))]
                matrix1[i,random_index] = 0
    return matrix1

def D_norm(D0):
    size = D0.shape[0]
    D1  = np.zeros((size,size)) # Preallocate matrix
    for i in np.arange(size):
        D1[:,i]  = (D0[:,i] / D0[:,i].sum()).T
    return D1
    
#! Function to generate a matrix that approximates a linear configuration
def D_linear(size, p_dispersal):
    D_linear = np.zeros((size,size))
    
    for i in np.arange(0,size):
        D_linear[i,i]= 1-p_dispersal*2
        if i != 0 and i != size-1:
            D_linear[i,i-1] = p_dispersal
            D_linear[i,i+1] = p_dispersal
        if i != 0 and i == size-1:
            D_linear[i,i-1] = p_dispersal
        if i == 0 and i != size-1:
            D_linear[i,i+1] = p_dispersal
            
    return D_linear  
    
#! Function to generate random matrix
def gen_rand_matrix(size, seed, connections=4):    
    # Network size = 20
    #! Note: directed = False produces a symmetric matrix
    random.seed(seed)

    L = size
    K = connections
    p0 = K/(L-1)
    regular = watts_strogatz(beta=0,directed=False,L=L,p0=p0)

    G = nx.Graph()
    G.add_nodes_from(np.arange(0,size))

    for i in np.arange(0,size):
        for j in np.arange(0,size):
            if regular[i,j] == 1:
                G.add_edge(i,j)
            
    reg_G = nx.to_numpy_matrix(G)
    di = np.diag_indices(size)

    random_G = nx.double_edge_swap(nx.from_numpy_matrix(reg_G), nswap=size, max_tries=1000)
    random_G2 = nx.to_numpy_matrix(random_G)

    di = np.diag_indices(size)
    random_G2[di]=1

    edges = connections+1
    random_G2_revised = np.asarray(preserve_self_loops(random_G2,edges=edges))
    
    random_G2_norm = D_norm(random_G2_revised)
    
    return random_G2_norm
    
#! Function to generate regular matrix
def gen_reg_matrix(size, connections=4):    
    #! Note: directed = False produces a symmetric matrix
    L = size
    K = connections
    p0 = K/(L-1)
    regular = watts_strogatz(beta=0,directed=False,L=L,p0=p0)
    
    G = nx.Graph()
    G.add_nodes_from(np.arange(0,size))

    for i in np.arange(0,size):
        for j in np.arange(0,size):
            if regular[i,j] == 1:
                G.add_edge(i,j)
            
    reg_G = nx.to_numpy_matrix(G)
    di = np.diag_indices(size)
    reg_G[di]=1
    
    reg_G_norm = D_norm(reg_G)

    return reg_G_norm