import numpy as np
import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
import time
import random
from functions import *
import parameters as P

SST0 = generate_temps_cos_fun(P.size,min_SST=20,max_SST=30) 
spp_state = generate_state_fun(P.size, P.nsp, cover=0.25,random=False) 
trait_state = generate_traits_fun(P.nsp,P.size,SST0,P.mid,P.temp_range,trait_scenario='perfect_adapt') 


#! This function submits multiple "sub_jobs" to the cluster:
def bound(njob, job_i, array_size):
    """ Returns the range of values that the job job_i out of njob jobs
    should process.

    Args:
      njob: total number of jobs in the array.
      job_i: index of the current job.
      array_size: the size of the array to split.

    Returns:
      start: the index of array to start.
      stop: the index of the array to stop.
      Note that array[start:stop] returns array[start]...array[stop-1]; that is,
      array[stop] is excluded from the range.
      
    """
    step = np.ceil(float(array_size)/float(njob))
    print 'njob=', njob, 'job_i= ', job_i, 'array_size= ', array_size, 'step= ', step
    start = int(job_i*step)
    stop = int(min((job_i+1)*step, array_size))
    print 'start= ', start, 'stop= ', stop
    return start, stop


if __name__ == '__main__':
    # get job_i from the command line argument
    arg = sys.argv
    njob = int(arg[1])
    job_i  = int(arg[2])
    
    #SET NUMBER OF ITERATIONS

    seed_global = np.arange(0,P.iterations) # Total number of iterations
    start, stop = bound(njob, job_i, seed_global.size)
    seed_values = seed_global[start:stop] 
    
    for seed in seed_values:
        np.random.seed(seed)
    
        #! Create the connectivity matrix
        #! Choose the baseline connectivity matrix (source reef does not contribute anything):
        connections=4 # Number of patches (NOT including itself) that a patch is connected to
        #! REGULAR MATRIX
        D0 = gen_reg_matrix(size=P.size, connections=connections)
        #! Choose sites to restore
        restore_status = set_restore_fun(SST0,spp_state,P.species_type,P.size,P.amount,P.strategy)
        num_restore = np.count_nonzero(restore_status)
        # Matrix with restoration site row and column
        D1 = np.zeros((D0.shape[0]+num_restore,D0.shape[1]+num_restore))
        D1[:-num_restore,:-num_restore] = D0
        D_no_restore = np.zeros(D1.shape)
        D_no_restore[:,:] = D1
        #Set the 'no restore' matrix such that extra source reefs are self_recruiting
        for i in np.arange(P.size,P.size+num_restore):
            D_no_restore[i,i] = 1.0
        D_restore = np.zeros(D_no_restore.shape)
        D_restore[:,:] = D_no_restore
        restore_array = np.where(restore_status == 1)[1]
        i=P.size
        for site in restore_array:
            D_restore[site,i] = 1.0
            i += 1

        SST0_restore = np.ones((SST0.shape[0]+num_restore))
        SST0_restore[:-num_restore] = SST0
        SST0_restore[-num_restore:] = SST0[-1]
        #shape: number of reefs

        anomalies_burn = np.tile(np.random.normal(0,P.temp_stoch,P.burnin),P.size).reshape((P.size,P.burnin)) 
        anomalies_burn_restore = np.ones((anomalies_burn.shape[0]+num_restore,anomalies_burn.shape[1]))
        anomalies_burn_restore[:-num_restore,:] = anomalies_burn
        anomalies_burn_restore[-num_restore:,:] = anomalies_burn[-1,:]
        #shape: number of reefs x time steps 

        anomalies_run = np.tile(np.random.normal(0,P.temp_stoch,P.runtime),P.size).reshape((P.size,P.runtime)) 
        anomalies_run_restore = np.ones((anomalies_run.shape[0]+num_restore,anomalies_run.shape[1]))
        anomalies_run_restore[:-num_restore,:] = anomalies_run
        anomalies_run_restore[-num_restore:,:] = anomalies_run[-1,:]
        #shape: number of reefs x time steps 

        algaemort = np.random.uniform(P.alg_mort,P.alg_mort,(P.runtime+P.burnin)*P.size).reshape((P.size,P.runtime+P.burnin))
        algaemort_restore = np.ones((algaemort.shape[0]+num_restore, algaemort.shape[1]))
        algaemort_restore[:-num_restore,:] = algaemort
        algaemort_restore[-num_restore:,:] = algaemort[-1,:]
        #shape: number of reefs x time steps 

        spp_state_restore = np.ones((spp_state.shape[0]+num_restore, spp_state.shape[1]))
        spp_state_restore[:-num_restore,:] = spp_state
        spp_state_restore[-num_restore:,:] = 1.0
        #shape: number of reefs x number of species

        trait_state_restore = np.ones((trait_state.shape[0]+num_restore, trait_state.shape[1]))
        trait_state_restore[:-num_restore,:] = trait_state
        trait_state_restore[-num_restore:,:] = 35
        #shape: number of reefs x number of species

        mpa_status = set_MPA_fun(SST0,spp_state,P.species_type,P.size,amount=P.mpa_amount,strategy='none')
        mpa_status_restore = np.ones((mpa_status.shape[0],mpa_status.shape[1]+num_restore))
        mpa_status_restore[:,:-num_restore] = mpa_status
        mpa_status_restore[:,-num_restore:] = mpa_status[:,-1]
        #shape: 1 x number of reefs

        #! BURN-IN
        spp_state_restore[-num_restore:,:] = P.source_cover
        trait_state_restore[-num_restore,:] = np.repeat(P.value,P.nsp)

        # This is where you tell the model when to use the restoration matrix; we can write something more sophisticated to
        # generate this later on. For example, something that takes in a "frequency" instead of having to manually input years
        restoration_years = []
        
        time_steps = P.burnin
        timemod = 0 #to offset algae mortality index
        parameters_dict = {'nsp': P.nsp, 
                            'size': P.size, 
                            'time_steps': P.burnin, 
                            'species_type': P.species_type, 
                            'V': P.V, 
                            'D0': D_no_restore, 
                            'D1': D_restore,
                            'beta': P.beta,
                            'r_max': P.r_max,
                            'alphas': P.alphas,
                            'mortality_model': P.mortality_model,
                            'mpa_status': mpa_status,
                            'w': P.w,
                            'm_const': P.m_const,
                            'maxtemp': P.maxtemp,
                            'annual_temp_change': P.annual_temp_change,
                            'timemod': timemod,
                            'restoration_years': restoration_years,
                            'source_cover': P.source_cover,
                            'trait_strategy': P.trait_strategy,
                            'value': P.value,
                            'scaling_frac': P.scaling_frac,
                            'percentile': P.percentile,
                            'num_restore': num_restore,
                            'restore_array': restore_array
                            }

        N0, Z0, SST_burnin = coral_restore_fun2(parameters_dict,spp_state_restore,trait_state_restore,
                                                           SST0_restore,anomalies_burn_restore,
                                                           algaemort_restore,temp_change="constant",
                                                           burnin=True)
        #! RUNTIME
        # Alter the baseline percent cover on the source reef in order to set the amount of coral cover to send to 
        # restoration reefs. The amount in the second column doesn't matter since macroalgae don't 
        # experience recruitment from other sites
        # Note: the code does NOT like having a zero value in one of these; use 1e-6 to approximate zero

        #runtime parameters: all restoration years
        mpa_status = set_MPA_fun(SST0,spp_state,P.species_type,P.size,amount=P.mpa_amount,strategy=P.mpa_strategy)
        time_steps = P.runtime

        # This is where you tell the model when to use the restoration matrix; we can write 
        #something more sophisticated to generate this later on. For example, something that takes 
        # in a "frequency" instead of having to manually input years

        restoration_years = P.restoration_years #restoration occurs during all years in runtime

        timemod = P.burnin #to offset algae mortality index

        parameters_dict = {'nsp': P.nsp, 
                            'size': P.size, 
                            'time_steps': P.runtime, 
                            'species_type': P.species_type, 
                            'V': P.V, 
                            'D0': D_no_restore, 
                            'D1': D_restore,
                            'beta': P.beta,
                            'r_max': P.r_max,
                            'alphas': P.alphas,
                            'mortality_model': P.mortality_model,
                            'mpa_status': mpa_status,
                            'w': P.w,
                            'm_const': P.m_const,
                            'maxtemp': P.maxtemp,
                            'annual_temp_change': P.annual_temp_change,
                            'timemod': timemod,
                            'restoration_years': restoration_years,
                            'source_cover': P.source_cover,
                            'trait_strategy': P.trait_strategy,
                            'value': P.value,
                            'scaling_frac': P.scaling_frac,
                            'percentile': P.percentile,
                            'num_restore': num_restore,
                            'restore_array': restore_array
                            }               
                                    
        N1, Z1, SST_runtime = coral_restore_fun2(parameters_dict,N0[:,:,-1],Z0[:,:,-1],
                                                           SST_burnin[:,-1],anomalies_run_restore,
                                                           algaemort_restore,temp_change="sigmoid",
                                                           burnin=False)
                          
        
        
    

        np.save("N0_"+P.trait_strategy+"_"+P.strategy+"_"+str(P.value)+"_"+str(P.source_cover[0])+"_"+str(P.V[0][0])+"_"+str(seed)+".npy", N0)
        np.save("Z0_"+P.trait_strategy+"_"+P.strategy+"_"+str(P.value)+"_"+str(P.source_cover[0])+"_"+str(P.V[0][0])+"_"+str(seed)+".npy", Z0)
        np.save("SST0"+P.trait_strategy+"_"+P.strategy+"_"+str(P.value)+"_"+str(P.source_cover[0])+"_"+str(P.V[0][0])+"_"+str(seed)+".npy", SST_burnin)

        np.save("N1_"+P.trait_strategy+"_"+P.strategy+"_"+str(P.value)+"_"+str(P.source_cover[0])+"_"+str(P.V[0][0])+"_"+str(seed)+".npy", N1)
        np.save("Z1_"+P.trait_strategy+"_"+P.strategy+"_"+str(P.value)+"_"+str(P.source_cover[0])+"_"+str(P.V[0][0])+"_"+str(seed)+".npy", Z1)
        np.save("SST1_"+P.trait_strategy+"_"+P.strategy+"_"+str(P.value)+"_"+str(P.source_cover[0])+"_"+str(P.V[0][0])+"_"+str(seed)+".npy", SST_runtime)
        np.save("restore_status_"+P.trait_strategy+"_"+P.strategy+"_"+str(P.value)+"_"+str(P.source_cover[0])+"_"+str(P.V[0][0])+"_"+str(seed)+".npy", restore_status)