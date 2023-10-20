""" Generate data from the coupled tank mathematical model for neural network model 
training. Save data in files. 

(c) Martin Doff-Sotta, University of Oxford (martin.doff-sotta@eng.ox.ac.uk)

"""

import numpy as np
from scipy.linalg import block_diag
import matplotlib
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter
import cvxpy as cp
import mosek
import time
import os
import param_init as param
from tank_model import linearise, f, f1, f2, terminal
from control_custom import eul, dlqr, dp  


# Initialisation
N_sample = 50000                               # number of samples to generate
N = 50                                         # horizon 
T = 70                                         # terminal time
delta = T/N                                    # time step
N_state = param.x_init.size                    # number of states
N_input = param.u_init.size                    # number of inputs
x_0 =  param.x_init                            # initial condition
u = np.zeros((N_input, N))                     # control input                     
X = np.zeros((N_sample, N, N_input))           # store inputs 
Y = np.zeros((N_sample, N+1, N_state))         # store outputs
k = 0

# Generate samples
print("Generating {} samples".format(N_sample))
while k < N_sample:
    
    # Random initial condition
    x_0 = np.random.rand(x_0.shape[0])*(param.x_max[0]-param.x_min[0]) + param.x_min[0]
    
    # Random control input sequence
    u = np.random.rand(*u.shape)*param.u_max
        
    # Compute trajectory
    x = eul(f, u, x_0, delta, param) 
    
    # Check if trajectory is valid (i.e. no nan or infinite states)
    if np.isnan(x).any() or np.isinf(x).any():
        print("Invalid value encountered. Result was ignored")
          
    else: 
        # Store result
        X[k, :, :] = u.T
        Y[k, :, :] = x.T
        k += 1
    
        # Redefine initial condition
        x_0 = np.random.rand(x_0.shape[0])*(param.x_max[0]-param.x_min[0]) + param.x_min[0]
    
        # Redefine control input
        u = np.random.rand(*u.shape)*param.u_max

# Save training data in files (input / output)
np.save('input.npy', X)
np.save('output.npy', Y)