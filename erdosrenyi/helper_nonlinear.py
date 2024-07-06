# -*- coding: utf-8 -*-
import numpy as np

from External.MissDAG.data_loader.simulate_single_equation import simulate_single_equation
    
def get_X_j(X_pa, w_ub):
        
    equal_variances = 1
    n = len(X_pa)
#    X = np.random.rand(n,2)-0.5
    scale = None
    sem_type= 'mlp'
    
    
    X_j_0 = simulate_single_equation(X_pa, 
                                     scale, 
                                     equal_variances,
                                     n,
                                     sem_type,
                                     w_ub = w_ub
                                     )
#    X_j = X_j_0 - np.mean(X_j_0)
    X_j = X_j_0 
    return X_j

