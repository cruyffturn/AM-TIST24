# -*- coding: utf-8 -*-
import os
import numpy as np
from joblib import Parallel, delayed, parallel_backend
import helper_sim

import helper_local

N_jobs = helper_local.n_jobs
#%%                     
def main_sim_par(bool_repeat = 0,
                 ):
    
    savePath = helper_local.get_savePath(bool_repeat)
        
    print(savePath)
    temp_folder = None
    
    with parallel_backend('multiprocessing',
                          n_jobs = N_jobs):
                            
        trainable = delayed(helper_sim.main)
        Parallel(verbose=11,
                 temp_folder=temp_folder)(trainable(savePath,
                                                    kwargs_sim,
                                                    kwargs_model)
                          for kwargs_sim, kwargs_model in helper_sim.get_cfg())