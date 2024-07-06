# -*- coding: utf-8 -*-
import os
import inspect
#import matplotlib.pyplot as plt

import shlex
import subprocess
import numpy as np


filename = inspect.getframeinfo(inspect.currentframe()).filename
currPath = os.path.dirname(os.path.abspath(filename))

from helper_local import get_sharedPath, get_condaPath


def estimate_baseline_wrapped(X, seed_baseline, 
                              w_threshold,
                              bool_linear):
    
    str_arg = '%i %f %i'%(seed_baseline, w_threshold, bool_linear)
    script_name = 'call_baseline'
    
    load = get_script_wrapped(X,
                              str_arg,
                              script_name,
                              ['W','A'],
                              str_env = ['_mlp',''][bool_linear]
                              )
    
    W, A = load#['W'],load['A']
    
    return W, A

def estimate_missdag_wrapped(X_miss, 
                             seed, 
                             bool_linear,
                             w_threshold=0.1):
    
#    str_arg = '%i %f'%(seed, w_threshold)
    str_arg = '%i %f %i'%(seed, w_threshold, bool_linear)
    script_name = 'call_missdag'
    
    load = get_script_wrapped(X_miss,
                              str_arg,
                              script_name,
                              ['W','cov_est'],
                              str_env = ['_mlp',''][bool_linear]
                              )
    
    W, cov_est = load#['W'],load['cov_est']
    
    return W, cov_est

def get_script_wrapped(X,
                       str_arg,
                       script_name,
                       key_l,
                       str_env = ''):
    '''
    
    '''
    sharedPath = get_sharedPath()
    miniconda = get_condaPath()
    
    script_path = os.path.join(currPath,script_name+'.py')
    savePath = os.path.join(sharedPath,'temp_%s_%s'%(script_name,str(os.getpid())))
    
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    
    file_x = os.path.join(savePath,'X.npy')
    if os.path.exists(file_x):
        os.remove(file_x)
        
    with open(file_x, 'wb') as f:
        np.save(f, X)    
    
    file_z = os.path.join(savePath,'temp.npyz')
    if os.path.exists(file_z):
        os.remove(file_z)

    command = "%s %s %s %s"%(miniconda,
                             script_path,
                             savePath,
                             str_arg)
                                                     
    print(command)
#    import ipdb;ipdb.set_trace()
    args = shlex.split(command)
    my_subprocess = subprocess.Popen(args)
    my_subprocess.wait()
#    os.system(command)
    
    
    with open(file_z, 'rb') as f:
        load_0 = np.load(f)
        load = [load_0[key] for key in key_l]
    
    os.remove(file_z)
    os.remove(file_x)
    
    return load