# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import inspect
from External.notears.linear import notears_linear


filename = inspect.getframeinfo(inspect.currentframe()).filename
currPath = os.path.dirname(os.path.abspath(filename))
    
def get_sachs():
    
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    currPath = os.path.dirname(os.path.abspath(filename))
    
    df = pd.read_excel(os.path.join(currPath,'1. cd3cd28.xls'))
    
    X_0 = df.to_numpy()
    X = X_0-np.mean(X_0,0)
    
    N, p = X.shape
    S_hat = X.T@X/N
    
    mu_hat = np.zeros(p)
    
    return X, mu_hat, S_hat

def get_sachs_columns():
    
    columns = ['raf', 'mek', 'plc', 
               'pip2', 'pip3', 'erk', 
               'akt', 'pka', 'pkc', 
               'p38', 'jnk']
    
    return columns

def get_hat_sachs(X):
    
    lambda1 = 0.1
    loss_type = 'l2'
#    X = X_0-np.mean(X_0,0)
    W_est = notears_linear(X, lambda1, loss_type, 
                           max_iter=100, 
                           h_tol=1e-8, 
                           rho_max=1e+16, 
                           w_threshold=0.3)
    
    return W_est
