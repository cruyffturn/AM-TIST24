# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LinearRegression

import networkx as nx

from helper_nonlinear import get_X_j

def get_cov(W, sigma_sq, mu_z = None):
    
    p = len(W)
    
    A = np.linalg.inv(np.eye(p)-W.T)
    
    if np.isscalar(sigma_sq):
        S = sigma_sq*A @ A.T
    else:        
        S = A @ np.diag(sigma_sq) @ A.T
    
    if mu_z is None:
        return S
    
    else:
        mu = A.dot(mu_z)
        return S, mu

def get_W_from_dag(X, A, bool_ev = False,
                   return_mean = False):
    
    '''
    Inspired by peters 2014 
    "Identifiability of Gaussian structural equation models with equal error variances"
        
    In:
        X:  N,p
        A:  p,p        
    '''
    
    p = len(A)
    
    W = np.zeros([p,p])
    
    if return_mean:
        mu_z = np.zeros(p)
        
    if not bool_ev:
        sigma_sq = np.zeros([p])
    else:
        sigma_sq = 0
    
    for j in range(X.shape[1]):
        
        idx_pa = np.where(A[:,j])[0]
        if len(idx_pa) != 0:
            model = LinearRegression()
            model.fit(X[:,idx_pa],X[:,j])
            
            W[idx_pa,j] = model.coef_
            
            res = np.mean((model.predict(X[:,idx_pa])-X[:,j])**2)            
            
            if return_mean:
                mu_z[j] = model.intercept_
                
        else:
#            sigma_sq[j] = np.std(X[:,j])**2
            res = np.std(X[:,j])**2
            
            if return_mean:
                mu_z[j] = np.mean(X[:,j])
            
        if not bool_ev:
            sigma_sq[j] = res
        else:
            sigma_sq += res
    
    if not return_mean:    
        return W, sigma_sq
    else:
        return W, sigma_sq, mu_z

def get_pear(S):
    
    pear = S/np.sqrt(np.outer(np.diag(S),np.diag(S)))
    
    return pear

def sample_dag(W, sigma_sq, N = 1000, seed = None):
    
    '''
    Inspired by peters 2014 
    "Identifiability of Gaussian structural equation models with equal error variances"
    
    In:
        X:  N,p
        A:  p,p        
    '''
    
    if seed is not None:
        np.random.seed(seed)
        
    p = len(W)
    A = np.abs(W)>0

    Z = np.random.normal(size=[N,p])
    X = np.full([N,p], np.nan)
#    X = np.nan
    done = np.zeros(p, bool)

    while not np.all(done):
        
#        import ipdb;ipdb.set_trace()
        for j in range(p):
            
            if not done[j]:
                idx_pa = np.where(A[:,j])[0]
                
                if len(idx_pa) == 0 or np.all(done[idx_pa]):
                    
                    if np.isscalar(sigma_sq):
                        noise = Z[:,j]*np.sqrt(sigma_sq)
                    else:
                        noise = Z[:,j]*np.sqrt(sigma_sq[j])
                        
                    if len(idx_pa) != 0:
                        X[:,j] = X[:,idx_pa].dot(W[idx_pa,j]) + noise
                    else:
                        X[:,j] = noise
                    done[j] = True
        
    return X

def sample_dag_fast(W, sigma_sq, N = 1000, seed = None):
    
    '''
    Inspired by peters 2014 
    "Identifiability of Gaussian structural equation models with equal error variances"
    
    In:
        X:  N,p
        A:  p,p        
    '''
    
    if seed is not None:
        np.random.seed(seed)
        
    p = len(W)
    A = np.abs(W)>0

    Z = np.random.normal(size=[N,p])
    X = np.full([N,p], np.nan)

    done = np.zeros(p, bool)

    g = nx.DiGraph(A)
    idx = list(nx.topological_sort(g))
#        import ipdb;ipdb.set_trace()
    
    for j in idx:
            
        if not done[j]:
            idx_pa = np.where(A[:,j])[0]
            
            if len(idx_pa) == 0 or np.all(done[idx_pa]):
                
                if np.isscalar(sigma_sq):
                    noise = Z[:,j]*np.sqrt(sigma_sq)
                else:
                    noise = Z[:,j]*np.sqrt(sigma_sq[j])
                    
                if len(idx_pa) != 0:
                    X[:,j] = X[:,idx_pa].dot(W[idx_pa,j]) + noise
                else:
                    X[:,j] = noise
                done[j] = True
        
    return X

def legacy_sample_dag_fast_ng(W, sigma_sq, 
                       type_noise,
                       type_sem,
                       N = 1000, 
                       seed = None,
#                       type_sem == 'linear',
                       ):
    
    '''
    Inspired by peters 2014 
    "Identifiability of Gaussian structural equation models with equal error variances"
    
    In:
        X:  N,p
        A:  p,p        
    '''
    print('using noise', type_noise)
    if seed is not None:
        np.random.seed(seed)
        
    p = len(W)
    A = np.abs(W)>0

#    Z = np.random.normal(size=[N,p])
    X = np.full([N,p], np.nan)

    done = np.zeros(p, bool)

    g = nx.DiGraph(A)
    idx = list(nx.topological_sort(g))
#        import ipdb;ipdb.set_trace()
    
    for j in idx:
            
        if not done[j]:
            idx_pa = np.where(A[:,j])[0]
            
            if len(idx_pa) == 0 or np.all(done[idx_pa]):
                
                if np.isscalar(sigma_sq):
                    sigma = np.sqrt(sigma_sq)
                else:
                    sigma = np.sqrt(sigma_sq[j])
                
                noise = get_noise(type_noise, sigma, N)
                
                if len(idx_pa) != 0:
                    if type_sem == 'linear':
                        X_j_0 = X[:,idx_pa].dot(W[idx_pa,j])
                    elif type_sem == 'mlp':
                        X_j_0 = get_X_j(X[:,idx_pa])
                        
                    X[:,j] = X_j_0 + noise
                else:
                    X[:,j] = noise
                done[j] = True
        
    return X

def sample_dag_fast_ng(W, sigma_sq, 
                       type_noise,
                       type_scm,
                       N = 1000, 
                       seed = None,
                       w_ub = 2
                       ):
    
    '''
    Inspired by peters 2014 
    "Identifiability of Gaussian structural equation models with equal error variances"
    
    In:
        X:  N,p
        A:  p,p        
    '''
    print('using noise', type_noise)
    if seed is not None:
        np.random.seed(seed)
        
    p = len(W)
    A = np.abs(W)>0

#    Z = np.random.normal(size=[N,p])
    X = np.full([N,p], np.nan)

    done = np.zeros(p, bool)

    g = nx.DiGraph(A)
    idx = list(nx.topological_sort(g))
    
    #Samples the noise
    noise_mat = np.zeros([N,p])
    
    for j in range(p):
        
        if np.isscalar(sigma_sq):
            sigma = np.sqrt(sigma_sq)
        else:
            sigma = np.sqrt(sigma_sq[j])
        
        noise_mat[:,j] = get_noise(type_noise, sigma, N)
    
    if seed is not None:        #Seed for the SCM
        np.random.seed(seed)
        
    #Samples X
    for j in idx:
            
        if not done[j]:
            idx_pa = np.where(A[:,j])[0]
            
            if len(idx_pa) == 0 or np.all(done[idx_pa]):                
                
                noise = noise_mat[:,j]
                
                if len(idx_pa) != 0:
                    if type_scm == 'linear':
                        X_j_0 = X[:,idx_pa].dot(W[idx_pa,j])
                    elif type_scm == 'mlp':
                        X_j_0 = get_X_j(X[:,idx_pa],w_ub)
                        
                    X[:,j] = X_j_0 + noise
                else:
                    X[:,j] = noise
                done[j] = True
        
    return X

def get_noise(type_noise, 
              sigma, N):
    
    if type_noise == 'gumbel':
        scale = np.sqrt(6)/np.pi*sigma
        noise = np.random.gumbel(loc=-np.euler_gamma*scale, 
                                 scale=scale, size=N)
        print('making gumbel zero mean')
        
    elif type_noise == 'normal':
        scale = sigma
        noise = np.random.normal(scale=scale, size=N)
        
    elif type_noise == 'laplace':
        print('scale corrected')
        scale = sigma/np.sqrt(2)
        noise = np.random.laplace(scale=scale, 
                                  size=N)

    return noise
        

def get_cov_nindep(W, S_Z, mu_z = None):
    
    p = len(W)
    
    A = np.linalg.inv(np.eye(p)-W.T)
    
    S = A @ S_Z @ A.T
    
    if mu_z is None:
        return S
    
    else:
        raise ValueError
        mu = A.dot(mu_z)
        return S, mu