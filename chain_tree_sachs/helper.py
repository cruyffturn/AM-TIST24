# -*- coding: utf-8 -*-
import numpy as np
import copy
import os
import tensorflow as tf

import networkx as nx
from sklearn.impute import SimpleImputer

from sklearn.datasets import make_spd_matrix
from scipy.stats import invwishart
from joblib import Parallel, delayed, parallel_backend

from causallearn.search.ConstraintBased.PC import get_adjacancy_matrix, pc
from causallearn.utils.cit import fisherz, mv_fisherz
from External.notears.linear import notears_linear

import helper_em_tf, helper_em
from helper_prob.models import helper_mvn


def get_graph_error_total(K, K_est_all, thres):
    
    E_est_all = np.abs(K_est_all) > thres
    E = np.abs(K) > thres
    
    total = np.sum(E != E_est_all, (1,2))/2
    
    return total

def get_graph_error_sub(K_a, K_true, K_est_all, thres):
    
    E_est_all = np.abs(K_est_all) > thres
    E_a = np.abs(K_a) > thres
    E_true = np.abs(K_true) > thres
    
    E_target = E_true != E_a
    
    E_rem = E_true == E_a
        
    diff = E != E_est_all
    
    np.sum(diff*E_target, (1,2))/2
    np.sum(diff*E_rem, (1,2))/2
    
#    total = np.sum(, (1,2))/2
    
    return total

def get_graph_error(K, K_est_all, thres):
    
    E_est_all = np.abs(K_est_all) > thres
    E = np.abs(K) > thres
    
    n_rem = np.sum(E & (~E_est_all), (1,2))/2
    n_add = np.sum((~E) & E_est_all, (1,2))/2
    
    return n_rem, n_add

#def get_KL_error()
def multiple_em(X, p2, n_rep,
                init_mode,
                bool_full, S, mu,
                n_steps = 10,
                idx_adv = None,
                bool_sparse = False,
                alpha = None,
                bool_while = False,
                eps = .001,
                seed = 42,
                bool_history = False,
                bool_dag = False,
                bool_ev = False,
                verbose = True,
                **kwargs_dag):
    
    '''
    Calls missDAG with n_rep different missing data masks.
    In:
        p2:   N,p
    '''
    if idx_adv is None:
        raise TypeError
        
    np.random.seed(seed)
    tf.random.set_seed(seed)
    cat = tf.random.categorical(np.log(p2), n_rep).numpy()#[:,0]
    
    if not bool_history:
        p = X.shape[1]
        mu_est_all = np.zeros((n_rep,p))
        S_est_all = np.zeros((n_rep,p,p))
        K_est_all = np.zeros((n_rep,p,p))
        lb_all = np.zeros(n_rep)
        
        for i in range(n_rep):
            
            print('i/n_rep',i,n_rep)
            load = _multiple_em(X, cat[:,i], init_mode, 
                                bool_full, S, mu, 
                                n_steps,
                                idx_adv,
                                bool_sparse,
                                alpha,
                                bool_while,
                                eps,
                                bool_dag=bool_dag,
                                bool_ev = bool_ev,
                                verbose = verbose,
                                **kwargs_dag)
            
            if not bool_sparse:
                if not bool_dag:
                    mu_est_all[i], S_est_all[i] = load[0]
                    K_est_all[i] = np.linalg.inv(S_est_all[i])
                else:
                    W, sigma_sq = load[0][:2]
                    A = np.linalg.inv(np.eye(len(W))-W.T)
                    S = sigma_sq*A@A.T
                    mu = np.zeros(len(W))
                    
                    mu_est_all[i] = mu
                    S_est_all[i] = S
            
                    K_est_all[i] = W
                
            else:
                mu_est_all[i], S_est_all[i], K_est_all[i] = load[0]
        
            lb_all[i] = load[1]
            
        return mu_est_all, S_est_all, K_est_all, lb_all
    
    else:
        
        like_all_l = []
        for i in range(n_rep):
            
            like_L, mu_L, S_L = _multiple_em(X, cat[:,i], init_mode, 
                                             bool_full, S, mu, 
                                             n_steps,
                                             idx_adv,
                                             bool_sparse,
                                             alpha,
                                             bool_while,
                                             eps,
                                             bool_history = bool_history,
                                             **kwargs_dag)
            like_all_l.append(np.array(like_L))
        
        return like_all_l

def _multiple_em(X, cat, init_mode, 
                 bool_full, S, mu, 
                 n_steps = 10,
                 idx_mask = None,
                 bool_sparse = False,
                 alpha = None,
                 bool_while = False,
                 eps = .001,
                 bool_history = False,
                 bool_dag = False,
                 bool_ev = False,
                 verbose = True,
                 seed_spd = None,
                 **kwargs_dag):
    
    '''
    Initializes the $\theta$ and calls the missDAG
    
    In:
        X:          N,p
        cat:        N,p or N,p_sub
        idx_mask:    p_sub,          #indices
    '''
    if seed_spd is None:
        kwargs_spd = {}
    else:
        print('using seed_spd')
        kwargs_spd = {'random_state':seed_spd}
        
    p = X.shape[1]
    
    if not bool_full:
#            mask = cat.reshape(x.shape).astype(bool)#[:,0]
#            mask = ~mask
        mask = helper_em_tf.get_mask_nfull(cat, X.shape[1], 
                                           idx_mask)
    else:
        mask = helper_em_tf.get_mask_full(cat, X.shape[1],
                                          idx_mask)
    
    print('% missing ' +'%.2f'%(np.mean(mask)*100))
    print('% missing target' +'%.2f'%(np.mean(mask[:,idx_mask])*100))
    print('% missing per column', mask[:,idx_mask].mean(0)*100)
#    import ipdb;ipdb.set_trace()
#    if not bool_full:
#        mask_0 = cat.reshape(X.shape).astype(bool)#[:,0]
#        mask_0 = ~mask_0
#    else:
#        mask_0 = helper_em_tf.get_mask_full(cat, X.shape[1])
    
#    if idx_mask is None:
#        mask = mask_0
#    else:
#        mask = np.zeros_like(X, bool)
#        mask[:,idx_mask] = mask_0
    
    X_miss = copy.deepcopy(X)
    X_miss[mask] = np.nan        
    
    if init_mode == 0:
        S_0 = S#np.eye(p)
        mu_0 = mu
    elif init_mode == 1:
        print('I init')
        S_0 = np.eye(p)
        mu_0 = np.zeros(p)
    elif init_mode == 2:
        print('emprical')
        idx = np.all(~np.isnan(X_miss),1)
                
#        linalg.cholesky(a)
        X_full = X_miss[idx]
        mu_0 = np.mean(X_full,0)
        S_0 = X_full.T@X_full/len(X_full) - np.outer(mu_0,mu_0)
        
    elif init_mode == 3:
        #random psd sklearn
        mu_0 = np.nanmean(X_miss,0)
        
        std = np.nanstd(X_miss,0)        
        S_0_0 = make_spd_matrix(len(mu_0),**kwargs_spd)
#        diag = std/np.sqrt(np.diag(S_0_0))
#        D = np.diagflat(diag)
#        S_0 = D @ S_0_0 @ D
        S_0 = get_scaled(std, S_0_0)
        print(np.round(S_0_0[:3,:3],2))
#        import ipdb;ipdb.set_trace()
#        S_0 = X_full.T@X_full/len(X_full) - np.outer(mu_0,mu_0)
    
    elif init_mode == 4:
        mu_0 = np.nanmean(X_miss,0)
        std = np.nanstd(X_miss,0)
        S_0 = np.diagflat(std**2)
        
    elif init_mode == 5:
        #random psd sklearn
        mu_0 = np.nanmean(X_miss,0)
        
        S_0 = make_spd_matrix(len(mu_0),**kwargs_spd)
    
    elif init_mode == 6:
#        p = 3
        mu_0 = np.nanmean(X_miss,0)
        std = np.nanstd(X_miss,0)
        
        scale = make_spd_matrix(p,**kwargs_spd)
        S_0_0 = invwishart(df=p, scale=scale).rvs(**kwargs_spd)
        S_0 = get_scaled(std, S_0_0)
    
    if bool_dag:
       mu_0 = np.zeros_like(mu_0)
       
    if not bool_sparse:
        load = helper_em.get_em(X_miss, mu_0, S_0, 
                                n_steps = n_steps,
                                bool_sparse = bool_sparse,
                                alpha = alpha,
                                bool_while = bool_while,
                                eps = eps,
                                bool_history = bool_history,
                                bool_dag = bool_dag,
                                bool_ev = bool_ev,
                                verbose = verbose,
                                **kwargs_dag)
    else:
        load = helper_em.get_em_tuned(X_miss, mu_0, S_0, 
                                      n_alphas = 10,
                                      n_steps = n_steps,
#                                verbose = True,
                                   bool_while = bool_while,
                                   eps = eps)
        
    if not bool_history:
        idx_miss = np.isnan(X_miss)
        unq, inverse = np.unique(idx_miss, axis=0, 
                                 return_inverse=True)
        
        if not bool_dag:
            mu, S = load[:2]
        else:
            W, sigma_sq = load[:2]
            A = np.linalg.inv(np.eye(len(W))-W.T)
            S = sigma_sq*A@A.T
            mu = np.zeros(len(W))
        
        lb = helper_em.get_lower_bound(X_miss, unq, inverse, mu, S)
        
        return load, lb
    else:
        return load

def get_scaled(std, S_0):
        
    diag = std/np.sqrt(np.diag(S_0))
    D = np.diagflat(diag)
    S = D @ S_0 @ D
    
    return S
        
def get_stats( 
              mu_est_all, S_est_all,
              mu_a, S_a, 
              mu, S,
              bool_hat_first = True):
    
#    idx_miss = np.isnan(X)
#    unq, inverse = np.unique(idx_miss, axis=0, 
#                             return_inverse=True)
#    
    load_L = []
    
    for i in range(len(mu_est_all)):
        load_L.append(_get_stats( 
                                 mu_est_all[i], S_est_all[i], 
                                 mu_a, S_a, 
                                 mu, S,
                                 bool_hat_first=bool_hat_first))
    
    stats = np.stack(load_L,0)
    
    return stats
    
def _get_stats(
               mu_est, S_est, 
               mu_a, S_a, 
               mu, S,
               bool_hat_first = True):
    
    
#    lb = helper_em.get_lower_bound(X, unq, inverse, 
#                                   mu_est, S_est)
    
    if bool_hat_first:
        KL_p = helper_mvn.get_KL(mu_est, S_est, mu, S)
        KL_a = helper_mvn.get_KL(mu_est, S_est, mu_a, S_a)
    else:
        print('KL: hat second')
        KL_p = helper_mvn.get_KL(mu, S, mu_est, S_est)
        KL_a = helper_mvn.get_KL(mu_a, S_a, mu_est, S_est)
        
    
    return KL_p, KL_a

#    KL_2 = helper_mvn.get_KL(mu, S, mu_est, S_est)

def multiple_est(X, p2, n_rep,
                bool_full, 
#                S, mu,
                bool_impute,
                mode,
                n_steps = 10,
                idx_mask = None,
                seed = 42,
                n_jobs = 1,
                **kwargs_dag):
    
    '''
    In:
        p2:   N,p
    '''
    if idx_mask is None:
        raise TypeError
        
    np.random.seed(seed)
    tf.random.set_seed(seed)
    cat = tf.random.categorical(np.log(p2), n_rep).numpy()#[:,0]
    
    p = X.shape[1]
    
    A_est_all = np.zeros((n_rep,p,p))
    
    if n_jobs == 1:
        for i in range(n_rep):
            
            A_est_all[i] = _multiple_est(X, cat[:,i], 
                                     bool_full,
                                     bool_impute,
                                     mode,
                                     idx_mask,
                                     **kwargs_dag
                                     )
    else:
        
        
        temp_folder = None    
        with parallel_backend('multiprocessing',
        	                      n_jobs = n_jobs):            
            trainable = delayed(_multiple_est)
            
            load_L = Parallel(verbose=11,
                              max_nbytes=None,
                              temp_folder=temp_folder)(trainable(X, cat[:,i], 
                                                         bool_full,
                                                         bool_impute,
                                                         mode,
                                                         idx_mask,
                                                         **kwargs_dag)
        	                    for i in range(n_rep))

        count_val = 0
        for i, load in enumerate(load_L):
            if load is not None:
                A_est_all[i] = load
                count_val += 1
#            else:

#                A_est_all[i] = np.nan

        if count_val != len(A_est_all):
#            import ipdb;ipdb.set_trace()
            print('# errors',len(A_est_all)-count_val)
                  
            A_est_all = A_est_all[:count_val]

        
    return A_est_all

def _multiple_est(X, cat,
                  bool_full,
                  bool_impute,
                  mode,
                  idx_mask = None,
                  pc_alpha = 0.01,
                  lambda1 = 0.1
                  ):
    
    '''
    In:
        X:          N,p
        cat:        N,p or N,p_sub
        idx_mask:    p_sub,          #indices
    '''
    
#    p = X.shape[1]
    
    if not bool_full:
        mask = helper_em_tf.get_mask_nfull(cat, X.shape[1], 
                                           idx_mask)
    else:
        mask = helper_em_tf.get_mask_full(cat, X.shape[1],
                                          idx_mask)
    
    print('% missing ' +'%.2f'%(np.mean(mask)*100))
    print('% missing target' +'%.2f'%(np.mean(mask[:,idx_mask])*100))
    print('% missing per column', mask[:,idx_mask].mean(0)*100)

    
    X_miss = copy.deepcopy(X)
    X_miss[mask] = np.nan        
    
    try:
        if not bool_impute:
            if mode == 'pc':
                
                #Test wise??
                load = pc(X_miss, pc_alpha, 
                          mv_fisherz, True, 0,
                          -1, True)
                
                A = get_adj(load)
        else:
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            imp.fit(X_miss)
            X_hat = imp.transform(X_miss)
            
            if mode == 'pc':
                load = pc(X_hat, pc_alpha, 
                           fisherz, True, 0,
                           -1)
                A = get_adj(load)
            elif mode == 'nt':
                
    #            lambda1 = 0.0
                loss_type = 'l2'
                A = notears_linear(X_hat, lambda1, loss_type, 
                                   max_iter=100, 
                                   h_tol=1e-8, 
                                   rho_max=1e+16, 
                                   w_threshold=0.3)
    except Exception as e:
        print(str(e))
        A = None
        
    return A


def get_adj(load):
        
    load.to_nx_graph()
    A = nx.to_numpy_array(load.nx_graph)
    
    return A
    
def get_E(A):
    
    '''
    In:
        A:  p,p
        
    Out:
        edges: p*(p-1)/2,       #0: no edge,
                                1:  directed,
                                2:  directed,
                                3:  undir.
    '''
    
    idx_triu = np.triu_indices(len(A), k=1)
    temp = np.stack([A[idx_triu[0],idx_triu[1]],
                     A[idx_triu[1],idx_triu[0]]],1)
    
    E = np.zeros(len(idx_triu[0]))
    
    bool_dir = (temp[:,0] != temp[:,1]) & np.any(temp != 0,1)
    bool_und = (temp[:,0] == temp[:,1]) & np.any(temp != 0,1)
    
    E[bool_dir] = np.argmax(temp[bool_dir],1)+1     #1, or 2
    E[bool_und] = 3

    return E

def _get_pdag_dist(E, E_hat, allow_pdag = False, 
                   return_err = False,
                   legacy_pdag = False
                   ):
    
    '''
    Assumes E is dag but E_hat possibly pdag
    '''
    
    if not allow_pdag:
        #The existence and (if exists) direction has to match.
        err = (E != E_hat)
        
    else:            
        err = np.zeros(len(E), bool)
        
        if legacy_pdag:
            #For estimated directed edges,
            #   >The existence and (if exists) direction has to match.
            #For estimated undirected edges,
            #   >Only the existence has to match.   
        
            bool_und = E_hat == 3
            bool_dir = ~bool_und
            
            err[bool_und] = E[bool_und]==0
            err[bool_dir] = E[bool_dir] != E_hat[bool_dir]
        else:
            print('using new pdag hd')
            #When either graph contains an undirected edge,
            #   >Only the existence has to match.   
            #Remaining cases
            #   >The existence and (if exists) direction has to match.
            
            bool_und = (E_hat == 3) | (E == 3)
            bool_dir = ~bool_und
            
            exists = E != 0
            exists_hat = E_hat != 0
            
#            import ipdb;ipdb.set_trace()
            err[bool_und] = exists[bool_und] != exists_hat[bool_und]
            err[bool_dir] = E[bool_dir] != E_hat[bool_dir]
            
        
    n_err = np.sum(err)
    if not return_err:
        return n_err
    else:
        return n_err, err.astype(int)

def get_pdag_dist_single(A, A_est, 
                         allow_pdag = False, return_err = False):
    
    E = get_E(A)
    E_est = get_E(A_est)
    
    return _get_pdag_dist(E, E_est, 
                          allow_pdag = allow_pdag, 
                          return_err = return_err)
    
def get_pdag_dist(A, A_est_all, allow_pdag = False):
    
    '''
    "Entry i,j corresponds to an edge from i to j." networkx
    
    In:
        A:          p,p
        A_est_all:  L,p,p
    '''
    
    E = get_E(A)
    E_est_all = np.stack([get_E(A_i) for A_i in A_est_all],0)
    
    temp = []
    for E_i in E_est_all:
        temp.append(_get_pdag_dist(E, E_i, allow_pdag = allow_pdag))
        
    n_err = np.stack(temp,0)
    
    return n_err

def get_adv_err(A_p, A_a, A_est_all, 
                allow_pdag = False,
                return_err = False):
    
    '''
    for visual test
    np.stack([A_est_all[:,idx_r,idx_c],A_est_all[:,idx_c,idx_r]],1)
    
    Compares the estimated graph and adv. graph
    
    Inter:
        temp:       K,2
        E_a:        K,
        E_L:        M;K,
        E_est_all:  M,K
    
    Out:
        err_rate:   M,
        err:        M,K
    '''
    idx_r, idx_c = np.where(A_a!=A_p)
    
#    temp_0 = A_est_all[:,idx_r,idx_c]    
#    temp_1 = A_est_all[:,idx_c,idx_r]
    
    temp = np.stack([A_a[idx_r,idx_c],
                     A_a[idx_c,idx_r]],1)
    
    E_a = _get_E(temp)
    
#    import ipdb;ipdb.set_trace()
    E_L = []
    for A in A_est_all:
        temp = np.stack([A[idx_r,idx_c],
                         A[idx_c,idx_r]],1)
    
        E_L.append(_get_E(temp))
    
    E_est_all = np.stack(E_L,0)
    
    temp = []
    for E_i in E_est_all:
        temp.append(_get_pdag_dist(E_a, E_i, allow_pdag = allow_pdag,
                                   return_err = return_err))
        
    if not return_err:
        n_err = np.stack(temp,0)
        err_rate = n_err/len(E_a)
        
        return err_rate
    else:
        err = np.stack([temp_i[1] for temp_i in temp],0)        
        
        idx_edges = np.stack([idx_r, idx_c],1)
        return err, idx_edges
#    import ipdb;ipdb.set_trace()
    
    
    
def _get_E(temp):
    
    '''
    gets the edge type
    In:
        temp:       K,2
    Inter:
        bool_dir:   K,
        bool_und:   K,
    Out:
        E:          K,  #E[i] denotes the edge type of temp[i]
                        #0: no edge, 1 or 2: directed 3: und
    '''
    
    E = np.zeros(len(temp))
    
    bool_dir = (temp[:,0] != temp[:,1]) & np.any(temp != 0,1)
    bool_und = (temp[:,0] == temp[:,1]) & np.any(temp != 0,1)
    
    E[bool_dir] = np.argmax(temp[bool_dir],1)+1     #1, or 2
    E[bool_und] = 3
    
    return E

#%%
def multiple_em_par(X, p2, n_rep,
                init_mode,
                bool_full, S, mu,
                n_steps = 10,
                idx_mask = None,
                bool_sparse = False,
                alpha = None,
                bool_while = False,
                eps = .001,
                seed = 42,
                bool_history = False,
                bool_dag = False,
                bool_ev = False,
                verbose = True,
                n_jobs = 1,
                legacy_random = True,
                **kwargs_dag):
    
    '''
    Calls missDAG with n_rep different missing data masks.
    In:
        p2:   N,p
        legacy_random:      #If False, parallel runs becomes random
    '''
    if idx_mask is None:
        raise TypeError
        
#    import ipdb;ipdb.set_trace()
    np.random.seed(seed)
    tf.random.set_seed(seed)
    cat = tf.random.categorical(np.log(p2), n_rep).numpy()#[:,0]
    
    if legacy_random:
        seed_l = [None]*n_rep
    else:
        print('using corrected randomness')
        seed_l = np.random.randint(np.iinfo(np.int32).max, size=n_rep)
        
    if not bool_history:
        p = X.shape[1]
        mu_est_all = np.zeros((n_rep,p))
        S_est_all = np.zeros((n_rep,p,p))
        K_est_all = np.zeros((n_rep,p,p))
        lb_all = np.zeros(n_rep)
        
        if n_jobs == 1:
            
            load_l = []
            for i in range(n_rep):
                
                print('i/n_rep',i,n_rep)
                load = _multiple_em(X, cat[:,i], init_mode, 
                                    bool_full, S, mu, 
                                    n_steps,
                                    idx_mask,
                                    bool_sparse,
                                    alpha,
                                    bool_while,
                                    eps,
                                    bool_dag=bool_dag,
                                    bool_ev = bool_ev,
                                    verbose = verbose,
                                    seed_spd = seed_l[i],
                                    **kwargs_dag)
                load_l.append(load)
        else:
            
            temp_folder = None    
            with parallel_backend('multiprocessing',
                                  n_jobs = n_jobs):            
                trainable = delayed(_multiple_em)
                
                load_l = Parallel(verbose=11,
                                  max_nbytes=None,
                                  temp_folder=temp_folder)(trainable(X, cat[:,i], init_mode, 
                                                                bool_full, S, mu, 
                                                                n_steps,
                                                                idx_mask,
                                                                bool_sparse,
                                                                alpha,
                                                                bool_while,
                                                                eps,
                                                                bool_dag=bool_dag,
                                                                bool_ev = bool_ev,
                                                                verbose = verbose,
                                                                seed_spd = seed_l[i],
                                                                **kwargs_dag)
            	                    for i in range(n_rep))
                              
        for i, load in enumerate(load_l):
            if not bool_sparse:
                if not bool_dag:
                    mu_est_all[i], S_est_all[i] = load[0]
                    K_est_all[i] = np.linalg.inv(S_est_all[i])
                else:
                    W, sigma_sq = load[0][:2]
                    A = np.linalg.inv(np.eye(len(W))-W.T)
                    S = sigma_sq*A@A.T
                    mu = np.zeros(len(W))
                    
                    mu_est_all[i] = mu
                    S_est_all[i] = S
            
                    K_est_all[i] = W
                
            else:
                mu_est_all[i], S_est_all[i], K_est_all[i] = load[0]
        
            lb_all[i] = load[1]
            
        return mu_est_all, S_est_all, K_est_all, lb_all
    
    else:
        
        like_all_l = []
        for i in range(n_rep):
            
            like_L, mu_L, S_L = _multiple_em(X, cat[:,i], init_mode, 
                                             bool_full, S, mu, 
                                             n_steps,
                                             idx_mask,
                                             bool_sparse,
                                             alpha,
                                             bool_while,
                                             eps,
                                             bool_history = bool_history,
                                             **kwargs_dag)
            like_all_l.append(np.array(like_L))
        
        return like_all_l