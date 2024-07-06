# -*- coding: utf-8 -*-
import os
import copy
import numpy as np
import networkx as nx

import pickle
import tensorflow as tf
from tensorflow import keras
import json
import scipy.stats

import helper_dag
from helper_prob.models import helper_mvn
import helper_tf_model
import helper_data

#%% Simulation parameters
import cfgs

def get_cfg():
    
    debug = 0
    
    if debug:
        print('debug debug debug debug')    
    
#    type_graph = 'tree'
    type_graph = 'sachs'
#    type_graph = 'chain'
    
    if type_graph in ['chain','tree','erdos']:
        
        return _get_cfg_sim(debug, type_graph)
    
    elif type_graph == 'sachs':
        
        return _get_cfg_sachs(debug)
        
def _get_cfg_sim(debug, type_graph):    
    
    if debug:
        print('debug debug debug debug')
    seed_sim = 42
    
#    N = 300
    N = 1000
    p = 15
    
    bool_ev = True
    sigma = 1    
    
    module = getattr(cfgs,type_graph)
    
    (seed_sim_l, i_l,
     degree, w_lb, w_ub, 
     bool_flip_sign, 
     reg_lmbda, lr,
     epochs,
     seed_model_l,
     type_masking) = module.get_cfg()
    
            
    if debug:
#        epochs = 300
        epochs = 1
#        seed_sim_l = range(1)
#        i_l = range(1)
        seed_sim_l = [1]
#        i_l = [0]
        i_l = [2]
#        i_l = range(1,3)
    
    if type_graph == 'erdos':
        gen = ((i, seed_sim) for i in i_l for seed_sim in seed_sim_l)
    else:
        #Iterates over target nodes first
        gen = ((i, seed_sim) for seed_sim in seed_sim_l for i in i_l)
    
    bool_model_first = 1
    
    if bool_model_first:        
        gen = ((i, seed_sim,seed_model) for i, seed_sim in gen 
                                           for seed_model in seed_model_l)
        
    else:
        gen = ((i, seed_sim,seed_model) for seed_model in seed_model_l
                                           for i, seed_sim in gen)
        
            
    for i, seed_sim,seed_model in gen:

        if (type_masking == 'auto') and \
            (type_graph == 'tree'):
            
            if i == 0:
                type_masking_in = 'skip'
            elif i == 2:
                type_masking_in = 'des'
            else:
                type_masking_in = None
                
        else:
            type_masking_in = type_masking
            
        kwargs_sim = dict(seed_sim=seed_sim, 
                          degree=degree,
                          N=N, 
                          p=p,
                          bool_ev=bool_ev, 
                          sigma=sigma,
                          w_lb=w_lb, 
                          w_ub=w_ub,
                          i=i,
                          type_graph=type_graph,
                          bool_flip_sign=bool_flip_sign,
                          type_masking=type_masking_in
                          )
        if kwargs_sim['type_graph'] == 'erdos':
            kwargs_sim.pop('type_graph')                                    
            kwargs_sim.pop('bool_flip_sign')        
                
            
        kwargs_model = dict(reg_lmbda=reg_lmbda, 
                            lr=lr, 
                            epochs=epochs,
                            seed_model=seed_model)
        
        yield (kwargs_sim, kwargs_model)
        
def _get_cfg_sachs(debug):
    
    if debug:
        print('no debug sachs')

    lr = 0.5e-2
    epochs = 300
    seed_model = 42
    
#    i = 0
    i = 1
        
    kwargs_sim = dict(type_graph='sachs',
                      i=i                      
                      )
    if i == 1:
        kwargs_sim['type_masking'] = 'des'
        reg_lmbda = 1e-2
    else:
        reg_lmbda = 0
    kwargs_model = dict(reg_lmbda=reg_lmbda, 
                            lr=lr, 
                            epochs=epochs,
                            seed_model=seed_model)
    
    yield (kwargs_sim, kwargs_model)
    
def get_A(p, degree, seed):
    
    '''
    degree is not actually degree
    
    https://networkx.org/documentation/stable/reference/generated/networkx.linalg.graphmatrix.adjacency_matrix.html
    "For directed graphs, entry i,j corresponds to an edge from i to j."
    
    Selecting the upper triangular
    https://search.r-project.org/CRAN/refmans/gmat/html/rgraph.html
    '''
    np.random.seed(seed)
    
    param = 2*degree/(p-1)
    

    G = nx.erdos_renyi_graph(p, param, 
                             seed=seed,
                             directed=True
                             )    
            
    A = nx.to_numpy_array(G)
    A = np.triu(A,1)
    
    return A

def get_A_chain(p):

    A = np.zeros((p,p))
    
    A[np.arange(p-1),np.arange(1,p)] = 1
    
    return A
                                                 
def get_A_tree():
    
    r = 2
    h = 3
    
    g_0 = nx.balanced_tree(r, h, 
                           create_using=None)

    A = nx.to_numpy_array(g_0)
    A = np.triu(A)
    
    return A



def sample(A, bool_ev, 
           sigma, N, 
           w_lb, w_ub, 
           seed,
           bool_flip_sign = True):
    
    np.random.seed(seed)
    
    p = len(A)
    
    W = get_W(A, w_lb, w_ub, 
              bool_flip_sign = bool_flip_sign)
    sigma_sq = get_sigma_sq(p, sigma, bool_ev)    

#    import ipdb;ipdb.set_trace()
    X = helper_dag.sample_dag_fast(W, sigma_sq, N)
    S = helper_dag.get_cov(W, sigma_sq)
    
    mu = np.zeros(len(S))
    
    return  X, mu, S, sigma_sq, W
    
    
    
def get_W(A, lb, ub, seed = None, 
          bool_flip_sign = True):
    
    if seed is not None:
        np.random.seed(seed)
        
    W = np.zeros_like(A, float)
    
    rows, cols = np.where(A)
    
    for idx_r, idx_c in zip(rows, cols):
        
        amp = np.random.uniform(lb, ub)
        
        if bool_flip_sign:
            if np.random.binomial(1, 0.5):
                sign = 1
            else:
                sign = -1
        else:
            sign = 1
        
        W[idx_r,idx_c] = sign*amp
    
    return W
    
def get_sigma_sq(p, sigma_in, bool_ev, seed = None):
    
    if seed is not None:
        np.random.seed(seed)
        
    if bool_ev:
        sigma_sq = sigma_in**2
    else:
        sigma = np.random.uniform(sigma_in, sigma_in+1, p)
        sigma_sq = sigma**2
    
    return sigma_sq
        
def train(X, 
          idx_adv_train,          
          mu, S,
          mu_a, S_a,
          reg_lmbda, 
          lr, epochs,
          seed_model,
          idx_mask = None):
    
    
    print('tf seed', seed_model)
    tf.random.set_seed(seed_model)

    model_cfg = 4
    loss_type = 9
    bool_full = 1
    bool_ratio = 0
    n_steps = 40
    bool_sub = True
    bool_draw = 0
    bool_l1 = 0
    
    N, p = X.shape
        
    model = helper_tf_model.get_model(p, model_cfg, 
                                      loss_type,
                                      idx_adv_train,
                                      bool_full,
                                      mu = mu, S = S,
                                      mu_a = mu_a, S_a = S_a,
                                      bool_ratio = bool_ratio,
                                      X = X,
                                      bool_sub = bool_sub,
                                      idx_mask = idx_mask)
    
    model._set_param(S, mu,S_a, mu_a, n_steps, loss_type, bool_full,
                     bool_draw = bool_draw,
                     idx_adv = idx_adv_train,
                     reg_lmbda = reg_lmbda,
                     bool_l1 = bool_l1,
                     bool_sub = bool_sub,
                     idx_mask = idx_mask)                
    
    model.compile(
                  optimizer=keras.optimizers.Adam(lr),
                  run_eagerly = 1,
                  )
            

    min_delta = 1e-4
    callbacks = [keras.callbacks.EarlyStopping(monitor='avg_loss', 
                                               min_delta = min_delta,
#                                               patience=10,
                                               patience=20,
                                               verbose=1,
                                               restore_best_weights=True)]
        
    history = model.fit(X, 
                        epochs=epochs,
                        batch_size=N,
                        callbacks=callbacks
                        )
    
    return model, history

def get_adv_par(idx_s, idx_t, 
                idx_copa, 
                mu, S,
                W, sigma_sq):
    
    '''
    Calculates the minimum distance \theta_adversarial 
    '''
#    import ipdb;ipdb.set_trace()
    p = len(S)
    
    W_a = copy.deepcopy(W)
    W_a[idx_s, idx_t] = 0
    
    if len(idx_copa)>0: #If there are co-parents, new coefficients are calc.
        w_new, sigma_sq_new  = _get_adv_par(idx_t, idx_copa, 
                                            mu, S)
        
        W_a[idx_copa, idx_t] = w_new
        
    else:   #If no co-parents, node's variance is the MLE
        sigma_sq_new = S[idx_t,idx_t]
    
    if np.isscalar(sigma_sq):
        sigma_sq_a = np.full(p, sigma_sq)   #It's no longer scaler because
                                            #variance changed
    else:
        sigma_sq_a = copy.deepcopy(sigma_sq)
        
    sigma_sq_a[idx_t] = sigma_sq_new
    
    return W_a, sigma_sq_a

def _get_adv_par(idx_t, idx_copa, 
                 mu, S):
    
    w_new, sigma_sq_new = helper_mvn.get_cond_prob(1., np.array([idx_t]), 
                                                   idx_copa, 
                                                   mu, S,
                                                   return_a1 = True)
    
    w_new = np.squeeze(w_new)
    sigma_sq_new = np.squeeze(sigma_sq_new)

    return w_new, sigma_sq_new 

def get_adv(idx_s, idx_t, 
            mu, S, 
            W, sigma_sq):
    
    '''
    source,
    target
    '''
    A = np.abs(W)>0
        
    #Identifies all co-parents
    idx_copa = np.where(A[:,idx_t])[0]
    idx_copa = idx_copa[idx_copa!=idx_s]        
    
    #Gets the adv. \theta
    W_a, sigma_sq_a = get_adv_par(idx_s, idx_t, 
                                  idx_copa, 
                                  mu, S,
                                  W, sigma_sq)
    
    S_a = helper_dag.get_cov(W_a, sigma_sq_a)
    mu_a = mu
    
    idx_adv = np.hstack([[idx_s,idx_t],idx_copa])
    
    return idx_adv, S_a, mu_a, W_a
    
def get_idx(A, i, max_degree = 2):
    
    rows, cols = np.where(A)
    
    degree = np.sum(A[:,cols],0)
    bool_check = degree<=max_degree
    
    rows = rows[bool_check]
    cols = cols[bool_check]
    
    np.random.seed(42)
    idx = np.arange(len(rows))
    np.random.shuffle(idx)
    
    idx_s = rows[idx[i]]
    idx_t = cols[idx[i]]
    
    return idx_s, idx_t

def get_idx_chain(A, i):
        
    p = len(A)
    
    idx_s = int(p*(0.25*(i+1)))
    idx_t = idx_s+1
    
    return idx_s, idx_t
    
def get_idx_tree(i):
    
    if i == 0:
        idx_s = 0
        idx_t = 1
        
    elif i == 1:
        idx_s = 1
        idx_t = 4
    
    elif i == 2:
        idx_s = 4
        idx_t = 10
    
    return idx_s, idx_t

def sim(seed_sim, degree,
        N, p,
        bool_ev, sigma,
        w_lb, w_ub,
        i,
        return_W = False,
        type_graph = 'er',
        bool_flip_sign = True,
        type_masking = 'legacy'):
    
    '''
    if type_graph == 'er':        
    elif type_graph == 'chain':
        
    type_masking \in {'legacy','descent',}
    '''

        
    if type_graph == 'er':
        #Creates a random graph
        A = get_A(p, degree, seed_sim)
        
    elif type_graph == 'chain':
        A = get_A_chain(p)
    
    elif type_graph == 'tree':
        A = get_A_tree()
    
    #Creates the SCM and samples the data
    X, mu, \
    S, sigma_sq, \
    W = sample(A, bool_ev, 
               sigma, N, 
               w_lb, w_ub, 
               seed_sim,
               bool_flip_sign = bool_flip_sign)
    
#    import ipdb;ipdb.set_trace()
    #Selects the adversarial target and parameters
    try:
        if type_graph == 'er':            
            idx_s, idx_t = get_idx(A, i)
            
        elif type_graph == 'chain':
            idx_s, idx_t = get_idx_chain(A, i)
            
        elif type_graph == 'tree':
            idx_s, idx_t = get_idx_tree(i)
            
        if type_masking == 'legacy':
            idx_adv_train, S_a, \
            mu_a, W_a = get_adv(idx_s, idx_t, 
                                mu, S, 
                                W, sigma_sq)
            
            print('masking', idx_adv_train,
                  'W_{s,t}', np.round(W[idx_s,idx_t],2))
            idx_mask = None
        else:
            if type_graph == 'tree':
                if type_masking == 'des':
                    if type_graph == 'tree' and i ==2:
                        edge_adv = np.array([[idx_s,idx_t]])        
                        
                        idx_adv_train = np.array([idx_s, idx_t])
                        idx_mask = np.array([idx_t])
                
                elif type_masking == 'skip':
                    if type_graph == 'tree' and i == 0:
                        edge_adv = np.array([[idx_s,idx_t],
                                             [2,1]])
                        idx_adv_train = np.array([idx_s, idx_t, 2])
                        idx_mask = np.array([idx_s])
                    
            elif type_graph == 'chain':
                if type_masking == 'skip':                    
                    edge_adv = np.array([[idx_s, idx_t],
                                         [idx_s-1, idx_t]])
                    idx_adv_train = np.array([idx_s, idx_t, idx_s-1])
                    idx_mask = np.array([idx_s])
                    
                    print('adding a skip to chain')
                    
            S_a, mu_a, W_a = get_adv_all(edge_adv, 
                                         mu, S, 
                                         W, sigma_sq)
        
        if not return_W:
            return (X, mu, S, 
                    sigma_sq, W, 
                    idx_adv_train, 
                    idx_mask,
                    mu_a, S_a)
        else:
            return (X, mu, S, 
                    sigma_sq, W, 
                    idx_adv_train, 
                    idx_mask,
                    mu_a, S_a, W_a)
    except:
        return None

    
def main(savePath,
         kwargs_sim,
         kwargs_model,
         ):
    
    '''
    Trains the model and saves it
    
    In:
        k:      #The seed for the inner
            
    '''    
    
    print('starting to simulate')

    load = get_wrap(**kwargs_sim,
                    )
        
    if load is not None:
        
        (X, mu, S, 
         sigma_sq, W, 
         idx_adv_train, 
         idx_mask,
         mu_a, S_a) = load
    
        print('starting to train')
        #Trains the model
        model, history = train(X, 
                               idx_adv_train,
                               mu, S,
                               mu_a, S_a,
                               idx_mask = idx_mask,
                               **kwargs_model)
        
#        import ipdb;ipdb.set_trace()
        #%%    
        str_ = '_'.join([str(key)+'_'+str(val) for key, val in kwargs_sim.items()])
    
        if '07_13' not in savePath:
            simPath = os.path.join(savePath, str_)
        else:
            simPath = os.path.join(savePath, kwargs_sim['type_graph'],
                                   str_)
        
        print('saving to',simPath)
        
        str_ = '_'.join([str(key)+'_'+str(val) for key, val in kwargs_model.items()])
        
        modelPath = os.path.join(simPath,str_)
        
        if not os.path.exists( modelPath):
            os.makedirs( modelPath)
            
        #Saves the model
        with open(os.path.join(modelPath,'history.p'), "wb") as f:
            pickle.dump([history.history], f)
        with open(os.path.join(modelPath,'optim.p'), "wb") as f:
            pickle.dump([model.optimizer.get_weights(),
                         model.optimizer.get_config()], f)
        
        model.save(os.path.join(modelPath,'model'))
        
        #Saves the data
        with open(os.path.join(simPath,'data.p'), "wb") as f:
            pickle.dump(load, f)
    
    
def load_model(modelPath):
    
    loss_type = 9
    
    if loss_type != 9:    
        custom_objects = dict(Custom=helper_tf_model.Custom)
    else:
        custom_objects = dict(Custom2=helper_tf_model.Custom2)
        

    savePath2 = os.path.join(modelPath,'model')
    
        
    model = tf.keras.models.load_model(savePath2, 
                                       custom_objects=custom_objects)

    return model

def get_adv_par_all(A, A_a,
                    mu, S,
                    W, sigma_sq):
    
    '''
    Calculates the minimum distance \theta_adversarial 
    if equal variance not minimum distance
    
    '''
    
#    import ipdb;ipdb.set_trace()
    p = len(S)
    
    W_a = np.zeros_like(W)
    
    if np.isscalar(sigma_sq):
        sigma_sq_a = np.zeros(p)   #It's no longer scaler because
                                                    #variance changed
    else:
        sigma_sq_a = copy.deepcopy(sigma_sq)
                
    for i in range(p):
        
        idx_pa = np.where(A[:,i])[0]
        idx_pa_a = np.where(A_a[:,i])[0]
        
        if not np.array_equal(idx_pa, idx_pa_a):
            
            if len(idx_pa_a)>0: #If there are parents, use projection
                w_new, sigma_sq_new  = _get_adv_par(i, idx_pa_a, 
                                                    mu, S)
                
                W_a[idx_pa_a, i] = w_new
                
            else:   #If no parents, node's variance is the MLE
                sigma_sq_new = S[i,i]
                                    
            sigma_sq_a[i] = sigma_sq_new
    
        else:
            W_a[:,i] = W[:,i]
            
            if np.isscalar(sigma_sq):
                sigma_sq_a[i] = sigma_sq
            else:
                sigma_sq_a[i] = sigma_sq[i]
            
    if np.all(sigma_sq_a == sigma_sq_a[0]):#maybe we should output a scaler
        raise TypeError
        
    return W_a, sigma_sq_a

def get_adv_all(edge_adv, 
                mu, S, 
                W, sigma_sq):
    
    '''
    In:
        edge_adv:   K,2     #[i] 0:source, 1:target
    
    '''
    A = np.abs(W)>0
        
    A_a = get_A_adv(A, edge_adv)
    
    #Gets the adv. \theta
    W_a, sigma_sq_a = get_adv_par_all(A, A_a,
                                      mu, S,
                                      W, sigma_sq)
    
    S_a = helper_dag.get_cov(W_a, sigma_sq_a)
    mu_a = mu        
    
    return S_a, mu_a, W_a

def get_A_adv(A,
              edge_adv):
    
    '''
    In:
        edge_adv:   K,2     #[i] 0:source, 1:target
    
    '''
    A_a = copy.deepcopy(A)
    
    for edge in edge_adv:
        A_a[edge[0],edge[1]] = not bool(A[edge[0],edge[1]])
        
    return A_a

def get_like(X, mu, S, 
             unq, inverse):
    
    like_l = []
    like_o = []
    for i, unq_ in enumerate(unq):
        
        idx = np.where(inverse == i)[0]
        
        idx_m = np.where(unq_)[0]
        idx_o = np.where(~unq_)[0]
        
        if len(idx_o) != 0:
            X_o = X[idx][:,idx_o]
            
            mu_o = mu[idx_o]
            S_o = S[idx_o][:,idx_o]
            
            like_p = scipy.stats.multivariate_normal.logpdf(X_o, mu_o, S_o)
            like_l.append(like_p)
            like_o.append(np.mean(like_p))
            
    like_vec = np.hstack(like_l)
    
    like = np.mean(like_vec)
    
    return like, like_o

def get_wrap(type_graph,
             **kwargs_sim,
             ):
        
    if type_graph in ['chain','tree','erdos']:
        load = sim(**kwargs_sim, type_graph=type_graph)
        
    elif type_graph == 'sachs':
        print('using sachs')
        load = sachs(**kwargs_sim)
        
    return load

def sachs(i,
          return_W = False,
          type_masking = 'legacy',
          return_col = False):
    
    '''        
    type_masking \in {'legacy','descent',}
    '''            
    X, mu, S = helper_data.get_sachs()
    columns = helper_data.get_sachs_columns()
    
    #%%
    if i == 0:
        edge = ['plc','pip2']
        copar = 'pip3'
    elif i == 1:
        edge = ['pip3','pip2']
        copar = 'plc'
    
    idx_s = columns.index(edge[0])
    idx_t = columns.index(edge[1])
    idx_co = columns.index(copar)    
    
    edge_adv = np.array([[idx_s,idx_t]])
    
    idx_adv_train = np.array([idx_s, idx_t, idx_co])
    
    if (type_masking == 'des' or type_masking == 'legacy'):
        idx_mask = np.array([idx_t])
        print('masking t')
        

    #Sets the adversarial parameters
    S_a = copy.deepcopy(S)    
    S_a[idx_s,idx_t] = 0
    S_a[idx_t,idx_s] = 0    
    
    mu_a = mu
        
    #Used for A
    W = helper_data.get_hat_sachs(X)
    sigma_sq = None
        
    #Removes the edge, used for A_a (only?)
    W_a = copy.deepcopy(W)
    W_a[idx_s,idx_t] = 0
    
    if not return_W:
        return (X, mu, S, 
                sigma_sq, W, 
                idx_adv_train, 
                idx_mask,
                mu_a, S_a)
    else:        
        if not return_col:
            return (X, mu, S, 
                    sigma_sq, W, 
                    idx_adv_train, 
                    idx_mask,
                    mu_a, S_a, W_a)    
        else:
            return (X, mu, S, 
                    sigma_sq, W, 
                    idx_adv_train, 
                    idx_mask,
                    mu_a, S_a, W_a,
                    columns)