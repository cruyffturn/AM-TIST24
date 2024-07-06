# -*- coding: utf-8 -*-
import numpy as np

try:
    from External.MissDAG.dag_methods import Notears_ICA_MCEM, Notears_ICA
    from External.MissDAG.miss_methods.miss_dag_nongaussian import miss_dag_nongaussian
    from External.MissDAG.utils.utils import postprocess
    
    from External.MissDAG.dag_methods import Notears_MLP_MCEM_INIT, Notears_MLP_MCEM
    from External.MissDAG.miss_methods.miss_dag_nonlinear import miss_dag_nonlinear    

except:
    pass


def estimate_baseline(X, seed_baseline, 
                      w_threshold,
                      bool_linear):
    
    if bool_linear:
        W_nt_0  = Notears_ICA(seed_baseline).fit(X)
    else:
#        missdag_ref = Notears_MLP_MCEM_INIT()
        print('using no tears mlp init')
        W_nt_0, _ = Notears_MLP_MCEM_INIT().fit(X)
        
    _, A_nt = postprocess(W_nt_0, w_threshold)
    W_nt = W_nt_0*A_nt
    
    return A_nt, W_nt

    
def estimate_missdag(X_miss, seed_modeler, 
                     w_threshold,
                     bool_linear):
    
    if bool_linear:
        print('running missdag')
        dag_init_method = Notears_ICA(seed_modeler, 
        #                              args.MLEScore
                                      )
        dag_method = Notears_ICA_MCEM(seed_modeler
        #                              , args.MLEScore
                                      )
        
        W_0, cov_est, \
        histories = miss_dag_nongaussian(X_miss, 
                                         dag_init_method,
                                         dag_method, 
        #                                 em_iter, 
        #                                 MLEScore, 
        #                                   num_sampling, 
                                           )

    else:
        print('using missdag mlp')
        dag_init_method = Notears_MLP_MCEM_INIT()

        dag_method = Notears_MLP_MCEM()
        
        W_0, cov_est, \
        histories = miss_dag_nonlinear(X_miss.astype(np.float32), dag_init_method,
                                       dag_method, 
    #                                 em_iter, 
    #                                 MLEScore, 
    #                                   num_sampling, 
                                       )
    
            
    _, A = postprocess(W_0, w_threshold)
    W = W_0*A
    
    return W, cov_est
    
