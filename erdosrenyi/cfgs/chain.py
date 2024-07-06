# -*- coding: utf-8 -*-

def get_cfg():
    
    degree = 1
    w_lb = 0.5
    w_ub = 0.9
    
    bool_flip_sign = False

    
    
#    seed_sim_l = range(7)
#    seed_sim_l = range(20,)
#    seed_sim_l = range(1,2)
#    i_l = range(3)
#    lr = 0.5e-2
#    reg_lmbda = 1e-2
#    epochs = 300
#    lr = 0.2e-2
#    reg_lmbda = 1e-2
#    epochs = 600
     
#    reg_lmbda = 0
#    epochs = 600
#    lr = 0.1e-2

    lr = 0.5e-2
    reg_lmbda = 7.5e-2
#    reg_lmbda = 1e-1
    epochs = 500
    
    
    if 1:
#        seed_sim_l = range(20,)
        seed_sim_l = range(5,)
#        seed_sim_l = range(8,);print('up to 8')
        i_l = range(3)
        type_masking = 'skip'
    else:    
        seed_sim_l = range(1)
        i_l = range(1,2)
#        type_masking = 'bivar'
        type_masking = 'skip'
    
    if 0:
        reg_lmbda = 1e-2    
    #    seed_sim_l = range(7)
        seed_sim_l = range(1)
        i_l = range(1)
        lr = 0.5e-2
        epochs = 300 
    
#    seed_model_l = range(1,2)
    seed_model_l = range(1)
        
    
    type_noise = 'normal'
#    type_noise = 'laplace'
#    type_noise = 'gumbel'
    
    bool_emp_covar = False
    
    return (seed_sim_l, i_l,
            degree, w_lb, w_ub, 
            bool_flip_sign, 
            reg_lmbda, lr,
            epochs, seed_model_l,
            type_masking,
            type_noise,
            bool_emp_covar)