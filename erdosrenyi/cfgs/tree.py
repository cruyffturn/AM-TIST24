# -*- coding: utf-8 -*-

def get_cfg():
    
    degree = 1
    w_lb = 0.5
    w_ub = 2
    
    bool_flip_sign = True
        
    reg_lmbda = 1e-2
    degree = 1
                
#    seed_sim_l = range(10,20)
#    seed_sim_l = range(20)
#    i_l = range(3)
#    lr = 0.5e-2
#    epochs = 300 
                         
    seed_sim_l = range(7,)
    lr = 0.5e-2
    reg_lmbda = 7.5e-2
#    reg_lmbda = 1e-1
    epochs = 500
        
    seed_model_l = range(1)
    
#    type_masking = 'des'
    type_masking = 'auto'
#    type_masking = 'skip'
    
    if type_masking == 'skip':
        i_l = range(1)
    elif type_masking == 'des':
        i_l = range(2,3)
    elif type_masking == 'auto':
        i_l = [0,2]
    else:
        i_l = range(3)
        
#    type_noise = 'normal'
    type_noise = 'laplace'
    bool_emp_covar = False
    
    return (seed_sim_l, i_l,
            degree, w_lb, w_ub, 
            bool_flip_sign, 
            reg_lmbda, lr,
            epochs, seed_model_l,
            type_masking,
            type_noise,
            bool_emp_covar)