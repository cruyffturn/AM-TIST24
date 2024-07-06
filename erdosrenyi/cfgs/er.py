# -*- coding: utf-8 -*-
    
def get_cfg():    

    w_lb = 0.5  
    w_ub = 2
    
    bool_flip_sign = True

    seed_sim_l = range(1,20)

    i_l = range(1,2)
    
    lr = 0.4e-2
    reg_lmbda = 1.25e-1
    epochs = 500
#    
    seed_model_l = range(1)
    
    type_noise = 'normal'
#    type_noise = 'laplace'
#    type_noise = 'gumbel'
        
    bool_emp_covar = False
    type_masking = 'auto'
    
                
    return (seed_sim_l, i_l,
            w_lb, w_ub, 
            bool_flip_sign, 
            reg_lmbda, lr,
            epochs, seed_model_l,
            type_masking,
            type_noise,
            bool_emp_covar)