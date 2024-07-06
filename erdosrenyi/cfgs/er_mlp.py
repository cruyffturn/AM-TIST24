# -*- coding: utf-8 -*-

def get_cfg():
    
    w_lb = 0.5  
    w_ub = 1
    
    bool_flip_sign = True        
    
    bool_low_pearson = 1
    low_pearson_l = [2, 4, 5, 6, 7, 14, 15, 19]
    
    if not bool_low_pearson:
        print('running seeds with high pearson correlation')

        seed_sim_l = [i for i in range(1,20) if i not in low_pearson_l]
        i_l = range(1)    
    
        lr = 0.5e-2
        reg_lmbda = 2e-1
        epochs = 400
    else:
        print('running seeds with low pearson correlation')
        seed_sim_l = low_pearson_l
        i_l = range(1)
    
        lr = 0.5e-2    
        reg_lmbda = 7.5e-2
        epochs = 400    
    
    seed_model_l = range(1)
    type_noise = 'normal'
    bool_emp_covar = True
    type_masking = 'auto'
                
    return (seed_sim_l, i_l,
            w_lb, w_ub, 
            bool_flip_sign, 
            reg_lmbda, lr,
            epochs, seed_model_l,
            type_masking,
            type_noise,
            bool_emp_covar)