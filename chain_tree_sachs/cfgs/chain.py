# -*- coding: utf-8 -*-

def get_cfg():
    
    degree = 1
    w_lb = 0.5
    w_ub = 0.9
    
    bool_flip_sign = False
    reg_lmbda = 1e-2
    
#    seed_sim_l = range(7)
    seed_sim_l = range(20,)
    i_l = range(3)
    lr = 0.5e-2
    epochs = 300    
    
#    seed_model_l = range(1,2)
    seed_model_l = range(1)
    
    type_masking = 'skip'
    
    return (seed_sim_l, i_l,
            degree, w_lb, w_ub, 
            bool_flip_sign, 
            reg_lmbda, lr,
            epochs, seed_model_l,
            type_masking)