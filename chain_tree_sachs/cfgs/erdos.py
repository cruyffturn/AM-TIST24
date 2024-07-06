# -*- coding: utf-8 -*-

def get_cfg():
    degree = 1
    w_lb = 0.5    
    w_ub = 2    
    bool_flip_sign = True
        
#   reg_lmbda = 1e-3
    reg_lmbda = 1e-2
    degree = 1
                
    if degree == 1:
        seed_sim_l = range(3+14)
#            seed_sim_l = range(3,3+14)
        i_l = range(2)#3
        lr = 1e-2
        epochs = 200    
        
    elif degree == 2:
        lr = 0.5e-2
        epochs = 300 
        seed_sim_l = range(15)
        i_l = range(2)
                
    return (seed_sim_l, i_l,
            degree, w_lb, w_ub, 
            bool_flip_sign, 
            reg_lmbda, lr,
            epochs)