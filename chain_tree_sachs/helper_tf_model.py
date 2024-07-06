# -*- coding: utf-8 -*-
import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.datasets import make_spd_matrix

import os
import matplotlib.pyplot as plt
    
import helper_em_tf
from helper_prob.models.helper_mvn_tf import get_KL

import helper_dag
global count
count = 0
bool_ent = 0
#bool_sub = True
#bool_sub = False
#bool_reg = True
#bool_l1 = 1
#reg_l1 = 1
class Custom(keras.Sequential):
  
    def _set_param(self, S_init, mu_init, 
                   S_a, mu_a, n_steps,
                   loss_type,
                   bool_full = 0,
                   bool_draw = 0,
                   idx_adv = None,
                   reg_lmbda = 0,
                   bool_l1 = 0,
                   bool_sub = False,
                   idx_mask = None):
        
        self.S = S_init
        self.mu = mu_init
        
        self.S_a = tf.constant(S_a, tf.float32)
        self.mu_a = tf.constant(mu_a, tf.float32)
        self.n_steps = n_steps
        self.loss_type = loss_type
        
        self.bool_full = bool_full
        
        if idx_adv is None:
            self.idx_adv = np.arange(len(mu_a)).astype(np.int32)
        else:
            self.idx_adv = idx_adv
            
        self.date = datetime.now().strftime("_time_%H_%M_%m_%d_%Y")
        self.bool_draw = bool_draw
        
        self.reg_lmbda = reg_lmbda
        self.bool_l1 = bool_l1
        self.bool_sub = bool_sub
        
        if idx_mask is None:
            self.idx_mask = self.idx_adv
        else:
            print('Custom: using idx_mask')
            self.idx_mask = idx_mask
        
#        self._alpha = alpha        

def loss_cat_1(x, p_r_x,
               bool_full,
               loss_type, 
               mu_0, S_0, 
               mu_a, S_a, 
               n_steps,
               idx_mask,
               return_S = False):
    
    '''
                            
    '''
    if loss_type == 8:
        bool_while = True
    else:
        bool_while = 0
        
    mu_new, S_new = helper_em_tf.get_em_enum(x, p_r_x, 
                                             idx_mask,
                                             mu_0 = mu_0, 
                                             S_0 = S_0,
                                             n_steps = n_steps,
                                             bool_full = bool_full,
                                             bool_while = bool_while
                                             ) 
        
    if loss_type == 0:
        pass
    else:
#                loss = get_KL(self.mu_a, self.S_a, 
#                              mu_new, S_new, K2=None)
        loss = get_KL(mu_new, S_new, 
                      mu_a, S_a, K2=None)
    
    K_new = tf.linalg.inv(S_new)
    K_a = tf.linalg.inv(S_a)
    
    if loss_type == 3:
        pass        
    elif loss_type in [5,6]:                    
        pass
                
    if not return_S:
        return loss, K_new
    else:
        return loss, K_new, S_new

                
import helper
def get_KL_L(mu_L, S_L,
             mu_a, S_a,
             mu, S):
    
    stat_L = []
    for mu_, S_ in zip(mu_L,S_L):
        stat_L.append(helper._get_stats(mu_, S_, 
                                        mu_a, S_a, 
                                        mu, S))
    
    return np.stack(stat_L,0)
            

def rand_init(std, seed):
                
#    std = np.nanstd(X_miss,0)
    
    S_0_0 = make_spd_matrix(len(std), random_state = seed)
    diag = std/np.sqrt(np.diag(S_0_0))
    D = np.diagflat(diag)
    S_0 = D @ S_0_0 @ D
    
    return S_0

class Custom2(Custom):
      
    '''
    Extends the keras.Sequential for implementing a custom loss function
    '''
    def train_step(self, data): 
        
        '''
        '''
                
        debug = 0
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x = data
        bool_full = self.bool_full
        global count
        if self.loss_type == 9:
            
            trainable_vars = self.trainable_variables
            init = 1
            
            plain_loss = 0
#            n_init = 10#5
            n_init = 5
            avg_l1 = 0
            avg_loss = 0
            
            for seed_i in range(n_init):
                with tf.GradientTape() as tape:
                    
                    if not self.bool_sub:
                        p_r_x = self(x, training=True)  # Forward pass
                    else:
                        x_sub = tf.constant(x.numpy()[:,self.idx_adv])                        
                        p_r_x = self(x_sub, training=True)  # Forward pass
                        
                    mu_0 = self.mu
                    std = np.sqrt(np.diag(self.S))
                    
                    
                    
                    S_0 = rand_init(std, seed_i)    #Fix maybe set seedonce
                    
                    loss, K_new, S_new = loss_cat_1(x, p_r_x,
                                             bool_full, 8, 
                                             mu_0, S_0, 
                                             self.mu_a, self.S_a, 
                                             self.n_steps,
                                             self.idx_mask,
                                             return_S = True)
                    
                    if x.shape[1] == 3:
                        print(seed_i, 
                              np.round(K_new,2)[0,1],
                              np.round(K_new,2)[1,2])
                    elif x.shape[1] > 3:
#                        import ipdb;ipdb.set_trace()
#                        S_new_ = np.linalg.inv(K_new)
                        P_new = helper_dag.get_pear(S_new.numpy())
                        
                        if len(self.idx_adv) <3:
                            print(seed_i, 
                                  np.round(P_new,2)[self.idx_adv[0],self.idx_adv[1]]
                                  )
                        else:                                  
                            print(seed_i, 
                                  np.round(P_new,2)[self.idx_adv[0],self.idx_adv[1]],
                                  np.round(P_new,2)[self.idx_adv[0],self.idx_adv[2]]
                                  )
                    
#                    loss = loss_i
                    
#                    if bool_full:
#                        p_miss = tf.gather(p_r_x, indices=np.array([p_r_x.shape[1]-1]), 
#                                           axis = 1)
#                        loss = loss + 0*tf.math.reduce_mean(p_miss)
                        

#                    lmbda = 1e-2
#                    lmbda = 1e-4
                    plain_loss += loss.numpy()
                    if bool_full:
#                        print('using reg. with',self.reg_lmbda)
                        prob_obs = helper_em_tf.get_obs_prob(p_r_x, len(self.idx_mask))
                        
                        if self.reg_lmbda != 0:
                            loss = loss + self.reg_lmbda*(1-prob_obs)
                        
                    if self.bool_l1 != 0:
                        pass
                    
                    if bool_ent:
                        pass
                        
                    avg_loss += loss.numpy()

#                import ipdb;ipdb.set_trace() 
                gradients_i = tape.gradient(loss, trainable_vars)
                # Compute gradients
                if init:    
                    gradients = gradients_i
                    init = 0
                else:
                    for i in range(len(gradients)):
                        gradients[i] = gradients[i] + gradients_i[i]
                                    
        # Update weights
        if not debug:
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        else:
            print('debug:not grad')        

        if not bool_full:
            exp_obs_ratio = tf.math.reduce_mean(tf.math.reduce_mean(p_r_x,1))
            p_miss_row = tf.math.reduce_mean(tf.math.reduce_prod(1-p_r_x,1))
        else:
            exp_obs_ratio = np.nan
            p_miss_row = tf.math.reduce_mean(p_r_x[:,-1])
                
        
        if self.bool_draw:                                                     
            pass
        else:
            S_est = None

#        n_rem, n_add = get_graph_error(K_a.numpy(), K_new.numpy()[np.newaxis,:,:], 
#                                       0.1)
        
        plain_loss = plain_loss/n_init
        
        exp_obs_ratio = (prob_obs*len(self.idx_mask)+x.shape[1]-len(self.idx_mask))/x.shape[1]
        
        if not bool_ent:
            avg_loss = plain_loss + self.reg_lmbda*(1-prob_obs)
            
            if self.bool_l1 != 0:
                pass
        else:
            pass
        
        loss_dic = {'loss':loss,
                    'exp_obs_ratio':exp_obs_ratio,
                    'p_miss_row':p_miss_row,
                    'plain_loss':plain_loss,
                    'prob_obs':prob_obs,
                    'avg_loss':avg_loss
#                    'n_add_tilde':n_add,
#                'ent':ent
                }
        
#        tf.summary.scalar('learning rate', 
#                          data=loss_dic['avg_loss'], 
#                          step=count)


        if self.bool_l1 != 0:
            pass
        
        if bool_ent:
            pass
            
        if x.shape[1] == 2:
            loss_dic['corr']=S_est[0,1]
            
        if x.shape[1] == 3:
            if S_est is not None:
                K_est = np.linalg.inv(S_est)
                loss_dic['K_01']=K_est[0,1]
                loss_dic['K_12']=K_est[1,2]
                
            if self.loss_type != 7:
                loss_dic['K_01_tilde']=K_new.numpy()[0,1]
                loss_dic['K_12_tilde']=K_new.numpy()[1,2]
                
        elif x.shape[1] > 3:
            if S_est is not None:
                loss_dic['S_0']=S_est[:,:5]
            
            S_new = np.linalg.inv(K_new.numpy())            
            loss_dic['S_0_tilde']=S_new[0,:5]
            print(np.round(S_new,3)[0,:5])            
#            loss_dic['K_12']=K_est[1,2]
#        else:
        if self.loss_type == 5:
            pass
            
        return loss_dic
    
    

def get_model(p, model_cfg, 
              loss_type,
              idx_adv_train,
              bool_full,
              mu = None, S = None,
              mu_a = None, S_a = None,
              bool_ratio = False,
              X = None,
              bool_sub = False,
              idx_mask = None):
    
    '''
    idx_adv_train:  #idx of input nodes
    idx_mask:       #idx of masked nodes
        
    bool_ratio:     #experimental adding ratio as the input
    '''
    if loss_type != 9:
        model = Custom()
    else:
        model = Custom2()
    
    if not bool_sub:
        model.add(keras.Input(shape=(p,)))
    else:
        model.add(keras.Input(shape=(len(idx_adv_train),)))
    
    if X is not None:
#        tf.keras.layers.experimental.preprocessing.Normalization
        layer = layers.experimental.preprocessing.Normalization(axis=1)
        if not bool_sub:
            layer.adapt(X)
        else:
            layer.adapt(X[:,idx_adv_train])
        model.add(layer)
#        import ipdb;ipdb.set_trace()
        
    if bool_ratio:
        pass
    if model_cfg == 1:
        model.add(layers.Dense(10,activation='relu'))
    elif model_cfg == 2:
        model.add(layers.Dense(100,activation='relu'))
    elif model_cfg == 3:
        model.add(layers.Dense(10,activation='relu'))
        model.add(layers.Dense(10,activation='relu'))
    elif model_cfg == 4:
        model.add(layers.Dense(100,activation='relu'))
        model.add(layers.Dense(100,activation='relu'))
        
    #model.add(layers.Dense(p,activation='sigmoid'))
    #model._set_param(S, mu,S_a, mu_a, n_steps, loss_type)
    
    if idx_mask is None:
        idx_mask = idx_adv_train
        
    p_sub = len(idx_mask)
    
    if not bool_full:
        raise TypeError
#        if p_sub != p:
#            raise TypeError
        model.add(layers.Dense(p,activation='sigmoid'))
    else:
        model.add(layers.Dense(int(2**p_sub),activation='softmax'))
    
    return model

def get_savePath(model_cfg, loss_type, 
                 bool_full, mode,data_dic,
                 reg_lmbda, seed,
                 currPath,
                 bool_retrain = False,
                 mode_adv = 0,
                 bool_l1 = 0,
                 attack_node = 'pip2',
                 bool_hat_true = True,
                 bool_sub = False,
                 bool_force_zero = False,
                 bool_sim = False):
    
    dict_sub = dict(
#                corr=corr,
                model_cfg=model_cfg,
                loss_type=loss_type,
                bool_full=bool_full,
                mode=mode,
                seed=seed)
    dict_sub = {**dict_sub,**data_dic}
    
    if reg_lmbda != 0:
        dict_sub['reg_lmbda'] = reg_lmbda
    if mode_adv != 0:
        dict_sub['mode_adv'] = mode_adv
    if bool_l1 != 0:
        dict_sub['bool_l1'] = bool_l1
    if attack_node != 'pip2':
        dict_sub['attack_node'] = attack_node
    if not bool_hat_true:
        dict_sub['bool_hat_true'] = bool_hat_true
    if bool_sub:
        dict_sub['bool_sub'] = bool_sub         
    if bool_force_zero:
        dict_sub['bool_force_zero'] = bool_force_zero
    if bool_sim:
        dict_sub['bool_sim'] = bool_sim
        
    figname = '_'.join(['%s_%s'%(a,b) for a, b in dict_sub.items()])
    
    savePath = os.path.join(currPath, 'train_'+figname)

    if bool_retrain:
        savePath = os.path.join(savePath, 'retrain')
        
    return savePath
def load_model(model_cfg, loss_type, 
               bool_full, mode,data_dic,
               reg_lmbda, seed,
               currPath,
               **kwargs):
    
#    dict_sub = dict(
##                corr=corr,
#                model_cfg=model_cfg,
#                loss_type=loss_type,
#                bool_full=bool_full,
#                mode=mode,
#                seed=seed)
#    dict_sub = {**dict_sub,**data_dic}
#    
#    if reg_lmbda != 0:
#        dict_sub['reg_lmbda'] = reg_lmbda
#    if mode_adv != 0:
#        dict_sub['mode_adv'] = mode_adv
#        
#    figname = '_'.join(['%s_%s'%(a,b) for a, b in dict_sub.items()])
#    
#    savePath = os.path.join(currPath, 'train_'+figname)
#
#    if bool_retrain:
#        savePath = os.path.join(savePath, 'retrain')
    
    savePath = get_savePath(model_cfg, loss_type, 
                            bool_full, mode,data_dic,
                            reg_lmbda, seed,
                            currPath,
                            **kwargs)
    if loss_type != 9:
    #    Custom = 
        custom_objects = dict(Custom=Custom)
    else:
        custom_objects = dict(Custom2=Custom2)
        

    savePath2 = os.path.join(savePath,'model')
    
        
    model = tf.keras.models.load_model(savePath2, 
                                       custom_objects=custom_objects)

    return model