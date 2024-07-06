'''
local_sim_path:     #where the model is saved
sim_path:           #where the results are saved

please Use OMP_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1


bool_overwrite:     Overrides and saves the modular estimates again
'''
import os
import numpy as np
import pickle
import pandas as pd    
import inspect
    
from itertools import product
from PIL import Image

import matplotlib.pyplot as plt
    
import helper_sim
import helper
import helper_draw
import helper_load
import helper_data
import helper_local


class Temp():
    def __init__(self,history):
        self.history = history[0]

filename = inspect.getframeinfo(inspect.currentframe()).filename
currPath = os.path.dirname(os.path.abspath(filename))
            
def main_load_sim(bool_save = 0,       #save modeler g_hat
                  exp_type = 0,        #modeler type
                  bool_ratio = 0,       #Draw ratio
                  bool_save_mask = 0,   #Outputs the LAMM masks for debugging
                  draw_hat = 0,         #Draw estimated graphs
                  bool_save_only_rel = 0,   #Used For debugging
                  bool_overwrite = 0,       #Used For debugging
                  bool_repeat = 0,          #Used For debugging
                  bool_repeat_model = 0,    #Used For debugging
                  node = 0,
                  max_node = 1,
                  debug = 0,
                  debug_draw = 0,
                  bool_draw_loss = 1,
                  bool_draw_scatter = 1):
        
#    bool_repeat_model = bool_repeat_model and not bool_repeat
    
    localPath = helper_local.get_localPath(bool_repeat_model)
    sharedPath = helper_local.get_sharedPath(bool_repeat=bool_repeat,
                                             bool_retrain=bool_repeat_model)
    
    gen = helper_sim.get_cfg()    
    
    total_count = [i for i,_ in enumerate(helper_sim.get_cfg())][-1]+1    
    split_l = np.array_split(np.arange(total_count),max_node)    
    
    if debug:
        print('debug debug debug debug debug')
            
        
    bool_draw = not bool_save            
    
    if bool_repeat:
        bool_ratio = 0
        bool_draw = 0
        
    figure_l = []
    figure_ratio_l = []
    figure_hat_l = []
    figure_scatter_l = []
    
    figure_train_l = []
    
    if not bool_save:
        row_l = []          #For anything but score_names
        row2_l = []         #For score_names
    
    score_names = ['Adv. Success',
                   'HD($\hat{\mathcal{G}},\mathcal{G}_{p}$)',
                   'HD($\hat{\mathcal{G}},\mathcal{G}_{ref}$)',                   
                   ]
    
    score_names_new = [r'KL($\theta_p||\hat{\theta}$)',
                       r'KL($\theta_{\alpha}||\hat{\theta}$)']
    
    score_names_mcar = [r'HD($\hat{\mathcal{G}}_{MCAR},\hat{\mathcal{G}}_{MNAR}$)',
                        'subgraph HD($\hat{\mathcal{G}}_{MCAR},\hat{\mathcal{G}}_{MNAR}$)',
                        'n_rep']
    
    score_names_subgraph = ['subgraph HD($\hat{\mathcal{G}},\mathcal{G}_{ref}$)']
    
    if exp_type == 0:
        score_names.append('undir-HD($\hat{\mathcal{G}},\mathcal{G}_{ref}$)')
        
    count = 0
    count_gen = -1
    
    for kwargs_sim, kwargs_model in gen:
        
        count_gen += 1
                
        if kwargs_sim['type_graph'] == 'tree':
            bool_short = True
            bool_split = False
#            n_rep = 20
#            n_rep = 2
            n_rep = 10
        
        elif kwargs_sim['type_graph'] in ['chain',
#                                          'sachs'
                                          ]:
            bool_short = True
            bool_split = False
            n_rep = 10   
#            n_rep = 2
            
        elif kwargs_sim['type_graph'] == 'sachs':
            bool_short = False
            bool_split = False
#            n_rep = 50
            n_rep = 1
        else:
            bool_short = True
            bool_split = False
#            n_rep = 10
            n_rep = 5
            print('using',n_rep)
            kwargs_dag = {}
            
        legacy_random = False
        
        bool_split = bool_split and bool_save
        
        kwargs_dag = {}
            
        if kwargs_sim['type_graph'] in ['chain','tree']:
            kwargs_dag['max_steps'] = 200
            
#        str_1 = '_'.join([str(key)+'_'+str(val) for key, val in kwargs_sim.items()]) 
#        str_1 = '_'.join([str(key)+'_'+str(val) for key, val in kwargs_sim.items() 
#                            if not ((key=='type_noise') and (val=='normal'))])
        str_1 = '_'.join([str(key)+'_'+str(val) for key, val in kwargs_sim.items() 
                            if not (((key=='type_noise') and (val=='normal')) or
                                    ((key=='type_scm') and (val=='linear')))])
    
        check_split = (not bool_split) or \
                      (bool_split and (count_gen in split_l[node].tolist()))
        
        if '07_13' not in localPath:
            local_sim_path = os.path.join(localPath, str_1)
        else:
            local_sim_path = os.path.join(localPath, kwargs_sim['type_graph'],
                                           str_1)
        
        sim_path = os.path.join(sharedPath,
                                'save_07_13',
                                kwargs_sim['type_graph'],
                                str_1)        
        
            
        if not os.path.exists( sim_path):
            os.makedirs( sim_path)
        
        if check_split:
            
            print('count_gen',count_gen,str_1)
            count += 1
            
            if count>-1:        #Used for debugging
                            
                (X, mu, S, 
                sigma_sq, W, 
                idx_adv_train,
                idx_mask,
                mu_a, S_a, W_a) = helper_sim.get_wrap(**kwargs_sim,
                                                      return_W = True
                                                      )
                                
                str_2 = '_'.join([str(key)+'_'+str(val) for key, val in kwargs_model.items()])                
                    
                modelPath = os.path.join(local_sim_path,str_2)
                
                savePath = os.path.join(sim_path, str_2)
                
                #%%
                A_a = (np.abs(W_a)>0).astype(int)
                A_p = (np.abs(W)>0).astype(int)
                
                idx_s,idx_t = idx_adv_train[:2]
                
                file = os.path.join(modelPath,'history.p')
                
                if bool_draw:       #Draws the ground truth graphs
                    drawPath = os.path.join(localPath,str_1,'fig')
                    
                    if not os.path.exists( drawPath):
                        os.makedirs( drawPath)                    
                    
                    fig = helper_load.draw_g3_comb(W, W_a,
                                                   idx_t)
                    fig.suptitle(str_1)
#                    fig = helper_load.draw_g2(A_p, W, idx_s,idx_t,
#                                              str_1)
#                    fig.set_size_inches( w = 10,h = 5)
                    drawfile = os.path.join(drawPath,'g.png')
                    fig.savefig(drawfile, dpi=200, bbox_inches='tight')
                    figure_l.append(drawfile)            
    
                if bool_ratio and os.path.exists(file) and \
                    (len(idx_adv_train) == 2):      #Draws the likelihood ratio 
                                                    #figure
                        
#                    drawPath2 = os.path.join(localPath,str_1,'fig_ratio')
                    drawPath2 = os.path.join(savePath,'fig_ratio')
                    
                    if not os.path.exists( drawPath2):
                        os.makedirs( drawPath2)                                            

                    model = helper_sim.load_model(modelPath)
                    
#                    import ipdb;ipdb.set_trace()
                    drawfile = helper_load.draw_ratio(drawPath2,
                                                   model, 
                                                   X, mu, S,                 
                                                   idx_adv_train,
                                                   idx_mask,
                                                   mu_a, S_a,
                                                   idx_s, idx_t,
                                                   str_1
                                                   )
                    figure_ratio_l.append(drawfile)                 
#                import ipdb;ipdb.set_trace()
                if (bool_save_mask or bool_draw_scatter) and os.path.exists(file): #Saves example masks
                    
                    model = helper_sim.load_model(modelPath)
                    mask = helper_load.get_mask(X, model, 
                                         kwargs_model['seed_model'],
                                         idx_adv_train,
                                         idx_mask,
                                         n_rep = 20,
                                         )
                    if bool_save_mask:
                        with open(os.path.join(savePath,'mask'+'.p'), "wb") as f:
                            pickle.dump(mask, f)
                
#                import ipdb;ipdb.set_trace()
                if bool_draw_scatter and os.path.exists(file):       #Draws the (s,t) scatter
#                    drawPath3 = os.path.join(localPath,str_1,'fig_scatter')
                    drawPath3 = os.path.join(savePath,'fig_scatter')
                    
                    if not os.path.exists( drawPath3):
                        os.makedirs( drawPath3)               
                    
                    temp_fig, temp_axs = plt.subplots(1,2)
                    temp_axs[0].scatter(X[:,idx_t],X[:,idx_s],alpha=0.5)
                    temp_axs[0].set_xlabel('X_t')

                    max_std= np.max([np.sqrt(S[idx_t,idx_t]),
                                     np.sqrt(S[idx_s,idx_s])])
            
                    if kwargs_sim['type_scm'] == 'linear':
                        temp_axs[0].set_xlim(-5*max_std,5*max_std)
                        temp_axs[0].set_ylim(-5*max_std,5*max_std)
                    else:
                        temp_axs[0].axis('equal')
#                    temp_axs[0].set_aspect('equal', 'box')
                    
                    
                                        
                    temp_axs[1].scatter(X[:,idx_t],X[:,idx_s],alpha=0.5,
                                        c=mask[0,:,idx_mask])
                    temp_axs[1].set_xlabel('X_t')

                    if kwargs_sim['type_scm'] == 'linear':
                        temp_axs[1].set_xlim(-5*max_std,5*max_std)
                        temp_axs[1].set_ylim(-5*max_std,5*max_std)
                    else:
                        temp_axs[1].axis('equal')
                        
#                    temp_axs[1].set_aspect('equal', 'box')
                    
                    
                    
                    
#                    temp_ax.set_label('X_s')
                    temp_fig.suptitle(str_1)
#                    fig = helper_load.draw_g2(A_p, W, idx_s,idx_t,
#                                              str_1)
                    temp_fig.set_size_inches( w = 10,h = 5)
                    drawfile = os.path.join(drawPath3,'scatter.png')
                    temp_fig.savefig(drawfile, dpi=200, bbox_inches='tight')
                    figure_scatter_l.append(drawfile)
                    
                plt.close('all')
                if os.path.exists(file): #Accessing the Reference graphs
                                        
                    base_file = os.path.join(sim_path,
                                             'baseline.p')                    
                        
                    if not os.path.exists(base_file):
                        #Calculates Reference Graphs
                        A_pc, A_nt = helper_load.get_baseline(X, 
                                                              kwargs_sim['type_noise'],
                                                              kwargs_sim['type_scm'],
                                                              pc_alpha = 0.01,
                                                              lambda1 = 0.1)
                        
#                        import ipdb;ipdb.set_trace()
                        with open(base_file, "wb") as f:
                            pickle.dump({'pc':A_pc,
                                         'nt':A_nt}, f)    
                    else:
                        with open(base_file, "rb") as f:
                            temp = pickle.load(f)
                            A_pc, A_nt = temp['pc'], temp['nt']
                                
#                import ipdb;ipdb.set_trace()
                if bool_draw_loss and os.path.exists(file):
                                                
                    if not os.path.exists( savePath):
                        os.makedirs( savePath)
                        
                    #Saves the loss function
                    with open(file, "rb") as f:
                        history = pickle.load(f)
                                
                    
                    temp = Temp(history)
                    loss_type = 9
                    fig = helper_draw.draw_loss(temp, loss_type, False)
                    
                    fig.suptitle(str_1)
                    drawfile_loss = os.path.join(savePath,'train_loss.png')
                    fig.savefig(drawfile_loss, dpi=200, bbox_inches='tight')
#                    fig.savefig(os.path.join(savePath,'train_loss.png'), 
#                    	            dpi=200, bbox_inches='tight')
                    
                    figure_train_l.append(drawfile_loss)
                    
                if bool_save and os.path.exists(file):
                                                
                    if not os.path.exists( savePath):
                        os.makedirs( savePath)
                                        
                    #Runs the Modeler algorithms Under both MCAR and MNAR settings
                    seed_model = kwargs_model['seed_model']
                    
                    model = helper_sim.load_model(modelPath)
                    
                    bool_init = 1        
                                    
                    for bool_mcar in [0,1]:
                        for exp_type in [exp_type]:

                            tempPath = helper_load.get_savePath(savePath,
                                                                bool_mcar, 
                                                                exp_type,
                                                                bool_short = bool_short
                                                                )
                            _done = os.path.exists(os.path.join(tempPath,'A_est.p'))
                                                        
                            done = _done and not bool_overwrite
                            if not bool_save_only_rel and not done:
                                
                                helper_load.main(savePath,
                                                 X, model, 
                                                 S, mu,
                                                 A_p, A_a,
                                                 S_a, mu_a,
                                                 idx_adv_train,
                                                 seed_model,
                                                 kwargs_sim['type_noise'],
                                                 kwargs_sim['type_scm'],
                                                 #Configuration parameters
                                                 bool_mcar, 
                                                 exp_type,
                                                 bool_init,
                                                 debug=debug,
                                                 bool_short=bool_short,
                                                 n_rep=n_rep,
                                                 idx_mask=idx_mask,
                                                 idx_s = idx_s, 
                                                 idx_t = idx_t,
                                                 legacy_random=legacy_random,
                                                 **kwargs_dag    
                                                 )
#                                import ipdb;ipdb.set_trace()
                            
                            #Saves the "relative" graph
                            savePath2 = helper_load.get_savePath(savePath,
                                                                 bool_mcar, 
                                                                 exp_type,
                                                                 bool_short = bool_short)
                            
                            if os.path.exists(os.path.join(savePath2,
                                                           'A_est.p')):
        
                                print('saving rel')
                                helper_load.save_rel(A_pc, A_nt, 
                                                     A_p, savePath,
                                                     exp_type, bool_mcar,
                                                     bool_short = bool_short
                                                     )
                                print('saving rel')
                                
                                helper_load.save_rel_sub(A_pc, A_nt, 
                                                         A_p, A_a,
                                                         savePath,
                                                     exp_type, bool_mcar,
                                                     bool_short = bool_short
                                                     )
                                
                    if os.path.exists(os.path.join(savePath2,
                                                   'A_est.p')):
                        
                        print('saving mnar mcar rel ')
                        helper_load.save_rel_mcar(A_p,
                                                  A_a,
                                                  savePath,
                                                  exp_type, 
                                                  bool_short = bool_short
                                                  )
                else:
                    #+
                    stats = helper_load.get_stats(A_p, W, 
                                                  mu, S,
                                                  mu_a, S_a,
                                                  idx_s, idx_t)
                    
                    score_l = []                    
                    if draw_hat:
                        
                        if kwargs_sim['type_graph'] != 'tree':
                            _fig,_axs = plt.subplots(2,1)
                        else:
                            _fig,_axs = plt.subplots(1,2)
                        
                    for bool_mcar in [0,1]:
                        for exp_type in [exp_type]:
                            
                            savePath2 = helper_load.get_savePath(savePath,
                                                                 bool_mcar, 
                                                                 exp_type,
                                                                 bool_short = bool_short)
                            if exp_type == 1:
                                name = 'summary_exp_type_1.csv'
                            else:
                                name = 'summary.csv'
                                    
                            if os.path.exists(os.path.join(savePath2,name)):                                                                
                                #Reads the Performance metrics
                                
                                df = pd.read_csv(os.path.join(savePath2,name))
                                score_0 = df[score_names[:2]].to_numpy()
                                model_names = df.iloc[:,0]
                                
#                                import ipdb;ipdb.set_trace()
                                df_rel = pd.read_csv(os.path.join(savePath2,
                                                                  'rel_summary.csv'))
                                
                                if (df[score_names[1]] != \
                                    df_rel[score_names[1]]).any():
                                    raise ValueError
                                    
                                score_1 = df_rel[score_names[2:]].to_numpy()
                                
                                if not bool_mcar:
                                    df_rel_mcar = pd.read_csv(os.path.join(savePath2,
                                                              'mcar_rel_summary.csv'))
                                
                                #Checks the references match
                                temp_score = df_rel_mcar['reference_%s'%(['mnar','mcar'][bool_mcar])]
                                if (df[score_names[1]] != \
                                    temp_score).any():
                                    raise ValueError                                
                                    
                                score_3 = df_rel_mcar[score_names_mcar].to_numpy()

                                df_rel_subgraph = pd.read_csv(os.path.join(savePath2,
                                                                  'subgraph_rel_summary.csv'))
#                                import ipdb;ipdb.set_trace()
                                score_4 = df_rel_subgraph[score_names_subgraph].to_numpy()
                                
                                if (df[score_names[1]] != \
                                    df_rel_subgraph[score_names[1]]).any():
                                    raise ValueError
                                
                                if exp_type == 0:   #Adds the KL scores
                                    score_2 = df[score_names_new].to_numpy()                                    
                                
                                    score = np.concatenate([score_0,score_1,
                                                            score_2,score_3,
                                                            score_4],1)
                                else:
                                    score = np.concatenate([score_0,score_1,
                                                            score_3,score_4],
                                                           1)
                                                                


                                score_l.append(score)
                                bool_exists = True
                                                                    
#                                import ipdb;ipdb.set_trace()
                                                                
                                if draw_hat:    #Draws the Estimated graphs
                                                                                                            		
                                    with open(os.path.join(savePath2,
                                                           'A_est.p'), "rb") as f:
                                                                                
                                        W_est_all_l, hue_l = pickle.load(f)

                                        if len(hue_l) > 2:
                                            if exp_type == 0:
                                                idx_modeler = 2
                                            else:
                                                idx_modeler = 0
                                        else:
                                            idx_modeler = 0
                                        
                                        A_est_all = np.abs(W_est_all_l[idx_modeler])>0
                                        hue = hue_l[idx_modeler]
                                        
                                        if kwargs_sim['type_graph'] == 'sachs':
                                            node_names = helper_data.get_sachs_columns()
                                        else:
                                            node_names = None
                                            
                                        helper_load.draw_g2_edge(A_est_all.mean(0), 
                                                                 W, idx_s,idx_t,
                                                                 _axs[bool_mcar],
                                                                 node_names = node_names,
                                                                 bool_mcar = bool_mcar,
                                                                 hue = hue,
                                                                 type_graph=kwargs_sim['type_graph'],
                                                                 exp_type=exp_type)
                                                                                

                                    
                            else:
                                bool_exists = 0
                    
                    if draw_hat:    #Formats the saved figures
                        _fig.suptitle(str_1)
                        if kwargs_sim['type_graph'] == 'chain':
                            _fig.set_size_inches( w = 15, h = 5 )
                        elif kwargs_sim['type_graph'] == 'tree':
                            _fig.set_size_inches( w = 20, h = 7)
                            
                        else:
                            _fig.set_size_inches( w = 15, h = 10)
                            
                        drawfile = os.path.join(savePath,
                                                'g_hat_exp_type%i.png'%exp_type)
                        _fig.savefig(drawfile, dpi=200, bbox_inches='tight')
                        figure_hat_l.append(drawfile)

                        
                    if bool_exists:                
                        
                        #Calculates the miss. %
                        temp = pd.read_csv(os.path.join(savePath2,'p_r.csv'))                    
                        prob_sum = helper_load.get_prob(temp)
                                        
                        #Calculates the training stats
                        with open(file, "rb") as f:
                            history = pickle.load(f)
                            
                        loss_best = np.min(history[0]['avg_loss'])
                        epoch_best = np.argmin(history[0]['avg_loss'])    
                        
                        err_pc = helper.get_pdag_dist(A_p, A_pc[np.newaxis,:,:], 
                                                      allow_pdag = True)[0]
                        err_nt = helper.get_pdag_dist(A_p, A_nt[np.newaxis,:,:], 
                                                      allow_pdag = False)[0]
                        
                        pc_success = A_pc[idx_s,idx_t] == 0
                        nt_success = A_nt[idx_s,idx_t] == 0
                        
#                        import ipdb;ipdb.set_trace()
                        names_mask = np.array([str(idx_i) for idx_i in idx_mask])
                        names_mask[idx_mask==idx_s] = 'S'
                        names_mask[idx_mask==idx_t] = 'T'
                        
                        row = [str_1,kwargs_model['seed_model']] + \
                                list(stats) + [loss_best,epoch_best]+ \
                                [err_pc,err_nt]+\
                                [pc_success,nt_success]+\
                                ((1-prob_sum)*100).tolist()+\
                                ['_'.join(names_mask.tolist())]
                        
                        row_l.append(row)
                        
                        row2_l.append(np.stack(score_l,-1))
    #            break
    
        if debug_draw:
            break
#        if count_gen > 20:
#            break
#        break
    #%%
    if exp_type == 0:    
        num_used = 20+4
    else:
        num_used = 18+4
        
    if not bool_save:    #Prepares the Output Excel files                
        temp_max = np.max([len(row)-num_used for row in row_l])
        columns = ['path','seed_model']
        columns += ['avg_deg (in&out)', 'n_copa', 
                   'weight', 'partial', 
                   'KL_min', 'n_out_s', 
                   'n_in_s','pearson']
        
        columns += ['train. loss','train. epoch']
        
        columns += ['SHD(pc, true)','SHD(nt, true)']
        columns += ['pc_success','nt_success']
        
        if idx_mask is None:
            print('!!check num_used correct')
            raise ValueError
            columns += ['% missing (s)','% missing (t)']
            if temp_max > 2:
                columns += ['% missing (co)']* (temp_max-2)
        else:
            columns += ['% missing '+'(%i)'%iii for iii in range(len(prob_sum))]
        
        columns += ['Masked Nodes']
        
        if not bool_save:
            df1 = pd.DataFrame(row_l,columns=columns)
            df1 = df1.round({'avg_deg (in&out)': 2, 
                           'weight': 2,
                           'partial':2,
                           'KL_min':2,
                           'pearson':2,
                           'train. loss':2,
                           '% missing (s)':1,
                           '% missing (t)':1,
                           '% missing (co)':1,
                           '% missing (0)':1,
                           })
            
            score_tensor = np.stack(row2_l,0)
            
            file_name = 'results_exp_type_%i_%.3f'%(exp_type,kwargs_model['reg_lmbda'])
            
            if 'type_graph' in kwargs_sim.keys():
                file_name += '_type_graph_%s'%kwargs_sim['type_graph']
                                        
            if 'type_noise' in kwargs_sim.keys():
                file_name += '_type_noise_%s'%kwargs_sim['type_noise']
            
            if 'type_scm' in kwargs_sim.keys():
                file_name += '_type_scm_%s'%kwargs_sim['type_scm']
            
            if kwargs_sim['p'] != 15:
                file_name += '_p_%i'%kwargs_sim['p']
                
            if kwargs_sim['degree'] != 1:
                file_name += '_degree_%i'%kwargs_sim['degree']
                
            if bool_short:
                file_name = file_name + '_short'
                            
            if bool_repeat:
                file_name = 'repeat_' + file_name
            elif bool_repeat_model:
                file_name = 'retrain' + file_name
                
            file_name += '.xlsx'
            
            if debug_draw:
                file_name = 'debug_' + file_name
                
            if exp_type == 0:
                score_names_sub = score_names + score_names_new + \
                                    score_names_mcar + score_names_subgraph
            else:
                score_names_sub = score_names + score_names_mcar + \
                                    score_names_subgraph
                
            tablePath = os.path.join(sharedPath,'summary_tables')
            
            if not os.path.exists( tablePath):
                os.makedirs( tablePath)
    
            with pd.ExcelWriter(os.path.join(tablePath,file_name)) as writer:
                
                for iii in range(score_tensor.shape[1]):
                    
                    flat = score_tensor[:,iii].reshape(len(score_tensor),-1)
                    
                    cols = [x+y for x, y in product(score_names_sub, ['MNAR','MCAR'])]
                    
                    df2 = pd.DataFrame(flat, columns = cols)
                    df_all = pd.concat([df1,df2],axis=1)
                    
                    sheet_name = '%s'%(model_names[iii]).replace('/','_')
                    sheet_name = sheet_name.replace('*','')
                    df_all.to_excel(writer, 
                                    sheet_name=sheet_name,
                                    index=False)
                    
                if bool_draw:
                    
                    workbook  = writer.book
                    worksheet = workbook.add_worksheet('figures')
                    row_no = 1
                    for i, figPath, in enumerate(figure_l):
                           
                        # Get the xlsxwriter workbook and worksheet objects.                                                                    
                        
                        worksheet.insert_image('A' + str(row_no),
                                               figPath)
                        
                        with Image.open(figPath) as img:
                            width, height = img.size
                        row_no += int(height*2.5e-2)
                        
                if bool_ratio:
                    
                    workbook  = writer.book
                    worksheet = workbook.add_worksheet('ratio figures')
                    row_no = 1
                    for i, figPath, in enumerate(figure_ratio_l):
                           
                        # Get the xlsxwriter workbook and worksheet objects.                                                                    
                        
                        worksheet.insert_image('A' + str(row_no),
                                               figPath)
                        
                        with Image.open(figPath) as img:
                            width, height = img.size
                        row_no += int(height*2.5e-2)
                        
                if draw_hat:
                    
                    workbook  = writer.book
                    worksheet = workbook.add_worksheet('hat')
                    row_no = 1
                    for i, figPath, in enumerate(figure_hat_l):
                           
                        # Get the xlsxwriter workbook and worksheet objects.                                                                    
                        
                        worksheet.insert_image('A' + str(row_no),
                                               figPath)
                        
                        with Image.open(figPath) as img:
                            width, height = img.size
                        row_no += int(height*2.5e-2)
                        
                if bool_draw_scatter:
                    
                    workbook  = writer.book
                    worksheet = workbook.add_worksheet('scatter plots')
                    row_no = 1
                    for i, figPath, in enumerate(figure_scatter_l):
                           
                        # Get the xlsxwriter workbook and worksheet objects.                                                                    
                        
                        worksheet.insert_image('A' + str(row_no),
                                               figPath)
                        
                        with Image.open(figPath) as img:
                            width, height = img.size
                        row_no += int(height*2.5e-2)
                        
                if bool_draw_loss:
                    
                    workbook  = writer.book
                    worksheet = workbook.add_worksheet('training loss')
                    row_no = 1
                    for i, figPath, in enumerate(figure_train_l):
                           
                        # Get the xlsxwriter workbook and worksheet objects.                                                                    
                        
                        worksheet.insert_image('A' + str(row_no),
                                               figPath)
                        
                        with Image.open(figPath) as img:
                            width, height = img.size
                        row_no += int(height*2.5e-2)