# -*- coding: utf-8 -*-
import os
import inspect

filename = inspect.getframeinfo(inspect.currentframe()).filename
currPath = os.path.dirname(os.path.abspath(filename))

sharedPath_0 = os.path.join(currPath,
                            '04_19'
                            )

n_jobs = 1

def get_localPath(bool_repeat=0):
    
    return get_savePath(bool_repeat)

def get_sharedPath(bool_repeat=0,
                   bool_retrain=0):
    
    str_ = []
    
    if bool_repeat:        
        str_.append('repeat')
    
    if bool_retrain:
        str_.append('retrain')
        
    str_.append('output_result')
    
    sharedPath = os.path.join(sharedPath_0,'_'.join(str_))
    print('sharedPath',sharedPath)
    return sharedPath

def get_savePath(bool_repeat):
    '''
    Saves the model training results
    '''
    
    sharedPath_1 = sharedPath_0
        
    savePath = os.path.join(sharedPath_1, 'output_model')
    
    if bool_repeat:
        savePath = savePath.replace('output_model','repeat_output_model')
        
    return savePath

def get_condaPath():
    
    return 'miniconda3/envs/missdag_mlp/bin/python'