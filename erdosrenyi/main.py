#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

from main_sim_par import main_sim_par
from main_load_sim import main_load_sim

import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('train',
                    type=int)
parser.add_argument('modeler',
                    type=int)
parser.add_argument('table',
                    type=int)
parser.add_argument('bool_repeat',                     
                    default=0,
                    type=bool,
#                    required=False,
                    nargs='?'
                    )
parser.add_argument('bool_repeat_model',                     
                    default=0,
                    type=bool,
                    nargs='?'
#                    required=False,
                    )

args = parser.parse_args()

print('repeat',args.bool_repeat)
#bool_repeat = 
if __name__ == '__main__':
    
    if args.train:
        main_sim_par(args.bool_repeat_model)      #Trains the LAMM models1
        
    
    if args.modeler:
        main_load_sim(bool_save = 1, exp_type = 0,
                      bool_overwrite = 1,
#                      bool_overwrite = 0,
                      bool_repeat = args.bool_repeat,
                      bool_repeat_model = args.bool_repeat_model
                      ) #Saves missDAG Estimated Graphs
        
    if args.table:
        main_load_sim(bool_save = 0, exp_type = 0, 
                      draw_hat = 1,
                      bool_ratio=1,
                      bool_repeat = args.bool_repeat,
                      bool_repeat_model = args.bool_repeat_model) #Loads missDAG Estimated Graphs