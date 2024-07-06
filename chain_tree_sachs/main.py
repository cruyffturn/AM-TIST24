#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

from main_sim_par import main_sim_par
from main_load_sim import main_load_sim

main_sim_par()      #Trains the LAMM models

main_load_sim(bool_save = 1, exp_type = 1) #Saves PC Estimated Graphs
main_load_sim(bool_save = 0, exp_type = 1) #Loads PC Estimated Graphs

main_load_sim(bool_save = 1, exp_type = 0) #Saves missDAG Estimated Graphs
main_load_sim(bool_save = 0, exp_type = 0) #Loads missDAG Estimated Graphs