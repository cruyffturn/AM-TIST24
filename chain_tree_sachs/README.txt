#Required packages

matplotlib                3.3.3
numpy                     1.19.5 
scipy                     1.5.3 
scikit-learn              1.0.2 
pandas (with excel read)  1.1.4 
tensorflow                2.4.0 
causallearn               0.1.3.1 
networkx                  2.7.1
joblib                    0.17.0
seaborn                   0.12.0
xlrd                      1.2.0

#List of core files

+main_sim_par:      Trains the LAMM model
+main_load_sim:     Runs and reports the modeler algorithms
+main:              Runs the overall pipeline


./cfgs/
+tree:              Configuration file for the binary tree experiments
+chain:             Configuration file for the chain graph experiments

#List of helper files

+helper_dag:        Subroutines relavent to sampling or estimating dags
+helper_data:       Sachs dataset helper
+helper_draw:       Plots the loss functions
                    
+helper_em_tf:      Implements the WEM, and other EM related functions in tf
+helper_em:         Implements missDAG
+helper_load:       Helper functions for loading the simulations
+helper_local:      Defines the paths for saving the pipeline Outputs
+helper_sim:        Helper functions for running the simulations
+helper_tf_model:   Custom Keras model for the LAMM algorithm

+helper:            Calls modeler algorithms (missDAG, missPC, etc) and 
                    calculates graph distance                

./helper_prob/models/
+helper_mvn_tf:     Subroutines relavent to mvn distribution in tf
+helper_mvn:        Subroutines relavent to mvn distribution

./helper_plot/
+helper_plot:       Helper functions for throwing a heat map
+plots:             Draws a heat map 

./etc/
+other_favs:         Obtaining a grid for visualization purposes

+External/notears:  Contains the modified external notears package accessed at
                    https://github.com/xunzheng/notears


#Steps for repeating the sachs results
-Sachs dataset requires downloading the supplementary files

1. Download https://www.science.org/doi/suppl/10.1126/science.1105809/suppl_file/sachs.som.datasets.zip
2. Unzip sachs.som.datasets.zip and copy "1. cd3cd28.xls" to the project directory.
3. Call main.py.
4. It will output two files, first for missDAG & second for missPC algorithms:
-results_exp_type_0_0.010_type_graph_sachs.xlsx
-results_exp_type_1_0.010_type_graph_sachs.xlsx

To attack a different edge:
5. Change the file helper_sim by seting i = 0 in function _get_cfg_sachs().

#Steps for repeating the chain graph results

1. Change the file helper_sim by seting type_graph = 'chain' in function get_cfg(). 
2. Call main.py.
3. It will output two files, first for missDAG & second for missPC algorithms:
-results_exp_type_0_0.010_type_graph_chain_short.xlsx
-results_exp_type_1_0.010_type_graph_chain_short.xlsx

#Steps for repeating the binary tree graph results

1. Change the file helper_sim by seting type_graph = 'tree' in function get_cfg(). 
2. Call main.py.
3. It will output two files, first for missDAG & second for missPC algorithms:
-results_exp_type_0_0.010_type_graph_tree.xlsx
-results_exp_type_1_0.010_type_graph_tree.xlsx

