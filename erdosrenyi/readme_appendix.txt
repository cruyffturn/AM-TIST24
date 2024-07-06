#List of core files

+main_sim_par:      Trains the LAMM model
+main_load_sim:     Runs and reports the modeler algorithms
+main:              Runs the overall pipeline


./cfgs/
+er:                Configuration file for the linear erdos-renyi graph experiments
+er_mlp:            Configuration file for the nonlinear erdos-renyi graph experiments

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

helper_nonlinear:   Simulates non-linearity by calling External/MissDAG

+call_baseline:     Reads the data matrix from disk, calls NOTEARS-ICA or NOTEARS-MLP, and saves the output to disk
+call_missdag:      Reads the data matrix from disk, calls MissDAG-ICA or MissDAG-MLP, and saves the output to disk
+helper_missdag:    Calls NOTEARS-ICA or MissDAG-ICA
+helper_missdag_mlp:Calls NOTEARS-MLP or MissDAG-MLP
+helper_missdag_wrap:Wraps call_missdag.py or call_baseline.py and reads the results

./helper_prob/
+metrics:           Calculates the correlation

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
+External/MissDAG:  Contains the modified external MissDAG package accessed at
                    https://github.com/ErdunGAO/MissDAG