Please see the readme_appendix.txt for a brief description of the files contained in
this folder.

#Preparing the virtual environments 

!!! Our code was written in Tensorflow 2 but the author implementation of 
one of the baseline methods, missDAG, requires an older version of Tensorflow. 
To run the modeler algorithm missDAG with non-Gaussian and non-linear ANMs, 
our code requires two seperate virtual environments (referred to as A and B). 
Please make sure to specify the path to environment B in the helper_local.py 
file as described below. !!!

#Required packages for the environment A

python                    3.7.11
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

#Required packages for the environment B

python                    3.7.13
numpy                     1.21.6
pandas                    1.3.5
scikit-learn              1.0.2
scipy                     1.7.3
tensorflow                1.15.0            #pip install tensorflow==1.15.0
protobuf                  3.20.3            #This step is required for tensorflow to work accurately
torch                     1.13.1+cpu        #Call pip install torch==1.13.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
networkx                  2.6

After creating the environment B accordingly, please update the function 
get_condaPath in helper_local.py to reflect the python path. Ex:
'miniconda3/envs/[name of your environment B]/bin/python'

#Steps for repeating the Erdos-Renyi graph experiments

As the initial step, set the working directory to this folder containing the readme.txt file.

i) Linear ANMs with Gaussian Noise

Step 1. Call the main.py with arguments 1 1 1 as follows:
python main.py 1 1 1

Step 2. The summary table will be in the folder 04_19/output_result/ for missDAG algorithm
-results_exp_type_0_0.125_type_graph_er_type_noise_normal_type_scm_linear_short.xlsx

ii) Linear ANMs with Laplace Noise

Step 1. Change the file cfgs/er.py by seting type_noise = 'laplace' in function get_cfg().

Step 2. Call the main.py with arguments 1 1 1 as follows:
python main.py 1 1 1

Step 3. The summary table will be in the folder 04_19/output_result/ for missDAG-ICA algorithm
-results_exp_type_0_0.125_type_graph_er_type_noise_laplace_type_scm_linear_short.xlsx

iii) Linear ANMs with Gumbel Noise

Step 1. Change the file cfgs/er.py by seting type_noise = 'gumbel' in function get_cfg().

Step 2. Call the main.py with arguments 1 1 1 as follows:
python main.py 1 1 1

Step 3. The summary table will be in the folder 04_19/output_result/ for missDAG-ICA algorithm
-results_exp_type_0_0.125_type_graph_er_type_noise_gumbel_type_scm_linear_short.xlsx

iv) Nonlinear ANMs with Gaussian Noise

Step 1. Change the file helper_sim by seting type_scm = 'mlp' in function get_cfg(). 

Step 2. Call the main.py with arguments 1 1 1 as follows:
python main.py 1 1 1

Step 3. The summary table will be in the folder 04_19/output_result/ for missDAG-MLP algorithm
-results_exp_type_0_0.075_type_graph_er_type_noise_normal_type_scm_mlp_short.xlsx

Step 4. Change the file cfgs/er_mlp.py by seting bool_low_pearson = 0 in function get_cfg().

Step 5. Call the main.py with arguments 1 1 1 as follows:
python main.py 1 1 1

Step 6. The summary table will be in the folder 04_19/output_result/ for missDAG-MLP algorithm
-results_exp_type_0_0.200_type_graph_er_type_noise_normal_type_scm_mlp_short.xlsx

    
Disclaimer:

The code provided in External/MissDAG contains a modified version of the github repository 
https://github.com/ErdunGAO/MissDAG corresponding to the Neurips '22 paper 
"MissDAG: Causal Discovery in the Presence of Missing Data with Continuous Additive Noise Models".
The original repo does not have an explicit license. This modified version is made available 
for academic research purposes only. 