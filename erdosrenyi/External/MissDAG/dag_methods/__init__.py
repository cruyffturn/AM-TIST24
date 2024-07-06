from .notears.notears import Notears
try:
    from .notears_ica_mcem.notears_ica_mcem import Notears_ICA_MCEM    
    from .notears_ica.notears_ica import Notears_ICA    
except RuntimeError as error:
    print('ica not loaded')
    print(error)
    
try:
    from .notears_mlp_mcem.notears_mlp_mcem import Notears_MLP_MCEM
    from .notears_mlp_mcem.notears_mlp_mcem_init import Notears_MLP_MCEM_INIT    
except RuntimeError as error:
    print('non-linear not loaded')
    print(error)