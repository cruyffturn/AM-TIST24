# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np

#import helper_missdag

parser = argparse.ArgumentParser()
parser.add_argument('savePath',
                    type=str)
parser.add_argument('seed',
                    type=int)
parser.add_argument('w_threshold',
                    type=float)
parser.add_argument('bool_linear',
                    type=int)
args = parser.parse_args()

if args.bool_linear:
    import helper_missdag
else:
    import helper_missdag_mlp as helper_missdag
    
if __name__ == '__main__':
    import sys
    print(sys.version)
    # Execute when the module is not initialized from an import statement.
    
#    del sys.modules["tf"]
#    del tf
     
    with open(os.path.join(args.savePath,'X.npy'), 'rb') as f:
        X_miss = np.load(f)
        
    if args.bool_linear:
        kwargs = {}
    else:
        kwargs = dict(bool_linear=args.bool_linear)
            
#    print(args.bool_classi)
    print('threshold+call missdag',args.w_threshold)
    W, cov_est = helper_missdag.estimate_missdag(X_miss, 
                                                 args.seed, 
                                                 args.w_threshold,
                                                 **kwargs
                                                 )
    
    with open(os.path.join(args.savePath,'temp.npyz'), 'wb') as f:
        np.savez(f, cov_est=cov_est, W=W)