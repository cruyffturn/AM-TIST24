# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np

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
    
    with open(os.path.join(args.savePath,'X.npy'), 'rb') as f:
        X = np.load(f)
            
#    print(args.bool_classi)
    if np.any(np.isnan(X)):
        raise ValueError
        
    if args.bool_linear:
        kwargs = {}
    else:
        kwargs = dict(bool_linear=args.bool_linear)
        
    A, W = helper_missdag.estimate_baseline(X, 
                                            args.seed, 
                                            args.w_threshold,
                                            **kwargs)
    
    with open(os.path.join(args.savePath,'temp.npyz'), 'wb') as f:
        np.savez(f, A=A, W=W)