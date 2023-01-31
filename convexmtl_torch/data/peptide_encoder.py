
'''
 From https://dmnfarrell.github.io/bioinformatics/mhclearning#:~:targetText=One%20hot%20encoding,corresponds%20to%20that%20amino%20acid.
'''
import os, sys, math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
# import epitopepredict as ep

codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def one_hot_encode(seq):
    o = list(set(codes) - set(seq))
    s = pd.DataFrame(list(seq))    
    x = pd.DataFrame(np.zeros((len(seq),len(o)),dtype=int),columns=o)    
    a = s[0].str.get_dummies(sep=',')
    a = a.join(x)
    a = a.sort_index(axis=1)
    # show_matrix(a)
    e = a.values.flatten()
    return e

def nlf_encode(seq):    
    x = pd.DataFrame([nlf[i] for i in seq]).reset_index(drop=True)  
    show_matrix(x)
    e = x.values.flatten()
    return e

def blosum_encode(seq):
    #encode a peptide into blosum features
    s=list(seq)
    x = pd.DataFrame([blosum[i] for i in seq]).reset_index(drop=True)
    show_matrix(x)
    e = x.values.flatten()    
    return e

