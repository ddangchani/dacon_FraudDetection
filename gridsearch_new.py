import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
import itertools

def gridsearch(model, param_grid, x_test, y_test):
    '''
    model : scikit-learn model,
    param_grid : dict form of grid parameters
    x_test, y_test : scoring data and label
    '''
    keys, values = zip(*param_grid.items())
    permutations_dicts = [dict(zip(keys,v)) for v in itertools.product(*values)]
    