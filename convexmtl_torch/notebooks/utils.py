import numpy as np
import os
import pickle
from sklearn.model_selection import GridSearchCV

from icecream import ic

import joblib

RESULTS_DIR = 'results'
def train_gs(estim, params, X, y, model_name, problem_name, cv=None, scoring=None, retrain=False):
    pathname = '{}/{}_{}.joblib'.format(RESULTS_DIR, problem_name, model_name)
    if retrain or not os.path.exists(pathname):
        gs = GridSearchCV(estimator=estim, param_grid=params, cv=cv, scoring=scoring)
        gs.fit(X, y)
        joblib.dump(gs, pathname)
    else:
        gs = joblib.load(pathname)
    return gs
