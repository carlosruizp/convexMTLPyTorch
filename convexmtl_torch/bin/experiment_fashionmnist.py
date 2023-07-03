from clearml import Task, Logger


from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import make_scorer, accuracy_score, f1_score, recall_score, precision_score
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import StratifiedKFold
import itertools

from convexmtl_torch.data.DataLoader import DataLoader
from convexmtl_torch.model.HardSharingMTLPytorchModel import HardSharingMTLPytorchClassifier
from convexmtl_torch.model.ConvexMTLPytorchModel import ConvexMTLPytorchClassifier
from convexmtl_torch.model.GraphLaplacianMTLPytorchModel import GraphLaplacianMTLPytorchClassifier
from convexmtl_torch.preprocessing import MTLStandardScaler

from convexmtl_torch.model.utils import ConvNet, ConvNetFeatLearn



from icecream import ic

import pandas as pd
import numpy as np

import utils

from utils import *

import joblib
import os
import sys



N_JOBS = 10

DATA_DIR = '_data'
RESULTS_DIR = 'results'

MAX_EPOCHS = 50


# class ClearMLLogger():
#     def __init__(self, task: Task):
#         self.task = task

#     def connect(self, param_f):
#         ret = param_f

def get_hyperparameters(estim_name, task):
    # param grid
    ic(estim_name)
    wd_l = [10**k for k in range(-2, 1)]
    lr_l = [10**k for k in range(-3, 2)]
    earlystop_l = [False]
    lamb_l = [0.2*k for k in range(6)]
    nu_l = [10**k for k in range(-1, 2)]
    mu_l = [10**k for k in range(-4, -1)]

    # param_grid = {'early_stopping': earlystop_l}
    param_grid = {'weight_decay': wd_l,
                  'early_stopping': earlystop_l}
    if 'cvx' in estim_name and 'trainable' in estim_name:
        param_grid['lambda_lr'] = lr_l
    elif 'cvx' in estim_name and 'trainable' not in estim_name:
        param_grid['lamb'] = lamb_l
    if 'gl' in estim_name:
        param_grid['mu'] = mu_l
        param_grid['nu'] = nu_l
    # param_grid = {'early_stopping': earlystop_lr}
    params_dic = {'param_grid': param_grid}
    if task is not None:
        task.connect(params_dic)

    ic(param_grid)
    return param_grid

def macro_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average='macro')

def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

def macro_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average='macro')


def get_estim(estim_name, task=None, progress_bar=False):
    # estimator
    strategy = estim_name.split('_')[0]
    if strategy == 'hs':
        estim = HardSharingMTLPytorchClassifier(enable_progress_bar=progress_bar, epochs=MAX_EPOCHS, common_module=ConvNetFeatLearn)
    elif strategy == 'cvx-trainable':
        estim = ConvexMTLPytorchClassifier(lambda_trainable=True, specific_lambda=False, enable_progress_bar=progress_bar, epochs=MAX_EPOCHS, common_module=ConvNet)
    elif strategy == 'cvx-trainable-sp':
        estim = ConvexMTLPytorchClassifier(lambda_trainable=True, specific_lambda=True, enable_progress_bar=progress_bar, epochs=MAX_EPOCHS, common_module=ConvNet)
    elif strategy == 'gl':
        estim = GraphLaplacianMTLPytorchClassifier(adj_trainable=True, enable_progress_bar=progress_bar, epochs=MAX_EPOCHS, common_module=ConvNetFeatLearn)
    elif strategy == 'cvx':
        estim = ConvexMTLPytorchClassifier(lambda_trainable=False, specific_lambda=False, enable_progress_bar=progress_bar, epochs=MAX_EPOCHS, common_module=ConvNet)
    elif strategy == 'common':
        estim = ConvexMTLPytorchClassifier(lambda_trainable=False, specific_lambda=False, lamb=1, enable_progress_bar=progress_bar, epochs=MAX_EPOCHS, common_module=ConvNet)
    elif strategy == 'ind':
        estim = ConvexMTLPytorchClassifier(lambda_trainable=False, specific_lambda=False, lamb=0, enable_progress_bar=progress_bar, epochs=MAX_EPOCHS, common_module=ConvNet)
    
    return estim

def get_datascaler(task):
    # scaler
    datascaler = MTLStandardScaler()
    if task is not None:
        task.upload_artifact(name='datascaler', artifact_object=datascaler)
    return datascaler

def get_targetscaler(task):
    # scaler
    datascaler = MinMaxScaler()
    if task is not None:
        task.upload_artifact(name='targetscaler', artifact_object=datascaler)
    return datascaler

def get_data(problem_name, task=None):
    # problem
    if task is not None:
        task.connect({'problem_name': problem_name})
    data_loader = DataLoader(DATA_DIR)
    df_data, df_target, cv_outer, cv_inner, task_info = data_loader.load_dataset(problem_name)
    
    # if task is not None:
    #     task.upload_artifact(name='data', artifact_object=DATA_DIR)
    return df_data, df_target, cv_outer, cv_inner, task_info

def get_scorer_val(task):
    # estimator
    scorer = make_scorer(macro_f1)
    if task is not None:
        task.upload_artifact(name='scorer_val', artifact_object=scorer)
    return scorer



def evaluate_estim_on_problem(estim_name, problem_name, scores_fun,  task_name=None, project_name=None, force_refit=False):

    if CLEARML:
        task = Task.init(project_name=project_name, task_name=task_name)
    else:
        task = None

    X, y, cv_outer, cv_inner, task_info = get_data(problem_name=problem_name, task=task)

    estim = get_estim(estim_name)
    datascaler = get_datascaler(task)
    pipe = generate_pipeline(estim, None, None)
    ic(pipe)

    param_grid = get_hyperparameters(estim_name, task)
    ic(param_grid)

    scorer_val = get_scorer_val(task)

    score_test = macro_f1

    if 2 == y.ndim:
        y_2d = y
    else:
        y_2d = y[:, None]

    df_strat = pd.DataFrame(data=np.concatenate((X[:,task_info][:, None].astype(str), y_2d.astype(str)), axis=1), columns=['task', 'label'])
    ic(df_strat)
    df_strat["strat"] = df_strat[['task', 'label']].agg('-'.join, axis=1)
    col_strat = df_strat.strat

    len_keys = len(list(scores_fun.keys()))
    scores_l = dict(zip( list(scores_fun.keys()), [[] for _ in range(len_keys)]  ))
    for fold, (trainval_idx, test_idx) in enumerate(cv_outer.split(X, col_strat)):
        ic(fold)
        X_trainval, X_test = X[trainval_idx], X[test_idx]
        y_trainval, y_test = y[trainval_idx].ravel(), y[test_idx].ravel()
        col_strat_trainval, col_strat_test = col_strat[trainval_idx], col_strat[test_idx]

        gs = load_cv_results(estim_name=estim_name, problem_name=problem_name, cv_fold=fold, results_dir=RESULTS_DIR ,retrain=force_refit)
        if gs is None:
            if isinstance(cv_inner, StratifiedKFold):
                cv_inner_splits = list(cv_inner.split(X_trainval, col_strat_trainval))
            else:
                cv_inner_splits = cv_inner
            gs = generate_gridsearch(pipe, cv_inner_splits, param_grid, scorer_val)
            gs.fit(X_trainval, y_trainval, new_shape=(1, 28, 28))
            print('FIT DONE')
            save_cv_results(gs, estim_name=estim_name, problem_name=problem_name, cv_fold=fold, results_dir=RESULTS_DIR)
        
        # df = pd.DataFrame.from_dict(gs.cv_results_)
        # ic(df)
        analyze_gs(gs, problem_name, estim_name, fold, task=task)

        for test_fold in range(N_TEST_REPEAT):
            filename_pred = '{}/pred__{}__{}__{}.csv'.format(PREDICTIONS_DIR, pname, ename, test_fold)
            filename_ytest = '{}/ytest__{}__{}__{}.csv'.format(PREDICTIONS_DIR, pname, ename, test_fold)
            if force_refit or not os.path.exists(filename_pred):
                best = clone(pipe)
                logs_dir = 'logs/{}/{}/{}'.format(problem_name, estim_name, test_fold)
                os.makedirs(logs_dir, exist_ok=True)
                aux_params = {'train_mode': 'lightning', 'log_dir': logs_dir}
                best_params = {**{'{}'.format(k): v for k, v in aux_params.items()}, 
                            **gs.best_params_}
                best.set_params(**best_params)
                best.fit(X_trainval, y_trainval, new_shape=(1, 28, 28)) 

                pred = best.predict(X_test)
                np.savetxt(filename_pred, pred)
            else:
                pred = np.loadtxt(filename_pred)  
                np.savetxt(filename_ytest, y_test)          
            
            for score_key, score_fun in scores_fun.items():
                score_value = score_fun(y_test, pred)
                scores_l[score_key].append(score_value)

    if task is not None:
        task.close()

    scores_mean = {k: np.mean(scoresk) for k, scoresk in scores_l.items()}
    return scores_mean


    


# SCRIPT    
version = '0.3'

refit= False

strategies_l = ['ind', 'common', 'cvx', 'cvx-trainable', 'hs', 'gl', 'cvx-trainable-sp'] # ['ind', 'common', 'cvx', 'cvx-trainable'] # ['cvx', 'common', 'ind'] # ['hs', 'cvx', 'gl']

estim_names_l = ['{}_nn'.format(s) for s in strategies_l]



problem_names_l = ["variations_fashionmnist"]

scores_fun = {'accuracy': accuracy_score, 'f1_macro': macro_f1, 'recall_macro': macro_recall}

for pname in problem_names_l:
    scores_mean = {}
    for ename in estim_names_l:
        task_name = '{}'.format(pname, ename)
        scores_mean[ename] = evaluate_estim_on_problem(ename, pname, scores_fun, task_name=ename, project_name='hais23_'+pname, force_refit=refit)

    scores_pname = {}
    for k in scores_fun.keys():
        scores_pname[k] = [scores_mean[ename][k] for ename in estim_names_l]

    df = pd.DataFrame.from_dict(data=scores_pname)
    df.index = estim_names_l
    ic(df)

    results_fname = '{}/results__{}.csv'.format(RESULTS_DIR, pname)
    
        
    if CLEARML:
        task = Task.init(project_name='hais23_'+pname, task_name='results')
    else:
        task = None

    df.index.name = "model"

    if os.path.exists(results_fname):
        df_old = pd.read_csv(results_fname, index_col=0)
        ic(df_old)
        df = pd.concat([df_old, df], sort=True)
    ic(df)

    if task is not None:           
        
        Logger.current_logger().report_table(
            "table results - {}".format(pname), 
            version, 
            table_plot=df
        )
        task.close()
    else:
        df.to_csv(results_fname)
