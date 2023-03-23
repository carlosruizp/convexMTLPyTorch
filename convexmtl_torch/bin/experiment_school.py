from clearml import Task, Logger


from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
import itertools

from convexmtl_torch.data.DataLoader import DataLoader
from convexmtl_torch.model.HardSharingMTLPytorchModel import HardSharingMTLPytorchRegressor
from convexmtl_torch.model.ConvexMTLPytorchModel import ConvexMTLPytorchRegressor
from convexmtl_torch.model.GraphLaplacianMTLPytorchModel import GraphLaplacianMTLPytorchRegressor



from icecream import ic

import pandas as pd
import numpy as np

import utils

import joblib
import os

DATA_DIR = '_data'
RESULTS_DIR = 'results'
N_JOBS = 6

MAX_EPOCHS = 50

# class ClearMLLogger():
#     def __init__(self, task: Task):
#         self.task = task

#     def connect(self, param_f):
#         ret = param_f




def get_hyperparameters(estim_name, task):
    # param grid
    
    wd_l = [10**k for k in range(-3, 0)]
    lr_l = [10**k for k in range(-3, 0)]
    earlystop_lr = [True, False]

    param_grid = {'weight_decay': wd_l}
                  # 'early_stopping': earlystop_lr}
    if 'cvx' in estim_name:
        param_grid['lambda_lr'] = lr_l
    # param_grid = {'early_stopping': earlystop_lr}
    params_dic = {'param_grid': param_grid}
    if task is not None:
        task.connect(params_dic)
    return param_grid

def get_estim(estim_name, task=None, progress_bar=False):
    # estimator
    strategy = estim_name.split('_')[0]
    if strategy == 'hs':
        estim = HardSharingMTLPytorchRegressor(enable_progress_bar=progress_bar, epochs=MAX_EPOCHS, early_stopping=True)
    elif strategy == 'cvx':
        estim = ConvexMTLPytorchRegressor(lambda_trainable=True, specific_lambda=False, enable_progress_bar=progress_bar, epochs=MAX_EPOCHS, early_stopping=True)
    elif strategy == 'gl':
        estim = GraphLaplacianMTLPytorchRegressor(adj_trainable=True, enable_progress_bar=progress_bar, epochs=MAX_EPOCHS, early_stopping=True)
    elif strategy == 'common':
        estim = ConvexMTLPytorchRegressor(lambda_trainable=False, specific_lambda=False, lamb=1, enable_progress_bar=progress_bar, epochs=MAX_EPOCHS, early_stopping=True)
    elif strategy == 'ind':
        estim = ConvexMTLPytorchRegressor(lambda_trainable=False, specific_lambda=False, lamb=0, enable_progress_bar=progress_bar, epochs=MAX_EPOCHS, early_stopping=True)
    
    # if task is not None:
    #     task.connect({'estim_name': estim_name})
    # if task is not None:
    #     task.upload_artifact(name='estim', artifact_object=estim)
    return estim

def get_datascaler(task):
    # scaler
    datascaler = StandardScaler()
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
    scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    if task is not None:
        task.upload_artifact(name='scorer_val', artifact_object=scorer)
    return scorer



def generate_gridsearch(estim, cv, param_grid, scorer_val, n_jobs=N_JOBS, verbose=2):
    
    prefix = ''
    if isinstance(estim, TransformedTargetRegressor):
        prefix += 'regressor__'
        model = estim.regressor
    else:
        model = estim

    if isinstance(model, Pipeline):
        prefix += 'estim__'

    pref_param_grid_ = {'{}{}'.format(prefix, k): v for k, v in param_grid.items()}
    gs = GridSearchCV(estimator=estim, param_grid=pref_param_grid_, scoring=scorer_val,
                        n_jobs=n_jobs, cv=cv, verbose=verbose, refit=False)
    return gs

def generate_pipeline(estim, data_scaler, transformer=None):
    if data_scaler is not None:
        pipe = Pipeline(steps=[('scaler', data_scaler), ('estim', estim)])
    else:
        pipe = estim

    if transformer is not None:
        pipe = TransformedTargetRegressor(regressor=pipe,
                                             transformer=transformer)

    return pipe

def save_cv_results(cv_estim, estim_name, problem_name, cv_fold, results_dir):
    file_name = '{}/{}_{}_cv{}.joblib'.format(results_dir, problem_name, estim_name, cv_fold)
    joblib.dump(cv_estim, file_name)


def load_cv_results(estim_name, problem_name, cv_fold, results_dir, retrain=False):
    file_name = '{}/{}_{}_cv{}.joblib'.format(results_dir, problem_name, estim_name, cv_fold)
    if retrain or not os.path.exists(file_name):
        return None
    cv_estim = joblib.load(file_name)
    return cv_estim


def analyze_gs(gs, fold, task=None):
    params = [p.split('__')[-1] for p in gs.best_params_.keys()]
    linear_scale = ['lamb', 'early_stopping']


    for p in params:
        if p in linear_scale:
            utils.plot_param(gs, p, fold, logscale=False)
        else:
            utils.plot_param(gs, p, fold, logscale=True)


def evaluate_estim_on_problem(estim_name, problem_name, scores_fun,  task_name=None, project_name=None, force_refit=False):

    task = Task.init(project_name=project_name, task_name=task_name)
    # task = None

    X, y, cv_outer, cv_inner, task_info = get_data(problem_name=problem_name, task=task)

    estim = get_estim(estim_name)
    datascaler = get_datascaler(task)
    targetscaler = get_targetscaler(task)
    pipe = generate_pipeline(estim, datascaler, targetscaler)
    ic(pipe)

    param_grid = get_hyperparameters(estim_name, task)
    ic(param_grid)

    scorer_val = get_scorer_val(task)

    score_test = mean_absolute_error

    len_keys = len(list(scores_fun.keys()))
    scores_l = dict(zip( list(scores_fun.keys()), [[] for _ in range(len_keys)]  ))
    for fold, (trainval_idx, test_idx) in enumerate(cv_outer.split(X, X[:,task_info])):
        ic(fold)
        X_trainval, X_test = X[trainval_idx], X[test_idx]
        y_trainval, y_test = y[trainval_idx].ravel(), y[test_idx].ravel()

        gs = load_cv_results(estim_name=estim_name, problem_name=problem_name, cv_fold=fold, results_dir=RESULTS_DIR ,retrain=True)
        if force_refit or gs is None:
            gs = generate_gridsearch(pipe, cv_inner, param_grid, scorer_val)
            gs.fit(X_trainval, y_trainval)
            print('FIT DONE')
            save_cv_results(gs, estim_name=estim_name, problem_name=problem_name, cv_fold=fold, results_dir=RESULTS_DIR)
        
        # df = pd.DataFrame.from_dict(gs.cv_results_)
        # ic(df)
        if task is not None:
            analyze_gs(gs, fold, task=task)

        best = clone(pipe)
        best.set_params(**gs.best_params_)
        # best.set_params(regressor__estim__train_mode='lightning')
        best.fit(X_trainval, y_trainval) 

        if 'cvx' in estim_name:
            utils.plot_lambda_hist(best.regressor_.steps[1][1], cv_fold=fold, task=task)

        pred = best.predict(X_test)
        
        for score_key, score_fun in scores_fun.items():
            score_value = score_fun(y_test, pred)
            scores_l[score_key].append(score_value)
        
        # ic(score)
        # ic(recall_score(y_test, pred, pos_label=1))
        # ic(recall_score(y_test, pred, pos_label=2))
        # ic(accuracy_score(y_test, pred))
        # ic(macro_f1(y_test, pred))
        # ic(macro_precision(y_test, pred))

    # for score_key, score_fun in scores_fun.items():
    #     score_value = score_fun(y_test, pred)
    #     scores_l[score_key].append(score_value)

    if task is not None:
        task.close()

    scores_mean = {k: np.mean(scoresk) for k, scoresk in scores_l.items()}
    return scores_mean


    


# SCRIPT    
version = '0.3'

refit= True

strategies_l = ['cvx', 'common', 'ind'] # ['hs', 'cvx', 'gl']

estim_names_l = ['{}_nn'.format(s) for s in strategies_l]



problem_names_l = ["school"]

scores_fun = {'mae': mean_absolute_error, 'mse': mean_squared_error}

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


    task = Task.init(project_name='hais23_'+pname, task_name='results')
    df.index.name = "model"
    Logger.current_logger().report_table(
        "table results - {}".format(pname), 
        version, 
        table_plot=df
    )
    task.close()
