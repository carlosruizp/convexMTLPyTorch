from clearml import Task, Logger


from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, recall_score, accuracy_score, f1_score, precision_score
from sklearn.base import clone
from sklearn.pipeline import Pipeline
import itertools

from convexmtl_torch.data.DataLoader import DataLoader
from convexmtl_torch.model.HardSharingMTLPytorchModel import HardSharingMTLPytorchRegressor
from convexmtl_torch.model.ConvexMTLPytorchModel import ConvexMTLPytorchRegressor
from convexmtl_torch.model.GraphLaplacianMTLPytorchModel import GraphLaplacianMTLPytorchRegressor



from icecream import ic

import pandas as pd
import numpy as np

import utils

import pickle
import os

DATA_DIR = '_data'
PROBLEM_NAME = 'school_mini'
RESULTS_DIR = 'results'

# class ClearMLLogger():
#     def __init__(self, task: Task):
#         self.task = task

#     def connect(self, param_f):
#         ret = param_f




def macro_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average='macro')

def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

def macro_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average='macro')


def get_hyperparameters(estim_name, task):
    # param grid
    
    wd_l = [0.2*k for k in range(6)]

    param_grid = {'weight_decay': wd_l}
    
    params_dic = {'param_grid': param_grid}
    if task is not None:
        task.connect(params_dic)
    return param_grid

def get_estim(estim_name, task=None):
    # estimator
    strategy = estim_name.split('_')[0]
    if strategy == 'hs':
        estim = HardSharingMTLPytorchRegressor()
    elif strategy == 'cvx':
        estim = ConvexMTLPytorchRegressor()
    elif strategy == 'gl':
        estim = GraphLaplacianMTLPytorchRegressor()
    
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
    scorer = make_scorer(macro_recall)
    if task is not None:
        task.upload_artifact(name='scorer_val', artifact_object=scorer)
    return scorer



def generate_gridsearch(estim, cv, param_grid, scorer_val, n_jobs=3, verbose=0):
    
    if hasattr(estim, 'steps'):
        param_grid_ = {'estim__{}'.format(p): pgrid for p, pgrid in param_grid.items()}
    else:
        param_grid_ = param_grid
    gs = GridSearchCV(estimator=estim, param_grid=param_grid_, scoring=scorer_val,
                        n_jobs=n_jobs, cv=cv, verbose=verbose)
    return gs

def generate_pipeline(estim, data_scaler):
    if data_scaler is not None:
        pipe = Pipeline(steps=[('scaler', data_scaler), ('estim', estim)])
    else:
        pipe = estim
    return pipe

def save_cv_results(cv_estim, estim_name, problem_name, cv_fold, results_dir):
    file_name = '{}/{}_{}_cv{}.p'.format(results_dir, problem_name, estim_name, cv_fold)
    with open(file_name, 'wb') as f:      
        pickle.dump(cv_estim, f)


def load_cv_results(estim_name, problem_name, cv_fold, results_dir):
    file_name = '{}/{}_{}_cv{}.p'.format(results_dir, problem_name, estim_name, cv_fold)
    if not os.path.exists(file_name):
        return None
    with open(file_name, 'rb') as f:      
        cv_estim = pickle.load(f)
    return cv_estim


def analyze_gs(gs, fold, task=None):
    params = [p.split('__')[-1] for p in gs.best_params_.keys()]
    linear_scale = ['lamb']


    for p in params:
        if p in linear_scale:
            utils.plot_param(gs, p, fold, logscale=False)
        else:
            utils.plot_param(gs, p, fold, logscale=True)


def evaluate_estim_on_problem(estim_name, problem_name, scores_fun,  task_name=None, project_name=None, force_refit=False):

    # task = Task.init(project_name=project_name, task_name=task_name)
    task = None

    X, y, cv_outer, cv_inner, task_info = get_data(problem_name=problem_name, task=task)

    estim = get_estim(estim_name)
    datascaler = get_datascaler(task)
    pipe = generate_pipeline(estim, datascaler)

    param_grid = get_hyperparameters(estim_name, task)
    ic(param_grid)

    scorer_val = get_scorer_val(task)

    score_test = macro_recall

    len_keys = len(list(scores_fun.keys()))
    scores_l = dict(zip( list(scores_fun.keys()), [[] for _ in range(len_keys)]  ))
    for fold, (trainval_idx, test_idx) in enumerate(cv_outer.split(X, X[:,task_info])):
        ic(fold)
        X_trainval, X_test = X[trainval_idx], X[test_idx]
        y_trainval, y_test = y[trainval_idx].ravel(), y[test_idx].ravel()

        gs = load_cv_results(estim_name=estim_name, problem_name=PROBLEM_NAME, cv_fold=fold, results_dir=RESULTS_DIR)
        if force_refit or gs is None:
            gs = generate_gridsearch(pipe, cv_inner, param_grid, scorer_val)
            gs.fit(X_trainval, y_trainval)

            save_cv_results(gs, estim_name=estim_name, problem_name=PROBLEM_NAME, cv_fold=fold, results_dir=RESULTS_DIR)
        
        # df = pd.DataFrame.from_dict(gs.cv_results_)
        # ic(df)
        if task is not None:
            analyze_gs(gs, fold, task=task)
        
        # ic(gs.best_estimator_)

        pred = gs.predict(X_test)
        
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

strategies_l = ['hs', 'cvx', 'gl']
# kernels_l = ['rbf']
# strategies_l = [0]
estim_names_l = ['{}_nn'.format(s) for s in strategies_l]



problem_names_l = [PROBLEM_NAME]

scores_fun = {'macro_f1': macro_f1, 'macro_recall': macro_recall, 'macro_precision': macro_precision, 'accuracy': accuracy_score}

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


    task = Task.init(project_name=pname, task_name='results')
    df.index.name = "model"
    Logger.current_logger().report_table(
        "table results - {}".format(pname), 
        version, 
        table_plot=df
    )
    task.close()
