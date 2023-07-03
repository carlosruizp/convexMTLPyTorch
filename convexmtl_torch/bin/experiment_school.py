from clearml import Task, Logger


from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import StratifiedKFold

import itertools

from convexmtl_torch.data.DataLoader import DataLoader
from convexmtl_torch.model.HardSharingMTLPytorchModel import HardSharingMTLPytorchRegressor
from convexmtl_torch.model.ConvexMTLPytorchModel import ConvexMTLPytorchRegressor
from convexmtl_torch.model.GraphLaplacianMTLPytorchModel import GraphLaplacianMTLPytorchRegressor
from convexmtl_torch.preprocessing import MTLStandardScaler



from icecream import ic

import pandas as pd
import numpy as np

import utils

import joblib
import os



# class utils.ClearMLLogger():
#     def __init__(self, task: Task):
#         self.task = task

#     def connect(self, param_f):
#         ret = param_f






def get_estim(estim_name, task=None, progress_bar=False):
    # estimator
    strategy = estim_name.split('_')[0]
    n_hidden = int(estim_name.split('_')[2])
    if strategy == 'hs':
        estim = HardSharingMTLPytorchRegressor(enable_progress_bar=progress_bar, epochs=utils.MAX_EPOCHS)
    elif strategy == 'cvx-trainable':
        estim = ConvexMTLPytorchRegressor(lambda_trainable=True, specific_lambda=False, enable_progress_bar=progress_bar, epochs=utils.MAX_EPOCHS, n_hidden_common=n_hidden)
    elif strategy == 'gl':
        estim = GraphLaplacianMTLPytorchRegressor(adj_trainable=True, enable_progress_bar=progress_bar, epochs=utils.MAX_EPOCHS)
    elif strategy == 'cvx':
        estim = ConvexMTLPytorchRegressor(lambda_trainable=False, specific_lambda=False, enable_progress_bar=progress_bar, epochs=utils.MAX_EPOCHS, n_hidden_common=n_hidden)
    elif strategy == 'common':
        estim = ConvexMTLPytorchRegressor(lambda_trainable=False, specific_lambda=False, lamb=1, enable_progress_bar=progress_bar, epochs=utils.MAX_EPOCHS, n_hidden_common=n_hidden)
    elif strategy == 'ind':
        estim = ConvexMTLPytorchRegressor(lambda_trainable=False, specific_lambda=False, lamb=0, enable_progress_bar=progress_bar, epochs=utils.MAX_EPOCHS, n_hidden_common=n_hidden)
    
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
    data_loader = DataLoader(utils.DATA_DIR)
    df_data, df_target, cv_outer, cv_inner, task_info = data_loader.load_dataset(problem_name)
    
    # if task is not None:
    #     task.upload_artifact(name='data', artifact_object=utils.DATA_DIR)
    return df_data, df_target, cv_outer, cv_inner, task_info

def get_scorer_val(task):
    # estimator
    scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    if task is not None:
        task.upload_artifact(name='scorer_val', artifact_object=scorer)
    return scorer





def evaluate_estim_on_problem(estim_name, problem_name, scores_fun,  task_name=None, project_name=None, force_refit=False):

    if utils.CLEARML:
        task = Task.init(project_name=project_name, task_name=task_name)
    else:
        task = None

    X, y, cv_outer, cv_inner, task_info = get_data(problem_name=problem_name, task=task)

    estim = get_estim(estim_name)
    ic(estim.get_params())
    datascaler = get_datascaler(task)
    targetscaler = get_targetscaler(task)
    pipe = utils.generate_pipeline(estim, datascaler, targetscaler)
    ic(pipe.get_params())
    param_grid = utils.get_hyperparameters(estim_name, task)
    ic(param_grid)

    scorer_val = get_scorer_val(task)

    score_test = mean_absolute_error

    len_keys = len(list(scores_fun.keys()))
    scores_l = dict(zip( list(scores_fun.keys()), [[] for _ in range(len_keys)]  ))
    for fold, (trainval_idx, test_idx) in enumerate(cv_outer.split(X, X[:,task_info])):
        ic(fold)
        X_trainval, X_test = X[trainval_idx], X[test_idx]
        y_trainval, y_test = y[trainval_idx].ravel(), y[test_idx].ravel()

        ic(X_trainval)

        gs = utils.load_cv_results(estim_name=estim_name, problem_name=problem_name, cv_fold=fold, results_dir=utils.RESULTS_DIR, retrain=force_refit)
        ic(gs)
        ic(force_refit)
        if gs is None:
            if isinstance(cv_inner, StratifiedKFold):
                cv_inner_splits = list(cv_inner.split(X_trainval, X_trainval[:,task_info]))
            else:
                cv_inner_splits = cv_inner
            gs = utils.generate_gridsearch(pipe, cv_inner_splits, param_grid, scorer_val)
            gs.fit(X_trainval, y_trainval)
            print('FIT DONE')
            utils.save_cv_results(gs, estim_name=estim_name, problem_name=problem_name, cv_fold=fold, results_dir=utils.RESULTS_DIR)
        
        # df = pd.DataFrame.from_dict(gs.cv_results_)
        # ic(df)
        utils.analyze_gs(gs, problem_name, estim_name, fold, task=task)

        best = clone(pipe)
        logs_dir = 'logs/{}/{}/{}'.format(problem_name, estim_name, fold)
        os.makedirs(logs_dir, exist_ok=True)
        aux_params = {'train_mode': 'lightning', 'log_dir': logs_dir}
        best_params = {**{'regressor__estim__{}'.format(k): v for k, v in aux_params.items()}, 
                       **gs.best_params_}
        best.set_params(**best_params)
        # best.set_params(regressor__estim__train_mode='lightning')
        best.fit(X_trainval, y_trainval) 

        if 'cvx-trainable' in estim_name:
            utils.plot_lambda_hist(best.regressor_.steps[1][1], cv_fold=fold, task=task,
                                   problem_name=problem_name, model_name=estim_name)

        
        ic(X_test)
        ic(X_test.shape)
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

strategies_l = ['ind', 'common', 'cvx', 'cvx-trainable', 'hs', 'gl'] # ['cvx', 'common', 'ind'] # ['hs', 'cvx', 'gl']
nhid_l = [4, 8]

estim_names_l = ['{}_nn_{}'.format(s, nhid) for nhid in nhid_l for s in strategies_l]



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

    if utils.CLEARML:
        task = Task.init(project_name='hais23_'+pname, task_name='results')
    else:
        task = None

    df.index.name = "model"

    if task is not None:           
        
        Logger.current_logger().report_table(
            "table results - {}".format(pname), 
            version, 
            table_plot=df
        )
        task.close()
    else:
        df.to_csv('{}/results__{}.csv'.format(utils.RESULTS_DIR, pname))
