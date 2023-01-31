import numpy as np
import numpy_indexed as npi
from copy import deepcopy
import types


def _getTasksListandData(X, task_info):
    n, m = X.shape
    if isinstance(task_info, (list, np.ndarray)):
        return task_info, X
    elif isinstance(task_info, int):
        data_columns = np.delete(range(m), task_info)
        return X[:, task_info], X[:, data_columns]


def getDataTask(X, y, cv, t, task_info, predefined=False, include_task=False):
    """Return the data corresponding to the task t."""
    n, m = X.shape
    task_list, X_data = _getTasksListandData(X, task_info)
    unique, groups_idx = npi.group_by(task_list, np.arange(n))
    dic_tasks = dict(zip(unique, groups_idx))
    # print(t)
    # print(dic_tasks)
    if include_task:
        X_aux = X
    else:
        X_aux = X_data
    if len(dic_tasks[t]) == 1:
        row_idx = dic_tasks[t]
        X_g = X_aux[row_idx].reshape(1, -1)
    else:
        row_idx = dic_tasks[t]
        # print(row_idx)
        X_g = X_aux[row_idx]
    y_g = y[dic_tasks[t]]

    cv_task = []
    if predefined is True:
        # print(cv)
        aux_fold = cv.test_fold
        # print(aux_fold)
        aux_fold_task = aux_fold[dic_tasks[t]]
        train_idx_task = np.where(aux_fold_task == -1)[0]
        test_idx_task = np.where(aux_fold_task != -1)[0]
        cv_task.append((train_idx_task, test_idx_task))
    else:
        aux_fold = np.zeros(n)
        for train_idx, test_idx in cv:
            aux_fold[train_idx] = -1
            aux_fold[test_idx] = 1
            aux_fold_task = aux_fold[dic_tasks[t]]
            train_idx_task = np.where(aux_fold_task == -1)[0]
            test_idx_task = np.where(aux_fold_task != -1)[0]
            cv_task.append((train_idx_task, test_idx_task))

    return X_g, y_g, cv_task


def getDataTask_test(X, y, t, task_info, predefined=False):
    """Return the data corresponding to the task t."""
    n, m = X.shape
    task_list, X_data = _getTasksListandData(X, task_info)
    unique, groups_idx = npi.group_by(task_list, np.arange(n))
    dic_tasks = dict(zip(unique, groups_idx))
    # print(t)
    # print(dic_tasks)
    if len(dic_tasks[t]) == 1:
        row_idx = dic_tasks[t]
        X_g = X_data[row_idx].reshape(1, -1)
    else:
        row_idx = dic_tasks[t]
        # print(row_idx)
        X_g = X_data[row_idx]
    y_g = y[dic_tasks[t]]
    return X_g, y_g


def getUniqueTaskList(X, task_info):
    n, m = X.shape
    task_list, X_data = _getTasksListandData(X, task_info)
    unique, groups_idx = npi.group_by(task_list, np.arange(n))
    return unique, groups_idx
