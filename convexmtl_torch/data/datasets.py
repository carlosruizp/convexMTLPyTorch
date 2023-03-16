from .variations_mnist import *
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import numpy as np
import pandas as pd

def load_dataset_colored_mnist(task_type='predefined',
                                save_pd=False, seed=42,
                                max_size=None,
                                data_dir='.'):
    """Load colored mnist"""

    colored_mnist = ColoredMNIST(data_dir, None, None)
    envs = [0.1, 0.2, 0.9]
    X_l = []
    y_l = []
    t_l = []
    for i, (X_tensor, y_tensor) in enumerate(colored_mnist):
        X, y = X_tensor.numpy()[:max_size], y_tensor.numpy()[:max_size]
        n = X.shape[0]
        X_table = X.reshape((n, -1))
        l = len(y)
        t = np.array([envs[i]]*l)
        X_l.append(X_table)
        y_l.append(y)
        t_l.append(t)
    X = np.concatenate(X_l, axis=0)
    y = np.concatenate(y_l, axis=0)
    t = np.concatenate(t_l, axis=0)

    # append task
    X = np.concatenate([X, t[:, None]], axis=1)
    
    n_splits_inner = 5
    n_splits_outer = 1
    inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=seed)
    outer_cv = StratifiedShuffleSplit(n_splits=n_splits_outer, test_size=0.3, random_state=seed)
    task_info = -1

    return X, y, inner_cv, outer_cv, task_info 

def load_dataset_rotated_mnist(task_type='predefined',
                                save_pd=False, seed=42,
                                max_size=None,
                                data_dir='.'):
    """Load rotated mnist"""

    rotated_mnist = RotatedMNIST(data_dir, None, None)
    envs = ['0', '15', '30', '45', '60', '75']
    X_l = []
    y_l = []
    t_l = []
    for i, (X_tensor, y_tensor) in enumerate(rotated_mnist):
        X, y = X_tensor.numpy()[:max_size], y_tensor.numpy()[:max_size]
        n = X.shape[0]
        X_table = X.reshape((n, -1))
        l = len(y)
        t = np.array([envs[i]]*l)
        X_l.append(X_table)
        y_l.append(y)
        t_l.append(t)
    X = np.concatenate(X_l, axis=0)
    y = np.concatenate(y_l, axis=0)
    t = np.concatenate(t_l, axis=0)

    # append task
    X = np.concatenate([X, t[:, None]], axis=1)

    n_splits_inner = 5
    n_splits_outer = 1
    inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=seed)
    outer_cv = StratifiedShuffleSplit(n_splits=n_splits_outer, test_size=0.3, random_state=seed)
    task_info = -1

    return X, y, inner_cv, outer_cv, task_info 


def load_dataset_variations_mnist(task_type='predefined',
                                  save_pd=False, seed=42,
                                  max_size=None,
                                  data_dir='.'):
    """Load rotated mnist"""

    rotated_mnist = MNISTVariations(data_dir, None, None)
    envs = ['standard', 'images', 'random']
    X_l = []
    y_l = []
    t_l = []
    for i, (X_tensor, y_tensor) in enumerate(rotated_mnist):
        X, y = X_tensor.numpy()[:max_size], y_tensor.numpy()[:max_size]
        ic(X.shape)
        n = X.shape[0]
        X_table = X.reshape((n, -1))
        l = len(y)
        t = np.array([envs[i]]*l)
        X_l.append(X_table)
        y_l.append(y)
        t_l.append(t)
    X = np.concatenate(X_l, axis=0)
    y = np.concatenate(y_l, axis=0)
    t = np.concatenate(t_l, axis=0)

    # append task
    X = np.concatenate([X, t[:, None]], axis=1)

    n_splits_inner = 5
    n_splits_outer = 1
    inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=seed)
    outer_cv = StratifiedShuffleSplit(n_splits=n_splits_outer, test_size=0.3, random_state=seed)
    task_info = -1

    return X, y, inner_cv, outer_cv, task_info 


def load_dataset_variations_mnist_mini(task_type='predefined',
                                  save_pd=False, seed=42,
                                  max_size=10,
                                  data_dir='.'):
    """Load rotated mnist"""

    rotated_mnist = MNISTVariations(data_dir, None, None)
    envs = ['standard', 'images', 'random']
    X_l = []
    y_l = []
    t_l = []
    for i, (X_tensor, y_tensor) in enumerate(rotated_mnist):
        X, y = X_tensor.numpy()[:max_size], y_tensor.numpy()[:max_size]
        ic(X.shape)
        n = X.shape[0]
        X_table = X.reshape((n, -1))
        l = len(y)
        t = np.array([envs[i]]*l)
        X_l.append(X_table)
        y_l.append(y)
        t_l.append(t)
    X = np.concatenate(X_l, axis=0)
    y = np.concatenate(y_l, axis=0)
    t = np.concatenate(t_l, axis=0)

    # append task
    X = np.concatenate([X, t[:, None]], axis=1)

    n_splits_inner = 5
    n_splits_outer = 1
    inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=seed)
    outer_cv = StratifiedShuffleSplit(n_splits=n_splits_outer, test_size=0.3, random_state=seed)
    task_info = -1

    return X, y, inner_cv, outer_cv, task_info 



def load_dataset_colored_fashionmnist(task_type='predefined',
                                save_pd=False, seed=42,
                                max_size=None,
                                data_dir='.'):
    """Load colored mnist"""

    colored_mnist = ColoredFashionMNIST(data_dir, None, None)
    envs = [0.1, 0.2, 0.9]
    X_l = []
    y_l = []
    t_l = []
    for i, (X_tensor, y_tensor) in enumerate(colored_mnist):
        X, y = X_tensor.numpy()[:max_size], y_tensor.numpy()[:max_size]
        n = X.shape[0]
        X_table = X.reshape((n, -1))
        l = len(y)
        t = np.array([envs[i]]*l)
        X_l.append(X_table)
        y_l.append(y)
        t_l.append(t)
    X = np.concatenate(X_l, axis=0)
    y = np.concatenate(y_l, axis=0)
    t = np.concatenate(t_l, axis=0)

    # append task
    X = np.concatenate([X, t[:, None]], axis=1)
    
    n_splits_inner = 5
    n_splits_outer = 1
    inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=seed)
    outer_cv = StratifiedShuffleSplit(n_splits=n_splits_outer, test_size=0.3, random_state=seed)
    task_info = -1

    return X, y, inner_cv, outer_cv, task_info 

def load_dataset_rotated_fashionmnist(task_type='predefined',
                                save_pd=False, seed=42,
                                max_size=None,
                                data_dir='.'):
    """Load rotated mnist"""

    rotated_mnist = RotatedFashionMNIST(data_dir, None, None)
    envs = ['0', '15', '30', '45', '60', '75']
    X_l = []
    y_l = []
    t_l = []
    for i, (X_tensor, y_tensor) in enumerate(rotated_mnist):
        X, y = X_tensor.numpy()[:max_size], y_tensor.numpy()[:max_size]
        n = X.shape[0]
        X_table = X.reshape((n, -1))
        l = len(y)
        t = np.array([envs[i]]*l)
        X_l.append(X_table)
        y_l.append(y)
        t_l.append(t)
    X = np.concatenate(X_l, axis=0)
    y = np.concatenate(y_l, axis=0)
    t = np.concatenate(t_l, axis=0)

    # append task
    X = np.concatenate([X, t[:, None]], axis=1)

    n_splits_inner = 5
    n_splits_outer = 1
    inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=seed)
    outer_cv = StratifiedShuffleSplit(n_splits=n_splits_outer, test_size=0.3, random_state=seed)
    task_info = -1

    return X, y, inner_cv, outer_cv, task_info 


def load_dataset_variations_fashionmnist(task_type='predefined',
                                save_pd=False, seed=42,
                                max_size=None,
                                data_dir='.'):
    """Load variations mnist"""

    variations_mnist = FashionMNISTVariations(data_dir, None, None)
    envs = ['standard', 'images', 'random']
    X_l = []
    y_l = []
    t_l = []
    for i, (X_tensor, y_tensor) in enumerate(variations_mnist):
        X, y = X_tensor.numpy()[:max_size], y_tensor.numpy()[:max_size]
        n = X.shape[0]
        X_table = X.reshape((n, -1))
        l = len(y)
        t = np.array([envs[i]]*l)
        X_l.append(X_table)
        y_l.append(y)
        t_l.append(t)
    X = np.concatenate(X_l, axis=0)
    y = np.concatenate(y_l, axis=0)
    t = np.concatenate(t_l, axis=0)

    # append task
    X = np.concatenate([X, t[:, None]], axis=1)

    n_splits_inner = 5
    n_splits_outer = 1
    inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=seed)
    outer_cv = StratifiedShuffleSplit(n_splits=n_splits_outer, test_size=0.3, random_state=seed)
    task_info = -1

    return X, y, inner_cv, outer_cv, task_info


def load_dataset_school(data_dir):
        """Load dataframe from csv for mnist variations."""

        path = '{}/school/'.format(data_dir)        
        df_data = pd.read_csv('{}/school_data.csv'.format(path), index_col=0)
        df_target = pd.read_csv('{}/school_target.csv'.format(path), index_col=0)

        outer_cv = StratifiedKFold(shuffle=True)
        inner_cv = StratifiedKFold(shuffle=True)
        task_info=-1

        return df_data.values, df_target.values, inner_cv, outer_cv, task_info


def load_dataset_school_mini(data_dir, max_size=400):
        """Load dataframe from csv for mnist variations."""

        path = '{}/school/'.format(data_dir)        
        df_data = pd.read_csv('{}/school_data.csv'.format(path), index_col=0)[:max_size]
        df_target = pd.read_csv('{}/school_target.csv'.format(path), index_col=0)[:max_size]

        outer_cv = StratifiedKFold(shuffle=True)
        inner_cv = StratifiedKFold(shuffle=True)
        task_info=-1

        return df_data.values, df_target.values, inner_cv, outer_cv, task_info

def load_dataset_landmine(data_dir):
    """Load dataframe from csv for landmine."""

    path = '{}/landmine/'.format(data_dir)

    df_data = pd.read_csv('{}/landmine_data.csv'.format(path), index_col=0)
    df_target = pd.read_csv('{}/landmine_target.csv'.format(path), index_col=0)

    outer_cv = StratifiedKFold(shuffle=True)
    inner_cv = StratifiedKFold(shuffle=True)
    task_info=-1

    return df_data.values, df_target.values, inner_cv, outer_cv, task_info

def load_dataset_landmine_mini(data_dir, max_size=400):
    """Load dataframe from csv for landmine."""

    path = '{}/landmine/'.format(data_dir)

    df_data = pd.read_csv('{}/landmine_data.csv'.format(path), index_col=0)[:max_size]
    df_target = pd.read_csv('{}/landmine_target.csv'.format(path), index_col=0)[:max_size]

    outer_cv = StratifiedKFold(shuffle=True)
    inner_cv = StratifiedKFold(shuffle=True)
    task_info=-1

    return df_data.values, df_target.values, inner_cv, outer_cv, task_info
