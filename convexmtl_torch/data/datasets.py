from .variations_mnist import *
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
# from .MTLSyntheticDataset import MTLClustersARegression, MTLClustersBRegression, MTLClustersCRegression, MTLFunctionsRegression, MTLCommonRegression, MTLIndependentRegression
from convexmtl_torch.data import MTLSyntheticDataset

import numpy as np
import pandas as pd


def _load_dataset_mnist_traintest(env_mnist_l, envs, seed, max_size, data_dir):
    X_tr_l = []
    y_tr_l = []
    t_tr_l = []
    X_te_l = []
    y_te_l = []
    t_te_l = []
    for i, ( (X_tr_tensor, X_te_tensor), (y_tr_tensor, y_te_tensor)) in enumerate(env_mnist_l):
        # train
        X_tr, y_tr = X_tr_tensor.numpy()[:max_size], y_tr_tensor.numpy()[:max_size]
        ic(X_tr.shape, y_tr.shape)
        X_tr_table = X_tr.reshape((X_tr.shape[0], -1))
        t_tr = np.array([envs[i]] * len(y_tr))
        X_tr_l.append(X_tr_table)
        y_tr_l.append(y_tr)
        t_tr_l.append(t_tr)
        # test
        X_te, y_te = X_te_tensor.numpy()[:max_size], y_te_tensor.numpy()[:max_size]
        X_te_table = X_te.reshape((X_te.shape[0], -1))
        t_te = np.array([envs[i]] * len(y_te))
        X_te_l.append(X_te_table)
        y_te_l.append(y_te)
        t_te_l.append(t_te)

    X_tr = np.concatenate(X_tr_l, axis=0)
    y_tr = np.concatenate(y_tr_l, axis=0)
    t_tr = np.concatenate(t_tr_l, axis=0)
    # append task
    X_tr = np.concatenate([X_tr, t_tr[:, None]], axis=1)

    X_te = np.concatenate(X_te_l, axis=0)
    y_te = np.concatenate(y_te_l, axis=0)
    t_te = np.concatenate(t_te_l, axis=0)
    # append task
    X_te = np.concatenate([X_te, t_te[:, None]], axis=1)

    X = np.concatenate([X_tr, X_te], axis=0)
    y = np.concatenate([y_tr, y_te], axis=0)
    test_fold = np.concatenate([-np.ones(len(y_tr)), np.zeros(len(y_te))], axis=0) 

    n_splits_inner = 5
    n_splits_outer = None
    inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=seed)
    outer_cv = PredefinedSplit(test_fold=test_fold)
    task_info = -1

    return X, y, outer_cv, inner_cv, task_info 

# def load_dataset_colored_mnist(seed=42,
#                                 max_size=None,
#                                 data_dir='.'):
#     """Load colored mnist"""

#     colored_mnist = ColoredMNIST(data_dir, None, None)
#     envs = [0.1, 0.2, 0.9]
#     X_l = []
#     y_l = []
#     t_l = []
#     for i, (X_tensor, y_tensor) in enumerate(colored_mnist):
#         X, y = X_tensor.numpy()[:max_size], y_tensor.numpy()[:max_size]
#         n = X.shape[0]
#         X_table = X.reshape((n, -1))
#         l = len(y)
#         t = np.array([envs[i]]*l)
#         X_l.append(X_table)
#         y_l.append(y)
#         t_l.append(t)
#     X = np.concatenate(X_l, axis=0)
#     y = np.concatenate(y_l, axis=0)
#     t = np.concatenate(t_l, axis=0)

#     # append task
#     X = np.concatenate([X, t[:, None]], axis=1)
    
#     n_splits_inner = 5
#     n_splits_outer = 5
#     inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=seed)
#     # outer_cv = StratifiedShuffleSplit(n_splits=n_splits_outer, test_size=0.3, random_state=seed)
#     outer_cv = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=seed)
#     task_info = -1

#     return X, y, outer_cv, inner_cv, task_info 

# def load_dataset_rotated_mnist(task_type='predefined',
#                                 save_pd=False, seed=42,
#                                 max_size=None,
#                                 data_dir='.'):
#     """Load rotated mnist"""

#     rotated_mnist = RotatedMNIST(data_dir, None, None)
#     envs = ['0', '15', '30', '45', '60', '75']
#     X_l = []
#     y_l = []
#     t_l = []
#     for i, (X_tensor, y_tensor) in enumerate(rotated_mnist):
#         X, y = X_tensor.numpy()[:max_size], y_tensor.numpy()[:max_size]
#         n = X.shape[0]
#         X_table = X.reshape((n, -1))
#         l = len(y)
#         t = np.array([envs[i]]*l)
#         X_l.append(X_table)
#         y_l.append(y)
#         t_l.append(t)
#     X = np.concatenate(X_l, axis=0)
#     y = np.concatenate(y_l, axis=0)
#     t = np.concatenate(t_l, axis=0)

#     # append task
#     X = np.concatenate([X, t[:, None]], axis=1)

#     n_splits_inner = 5
#     n_splits_outer = 5
#     inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=seed)
#     # outer_cv = StratifiedShuffleSplit(n_splits=n_splits_outer, test_size=0.3, random_state=seed)
#     outer_cv = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=seed)
#     task_info = -1

#     return X, y, outer_cv, inner_cv, task_info 

def load_dataset_rotated_mnist(task_type='predefined',
                                save_pd=False, seed=42,
                                max_size=None,
                                data_dir='.'):
    """Load rotated mnist"""
    rotated_mnist = RotatedMNISTTrTe(data_dir, None, None)
    envs = ['0', '15', '30', '45', '60', '75']
    return _load_dataset_mnist_traintest(rotated_mnist, envs, seed, max_size, data_dir)


def load_dataset_rotated_mnist_mini(task_type='predefined',
                                save_pd=False, seed=42,
                                max_size=200,
                                data_dir='.'):
    """Load rotated mnist"""

    rotated_mnist = RotatedMNISTTrTe(data_dir, None, None)
    envs = ['0', '15', '30', '45', '60', '75']
    return _load_dataset_mnist_traintest(rotated_mnist, envs, seed, max_size, data_dir)


def load_dataset_variations_mnist(task_type='predefined',
                                save_pd=False, seed=42,
                                max_size=None,
                                data_dir='.'):
    """Load variations mnist"""

    variations_mnist = MNISTVariationsTrTe(data_dir, None, None)
    envs = ['0', '15', '30', '45', '60', '75']
    return _load_dataset_mnist_traintest(variations_mnist, envs, seed, max_size, data_dir) 


def load_dataset_variations_mnist_mini(task_type='predefined',
                                save_pd=False, seed=42,
                                max_size=200,
                                data_dir='.'):
    """Load variations mnist"""

    variations_mnist = MNISTVariationsTrTe(data_dir, None, None)
    envs = ['0', '15', '30', '45', '60', '75']
    return _load_dataset_mnist_traintest(variations_mnist, envs, seed, max_size, data_dir)


# def load_dataset_variations_mnist(task_type='predefined',
#                                   save_pd=False, seed=42,
#                                   max_size=None,
#                                   data_dir='.'):
#     """Load rotated mnist"""

#     rotated_mnist = MNISTVariations(data_dir, None, None)
#     envs = ['standard', 'images', 'random']
#     X_l = []
#     y_l = []
#     t_l = []
#     for i, (X_tensor, y_tensor) in enumerate(rotated_mnist):
#         X, y = X_tensor.numpy()[:max_size], y_tensor.numpy()[:max_size]
#         ic(X.shape)
#         n = X.shape[0]
#         X_table = X.reshape((n, -1))
#         l = len(y)
#         t = np.array([envs[i]]*l)
#         X_l.append(X_table)
#         y_l.append(y)
#         t_l.append(t)
#     X = np.concatenate(X_l, axis=0)
#     y = np.concatenate(y_l, axis=0)
#     t = np.concatenate(t_l, axis=0)

#     # append task
#     X = np.concatenate([X, t[:, None]], axis=1)

#     n_splits_inner = 5
#     n_splits_outer = 5
#     inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=seed)
#     # outer_cv = StratifiedShuffleSplit(n_splits=n_splits_outer, test_size=0.3, random_state=seed)
#     outer_cv = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=seed)
#     task_info = -1

#     return X, y, outer_cv, inner_cv, task_info 




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
    n_splits_outer = 5
    inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=seed)
    # outer_cv = StratifiedShuffleSplit(n_splits=n_splits_outer, test_size=0.3, random_state=seed)
    outer_cv = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=seed)
    task_info = -1

    return X, y, outer_cv, inner_cv, task_info 


def load_dataset_rotated_fashionmnist(task_type='predefined',
                                save_pd=False, seed=42,
                                max_size=None,
                                data_dir='.'):
    """Load rotated fashionmnist"""
    rotated_fashionmnist = RotatedFashionMNISTTrTe(data_dir, None, None)
    envs = ['0', '15', '30', '45', '60', '75']
    return _load_dataset_mnist_traintest(rotated_fashionmnist, envs, seed, max_size, data_dir)


def load_dataset_rotated_fashionmnist_mini(task_type='predefined',
                                save_pd=False, seed=42,
                                max_size=200,
                                data_dir='.'):
    """Load rotated fashionmnist"""

    rotated_fashionmnist = RotatedFashionMNISTTrTe(data_dir, None, None)
    envs = ['0', '15', '30', '45', '60', '75']
    return _load_dataset_mnist_traintest(rotated_fashionmnist, envs, seed, max_size, data_dir)


def load_dataset_variations_fashionmnist(task_type='predefined',
                                save_pd=False, seed=42,
                                max_size=None,
                                data_dir='.'):
    """Load variations fashionmnist"""

    variations_fashionmnist = FashionMNISTVariationsTrTe(data_dir, None, None)
    envs = ['0', '15', '30', '45', '60', '75']
    return _load_dataset_mnist_traintest(variations_fashionmnist, envs, seed, max_size, data_dir) 


def load_dataset_variations_fashionmnist_mini(task_type='predefined',
                                save_pd=False, seed=42,
                                max_size=200,
                                data_dir='.'):
    """Load variations fashionmnist"""

    variations_fashionmnist = FashionMNISTVariationsTrTe(data_dir, None, None)
    envs = ['0', '15', '30', '45', '60', '75']
    return _load_dataset_mnist_traintest(variations_fashionmnist, envs, seed, max_size, data_dir)


def load_dataset_rotated_fashionmnist(task_type='predefined',
                                save_pd=False, seed=42,
                                max_size=None,
                                data_dir='.'):
    """Load rotated fashionmnist"""

    rotated_fashionmnist = RotatedFashionMNISTTrTe(data_dir, None, None)
    envs = ['0', '15', '30', '45', '60', '75']
    return _load_dataset_mnist_traintest(rotated_fashionmnist, envs, seed, max_size, data_dir) 


# def load_dataset_rotated_fashionmnist(task_type='predefined',
#                                 save_pd=False, seed=42,
#                                 max_size=None,
#                                 data_dir='.'):
#     """Load rotated mnist"""

#     rotated_mnist = RotatedFashionMNIST(data_dir, None, None)
#     envs = ['0', '15', '30', '45', '60', '75']
#     X_l = []
#     y_l = []
#     t_l = []
#     for i, (X_tensor, y_tensor) in enumerate(rotated_mnist):
#         X, y = X_tensor.numpy()[:max_size], y_tensor.numpy()[:max_size]
#         n = X.shape[0]
#         X_table = X.reshape((n, -1))
#         l = len(y)
#         t = np.array([envs[i]]*l)
#         X_l.append(X_table)
#         y_l.append(y)
#         t_l.append(t)
#     X = np.concatenate(X_l, axis=0)
#     y = np.concatenate(y_l, axis=0)
#     t = np.concatenate(t_l, axis=0)

#     # append task
#     X = np.concatenate([X, t[:, None]], axis=1)

#     n_splits_inner = 5
#     n_splits_outer = 5
#     inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=seed)
#     # outer_cv = StratifiedShuffleSplit(n_splits=n_splits_outer, test_size=0.3, random_state=seed)
#     outer_cv = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=seed)
#     task_info = -1

#     return X, y, outer_cv, inner_cv, task_info 


# def load_dataset_variations_fashionmnist(task_type='predefined',
#                                 save_pd=False, seed=42,
#                                 max_size=None,
#                                 data_dir='.'):
#     """Load variations mnist"""

#     variations_mnist = FashionMNISTVariations(data_dir, None, None)
#     envs = ['standard', 'images', 'random']
#     X_l = []
#     y_l = []
#     t_l = []
#     for i, (X_tensor, y_tensor) in enumerate(variations_mnist):
#         X, y = X_tensor.numpy()[:max_size], y_tensor.numpy()[:max_size]
#         n = X.shape[0]
#         X_table = X.reshape((n, -1))
#         l = len(y)
#         t = np.array([envs[i]]*l)
#         X_l.append(X_table)
#         y_l.append(y)
#         t_l.append(t)
#     X = np.concatenate(X_l, axis=0)
#     y = np.concatenate(y_l, axis=0)
#     t = np.concatenate(t_l, axis=0)

#     # append task
#     X = np.concatenate([X, t[:, None]], axis=1)

#     n_splits_inner = 5
#     n_splits_outer = 5
#     inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=seed)
#     # outer_cv = StratifiedShuffleSplit(n_splits=n_splits_outer, test_size=0.3, random_state=seed)
#     outer_cv = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=seed)
#     task_info = -1

#     return X, y, outer_cv, inner_cv, task_info


# def load_dataset_school(data_dir, random_state=42):
#         """Load dataframe from csv for mnist variations."""

#         path = '{}/school/'.format(data_dir)        
#         df_data = pd.read_csv('{}/school_data.csv'.format(path), index_col=0)
#         df_target = pd.read_csv('{}/school_target.csv'.format(path), index_col=0)

#         outer_cv = StratifiedKFold(random_state=random_state, shuffle=True)
#         inner_cv = StratifiedKFold(random_state=random_state, shuffle=True)
#         task_info=-1

#         return df_data.values, df_target.values, outer_cv, inner_cv, task_info


def load_dataset_school(data_dir, max_size=None, random_state=42):
        """Load dataframe from csv for mnist variations."""

        path = '{}/school/'.format(data_dir)        
        df_data = pd.read_csv('{}/school_data.csv'.format(path), index_col=0)[:max_size]# .sample(max_size, random_state=random_state)
        df_target = pd.read_csv('{}/school_target.csv'.format(path), index_col=0)[:max_size]# .sample(max_size, random_state=random_state)

        outer_cv = StratifiedKFold(random_state=random_state, shuffle=True)
        inner_cv = StratifiedKFold(random_state=random_state, shuffle=True)
        task_info=-1

        return df_data.values, df_target.values, outer_cv, inner_cv, task_info

# def load_dataset_landmine(data_dir, random_state=42):
#     """Load dataframe from csv for landmine."""

#     path = '{}/landmine/'.format(data_dir)

#     df_data = pd.read_csv('{}/landmine_data.csv'.format(path), index_col=0)
#     df_target = pd.read_csv('{}/landmine_target.csv'.format(path), index_col=0)

#     outer_cv = StratifiedKFold(random_state=random_state, shuffle=True)
#     inner_cv = StratifiedKFold(random_state=random_state, shuffle=True)
#     task_info=-1

#     return df_data.values, df_target.values, outer_cv, inner_cv, task_info

def load_dataset_landmine(data_dir, max_size=None, random_state=42):
    """Load dataframe from csv for landmine."""

    path = '{}/landmine/'.format(data_dir)    

    df_data = pd.read_csv('{}/landmine_data.csv'.format(path), index_col=0).sample(max_size, random_state=random_state)
    df_target = pd.read_csv('{}/landmine_target.csv'.format(path), index_col=0).sample(max_size, random_state=random_state)

    outer_cv = StratifiedKFold(random_state=random_state, shuffle=True)
    inner_cv = StratifiedKFold(random_state=random_state, shuffle=True)
    task_info=-1

    return df_data.values, df_target.values, outer_cv, inner_cv, task_info




def load_dataset_sarcos(data_dir, max_size=None, random_state=42):
    """Load dataframe from csv for sarcos."""
    
    file_train_data = '{}/sarcos/sarcos_train_data.csv'.format(data_dir)
    file_train_target = '{}/sarcos/sarcos_train_target.csv'.format(data_dir)
    file_test_data = '{}/sarcos/sarcos_test_data.csv'.format(data_dir)
    file_test_target = '{}/sarcos/sarcos_test_target.csv'.format(data_dir)
    ic(file_train_data)

    if os.path.exists(file_train_data) and \
    os.path.exists(file_train_target) and \
    os.path.exists(file_test_data) and \
    os.path.exists(file_test_target):
        df_train_data = pd.read_csv(file_train_data, index_col=0)
        df_train_target = pd.read_csv(file_train_target, index_col=0)
        df_test_data = pd.read_csv(file_test_data, index_col=0)
        df_test_target = pd.read_csv(file_test_target, index_col=0)
    else:
        raise NotImplementedError
    
    df_data = pd.concat([df_train_data, df_test_data], axis=0)
    ic(df_data)

    df_target = pd.concat([df_train_target, df_test_target], axis=0)
    ic(df_target)

    ic(df_data)
    ic(df_target)

    test_fold = np.concatenate([-np.ones(df_train_data.shape[0]), np.zeros(df_test_data.shape[0])], axis=0)

    inner_cv = StratifiedKFold(shuffle=True, random_state=random_state)
    outer_cv = PredefinedSplit(test_fold=test_fold)
    
    task_info = -1

    return df_data.values[:max_size], df_target.values[:max_size], outer_cv, inner_cv, task_info


def load_dataset_computer(data_dir, max_size=None, random_state=42):
    """Load dataframe from mat for computer dataset."""

    file_data = '{}/computer/computer_data.csv'.format(data_dir)
    file_target = '{}/computer/computer_target.csv'.format(data_dir)
    ic(file_data)
    if os.path.exists(file_data) and os.path.exists(file_target):
        df_data = pd.read_csv(file_data, index_col=0)
        df_target = pd.read_csv(file_target, index_col=0)
    else:    
        raise NotImplementedError

    outer_cv = StratifiedKFold(random_state=random_state, shuffle=True)
    inner_cv = StratifiedKFold(random_state=random_state, shuffle=True)
    task_info = -1

    return df_data.values[:max_size], df_target.values[:max_size], outer_cv, inner_cv, task_info


def load_dataset_parkinson(data_dir, max_size=None, random_state=42):
    """Load dataframe from txt for parkinson dataset."""
    
    file_data = '{}/parkinson/parkinson_data.csv'.format(data_dir)
    file_target = '{}/parkinson/parkinson_target.csv'.format(data_dir)
    if os.path.exists(file_data) and os.path.exists(file_target):
        df_data = pd.read_csv(file_data, index_col=0)
        df_target = pd.read_csv(file_target, index_col=0)
    else:
        raise NotImplementedError

    outer_cv = StratifiedKFold(random_state=random_state, shuffle=True)
    inner_cv = StratifiedKFold(random_state=random_state, shuffle=True)
    task_info = -1

    return df_data.values[:max_size], df_target.values[:max_size], outer_cv, inner_cv, task_info


def load_synthetic_dataset(dataset_name, max_size=None, random_state=42, **kwargs):
    problem_, suffix_ = dataset_name.split('-')
    ic(problem_)
    ic(suffix_)
    problem = problem_[0].upper() + problem_[1:]
    reg = (suffix_ == 'reg')
    if reg is True:
        suffix_class = 'Regression'
    else:
        suffix_class = 'Classification'
    data_class = getattr(MTLSyntheticDataset, 'MTL{}{}'.format(problem, suffix_class))
    mtlds = data_class(random_state=random_state, **kwargs)
    X, y = mtlds.X, mtlds.y

    if y.ndim == 2:
        y_2d = y
    else:
        y_2d = y[:, None]

    df_data = pd.DataFrame(data=X)
    df_target = pd.DataFrame(data=y_2d)

    outer_cv = StratifiedKFold(random_state=random_state, shuffle=True)
    inner_cv = StratifiedKFold(random_state=random_state, shuffle=True)
    task_info = -1

    return df_data.values[:max_size], df_target.values[:max_size], outer_cv, inner_cv, task_info


    

