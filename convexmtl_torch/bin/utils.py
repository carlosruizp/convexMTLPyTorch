import numpy as np 
import matplotlib.pyplot as plt

from icecream import ic

import pandas as pd


def plot_param(gs, param, fold, logscale=False, task=None):
    params_l = np.array(list(gs.best_params_.keys()))
    find_param = np.array([param in p for p in params_l])
    if (find_param == False).all():
        # print('Parameter ', param, 'not in GS object parameters.')
        return

    param_keys_l = np.array(['param_{}'.format(p) for p in params_l])
    keys_df = np.append(param_keys_l, ['mean_test_score', 'std_test_score'])
    # print(keys_df)
    df = pd.DataFrame.from_dict(gs.cv_results_)
    df = df[keys_df]

    param_name = params_l[find_param][0]
    param_key = param_keys_l[find_param][0]

    l_param = np.unique(pd.DataFrame.from_dict(gs.cv_results_)[param_key]).astype(float)

    best_param = gs.best_params_[param_name]
    # ic(best_param)
    # print('Best {}: {}'.format(param, best_param))
    param_idx = (df[param_key] == best_param)

    # ic(gs.best_params_)


    best_params_idx_dic = {}
    for p, v in gs.best_params_.items():
        # print('Best {}: {}'.format(p, v))
        idx = (df['param_{}'.format(p)] == v)
        best_params_idx_dic[p] = idx

    ########## plotting hyperparameters by pairs
    plt.figure( figsize=(8, 5))

    # plt.subplot(1, 3, 1)
    plt.title('Fold {}. param {}: {:.2f}'.format(fold, param, best_param))

    plt.xticks(l_param, np.round(l_param, 2), rotation=45)
    plt.xlabel(param)
    plt.ylabel('cv score')
    if logscale:
        plt.xscale('log')

    idx = [True] * len(param_idx)
    for k, k_idx in best_params_idx_dic.items():
        if k != param_name:
            # ic(k, param_name)
            idx = np.logical_and(idx, k_idx)

    X_axis = list(df[idx][param_key].values)
    sample_score_mean = df[idx]['mean_test_score'].values
    sample_score_std = df[idx]['std_test_score'].values
    # print(sample_score_mean - sample_score_std)
    plt.fill_between(X_axis, sample_score_mean - sample_score_std,
                     sample_score_mean + sample_score_std, alpha=0.1)
    _ = plt.plot( df[idx][param_key], df[idx]['mean_test_score'], '-', 
                df[idx][param_key], df[idx]['mean_test_score'], '*')

    # ic(df[idx][param_key].shape, df[idx]['mean_test_score'].shape)
    # ic(df[idx][param_key], df[idx]['mean_test_score'])
    best_index = np.argmax(sample_score_mean)
    best_score = np.max(sample_score_mean)

    # # Plot a dotted vertical line at the best score for that scorer marked by x
    plt.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', marker='x', markeredgewidth=3, ms=8)

    # # Annotate the best score for that scorer
    plt.annotate("%0.2f" % best_score,
                (X_axis[best_index], best_score + 0.005))
    if task is not None:
        task.logger.report_matplotlib_figure(title="Parameter {}".format(param), series="GridSearch Analysis", iteration=0, figure=plt)
    
    # plt.legend()
    plt.show(block=False)
    plt.close()

    return best_param

def plot_lambda_hist(best_estim, cv_fold, task=None):

    if best_estim.lambda_trainable:
        len_hist = len(best_estim.get_lamb_history())
        plt.figure( figsize=(8, 5))
        plt.xticks(range(len_hist), range(len_hist), rotation=45)
        plt.xlabel('iter')
        plt.ylabel('lambda value')
        if best_estim.specific_lambda:
            for t in best_estim.lamb.keys():
                arr_aux = [best_estim.get_lamb_history()[i][t] for i in range(len(best_estim.get_lamb_history()))]
                plt.plot(range(len(best_estim.get_lamb_history())), arr_aux, label='task {}'.format(t))
        else:
            plt.plot(range(len(best_estim.get_lamb_history())), best_estim.get_lamb_history())

    if task is not None:
        task.logger.report_matplotlib_figure(title="Fold {}. lambda history".format(cv_fold), series="GridSearch Analysis", iteration=0, figure=plt)
    
    # plt.legend()
    plt.show(block=False)
    plt.close()