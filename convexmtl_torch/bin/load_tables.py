import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from icecream import ic
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def macro_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average='macro')

def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


PREDICTIONS_DIR = 'predictions'
RESULTS_DIR = 'results'


def generate_results_df(models, problems, scores_fun, n_test, train=False):

    df_l = []

    for p in problems:
        scores_mean = {}
        scores_std = {}    
        for score_key, score_fun in scores_fun.items():
            scores_l = {k: [] for k in models}
            for m in models:            
                for i in range(n_test):
                    if train == True:
                        pred_fname = '{}/predtrain__{}__{}__{}.csv'.format(PREDICTIONS_DIR, p, m, i)
                    else:
                        pred_fname = '{}/pred__{}__{}__{}.csv'.format(PREDICTIONS_DIR, p, m, i)
                    pred = np.loadtxt(pred_fname)

                    if train == True:
                        ytest_fname = '{}/ytrain__{}__{}__{}.csv'.format(PREDICTIONS_DIR, p, m, i)
                    else:
                        ytest_fname = '{}/ytest__{}__{}__{}.csv'.format(PREDICTIONS_DIR, p, m, i)
                    y_test = np.loadtxt(ytest_fname)                
                    
                    score_value = score_fun(y_test, pred)
                    scores_l[m].append(score_value)

            scores_mean = {k: np.mean(s_l) for k, s_l in scores_l.items()}
            scores_std = {k: np.std(s_l) for k, s_l in scores_l.items()}

        # ic(scores_mean)

        df_mean = pd.DataFrame.from_dict(data=scores_mean, columns=['mean'], orient='index')  
        df_std = pd.DataFrame.from_dict(data=scores_std, columns=['std'], orient='index')

        df_p = pd.concat([df_mean, df_std], axis=1)

        df_l.append(df_p)

    df_concat = pd.concat(df_l, keys=problems, axis=1)
    df_concat_rank = add_ranks(df_concat, list(scores_fun.keys())[0])
    # ic(df_concat_rank)
    return df_concat_rank


def add_ranks(df_results, score_fun):
    columns = list(df_results)
    col_l = []
    # ic(df_results.keys())
    for i, col in enumerate(columns):
        if col[1] == 'mean':
            if score_fun in ['mse', 'mae']:
                ascending = True
            else:
                ascending = False
            rank = df_results[col].rank(ascending=ascending).rename((col[0], 'rank')).astype('int')
            col_l.append(df_results[col])
        else:
            col_l.append(df_results[col])
        if i%2 == 1:
            col_l.append(rank)

    df_ranked = pd.concat(col_l, axis=1)
    # ic(df_ranked)
    # df_ranked = df_ranked.rename(colnames, axis=1)
    return df_ranked


def save_matrix(models, problems, n_test=5):
    ncols = 3
    nrows = len(problems) / ncols
    
    fig, axn = plt.subplots(1, ncols, sharex=False, sharey=False)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])

    ic(axn)
        
    for k, p in enumerate(problems): 
        ic(k, p)
        for m in [m for m in models if 'gl' in m]:
            matrix_l = []
            for i in range(n_test):
                filename_matrix = '{}/matrix__{}__{}__{}.p'.format(RESULTS_DIR, p, m, i)
                try:
                    adj_matrix = joblib.load(filename_matrix)
                    matrix_l.append(adj_matrix)
                except:
                    break
            matrix_l = np.array(matrix_l)
            ic(matrix_l.shape)
            matrix_avg = np.mean(matrix_l, axis=0)
            ic(matrix_avg)
            if axn is None:
                plt.figure(figsize=(10, 8))
                ax = plt.gca()
            else:
                ax = axn[k]
            tasks = None
            sns.heatmap(matrix_avg, cmap='magma_r', vmin=0, vmax=1, ax=ax, 
                        xticklabels=False, yticklabels=False,
                        cbar_ax=None if i else cbar_ax,
                        cbar=i == 0,) # xticklabels=tasks, yticklabels=tasks, cmap='viridis'
            if axn is None:
                plt.tight_layout()
                filename_avgmatrix = '{}/matrixavg_{}__{}.pdf'.format(RESULTS_DIR, p, m)
                plt.savefig(filename_avgmatrix)

    fig.tight_layout(rect=[0, 0, .9, 1])
    filename_avgmatrix = '{}/matrixavg_full__{}.pdf'.format(RESULTS_DIR, m)
    plt.savefig(filename_avgmatrix)

strategies = ['ind', 'common', 'cvx', 'cvx-trainable', 'hs', 'gl', 'cvx-trainable-sp']
strategies = ['ind', 'common', 'cvx-trainable-sp', 'hs', 'gl']

models = ['{}_nn'.format(s) for s in strategies]
problems = ['variations_mnist', 'rotated_mnist', 'variations_fashionmnist', 'rotated_fashionmnist']
n_test = 5

scores_fun = {'accuracy': accuracy_score, 'f1_macro': macro_f1} # , 'recall_macro': macro_recall
# scores_fun = {'accuracy': accuracy_score}

for k, v in scores_fun.items():
    # ic(k)
    print(k)
    df_results = generate_results_df(models=models, problems=problems, scores_fun={k: v}, n_test=n_test)
    ic(df_results)
    print(df_results.to_latex(float_format="%.4f"))


strategies = ['ind', 'common', 'cvx-trainable-sp', 'hs', 'gl']
n_hidden = [32]
models = ['{}_nn_{}'.format(s, n) for n in n_hidden for s in strategies]
problems = ['clustersA-reg', 'clustersB-reg', 'clustersC-reg'] # 'functions-reg', 
# problems = ['clustersB-reg']

n_test = 5

scores_fun = {'mae': mean_absolute_error, 'mse': mean_squared_error, 'r2': r2_score}
# scores_fun = {'mse': mean_squared_error}

# save_matrix(models=models, problems=problems)

# for k, v in scores_fun.items():
#     # ic(k)
#     print(k)
#     df_results = generate_results_df(models=models, problems=problems, scores_fun={k: v}, n_test=n_test, train=False)
#     ic(df_results)
#     print(df_results.to_latex(float_format="%.4f"))
            
