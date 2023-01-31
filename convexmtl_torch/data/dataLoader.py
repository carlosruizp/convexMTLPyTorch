"""Module to load data for experiments."""
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split, PredefinedSplit
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from scipy.io import loadmat
from mtlskl.data.peptide_encoder import *
import itertools

from mtlskl.data.taskDefinition import *

from icecream import ic

import mtlskl

# Macros
module_path = '/'.join(mtlskl.__file__.split('/')[:-1])
data_dir = '{}/data/downloads/'.format(module_path)

prods = {'Mallorca': 72.46014, 'Tenerife': 107.6759}
dic_years = {'train': 2013, 'val': 2014, 'test': 2015}
dic_years_stv = {'train': 2016, 'val': 2017, 'test': 2018}

cities = {'majorca': 'Mallorca', 'tenerife': 'Tenerife'}
suf = {'majorca': 'maj', 'tenerife': 'ten'}





def dateparse_majorca(x):
    """Parse datetime data in format yyyymmddhh."""
    return pd.datetime.strptime(x, '%Y%m%d%H')


def dateparse_pm2_5beijing(x):
    """Parse datetime data in format yyyymmddhh."""
    return pd.datetime.strptime(x, '%Y %m %d %H')


def normalize_prods(y, c):
    """Scale productions as a percentage."""
    return y / prods[c] * 100


def _load_dataset_tenerife(data_goal, target=False):
    """
    Load dataframe from csv.

    data_goal: it must be either train, test or val
    year: year of data
    """
    c = cities['tenerife']
    s = suf['tenerife']
    if target:
        file = '{}/prods{}.csv'.format(dir, c)
    else:
        file = '{}/mtl{}_{}.csv'.format(dir, dic_years[data_goal], s)
    df = pd.read_csv(file, index_col=0, parse_dates=True,
                     date_parser=dateparse_majorca).between_time(start_time_tenerife, end_time_tenerife)

    # For target
    # if target:
    #     df = df / 100
    y = int(dic_years[data_goal])
    y_next = int(dic_years[data_goal]) + 1
    d = datetime(y, 1, 1)
    d_next = datetime(y_next, 1, 1)
    df = df.loc[(df.index >= d) & (df.index < d_next)]
    print(df.shape)
    return df

def _load_dataset_majorca(data_goal, target=False):
    """
    Load dataframe from csv.

    data_goal: it must be either train, test or val
    year: year of data
    """
    c = cities['majorca']
    s = suf['majorca']
    if target:
        file = '{}/prods{}.csv'.format(dir, c)
    else:
        file = '{}/mtl{}_{}.csv'.format(dir, dic_years[data_goal], s)
    df = pd.read_csv(file, index_col=0, parse_dates=True,
                     date_parser=dateparse_majorca).between_time(start_time_majorca, end_time_majorca)

    # For target
    # if target:
    #     df = df / 100
    y = int(dic_years[data_goal])
    y_next = int(dic_years[data_goal]) + 1
    d = datetime(y, 1, 1)
    d_next = datetime(y_next, 1, 1)
    df = df.loc[(df.index >= d) & (df.index < d_next)]
    return df


def _load_dataset_stv(data_goal, target=False):
    """
    Load dataframe from csv.
    data_goal: it must be either train, test or val
    """
    year = dic_years_stv[data_goal]
    if target:
        file_str = 'target'        
    else:
        file_str = 'data'
    filename = '{}/sptdf_sotavento_{}_{}.grib_1_0.pkl'.format(dir, year, file_str)
    df = pickle.load(open(filename, 'rb'))

    # print(df)
    # print(df.shape)
    # df.to_csv('../datos_stv/{}_stv_{}.csv'.format(file_str, year))
    # print(df.index.values[:24*31])
    # print(df[df.index.month== 1].index.values)
   

    # def compute_vel_height(height):
    #     def compute_vel(row):
    #         return np.sqrt(row['{}u'.format(height)]**2 + row['{}v'.format(height)]**2)
    #     return compute_vel

    # if not target:
    #     print(df)
    #     print(list(df))
    #     print(df.get_features())
    #     df.insert_feature_grid('vel10', compute_vel_height(10))
    #     df.insert_feature_grid('vel100', compute_vel_height(100))
    #     print(df)
    #     print(list(df))
    #     print(df.get_features(recompute=True))
    #     filename = '{}/sptdf_sotavento_{}_{}.grib_1_0.pkl'.format(dir, year, file_str)
    #     pickle.dump(df, open(filename, 'wb'))

    return df

def _load_dataset_realstv(data_goal, target=False):
    """
    Load dataframe from csv.
    data_goal: it must be either train, test or val
    """
    year = dic_years_stv[data_goal]
    filename = '{}/stv_{}.p'.format(dir, year)
    df = pickle.load(open(filename, 'rb'))
    df.fillna(method='ffill', inplace=True)
    # df.rename({'Speed': 'velocity', 'Direction': 'angle'})
    if target:
        return df['Energy']
    else:
        return df.drop('Energy', axis=1)
 


def _load_dataset_abalone():
    """Load dataframe from csv for abalone."""
    file = '{}/abalone.data'.format(dir)
    df = pd.read_csv(file, header=None)
    target_col = list(df)[-1]
    task_col = 0
    df['task'] = df[task_col]
    df['target'] = df[target_col]
    df = df.drop([task_col, target_col], axis=1)
    return df


def _load_dataset_pop_failures():
    """Load dataframe from csv for pop failure."""
    file = '{}/pop_failures.dat'.format(dir)
    df = pd.read_csv(file, header=0, sep='\s+')
    print(df)
    return df


def _load_dataset_pm2_5beijing():
    """Load dataframe from csv for pop failure."""
    file = '{}/PRSA_data_2010.1.1-2014.12.31.csv'.format(dir)
    df = pd.read_csv(file, header=0, sep=',', parse_dates={'Datetime':
                                                           ['year',
                                                            'month',
                                                            'day', 'hour']},
                     date_parser=dateparse_pm2_5beijing, index_col=False)
    df = df.drop(['No', 'cbwd'], axis=1).set_index('Datetime').dropna()
    d = datetime(2012, 1, 1)
    df = df.loc[df.index >= d]
    return df


def _load_dataset_communities():
    """Load dataframe from csv for pop failure."""
    file = '{}/communities.data'.format(dir)
    df = pd.read_csv(file, sep=',', index_col=False, header=None, dtype=str)
    target_col = list(df)[-1]
    task_col = 0
    df['task'] = df[task_col]
    df['target'] = df[target_col]
    df = df.drop([task_col, target_col], axis=1)

    df = df.drop([1, 2, 3, 4], axis=1)
    df = df.replace('?', np.NaN)
    nansum = df.isna().sum()
    colsToRemove = list(nansum[nansum > 1].index)
    # print(colsToRemove)
    df = df.drop(colsToRemove, axis=1).dropna()
    # print(df.shape)
    # Remove States with less examples
    limit = 50
    df_count = df.groupby('task').count()[5]
    statesToInclude = list(df_count[df_count > limit].index)
    # print(statesToInclude)
    df = df[df['task'].isin(statesToInclude)]
    # print(df.shape)
    print(df)
    return df


def _load_dataset_cal_housing():
    """Load dataframe from csv for pop failure."""
    file = '{}/cal_housing_processed.csv'.format(dir)
    df = pd.read_csv(file, sep=',', index_col=0, header=0)
    print(list(df))
    df = df[[c for c in df if c not in ['ocean_proximity', 'median_house_value']] + ['ocean_proximity', 'median_house_value']]
    df.replace('NEAR BAY', 'NEAR_BAY', inplace=True)
    df.replace('NEAR OCEAN', 'NEAR_OCEAN', inplace=True)
    df.replace('<1H OCEAN', '<1H_OCEAN', inplace=True)
    print(df)
    return df


def _load_dataset_bos_housing():
    """Load dataframe from csv for pop failure."""
    file = '{}/bos_housing.csv'.format(dir)
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    df = pd.read_csv(file, index_col=None, header=None,  delimiter=r"\s+", names=column_names, dtype=str)
    task_col = 'CHAS'

    df['task'] = df[task_col]
    df['target'] = df['MEDV']
    df = df.drop('MEDV', axis=1)
    df = df.drop(task_col, axis=1)
    print(df)
    return df

def _load_dataset_landmine():
    """Load dataframe from csv for pop failure."""
    file = '{}/LandmineData.mat'.format(dir)
    dmat = loadmat(file)
    list_df = []
    for i, (feat, label) in enumerate(zip(dmat['feature'][0], dmat['label'][0])):
        print('Task ', i)
        X = np.array(feat)
        y = np.array(label)
        df = pd.DataFrame(X, columns=None, index=None)
        df['task'] = int(i)
        df['label'] = y
        list_df.append(df.copy())
    df = pd.concat(list_df, ignore_index=True)

    print(df['label'].value_counts())
    return df

def _load_dataset_binding(encoding='one_hot'):
    """Load dataframe from csv for pop failure."""
    file = '{}/mhc-peptide_binding.csv'.format(dir)
    df = pd.read_csv(file, index_col=None, header=0, sep=None,  dtype=str)
    # df = df.iloc[:1000, :]
    df.drop(['ann', 'inequality', 'cv', 'length'], inplace=True, axis=1)
    
    # df.drop(['species'], inplace=True, axis=1)
    
    # We use dummy variables for the four species: ['chimpanzee' 'human' 'macaque' 'mouse']
    df_dummy = pd.get_dummies(df['species'], prefix='sp')
    df = pd.concat([df_dummy, df.drop('species', axis=1)], axis=1, join='inner')
    
    # We use the allele as task
    uniq = np.unique(df['allele'])

    # encoding of the sequences of peptides
    if 'one_hot' == encoding:
        encode = one_hot_encode
    elif 'nlf' == encoding:
        encode = nlf_encode
    elif 'blosum' == encoding:
        encode = blosum_encode

    file_enc = '{}/{}_binding.pkl'.format(dir, encoding)
    if os.path.isfile(file_enc):
        df_enc = pd.read_pickle(file_enc)
    else:
        df_enc = df['sequence'].apply(lambda pep: pd.Series(encode(pep)))
        df_enc.to_pickle(file_enc)
    df = pd.concat([df_enc, df.drop('sequence', axis=1)], axis=1)

    # label from ic50 : ic50 <= 500 -> 0 ; ic50 <= 500 -> 1
    df['label'] = df['ic50'].apply(lambda x: 0 if float(x) < 500 else 1)
    df.drop('ic50', inplace=True, axis=1)
    # print(df['label'].value_counts())
    # print(df)
    # print(df['allele'].value_counts().max())
    # print(df['allele'].value_counts().min())
    # print(df['allele'].value_counts().count())
    return df

def _load_dataset_adult():
    """Load dataframe from csv for pop failure."""
    file_data = '{}/adult/adult.data'.format(dir)
    file_test = '{}/adult/adult.test'.format(dir)
    file_names = '{}/adult/adult.names'.format(dir)

    with open(file_names) as file:          
        # loop to read iterate  
        # last n lines and print it 
        N = 14 # number of features
        columns = {}
        for line in (file.readlines() [-N:]):
            feat_name, desc = line.split(':')[0], line.split(':')[1]
            cont = ('continuous' in desc) # or feat_name == ''
            task_def = feat_name in ['race', 'sex']
            columns[feat_name] = cont or task_def

    column_names = list(columns.keys())
    
    column_names.append('target')

    df_data = pd.read_csv(file_data, index_col=False, header=None,  dtype=str, names=column_names)
    n_train = df_data.shape[0]
    df_test = pd.read_csv(file_test, index_col=False, header=None,  dtype=str, names=column_names, skiprows=1)

    df = pd.concat([df_data, df_test], ignore_index=True)


    for col, cond in columns.items():
        if not cond:
            # Get one hot encoding of columns col
            one_hot = pd.get_dummies(df[col], prefix=col)
            # Drop column col as it is now encoded
            df = df.drop(col, axis = 1)
            # Join the encoded df
            df = df.join(one_hot)


    df_target = df['target']    
    df = df.drop('target', axis=1)

    df_target.replace(' <=50K.', ' <=50K', inplace=True)
    df_target.replace(' >50K.', ' >50K', inplace=True)
    df_target = pd.get_dummies(df_target)[' >50K']

    dic_data = {}
    dic_data['train'] = df.iloc[:n_train]
    dic_data['test'] = df.iloc[n_train:]

    dic_target = {}
    dic_target['train'] = df_target.iloc[:n_train]
    dic_target['test'] = df_target.iloc[n_train:]
    return dic_data, dic_target


def _load_dataset_compas():
    """Load dataframe from csv for pop failure."""
    file_data = '{}/compas/compas-scores-two-years-violent.csv'.format(data_dir)
    raw_data = pd.read_csv(file_data)

    # df = pd.read_csv(file_data, index_col=0, header=0,  dtype=str)

    df = raw_data[((raw_data['days_b_screening_arrest'] <=30) & 
      (raw_data['days_b_screening_arrest'] >= -30) &
      (raw_data['is_recid'] != -1) &
      (raw_data['c_charge_degree'] != 'O') & 
      (raw_data['score_text'] != 'N/A')
     )]

    # ic('Num rows filtered: %d' % len(df))

    cols_to_keep = ['c_charge_degree','age_cat','race','sex', 'priors_count', 'two_year_recid']

    # Remove the race Native American
    df = df.query('race!=\'Native American\'')
    # Remove the race Asian
    df = df.query('race!=\'Asian\'')


    df_data = df[cols_to_keep]

    # df_crime = pd.get_dummies(df['c_charge_degree'],prefix='crimefactor',drop_first=True)
    # df_age = pd.get_dummies(df['age_cat'],prefix='age')
    # df_race = pd.get_dummies(df['race'],prefix='race')
    # df_gender = pd.get_dummies(df['sex'],prefix='sex',drop_first=True)

    # df_data = pd.concat([df_crime, df_age,df_race,df_gender,
    #                df['priors_count'],df['two_year_recid']
    #               ],axis=1)

    # ic(df_data)

    df_score = pd.get_dummies(df['score_text'] != 'Low',prefix='score_factor',drop_first=True)

    # df = pd.concat([df_data, df_score], axis=1, join='inner')


    return df_data, df_score


def _load_dataset_wine():
    """Load dataframe from csv for pop failure."""
    file_red = '{}/wine/winequality_red.csv'.format(dir)
    df_red = pd.read_csv(file_red)
    df_red['color'] = pd.Series(['red'] * len(df_red.index))

    file_white = '{}/wine/winequality_white.csv'.format(dir)
    df_white = pd.read_csv(file_white)
    df_white['color'] = pd.Series(['white'] * len(df_white.index))

    df = pd.concat([df_red, df_white], axis=0, ignore_index=True)
    
    return df


def _load_dataset_sarcos():
    """Load dataframe from csv for pop failure."""
    file_train = '{}/sarcos/sarcos_inv.mat'.format(dir)
    # Load training set
    train = loadmat(file_train)
    # Inputs  (7 joint positions, 7 joint velocities, 7 joint accelerations)
    Xtrain = train["sarcos_inv"][:, :21]
    # Outputs (7 joint torques)
    Ytrain = train["sarcos_inv"][:, 21:]

    print(Ytrain.shape)

    ntrain = Xtrain.shape[0]
    Xtrain_data = np.zeros((ntrain * 7, 21))
    ytrain_data = np.zeros((ntrain * 7, 1))
    ttrain_data = np.zeros((ntrain * 7, 1))
    for i in range(7):
        Xtrain_data[i*ntrain: (i+1)*ntrain, :] = Xtrain[:, :]
        ytrain_data[i*ntrain: (i+1)*ntrain, :] = Ytrain[:, i : (i+1)]
        ttrain_data[i*ntrain: (i+1)*ntrain, :] = i * np.ones((ntrain, 1)).astype(int)

    columns = [['X{}'.format(i), 'Y{}'.format(i), 'Z{}'.format(i)] for i in range(7)]
    columns = list(itertools.chain(*columns))

    df_train = pd.DataFrame(columns=columns, data=Xtrain_data)
    df_train['task'] = ttrain_data
    df_train['target'] = ytrain_data

    print(df_train)
    exit()

    file_test = '{}/sarcos/sarcos_inv_test.mat'.format(dir)
    test = loadmat(file_test)
    Xtest = test["sarcos_inv_test"][:, :21]
    Ytest = test["sarcos_inv_test"][:, 21:]

    ntest = Xtest.shape[0]
    Xtest_data = np.zeros((ntest * 7, 3))
    ytest_data = np.zeros((ntest * 7, 1))
    ttest_data = np.zeros((ntest * 7, 1))
    for i in range(7):
        Xtest_data[i*ntest: (i+1)*ntest, :] = Xtest[:, 3*i : 3*(i+1)]
        ytest_data[i*ntest: (i+1)*ntest, :] = Ytest[:, i : (i+1)]
        ttest_data[i*ntest: (i+1)*ntest, :] = i * np.ones((ntest, 1)).astype(int)

    df_test = pd.DataFrame(columns=['X', 'Y', 'Z'], data=Xtest_data)
    df_test['task'] = ttest_data
    df_test['target'] = ytest_data

    print(df_test)
    
    return df_train, df_test


def load_mnist_var(dir_path, var='', digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], max_size=-1):
    """
    Loads a binarized version of MNIST. 

    The data is given by a dictionary mapping from strings
    ``'train'``, ``'valid'`` and ``'test'`` to the associated pair of data and metadata.
    
    **Defined metadata:**

    * ``'input_size'``
    * ``'length'``

    """
    
    input_size=784
    # dir_path = '../data/'

    dic_variations = {'standard': '', 'random': 'background_random_', 'images': 'background_images_'}
    variation = dic_variations[var]
    print(variation)

    train_file,test_file = [os.path.join(dir_path, 'mnist_' + variation + ds + '.amat') for ds in ['train','test']]
    # Get data
    # train,valid,test = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file,test_file]]
    train,test = [np.loadtxt(f)[:max_size] for f in [train_file,test_file]]

    y_train = train[:, input_size]
    y_test = test[:, input_size]
    ix_train = np.array([False] * len(y_train))
    ix_test = np.array([False] * len(y_test))
    # Get subsample of digits
    for d in digits:
        ix_train = ix_train | (y_train == d)
        ix_test = ix_test | (y_test == d)

    X_train, y_train, X_test, y_test = train[ix_train, :input_size], train[ix_train, input_size], test[ix_test, :input_size], test[ix_test, input_size]

    del train
    del test
    
    return X_train, y_train, X_test, y_test

def _load_dataset_mnist(max_size=-1):
    """Load dataframe from csv for mnist variations."""

    trainX_l = []
    trainy_l = []
    testX_l = []
    testy_l = []
    data_dir = '../data/mnist/'
    for var in ['standard', 'random', 'images']:        
        X_train, y_train, X_test, y_test = load_mnist_var(data_dir, var, max_size=max_size)
        t_train = np.array([var] * len(y_train) )
        t_test = np.array([var] * len(y_test) )

        ic(X_train.shape)
        ic(X_test.shape)

        df_train = pd.DataFrame(data=X_train)
        df_train['task'] = t_train

        df_test = pd.DataFrame(data=X_test)
        df_test['task'] = t_test

        target_train = pd.Series(data=y_train)
        target_test = pd.Series(data=y_test)

        trainX_l.append(df_train)
        trainy_l.append(target_train)
        testX_l.append(df_test)
        testy_l.append(target_test)

    df_train = pd.concat(trainX_l)
    target_train = pd.concat(trainy_l)
    df_test = pd.concat(testX_l)
    target_test = pd.concat(testy_l)

    ic(df_train.shape)

    return df_train, target_train, df_test, target_test 



def load_fashionmnist_var(dir_path, var='', digits=[0, 1, 2, 3, 4, 5]):
    """
    Loads a binarized version of MNIST. 

    The data is given by a dictionary mapping from strings
    ``'train'``, ``'valid'`` and ``'test'`` to the associated pair of data and metadata.
    
    **Defined metadata:**

    * ``'input_size'``
    * ``'length'``

    """
    
    input_size=784
    # dir_path = '../data/'

    dic_variations = {'standard': '', 'random': 'background_random_'}
    variation = dic_variations[var]
    print(variation)

    train_file,test_file = [os.path.join(dir_path, 'images_' + variation + ds + '.csv') for ds in ['train','test']]
    data_train, data_test = np.loadtxt(train_file), np.loadtxt(test_file)

    print(train_file)
    
    train_file,test_file = [os.path.join(dir_path, 'labels_' + ds + '.csv') for ds in ['train','test']]
    y_train, y_test = np.loadtxt(train_file), np.loadtxt(test_file)
    print(np.unique(y_train))

    ix_train = np.array([False] * len(y_train))
    ix_test = np.array([False] * len(y_test))
    # Get subsample of digits
    for d in digits:
        ix_train = ix_train | (y_train == d)
        ix_test = ix_test | (y_test == d)

    X_train, y_train, X_test, y_test = data_train[ix_train], y_train[ix_train], data_test[ix_test], y_test[ix_test]

    del data_train
    del data_test
    
    return X_train, y_train, X_test, y_test


def _load_dataset_fashionmnist():
    """Load dataframe from csv for mnist variations."""

    trainX_l = []
    trainy_l = []
    testX_l = []
    testy_l = []
    data_dir = '../data/mnist-fashion/'
    
    for var in ['standard', 'random']:        
        X_train, y_train, X_test, y_test = load_fashionmnist_var(data_dir, var)
        t_train = np.array([var] * len(y_train) )
        t_test = np.array([var] * len(y_test) )

        print(X_train.shape)
        print(X_test.shape)

        df_train = pd.DataFrame(data=X_train)
        df_train['task'] = t_train

        df_test = pd.DataFrame(data=X_test)
        df_test['task'] = t_test

        target_train = pd.Series(data=y_train)
        target_test = pd.Series(data=y_test)

        trainX_l.append(df_train)
        trainy_l.append(target_train)
        testX_l.append(df_test)
        testy_l.append(target_test)

    df_train = pd.concat(trainX_l)
    target_train = pd.concat(trainy_l)
    df_test = pd.concat(testX_l)
    target_test = pd.concat(testy_l)

    return df_train, target_train, df_test, target_test 


def _load_dataset_school():
    """Load dataframe from csv for mnist variations."""

  
    data = loadmat('{}/school/school_b.mat'.format(data_dir))

    y = data['y']
    x = data['x'].T[:, :-1] # remove the bias column
    task_indexes = data['task_indexes']

    # ic(x.shape)
    # ic(y.shape)

    t_col = np.zeros(y.shape)

    for i, (t1, t2) in enumerate(zip(task_indexes[:-1], task_indexes[1:])):
        task1, task2 = t1[0]-1, t2[0]-1
        t_col[task1:task2] = i

    t_col[task2:] = i+1  

    # for n in range(1, 10):
    #      data = loadmat('{}/school_{}_indexes.mat'.format(data_dir, n))
    #      print(data.keys())

    dic_feat = {}
    dic_feat['year'] = 3
    # dic_feat['exam_score'] = 2
    dic_feat['per_fsm'] = 1
    dic_feat['per_vr1band'] = 1
    dic_feat['gender'] = 2
    dic_feat['vrband'] = 3
    dic_feat['ethnic_group'] = 11
    dic_feat['school_gender'] = 3
    dic_feat['school_denomination'] = 3

    columns = []
    for colname, n in dic_feat.items():
        for i in range(n):
            columns.append('{}-{}'.format(colname, i))

    # columns = [''] * x.shape[1]
    # columns[0] = 'year'
    # columns[1], columns[2], columns[3] = 'school1', 'school2', 'school3'
    # columns[4], columns[5] = 'exam_score1', 'exam_score2'
    # columns[6], columns[7] = 'per_fsm1', 'per_fsm2'
    # columns[8], columns[9] = 'per_vrband1', 'per_vrband2'
    # columns[10] = 'gender'
    # columns[11] = 'vrband'
    # columns[12], columns[13] = 'ethnic_group1', 'ethnic_group2'

    df_data = pd.DataFrame(x, columns=columns)

    # ic(df_data)
    df_data['task'] = t_col
    df_target = pd.Series(y.flatten())

    return df_data, df_target


def _load_dataset_computer():
    """Load dataframe from mat for computer dataset."""
    
    data = loadmat('{}/computer/conjointAnalysisComputerBuyers.mat'.format(data_dir))
    # ic(data.keys())


    design_matrix = data['designMarix']
    # ic(design_matrix)
    # ic(design_matrix.shape)

    like_buy = data['likeBuy']

    # ic(like_buy)
    # ic(like_buy.shape)

    # ic(data['computerDataZ'])

    computer_data_z = data['computerDataZ']
    # ic(computer_data_z.shape)

    X = np.tile(design_matrix, (like_buy.shape[0], 1))
    user_ids = np.expand_dims(range(like_buy.shape[0]), axis=1)
    # ic(user_ids)
    t = np.repeat(user_ids, like_buy.shape[1], axis=0)
    # # ic(t[0:20])
    y = like_buy.flatten()
    # ic(X[0:20])
    # ic(y[:20])
    # ic(X[20:40])
    # ic(X.shape)
    # ic(t.shape)
    # ic(y.shape)

    columns = [ 'bias',
                'hot_line',
                'ram',
                'screen_size',
                'cpu_speed',
                'hard_disk',
                'cd_rom',
                'cache',
                'color_of_unit',
                'availability',
                'warranty',
                'software',
                'guarantee',
                'price'    ]

    df_data = pd.DataFrame(data=X, columns=columns)
    df_data = df_data.drop('bias', axis=1)
    df_data['task'] = t
    # ic(df_data)

    df_target = pd.Series(y)
    # ic(df_target)


    return df_data, df_target
    

def load_dataset_parkinson():
    """Load dataframe from txt for parkinson dataset."""

    data_dir = '../data/parkinson'
    
    # data = np.loadtxt('{}/parkinsons_updrs.data'.format(data_dir), dtype='str')
    data = pd.read_csv('{}/parkinsons_updrs.data'.format(data_dir), header=0, index_col=None)
    print(data)

    df_data = data.drop(['motor_UPDRS', 'total_UPDRS', 'subject#'], axis=1)
    df_data['task'] = data['subject#']
    df_target = data['total_UPDRS']

    return df_data, df_target

def _load_dataset_parkinson():
    """Load dataframe from txt for parkinson dataset."""

    data_dir = '../data/parkinson'
    
    # data = np.loadtxt('{}/parkinsons_updrs.data'.format(data_dir), dtype='str')
    data = pd.read_csv('{}/parkinsons_updrs.data'.format(data_dir), header=0, index_col=None)
    print(data)

    df_data = data.drop(['motor_UPDRS', 'total_UPDRS', 'subject#', 'test_time'], axis=1)
    df_data['task'] = data['subject#']
    df_target = data['total_UPDRS']

    return df_data, df_target


def _load_dataset_isolet():
    """Load dataframe from txt for isolet dataset."""

    data_dir = '../data/isolet'
    
    # data = np.loadtxt('{}/parkinsons_updrs.data'.format(data_dir), dtype='str')
    data_train = pd.read_csv('{}/isolet1+2+3+4.data'.format(data_dir), header=None, index_col=None)
    data_test = pd.read_csv('{}/isolet5.data'.format(data_dir), header=None, index_col=None)
    data = pd.concat([data_train, data_test], ignore_index=True)
    target = data.iloc()[:,-1]

    # print(target[4 * 30 * 52 - 52 : 4 * 30 * 52 + 0])

    # There are two misses in ISOLET4
    # There is one miss in ISOLET 5
    misses = [0, 0, 0, 2, 1]
    tot_misses = [0, 0, 0, 0, 2, 3]

    # In ISOLET1 and ISOLET2 the letters are paired
    # from 2 * 30 * 52 = 3120 (first letter of first speaker of ISOLET 3) 
    # to   4 * 30 * 52 - 2 = 6238 (first letter of first speaker of ISOLET 5)
    # the letters are not paired    
    # The letters are again paired in ISOLET 5
    

    size_isolet = 30 * 52
    task = np.zeros(len(target))
    for i, i_next in zip(range(0, 5), range(1, 6)):
        target_i = target.iloc()[i * size_isolet - tot_misses[i] : i_next * size_isolet - tot_misses[i_next]]
        task[i * size_isolet - tot_misses[i] : i_next * size_isolet - tot_misses[i_next]] = i_next
        # print(len(target_i))
        # print(np.unique(target_i, return_counts=True))

    df_data = data.drop(columns=data.shape[1]-1)
    df_data['task'] = task
    df_target = target

    return df_data, df_target


def _load_data(name, max_size=-1):
    """Load train, test and validation datasets from data."""
    dic_data = {}
    dic_target = {}
    if name == 'majorca':
        for x in ['train', 'test', 'val']:
            dic_data[x] = _load_dataset_majorca(x)
            dic_target[x] = _load_dataset_majorca(x, target=True)
        return dic_data, dic_target
    elif name == 'tenerife':
        for x in ['train', 'test', 'val']:
            dic_data[x] = _load_dataset_tenerife(x)
            dic_target[x] = _load_dataset_tenerife(x, target=True)
        return dic_data, dic_target
    elif name == 'stv':
        for x in ['train', 'test', 'val']:
            dic_data[x] = _load_dataset_stv(x)
            dic_target[x] = _load_dataset_stv(x, target=True)
        return dic_data, dic_target
    elif name == 'realstv':
        for x in ['train', 'test', 'val']:
            dic_data[x] = _load_dataset_realstv(x)
            dic_target[x] = _load_dataset_realstv(x, target=True)
        return dic_data, dic_target
    elif name == 'abalone':
        df = _load_dataset_abalone()
        print(df.shape)
        X, y = df.values[:, :-1], df.values[:, -1]
        return X, y
    elif name == 'pop_failures':
        df = _load_dataset_pop_failures()
        print(df.shape)
        X, y = df.values[:, :-1], df.values[:, -1]
        return X, y
    elif name == 'pm2_5beijing':
        df = _load_dataset_pm2_5beijing()
        return df
    elif name == 'communities':
        df = _load_dataset_communities()
        X, y = df.values[:, :-1], df.values[:, -1].astype(float)
        return X, y
    elif name == 'cal_housing':
        df = _load_dataset_cal_housing()
        X, y = df.values[:, :-1], df.values[:, -1].astype(float)
        return X, y
    elif name == 'bos_housing':
        df = _load_dataset_bos_housing()
        # print(df)
        X, y = df.values[:, :-1], df.values[:, -1].astype(float)
        return X, y
    elif name == 'landmine':
        df = _load_dataset_landmine()
        # print(df)
        X, y = df.values[:, :-1], df.values[:, -1].astype(float)
        return X, y
    elif name == 'binding':
        df = _load_dataset_binding()
        # print(df)
        X, y = df.values[:, :-1], df.values[:, -1].astype(float)
        return X, y
    elif name == 'adult':
        dic_data, dic_target = _load_dataset_adult()
        return dic_data, dic_target
    elif name == 'compas':
        df_data, df_target = _load_dataset_compas()
        return df_data, df_target
    elif name == 'wine':
        df = _load_dataset_wine()
        y = df['quality'].values
        X = df.drop('quality', axis=1).values
        return X, y
    elif name == 'sarcos':
        df_train, df_test = _load_dataset_sarcos()
        return df_train, df_test
    elif name == 'mnist_variations':
        df_train, target_train, df_test, target_test = _load_dataset_mnist(max_size=max_size)
        return df_train, target_train, df_test, target_test
    elif name == 'fashionmnist_variations':
        df_train, target_train, df_test, target_test = _load_dataset_fashionmnist()
        return df_train, target_train, df_test, target_test
    elif name == 'school':
        df_data, df_target = _load_dataset_school()
        return df_data, df_target
    elif name == 'computer':
        df_data, df_target = _load_dataset_computer()
        return df_data, df_target
    elif name == 'parkinson':
        df_data, df_target = _load_dataset_parkinson()
        return df_data, df_target
    elif name == 'isolet':
        df_data, df_target = _load_dataset_isolet()
        return df_data, df_target


# ---------------------------------------------------------------------

def split_tasks_solar(dic_data, task_type, dataname):
    """Add task number to each example."""
    if task_type == 'predefined' or task_type == '':
        task_type = 'hour'
    for name, df in dic_data.items():
        df['task'] = build_task_column(df, dataname, task_type)

    return dic_data


def split_tasks_solar_mult(dic_data_mult, task_type, problems):
    """Add task number to each example."""
    keys = ['train', 'val', 'test']
    for k, dic_data in dic_data_mult.items():
        assert keys != dic_data.keys(), "{} has keys {}, not {}.".format(k, dic_data.keys(), keys)
        for key in keys:
            dic_data[key].columns = range(len(dic_data[key].columns))
    
    for key in keys:
        df_mult = pd.concat([dic_data_mult[name][key] for name in problems])
        col_problemTask = np.array([n * np.ones(dic_data_mult[name][key].shape[0], dtype='int') for n, name in enumerate(problems)]).reshape([-1,1])
        df_mult['problem_task'] = col_problemTask
        df_mult['task'] = build_task_column(df_mult, 'solarmult', task_type)
        df_mult.drop('problem_task', axis=1, inplace=True) # solo era útil para construir la tarea
        dic_data[key] = df_mult

    return dic_data


def remove_coords(colname):
    return colname.split('_')[0].strip()

def split_tasks_stv(dic_data, task_type):
    """Add task number to each example."""
    sotavento = (-7.8808, 43.3540)

    if task_type == 'predefined' or task_type == '':
        task_type = 'angle'
    
    for name, df in dic_data.items():
        coordinates = df.get_nClosestCoordinates(sotavento[0], sotavento[1], 1)
        columns = df.get_columnsByCoordinates(coordinates)
        df_nearest = df[columns]
        df_nearest.rename(mapper=remove_coords, axis='columns', inplace=True)
        df['task'] = build_task_column(df_nearest, 'stv', task_type)
    return dic_data

def split_tasks_realstv(dic_data, task_type):
    """Add task number to each example."""
    sotavento = (-7.8808, 43.3540)
    
    for name, df in dic_data.items():
        df['task'] = build_task_column(df, 'realstv', task_type)
    return dic_data


# def get_tasks(dic_data, task_type):
#     """Return task column."""
#     dic_tasks = {}
#     for name, df in dic_data.items():
#         dic_tasks[name] = df.index.to_series().map(task_id,
#                                                    task_type=task_type)

#     return dic_tasks


def load_datasets_majorca(task_type='predefined',
                          save_pd=False):
    """Load the datasets for the experiments."""
    # Get data
    dic_data, dic_target = _load_data('majorca')
    dic_taskData = split_tasks_solar(dic_data, task_type, 'majorca')
    # Data X
    df_train = dic_taskData['train']
    df_val = dic_taskData['val']
    df_test = dic_taskData['test']

    df_total = pd.concat([df_train, df_val, df_test])
    X = df_total.values
    # print(X.shape)

    df_train_val = pd.concat([df_train, df_val])
    test_fold_outer = np.append(-np.ones(df_train_val.shape[0]),
                                np.ones(df_test.shape[0]))
    # print(test_fold_outer.shape)

    print(df_total[test_fold_outer==-1])
    print(df_total[test_fold_outer==1])

    outer_cv = PredefinedSplit(test_fold_outer)

    test_fold_inner = np.append(-np.ones(df_train.shape[0]),
                                np.ones(df_val.shape[0]))


    # print(test_fold_inner.shape)
    inner_cv = PredefinedSplit(test_fold_inner)

    # X_train = df_train_val.values
    # min_max_scaler = MinMaxScaler()
    # X_train[:, :-1] = min_max_scaler.fit_transform(X_train[:, :-1])
    # X_test[:, :-1] = min_max_scaler.transform(X_test[:, :-1])

    # Target y
    target_train = dic_target['train']
    target_val = dic_target['val']
    target_test = dic_target['test']
    target_total = pd.concat([target_train, target_val, target_test])
    y = target_total.values

    print('y shape', y.shape)
    print('X shape', X.shape)
    task = X[:, -1].flatten()
    print('task shape', task.shape)
    df_aux = pd.DataFrame({'y': y.flatten(), 'task': task}, dtype=float)
    # print(df_aux)

    task_info = -1
    return X, y, inner_cv, outer_cv, task_info


def load_datasets_tenerife(task_type='predefined',
                           save_pd=False):
    """Load the datasets for the experiments."""
    # Get data
    dic_data, dic_target = _load_data('tenerife')
    dic_taskData = split_tasks_solar(dic_data, task_type, 'tenerife')
    # Data X
    df_train = dic_taskData['train']
    print('train shape', df_train.shape)
    df_val = dic_taskData['val']
    print('val shape', df_val.shape)
    df_test = dic_taskData['test']
    print('test shape', df_test.shape)

    df_total = pd.concat([df_train, df_val, df_test])
    X = df_total.values
    print(X.shape)

    df_train_val = pd.concat([df_train, df_val])
    test_fold_outer = np.append(-np.ones(df_train_val.shape[0]),
                                np.ones(df_test.shape[0]))
    print(test_fold_outer.shape)

    outer_cv = PredefinedSplit(test_fold_outer)

    test_fold_inner = np.append(-np.ones(df_train.shape[0]),
                                np.ones(df_val.shape[0]))
    print(test_fold_inner.shape)
    inner_cv = PredefinedSplit(test_fold_inner)

    # X_train = df_train_val.values
    # min_max_scaler = MinMaxScaler()
    # X_train[:, :-1] = min_max_scaler.fit_transform(X_train[:, :-1])
    # X_test[:, :-1] = min_max_scaler.transform(X_test[:, :-1])

    # Target y
    target_train = dic_target['train']
    target_val = dic_target['val']
    target_test = dic_target['test']
    target_total = pd.concat([target_train, target_val, target_test])
    y = target_total.values

    task_info = -1
    return X, y, inner_cv, outer_cv, task_info


def load_datasets_majten(task_type='predefined',
                           save_pd=False):
    # Get  data
    problems = ['majorca', 'tenerife']
    dic_data_mult = {}
    dic_target_mult = {}
    for problem in problems:
        dic_data, dic_target = _load_data(problem)
        dic_data_mult[problem] = dic_data
        dic_target_mult[problem] = dic_target

    
    # Concat data
    dic_taskData = split_tasks_solar_mult(dic_data_mult, task_type, problems)
    # Data X
    df_train = dic_taskData['train']
    print('train shape', df_train.shape)
    df_val = dic_taskData['val']
    print('val shape', df_val.shape)
    df_test = dic_taskData['test']
    print('test shape', df_test.shape)

    df_total = pd.concat([df_train, df_val, df_test])
    X = df_total.values
    print(X.shape)

    df_train_val = pd.concat([df_train, df_val])
    test_fold_outer = np.append(-np.ones(df_train_val.shape[0]),
                                np.ones(df_test.shape[0]))
    print(test_fold_outer.shape)

    outer_cv = PredefinedSplit(test_fold_outer)

    test_fold_inner = np.append(-np.ones(df_train.shape[0]),
                                np.ones(df_val.shape[0]))
    print(test_fold_inner.shape)
    inner_cv = PredefinedSplit(test_fold_inner)
    
    # Target y
    target_train = pd.concat([dic_target_mult[name]['train'] for name  in problems])
    target_val = pd.concat([dic_target_mult[name]['val'] for name  in problems])
    target_test = pd.concat([dic_target_mult[name]['test'] for name  in problems])
    target_total = pd.concat([target_train, target_val, target_test])
    y = target_total.values

    task_info = -1

    return X, y, inner_cv, outer_cv, task_info



def load_datasets_stv(task_type='predefined',
                           save_pd=False):
    """Load the datasets for the experiments."""
    # Get data
    dic_data, dic_target = _load_data('stv')
    dic_taskData = split_tasks_stv(dic_data, task_type)
    # Data X
    df_train = dic_taskData['train']
    print('train shape', df_train.shape)
    df_val = dic_taskData['val']
    print('val shape', df_val.shape)
    df_test = dic_taskData['test']
    print('test shape', df_test.shape)

    df_total = pd.concat([df_train, df_val, df_test])
    X = df_total.values
    print(X.shape)

    df_train_val = pd.concat([df_train, df_val])
    test_fold_outer = np.append(-np.ones(df_train_val.shape[0]),
                                np.ones(df_test.shape[0]))
    print(test_fold_outer.shape)

    outer_cv = PredefinedSplit(test_fold_outer)

    test_fold_inner = np.append(-np.ones(df_train.shape[0]),
                                np.ones(df_val.shape[0]))
    print(test_fold_inner.shape)
    inner_cv = PredefinedSplit(test_fold_inner)

    # X_train = df_train_val.values
    # min_max_scaler = MinMaxScaler()
    # X_train[:, :-1] = min_max_scaler.fit_transform(X_train[:, :-1])
    # X_test[:, :-1] = min_max_scaler.transform(X_test[:, :-1])

    # Target y
    target_train = dic_target['train']
    target_val = dic_target['val']
    target_test = dic_target['test']
    target_total = pd.concat([target_train, target_val, target_test])
    y = target_total.values * 100 / 17560 # Normalizing by the maximum installed power

    task_info = -1
    return X, y, inner_cv, outer_cv, task_info


def load_datasets_realstv(task_type='predefined',
                           save_pd=False):
    """Load the datasets for the experiments."""
    # Get data
    dic_data, dic_target = _load_data('realstv')
    dic_taskData = split_tasks_realstv(dic_data, task_type)
    # Data X
    df_train = dic_taskData['train']
    print('train shape', df_train.shape)
    df_val = dic_taskData['val']
    print('val shape', df_val.shape)
    df_test = dic_taskData['test']
    print('test shape', df_test.shape)

    df_total = pd.concat([df_train, df_val, df_test])
    X = df_total.values
    print(X.shape)

    df_train_val = pd.concat([df_train, df_val])
    test_fold_outer = np.append(-np.ones(df_train_val.shape[0]),
                                np.ones(df_test.shape[0]))
    print(test_fold_outer.shape)

    outer_cv = PredefinedSplit(test_fold_outer)

    test_fold_inner = np.append(-np.ones(df_train.shape[0]),
                                np.ones(df_val.shape[0]))
    print(test_fold_inner.shape)
    inner_cv = PredefinedSplit(test_fold_inner)

    # X_train = df_train_val.values
    # min_max_scaler = MinMaxScaler()
    # X_train[:, :-1] = min_max_scaler.fit_transform(X_train[:, :-1])
    # X_test[:, :-1] = min_max_scaler.transform(X_test[:, :-1])

    # Target y
    target_train = dic_target['train']
    target_val = dic_target['val']
    target_test = dic_target['test']
    target_total = pd.concat([target_train, target_val, target_test])
    y = target_total.values * 100 / 17560 # Normalizing by the maximum installed power

    task_info = -1
    return X, y, inner_cv, outer_cv, task_info


def load_datasets_abalone(task_type='predefined',
                          save_pd=False, seed=42):
        X, y = _load_data('abalone')
        task_info = -1

        n_splits_inner = 5
        n_splits_outer = 5
        inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=42)
        outer_cv = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)

        return X, y, inner_cv, outer_cv, task_info


def load_datasets_pm2_5beijing(task_type='predefined',
                               save_pd=False, seed=42):
        df = _load_data('pm2_5beijing')
        df['task'] = df.index.to_series().apply(task_id, task_type=task_type)
        print(df)

        X, y = df.drop('pm2.5', axis=1).values, df['pm2.5'].values
        d_test = datetime(2014, 1, 1)
        d_val = datetime(2013, 1, 1)
        df_train = df[df.index < d_val]
        df_val = df[(df.index >= d_val) & (df.index < d_test)]
        df_test = df[df.index >= d_test]

        df_train_val = pd.concat([df_train, df_val])
        test_fold_outer = np.append(-np.ones(df_train_val.shape[0]),
                                    np.ones(df_test.shape[0]))
        print(test_fold_outer.shape)

        outer_cv = PredefinedSplit(test_fold_outer)

        test_fold_inner = np.append(-np.ones(df_train.shape[0]),
                                    np.ones(df_val.shape[0]))
        print(test_fold_inner.shape)
        inner_cv = PredefinedSplit(test_fold_inner)

        task_info = -1

        return X, y, inner_cv, outer_cv, task_info


def load_datasets_communities(task_type='predefined',
                              save_pd=False, seed=42):
        X, y = _load_data('communities')
        # print(X[:, 0])
        task_info = -1
        n_splits_inner = 5
        n_splits_outer = 5
        inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=42)
        outer_cv = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)

        return X, y, inner_cv, outer_cv, task_info


def load_datasets_pop_failures(task_type='predefined',
                               save_pd=False, seed=42):
        X, y = _load_data('pop_failures')
        task_info = 0
        n_splits_inner = 5
        n_splits_outer = 5
        inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=42)
        outer_cv = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)

        return X, y, inner_cv, outer_cv, task_info


def load_datasets_cal_housing(task_type='predefined',
                              save_pd=False, seed=42):
        X, y = _load_data('cal_housing')
        task_info = -1
        n_splits_inner = 5
        n_splits_outer = 5
        inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=42)
        outer_cv = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)

        return X, y, inner_cv, outer_cv, task_info


def load_datasets_bos_housing(task_type='predefined',
                              save_pd=False, seed=42):
        X, y = _load_data('bos_housing')
        task_info = -1
        n_splits_inner = 5
        n_splits_outer = 5
        inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=42)
        outer_cv = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)
        # print(X)

        return X, y, inner_cv, outer_cv, task_info

def load_datasets_landmine(task_type='predefined',
                              save_pd=False, seed=42):
        X, y = _load_data('landmine')
        task_info = -1
        n_splits_inner = 5
        n_splits_outer = 5
        inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=42)
        outer_cv = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)
        # print(X)

        return X, y, inner_cv, outer_cv, task_info

def load_datasets_binding(task_type='predefined',
                              save_pd=False, seed=42):
        X, y = _load_data('binding')
        task_info = -1
        n_splits_inner = 5
        n_splits_outer = 5
        inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=42)
        outer_cv = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)
        # print(X)

        return X, y, inner_cv, outer_cv, task_info


def addTasks2df_adult(df, task_type, one_hot):

    task_col = build_task_column(df, 'adult', task_type)

    if isinstance(task_type, list):
        for col in ['race', 'sex']:
            if col not in task_type or one_hot is True:
                # Get one hot encoding of columns col
                one_hot_col = pd.get_dummies(df[col], prefix=col)
                # Join the encoded df
                df = df.join(one_hot_col)
            # Drop column col as it is now encoded
            df = df.drop(col, axis = 1)
                
    df['task'] = task_col

    return df

def load_datasets_adult(task_type='predefined',
                              save_pd=False, seed=42, one_hot=False):
        dic_data, dic_target = _load_data('adult')
        task_info = -1
        n_splits_inner = 5
        n_splits_outer = 5
        inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=42)
        outer_cv = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)

        df_train_val = dic_data['train']
        df_train_val = addTasks2df_adult(df_train_val, task_type, one_hot)
        df_test = dic_data['test']
        df_test = addTasks2df_adult(df_test, task_type, one_hot)

        target_train_val = dic_target['train']
        target_test = dic_target['test']

        # test_fold_outer = np.append(-np.ones(df_train_val.shape[0]),
        #                         np.ones(df_test.shape[0]))
        # outer_cv = PredefinedSplit(test_fold_outer)

        X = pd.concat([df_train_val, df_test], ignore_index=True).values
        y = pd.concat([target_train_val, target_test], ignore_index=True).values


        return X, y, inner_cv, outer_cv, task_info


def load_datasets_compas(task_type='predefined',
                              save_pd=False, seed=42, one_hot=False):
        df_data, df_target = _load_data('compas')

        # ic(df_data)
        task_info = -1
        n_splits_inner = 5
        n_splits_outer = 5
        inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=42)
        outer_cv = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)

        if task_type == 'predefined':
            task_type = ['sex', 'race']
        task_col = build_task_column(df_data, 'compas', task_type)
        # ic(task_col)

        if not isinstance(task_type, list):
            task_type_list = [task_type]
        else:
            task_type_list = task_type
        for col in list(df_data):
            if col not in [ 'priors_count', 'two_year_recid']: # then the column is categorical
                if col not in task_type_list or one_hot is True:
                    # Get one hot encoding of columns col
                    one_hot_col = pd.get_dummies(df_data[col], prefix=col)
                    # Join the encoded df_data
                    df_data = df_data.join(one_hot_col)
                # Drop column col as it is now encoded
                df_data = df_data.drop(col, axis = 1)
                    
        df_data['task'] = task_col

        uniq, counts = np.unique(task_col, return_counts=True)
        # ic(uniq, counts)

        X = df_data.values
        y = df_target.values.flatten()

        return X, y, inner_cv, outer_cv, task_info


def load_datasets_wine(task_type='predefined',
                              save_pd=False, seed=42):
        X, y = _load_data('wine')
        task_info = -1
        n_splits_inner = 5
        n_splits_outer = 5
        inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=42)
        outer_cv = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)
        # print(X)

        return X, y, inner_cv, outer_cv, task_info


def load_datasets_sarcos(task_type='predefined',
                              save_pd=False, seed=42):
        df_train, df_test = _load_data('sarcos')
        task_info = -1
        n_splits_inner = 5

        
        df = pd.concat([df_train, df_test], axis=0)
        
        test_fold_outer = np.append(-np.ones(df_train.shape[0]),
                                np.ones(df_test.shape[0]))

        outer_cv = PredefinedSplit(test_fold_outer)
        
        inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=42)

        X = df.drop('target', axis=1).values
        y = df['target'].values

        return X, y, inner_cv, outer_cv, task_info


def load_datasets_mnist(task_type='predefined',
                              save_pd=False, seed=42,
                              max_size=-1):
        df_train, target_train, df_test, target_test = _load_data('mnist_variations', max_size=max_size)
        task_info = -1
        n_splits_inner = 5

        
        df = pd.concat([df_train, df_test], axis=0)
        target = pd.concat([target_train, target_test], axis=0)
        
        test_fold_outer = np.append(-np.ones(df_train.shape[0]),
                                np.ones(df_test.shape[0]))

        outer_cv = PredefinedSplit(test_fold_outer)
        
        inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=42)

        X = df.values
        y = target.values

        return X, y, inner_cv, outer_cv, task_info

def load_datasets_fashionmnist(task_type='predefined',
                              save_pd=False, seed=42):
        df_train, target_train, df_test, target_test = _load_data('fashionmnist_variations')
        task_info = -1
        n_splits_inner = 5

        
        df = pd.concat([df_train, df_test], axis=0)
        target = pd.concat([target_train, target_test], axis=0)
        
        test_fold_outer = np.append(-np.ones(df_train.shape[0]),
                                np.ones(df_test.shape[0]))

        outer_cv = PredefinedSplit(test_fold_outer)
        
        inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=42)

        X = df.values
        y = target.values

        return X, y, inner_cv, outer_cv, task_info


def load_datasets_school(task_type='predefined',
                              save_pd=False, seed=42):
        df_data, df_target = _load_data('school')

        task_info = -1
        n_splits_inner = 5
        n_splits_outer = 5
        X, y = df_data.values, df_target.values
        inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=42)
        outer_cv = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)
        # print(X)

        return X, y, inner_cv, outer_cv, task_info


def load_datasets_computer(task_type='predefined',
                              save_pd=False, seed=42):
        df_data, df_target = _load_data('computer')

        task_info = -1
        n_splits_inner = 5
        n_splits_outer = 5
        X, y = df_data.values, df_target.values
        inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=42)
        outer_cv = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)
        # print(X)

        return X, y, inner_cv, outer_cv, task_info


def load_datasets_parkinson(task_type='predefined',
                              save_pd=False, seed=42):
        df_data, df_target = _load_data('parkinson')

        task_info = -1
        n_splits_inner = 5
        n_splits_outer = 5
        X, y = df_data.values, df_target.values
        inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=42)
        outer_cv = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)
        # print(X)

        return X, y, inner_cv, outer_cv, task_info


def load_datasets_isolet(task_type='predefined',
                              save_pd=False, seed=42):
        df_data, df_target = _load_data('isolet')

        task_info = -1
        n_splits_inner = 5
        n_splits_outer = 5
        X, y = df_data.values, df_target.values
        inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=42)
        outer_cv = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)
        # print(X)

        return X, y, inner_cv, outer_cv, task_info


def load_datasets_colored_mnist(task_type='predefined',
                                save_pd=False, seed=42,
                                max_size=-1):
    """Load colored mnist"""

    colored_mnist = ColoredMNIST('.', None, None)
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

def load_datasets_rotated_mnist(task_type='predefined',
                                save_pd=False, seed=42,
                                max_size=-1):
    """Load rotated mnist"""

    rotated_mnist = RotatedMNIST('.', None, None)
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


def load_datasets_variations_mnist(task_type='predefined',
                                save_pd=False, seed=42,
                                max_size=-1):
    """Load rotated mnist"""

    rotated_mnist = MNISTVariations('.', None, None)
    envs = ['standard', 'images', 'random']
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



def load_datasets_colored_fashionmnist(task_type='predefined',
                                save_pd=False, seed=42,
                                max_size=-1):
    """Load colored mnist"""

    colored_mnist = ColoredFashionMNIST('.', None, None)
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

def load_datasets_rotated_fashionmnist(task_type='predefined',
                                save_pd=False, seed=42,
                                max_size=-1):
    """Load rotated mnist"""

    rotated_mnist = RotatedFashionMNIST('.', None, None)
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


def load_datasets_variations_fashionmnist(task_type='predefined',
                                save_pd=False, seed=42,
                                max_size=-1):
    """Load rotated mnist"""

    rotated_mnist = FashionMNISTVariations('.', None, None)
    envs = ['standard', 'images', 'random']
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


def load_datasets(name, task_type='predefined', nested_cv=False, save_np=False,
                  save_pd=False, seed=42, max_size=-1):
    # fix random seed for reproducibility
    np.random.seed(seed)
    if name == 'majorca':
        X, y, inner_cv, \
         outer_cv, task_info = load_datasets_majorca(task_type,
                                                     save_pd)
    elif name == 'tenerife':
        X, y, inner_cv, \
         outer_cv, task_info = load_datasets_tenerife(task_type,
                                                     save_pd)
    elif name == 'maj+ten':
        X, y, inner_cv, \
         outer_cv, task_info = load_datasets_majten(task_type,
                                                     save_pd)
    elif name == 'stv':
        X, y, inner_cv, \
         outer_cv, task_info = load_datasets_stv(task_type,
                                                     save_pd)
    elif name == 'realstv':
        X, y, inner_cv, \
         outer_cv, task_info = load_datasets_realstv(task_type,
                                                     save_pd)
    elif name == 'abalone':
        X, y, inner_cv, \
         outer_cv, task_info = load_datasets_abalone(task_type,
                                                     save_pd)
    elif name == 'pm2_5beijing':
        X, y, inner_cv, \
         outer_cv, task_info = load_datasets_pm2_5beijing(task_type,
                                                          save_pd)
    elif name == 'pop_failures' or name == 'climate':
        X, y, inner_cv, \
         outer_cv, task_info = load_datasets_pop_failures(task_type,
                                                          save_pd)
    elif name == 'communities':
        X, y, inner_cv, \
         outer_cv, task_info = load_datasets_communities(task_type,
                                                         save_pd)
    elif name == 'cal_housing':
        X, y, inner_cv, \
         outer_cv, task_info = load_datasets_cal_housing(task_type,
                                                         save_pd)
    elif name == 'bos_housing':
        X, y, inner_cv, \
         outer_cv, task_info = load_datasets_bos_housing(task_type,
                                                         save_pd)
    elif name == 'landmine':
        X, y, inner_cv, \
         outer_cv, task_info = load_datasets_landmine(task_type,
                                                         save_pd)
    elif name == 'binding':
        X, y, inner_cv, \
         outer_cv, task_info = load_datasets_binding(task_type,
                                                         save_pd)
    elif name == 'adult':
        X, y, inner_cv, \
         outer_cv, task_info = load_datasets_adult(task_type,
                                                         save_pd)
    elif name == 'compas':
        X, y, inner_cv, \
         outer_cv, task_info = load_datasets_compas(task_type,
                                                         save_pd)
    elif name == 'wine':
        X, y, inner_cv, \
         outer_cv, task_info = load_datasets_wine(task_type,
                                                         save_pd)
    elif name == 'sarcos':
        X, y, inner_cv, \
         outer_cv, task_info = load_datasets_sarcos(task_type,
                                                         save_pd)
    elif name == 'mnist_variations':
        X, y, inner_cv, \
         outer_cv, task_info = load_datasets_mnist(task_type,
                                                         save_pd,
                                                         max_size=max_size)
    elif name == 'colored_mnist':
        X, y, inner_cv, \
         outer_cv, task_info = load_datasets_colored_mnist(task_type,
                                                         save_pd,
                                                         max_size=max_size)
    elif name == 'rotated_mnist':
       X, y, inner_cv, \
         outer_cv, task_info = load_datasets_rotated_mnist(task_type,
                                                         save_pd,
                                                         max_size=max_size)
    elif name == 'variations_mnist':
       X, y, inner_cv, \
         outer_cv, task_info = load_datasets_variations_mnist(task_type,
                                                         save_pd,
                                                         max_size=max_size)
    elif name == 'fashionmnist_variations':
        X, y, inner_cv, \
         outer_cv, task_info = load_datasets_fashionmnist(task_type,
                                                         save_pd)
    elif name == 'colored_fashionmnist':
        X, y, inner_cv, \
         outer_cv, task_info = load_datasets_colored_fashionmnist(task_type,
                                                         save_pd,
                                                         max_size=max_size)
    elif name == 'rotated_fashionmnist':
       X, y, inner_cv, \
         outer_cv, task_info = load_datasets_rotated_fashionmnist(task_type,
                                                         save_pd,
                                                         max_size=max_size)
    elif name == 'variations_fashionmnist':
        X, y, inner_cv, \
         outer_cv, task_info = load_datasets_variations_fashionmnist(task_type,
                                                         save_pd,
                                                         max_size=max_size)
    elif name == 'school':
        X, y, inner_cv, \
         outer_cv, task_info = load_datasets_school(task_type,
                                                         save_pd)
    elif name == 'computer':
        X, y, inner_cv, \
         outer_cv, task_info = load_datasets_computer(task_type,
                                                         save_pd)
    elif name == 'parkinson':
        X, y, inner_cv, \
         outer_cv, task_info = load_datasets_parkinson(task_type,
                                                         save_pd)
    elif name == 'isolet':
        X, y, inner_cv, \
            outer_cv, task_info = load_datasets_isolet(task_type,
                                                         save_pd)
    else:
        raise ValueError('{} is not a valid dataname'.format(name))

    if save_np:
        pass
        # np.savetxt('X_test_%s.csv' % name, X_test)
        # np.savetxt('y_test_%s.csv' % name, y_test)
        # np.savetxt('X_train_%s.csv' % name, X_train)
        # np.savetxt('y_train_%s.csv' % name, y_train)

    return X, y, inner_cv, outer_cv, task_info
