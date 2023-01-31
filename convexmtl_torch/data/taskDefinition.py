import numpy as np
import pandas as pd
from datetime import date, datetime
from itertools import combinations
import math

### SOLAR tasks ------------------------------------------------------

def _task_by_hour_tenerife(row):
    return _task_by_hour_solar(row)

def _task_by_hour_majorca(row):
    return _task_by_hour_solar(row)

def _task_by_hour_solarmult(row):
    return _task_by_hour_solar(row)

def _task_by_hour_solar(row):
    return int(row.name.hour)

def _task_by_season_tenerife(row):
    return _task_by_season_solar(row)

def _task_by_season_majorca(row):
    return _task_by_season_solar(row)

def _task_by_season_solarmult(row):
    return _task_by_season_solar(row)

def _task_by_season_solar(row):   
    date_row = row.name
    return task_from_season_solar(date_row)

def task_from_season_solar(_date):
    Y = 2000 # dummy leap year to allow input X-02-29 (leap day)
    date_aux = _date.replace(year=Y)
    seasons = [('winter', (date(Y,  1,  1),  date(Y,  2, 15))),
               ('spring', (date(Y,  2, 16),  date(Y,  5, 15))),
               ('summer', (date(Y,  5, 16),  date(Y,  8, 15))),
               ('autumn', (date(Y,  8, 16),  date(Y, 11, 15))),
               ('winter', (date(Y, 11, 16),  date(Y, 12, 31)))]
    return next(season for season, (start, end) in seasons if start <= date_aux <= end)
    

def _task_by_location_solarmult(row):
    return int(row['problem_task'])

start_time_majorca = '06:00'
end_time_majorca = '19:00'
start_time_tenerife = '07:00'
end_time_tenerife = '20:00'
start_time_mid = '10:00'
end_time_mid = '14:00'
start_time_mor = '06:00'
end_time_mor = '09:00'
start_time_eve = '15:00'
end_time_eve = '19:00'

def between_time(time, start, end):
    """Return true only if the time hour is between start and end."""
    start_dt = datetime.strptime(start, '%H:%M')
    end_dt = datetime.strptime(end, '%H:%M')
    return time.hour >= start_dt.hour and time.hour <= end_dt.hour


def _predefined_tasks(data):
    tasks_by_time = [[(start_time_mid, end_time_mid)],
                     [(start_time_mor, end_time_mor),
                      (start_time_eve, end_time_eve)]]
    ret_err = -1
    for i, task in enumerate(tasks_by_time):
        for (start, end) in task:
            if between_time(data, start, end):
                return i
    return ret_err


def _task_by_hour(data):
    return int(data.hour)


def task_id_solar(data, task_type):
    """Return the appropiate task id for each row."""
    return _task_by_hour(data)
    # if task_type == 'byHour':
    #     return _task_by_hour(data)
    # else:
    #     return _predefined_tasks(data)

## STV tasks --------------------------------------------------------

def task_from_season_stv(_date):
    Y = 2000 # dummy leap year to allow input X-02-29 (leap day)
    date_aux = _date.replace(year=Y)
    seasons = [('winter', (date(Y,  1,  1),  date(Y,  2, 15))),
               ('spring', (date(Y,  2, 16),  date(Y,  5, 15))),
               ('summer', (date(Y,  5, 16),  date(Y,  8, 15))),
               ('autumn', (date(Y,  8, 16),  date(Y, 11, 15))),
               ('winter', (date(Y, 11, 16),  date(Y, 12, 31)))]
    return next(season for season, (start, end) in seasons if start <= date_aux <= end)

# STV
def _task_by_timeOfDay_stv(row):
    time = _timeOfDay_stv(row)
    return taskFromTime(time)

def _timeOfDay_stv(row):
    name = row.name
    time = name.time().hour
    return time

def _task_by_season_stv(row):
    date = _date_stv(row)
    return task_from_season_stv(date)

def _task_by_month_stv(row):
    date = _date_stv(row)
    return date.month

def _date_stv(row):
    date_row = row.name
    return date_row

nTasks_angle_stv = 4
inc = 360 / nTasks_angle_stv
start = 20
dic_angle_stv = dict(zip(range(nTasks_angle_stv), [(start + inc * n)%360 for n in range(nTasks_angle_stv)]))
def _task_by_angle_stv(row):
    angle = _angle_stv(row)
    return task_from_angle(angle, dic_angle_stv)

def _angle_stv(row):
    angle = np.arctan2(row['100u'], row['100v'])
    angle = math.degrees(angle) + 180
    return angle

dic_velocity_stv = {'l': 4, 'm': 10}
def _task_by_velocity_stv(row):    
    vel = _velocity_stv(row)
    return task_from_velocity(vel, dic_velocity_stv)

def _velocity_stv(row):
    vel = np.sqrt(row['100u']**2 +  row['100v']**2) 
    return vel

# REALSTV
def _task_by_timeOfDay_realstv(row):
    name = row.name
    time = name.time().hour
    return taskFromTime(time)


nTasks_angle_realstv = 4
inc = 360 / nTasks_angle_realstv
start = 45    
dic_angle_realstv = dict(zip(range(nTasks_angle_realstv), [(start + inc * n)%360 for n in range(nTasks_angle_realstv)]))
def _task_by_angle_realstv(row):    
    angle = _angle_realstv(row)
    ret = task_from_angle(angle, dic_angle_realstv)
    return ret

def _angle_realstv(row):
    angle = row['Direction']
    return angle

# lim_l = 4
# lim_m = 7
dic_velocity_realstv = {'l': 5, 'm': 12}
def _task_by_velocity_realstv(row):
    vel = _velocity_realstv(row)
    return task_from_velocity(vel, dic_velocity_realstv)

def _velocity_realstv(row):
    return row['Speed']


# def task_from_angle(angle, nTasks_angle, start, scalingf):
#     step = 2 * np.pi / nTasks_angle
#     angle_scaled = angle * scalingf + np.pi
#     for i in range(nTasks_angle-1):
#         taskStart = (start * scalingf + i * step) 
#         taskEnd = (start * scalingf + (i+1) * step)
#         if angle_scaled > taskStart and angle_scaled  < taskEnd:
#             return i

#     return nTasks_angle-1

def task_from_angle(angle, dic_angle):
    for i, (k, a) in enumerate(dic_angle.items()):
        if angle < a:
            return i
    return 0


def task_from_velocity(vel, dic_vel):
    nTasks_vel = len(dic_vel)+1
    for i, (k, v) in enumerate(dic_vel.items()):
        if vel < v:
            return i
    return len(dic_vel)


def taskFromTime(hour):
    if 7 < hour < 20:
        return 1
    else:
        return 0


def get_dict(dataname, task):
    dic_name = 'dic_{}_{}'.format(task, dataname)
    possibles = globals().copy()
    possibles.update(locals())
    dic = possibles.get(dic_name)
    if not dic:
     raise NotImplementedError("Dictionary %s does not exist" % dic_name)
    return dic

# def task_from_angle_stv(row):
#     nTasks_angle = 4
#     start = math.radians(20)
#     deg2rad = 1
#     angle = np.arctan2(row['100u'], row['100v']) 
#     return task_from_angle(angle, nTasks_angle, start, deg2rad)

# def task_from_angle_realstv(row):
#     nTasks_angle = 4
#     start = 45
#     deg2rad =  (2 * np.pi) / 360
#     ret = task_from_angle(row['Direction'], nTasks_angle, start, deg2rad)
#     return ret

# def task_from_velocity_stv(row):
#     lim_l = 4
#     lim_m = 10
#     dic_vel = {'l': lim_l, 'm': lim_m}
#     vel = np.sqrt(row['100u']**2 +  row['100v']**2) 
#     return task_from_velocity(vel, dic_vel)

# def task_from_velocity_realstv(row):
#     lim_l = 4
#     lim_m = 7
#     dic_vel = {'l': lim_l, 'm': lim_m}
#     return task_from_velocity(row['Speed'], dic_vel) 



# def task_id_stv(data, task_type):
#     """Return the appropiate task id for each row."""
#     if task_type == 'timeOfDay':
#         pass
#         # taskDef = taskFromTime_stv
#     elif task_type == 'angle':
#         taskDef = task_from_angle_stv
#     elif task_type == 'velocity':
#         taskDef = task_from_velocity_stv
#     # elif task_type == 'ctlPred':
#     #     taskDef = taskCTLpred
#     #     nTasks = nTasks_pred
#     else:
#         print('{} is not a task definition'.format(task_type))
#     return taskDef(data)


# def task_id_realstv(data, task_type):
#     """Return the appropiate task id for each row."""
#     if task_type == 'timeOfDay':
#         pass
#         # taskDef = taskFromTime_realstv
#     elif task_type == 'angle':
#         taskDef = task_from_angle_realstv
#     elif task_type == 'velocity':
#         taskDef = task_from_velocity_realstv
#     # elif task_type == 'ctlPred':
#     #     taskDef = taskCTLpred
#     #     nTasks = nTasks_pred
#     else:
#         print('{} is not a task definition'.format(task_type))
#     return taskDef(data)


# adult

def _task_by_sex_adult(row):
    return row['sex']

def _task_by_race_adult(row):
    return row['race']


# compas

def _task_by_sex_compas(row):
    return row['sex']

def _task_by_race_compas(row):
    return row['race']


# BUILD TASK COLUMN -----------------------------------------------------------------------------------------------

def unified_task(row):
    col_names = list(row.index)
    task_name = ';'.join(['{}={}'.format(c, row[c]) for c in col_names])
    # print(task_name)
    return task_name

def build_task_column(df, dataname, task_type):
    '''
    Builds a task column from either single or multiple task type definitions
    '''
    # print(task_type)
    if isinstance(task_type, ((list, np.ndarray))):
        task_col_d = {}
        for t_type in task_type:
            task_col_d[t_type] = get_task_column(df, dataname, t_type)
        result_df = pd.concat(task_col_d.values(), axis=1, join='inner')
        # print(result_df)
        task_col = result_df.apply(unified_task, axis=1).to_frame(name='task')
        # print(task_col)
    else:
        task_col = get_task_column(df, dataname, task_type)
    return task_col

def get_task_column(df, dataname, task_type):
    '''
    Given a single task type definition, the task column is built
    '''
    try:
        _dataname = get_internal_name(dataname)
    except Exception as exp:
        print(exp)
        print('{}'.format(dataname))
    fun_name = '_task_by_{}_{}'.format(task_type, _dataname)
    possibles = globals().copy()
    possibles.update(locals())
    fun = possibles.get(fun_name)
    if not fun:
     raise NotImplementedError("Function %s not implemented" % fun_name)
    task_col = df.apply(lambda row: fun(row), axis=1).to_frame(name=task_type)
    return task_col

# GET TASKS LIST ---------------------------------------------------------------------------------------------------

def get_internal_name(problem_name):
    internal_names = {
        'majorca': ['majorca'],
        'tenerife': ['tenerife'],
        'solarmult': ['solarmult', 'maj+ten'],
        'stv': ['stv'],
        'realstv': ['realstv'],
        'adult': ['adult'],
        'compas': ['compas']
    }
    for name, l in internal_names.items():
        if problem_name in l:
            return name
    raise Exception('Invalid problem_name')


def getList_task_type(problem_name):
    tt_list_dic = {
        'majorca': ['hour', 'season'],
        'tenerife': ['hour', 'season'],
        'solarmult': ['hour', 'season', 'location'],
        'stv': ['timeOfDay', 'angle', 'velocity'], # season and month can be also a task definition
        'realstv': ['timeOfDay', 'angle', 'velocity'],
        'adult': ['sex', 'race'], 
        'compas': ['sex', 'race']
    }
    try:
        _problem_name = get_internal_name(problem_name)
    except Exception as exp:
        print(exp)
        print('{}'.format(problem_name))

    return tt_list_dic[_problem_name]


def get_comb_fromList(full_list):
    n = len(full_list)
    full_combs = []
    for k in range(1, n+1):
        combs = get_kcomb_fromList(full_list, k)
        full_combs.extend(combs)
    return full_combs


def get_kcomb_fromList(full_list, k):
    '''Returns a list with all sublists of k elements in full_list'''
    try:
        combs = combinations(full_list, k)
    except Exception as ex:
        print('Error in combinations: {}'.format(ex))
        exit()
    else:
        els = [list(x) for x in combs]
        return els


# l = getList_task_type('stv')
# print(l)
# comb = get_comb_fromList(l)
# print(comb)