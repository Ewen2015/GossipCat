#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""

general_parameters = {
    'eta': 0.1,
    'gamma': 0,
    'max_depth': 3,
    'min_child_weight': 0.01,
    'subsample': 0.75,
    'colsample_bytree': 0.75, 
    'colsample_bylevel': 0.75, 
    'colsample_bynode': 1,
    'lamdba': 5,
    'alpha': 0.2,
    'tree_method': 'hist',
    'scale_pos_weight': scale_pos_weight
}

learning_task_parameters = {
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',
    'tree_method': 'hist'
}

cv_parameters = {
    'num_boost_round': 9000,
    'nfold': 12,
    'stratified': True,
    'metrics': ('aucpr'),
    'maximize': True,
    'early_stopping_rounds': 20,
    'verbose_eval': True,
    'shuffle': True,
    'seed': 0
}

params = {**general_parameters, **learning_task_parameters}

def date_delta(date, delta, ago=True, format='%Y-%m-%d'):
    """
    arg:
        date: anchor date, str
        delta: delta days, int
        ago: before the anchor date or after, boolean
        format: return date format, default '%Y-%m-%d'
    return:
        target date: str
    """
    from datetime import datetime, timedelta
    if ago:
        return (datetime.strptime(date, format) - timedelta(days=delta)).strftime(format)
    else:
        return (datetime.strptime(date, format) + timedelta(days=delta)).strftime(format)


def time_fold(df, col_date, n_splits=12, delta=30*3, step=30*1):

    date_max = df[col_date].max()
    date_min = df[col_date].min()
    date_range = round((pd.to_datetime(date_max) - pd.to_datetime(date_min)).days/30, 2)
    print('time window: {} to {}'.format(date_max, date_min))
    print('time window coverage: {} months'.format(date_range))

    if date_range < n_splits:
        return 'ERROR: time window does not cover split range.'

    date_split = list()
    date_end = list()
    date_end.append(date_max)

    folds = list()

    for i in range(n_splits):
        d_split = date_delta(date_end[i], delta)
        d_end = date_delta(date_end[i], step)
        date_split.append(d_split)
        date_end.append(d_end)
        
        print('{} to {}'.format(date_split[i], date_end[i]))
        
        dtrain = df[(df[col_date] < date_split[i])].index
        dvalid = df[(df[col_date] >= date_split[i]) & (df[col_date] < date_end[i])].index
        
        folds.append((dtrain, dvalid))
    return folds


def cv_rounds(data, target, features, folds, params, cv_parameters, path_cv_history):
    dtrain = xgb.DMatrix(data[features], label=data[target])

    cvr = xgb.cv(params, 
               dtrain, 
               num_boost_round=cv_parameters['num_boost_round'], 
               nfold=len(folds), 
               stratified=False, 
               folds=folds, 
               metrics=cv_parameters['metrics'], 
               obj=None, 
               feval=None, 
               maximize=cv_parameters['maximize'], 
               early_stopping_rounds=cv_parameters['early_stopping_rounds'], 
               fpreproc=None, 
               as_pandas=True, 
               verbose_eval=cv_parameters['verbose_eval'], 
               show_stdv=True, 
               seed=cv_parameters['seed'], 
               callbacks=None, 
               shuffle=cv_parameters['shuffle'])
    print('total rounds: {}'.format(cvr.shape[0]))
    
    best_round = cvr.shape[0] - cv_parameters['early_stopping_rounds']
    print('best round: {}'.format(best_round))
    
    cvr.to_csv(path_cv_history)
    print('cv history saved to {}'.format(path_cv_history))
    return best_round


def search(train, 
           target, 
           features,
           folds,
           cv_parameters=cv_parameters, 
           params=params, 
           range_max_depth=range(3, 10, 1),
           range_subsample=range(50, 91, 5),
           range_colsample_bytree=range(50, 91, 5),
           log_path=None):

    esr = cv_parameters['early_stopping_rounds']
    dtrain = xgb.DMatrix(data=train[features], label=train[target], silent=False, nthread=-1)

    with open(log_path, 'w') as f:
        f.write('max_depth,subsample,colsample_bytree,best_round,train_aucpr_mean,train_aucpr_std,test_aucpr_mean,test_aucpr_std\n')

    for d in range_max_depth:
        for s in range_subsample:
            for c in range_colsample_bytree:
                params['max_depth'] = d
                params['subsample'] = s/100
                params['colsample_bytree'] = c/100

                cvr = xgb.cv(params, 
                            dtrain, 
                            num_boost_round=cv_parameters['num_boost_round'],
                            stratified=False, 
                            folds=folds, 
                            metrics=cv_parameters['metrics'], 
                            obj=None, 
                            feval=None, 
                            maximize=cv_parameters['maximize'], 
                            early_stopping_rounds=esr, 
                            fpreproc=None, 
                            as_pandas=True, 
                            verbose_eval=cv_parameters['verbose_eval'], 
                            show_stdv=True, 
                            seed=cv_parameters['seed'], 
                            callbacks=None, 
                            shuffle=cv_parameters['shuffle'])

                total_rounds = cvr.shape[0] + esr
                print('total rounds: {}'.format(total_rounds))
                print('early stopping rounds: {}'.format(esr))
                print('best round: {}'.format(cvr.shape[0]))
                with open(log_path, 'a') as f:
                    f.write('%d,%f,%f,%d,%f,%f,%f,%f\n' % (params['max_depth'], 
                                                           params['subsample'], 
                                                           params['colsample_bytree'], 
                                                           cvr.index[-1],
                                                           cvr.tail(1)['train-aucpr-mean'],
                                                           cvr.tail(1)['train-aucpr-std'],
                                                           cvr.tail(1)['test-aucpr-mean'],
                                                           cvr.tail(1)['test-aucpr-std']))
    print('done.')
    return None




