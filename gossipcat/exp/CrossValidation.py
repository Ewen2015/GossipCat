#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
import pandas as pd
import xgboost as xgb
import pickle
from sklearn.metrics import recall_score

from gossipcat.Logging import get_logger
from gossipcat.Report import Visual

def date_delta(date, delta, ago=True, format='%Y-%m-%d'):
    from datetime import datetime, timedelta
    if ago:
        return (datetime.strptime(date, format) - timedelta(days=delta)).strftime(format)
    else:
        return (datetime.strptime(date, format) + timedelta(days=delta)).strftime(format)

def time_fold(df, col_date, n_splits=12, delta=30*3, step=30*1):
    """provides train/valid indices to split time series data samples that are observed at 
    fixed time intervals, in train/valid sets.

    Arg:
        df: training data frame
        col_date: date column for splitting
        n_splits: number of splits
        delta: test size in days
        step: length of sliding window

    Returns:
        folds: train/valid indices
    """
    date_max = df[col_date].max()
    date_min = df[col_date].min()
    date_range = round((pd.to_datetime(date_max) - pd.to_datetime(date_min)).days/30, 2)
    print('time window: {} to {}'.format(date_min, date_max))
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

def grid_search(train, 
                target, 
                features,
                folds, 
                general_parameters=None,
                learning_parameters=None,
                cv_parameters=None, 
                range_max_depth=range(3, 10, 1),
                range_subsample=range(50, 91, 5),
                range_colsample_bytree=range(50, 91, 5),
                log_path='log_grid_search.log'):
    """ grid search for hyter parameters of xgboost.

    Args:
        train: training data, df
        target: target column, str
        features: feature columns, list
        folds: cross-validation folds indices
        general_parameters: general parameters for gradient boosting decision trees, dict
        learning_parameters: optimization parameters for gradient boosting decision trees, dict 
        cv_parameters: cross-validation parameters, dict
        range_max_depth: range of max_depth, range
        range_subsample: range of subsample, range
        range_colsample_bytree: range of colsample_bytree, range
        log_path: logging path, path

    Returns:
        None
    """
    esr = cv_parameters['early_stopping_rounds']
    dtrain = xgb.DMatrix(data=train[features], label=train[target], silent=False, nthread=-1, enable_categorical=True)

    if general_parameters == None:
        general_parameters = {
            'eta': 0.1,
            'gamma': 0,
            'max_depth': 3,
            'min_child_weight': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.5, 
            'colsample_bylevel': 0.75, 
            'colsample_bynode': 1,
        #     'lamdba': 5,
            'alpha': 0.2,
            'tree_method': 'hist',
            'scale_pos_weight': scale_pos_weight
        }
    if learning_parameters == None:
        learning_task_parameters = {
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'tree_method': 'hist'
        }
    if cv_parameters == None:
        cv_parameters = {
            'num_boost_round': 1000,
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

def cv_rounds(data, 
                target, 
                features, 
                folds, 
                logger,
                general_parameters=None,
                learning_parameters=None,
                cv_parameters=None, 
                path_cv_history='log_cv_bst_rnd.log'):
    """ cross validation to find best round of xgboost.

    Args:
        data: training data, df
        target: target column, str
        features: feature columns, list
        folds: cross-validation folds indices
        general_parameters: general parameters for gradient boosting decision trees, dict
        learning_parameters: optimization parameters for gradient boosting decision trees, dict 
        cv_parameters: cross-validation parameters, dict
        range_max_depth: range of max_depth, range
        range_subsample: range of subsample, range
        range_colsample_bytree: range of colsample_bytree, range
        path_cv_history: logging path, path
        logger: gossipcat.Logging.get_logger

    Returns:
        best_round: best round in given parameters
    """
    dtrain = xgb.DMatrix(data[features], label=data[target], enable_categorical=True)
    
    if general_parameters == None:
        general_parameters = {
            'eta': 0.1,
            'gamma': 0,
            'max_depth': 3,
            'min_child_weight': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.5, 
            'colsample_bylevel': 0.75, 
            'colsample_bynode': 1,
        #     'lamdba': 5,
            'alpha': 0.2,
            'tree_method': 'hist',
            'scale_pos_weight': scale_pos_weight
        }
    if learning_parameters == None:
        learning_task_parameters = {
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'tree_method': 'hist'
        }
    if cv_parameters == None:
        cv_parameters = {
            'num_boost_round': 1000,
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
    logger.info('total rounds: {}'.format(cvr.shape[0]))
    
    best_round = cvr.shape[0]
    logger.info('best round: {}'.format(best_round))
    
    cvr.to_csv(path_cv_history)
    logger.info('cv history saved to {}'.format(path_cv_history))
    return best_round