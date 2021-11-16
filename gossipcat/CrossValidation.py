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

def search(train, 
           target, 
           features,
           folds,
           cv_parameters, 
           params, 
           range_max_depth=range(3, 10, 1),
           range_subsample=range(50, 91, 5),
           range_colsample_bytree=range(50, 91, 5),
           log_path=None):
    esr = cv_parameters['early_stopping_rounds']
    dtrain = xgb.DMatrix(data=train[features], label=train[target], silent=False, nthread=-1, enable_categorical=True)

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

def cv_rounds(data, target, features, folds, params, cv_parameters, path_cv_history, logger):
    dtrain = xgb.DMatrix(data[features], label=data[target], enable_categorical=True)

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

def train(dtrain, params, cv_parameters, num_boost_round, path_model, logger):

    booster = xgb.train(params, 
                          dtrain, 
                          num_boost_round, 
                          evals=[(dtrain, 'train')], 
                          obj=None, 
                          feval=None, 
                          maximize=cv_parameters['maximize'], 
                          early_stopping_rounds=cv_parameters['early_stopping_rounds'], 
                          evals_result=None, 
                          verbose_eval=True, 
                          xgb_model=None, 
                          callbacks=None)

    pickle.dump(booster, open(path_model, 'wb'))

    logger.info('best score: {}'.format(booster.best_score))
    logger.info('best iteration: {}'.format(booster.best_iteration))
    return booster


def predict(indcol, features, path_pred, path_model, path_result, logger):
    
    logger.info('load data from: {}'.format(path_pred))
    df_pred = pd.read_csv(path_pred)
    dpred = xgb.DMatrix(df_pred[features], enable_categorical=True)

    logger.info('load model from: {}'.format(path_model))
    booster = pickle.load(open(path_model, 'rb'))
    
    results = pd.DataFrame()
    results[indcol] = df_pred[indcol]

    logger.info('predicting')
    results['prob'] = booster.predict(dpred)
    results['pred'] = results['prob'].apply(lambda x: 1 if x >= 0.5 else 0)

    results.to_csv(path_result, index=False)
    logger.info('results saved in path: {}'.format(path_result))
    
    return results


def test(path_result, path_pred, logger):
    logger.info('load data from: {}'.format(path_result))
    results = pd.read_csv(path_result)

    logger.info('load data from: {}'.format(path_pred))
    df_pred = pd.read_csv(path_pred)

    results['target'] = df_pred['target']

    recall = recall_score(results['target'], results['pred'], average='binary')
    logger.info('test recall score: {}'.format(recall))

    vis = Visual(results['target'], results['prob'])
    vis.combo()
    return recall




