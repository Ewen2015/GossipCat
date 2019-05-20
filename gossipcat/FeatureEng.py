#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""

def featureEngineering(df):
    return df 


def scoreIt(df, target, features):
    print('[score]starting...')
    import xgboost as xgb

    dtrain = xgb.DMatrix(df[features], label=df[target])

        
    nfold = 5
    learning_rate = 0.01
    n_rounds = 3000
    early_stopping = 1000
    verbose = 1
    seed = 2019

    message = '[score]cross validation started and will stop if performace did not improve in %d rounds.' % early_stopping
    print(message)
    params = {
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'eval_metric': 'auc',
        'eta': learning_rate,
        'gamma': 0,
        'min_child_weight': 0.01,
        'max_depth': 5,
        'max_delta_depth': 1,
        'subsample': 0.85,
        'colsample_bytree': 0.75,
        'colsample_bylevel': 0.75,
        'colsample_bynode': 1.0,
        'lambda': 5,
        'alpha': 0.2
    }

    cvr = xgb.cv(params=params,
                  dtrain=dtrain,
                  num_boost_round=n_rounds,
                  nfold=nfold,
                  stratified=True,
                  metrics='aucpr',
                  maximize=True,
                  early_stopping_rounds=early_stopping,
                  verbose_eval=verbose,
                  seed=seed)
    message = '[score]test %s: %.3f' % (params['eval_metric'], cvr.iloc[-1, 2])
    print(message)
    print('[score]baseline: ')
    return None













