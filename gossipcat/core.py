#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wang.enqun@outlook.com
license:    Apache License 2.0
"""
from lightgbm import LGBMClassifier
from .Feature import Feature
from .SimAnneal import SimulatedAnneal
from .Report import Report


def simAnneal(train, target, predictors, results=True, seed=2017):
    """ Hyper parameter tuning with simulated annealing.

    Employes the simulated annealing to find the optimal hyper parameters and 
    return an optimized classifier.

    Args:
        train: A training set of your machine learning project.
        target: The target variablet; limited to binary.
        predictors: The predictors.
        results: Whether print the progress out; default with True.

    Returns:
        optimized_clf: An optimized classifier after hyper parameter tuning.
    """
    params = {
        'max_depth': [i for i in range(1, 11, 1)],
        'subsample': [i / 10.0 for i in range(1, 11, 1)],
        'colsample_bytree': [i / 10.0 for i in range(1, 11, 1)],
    }

    print('simulating...')

    gbm = LGBMClassifier(
        learning_rate=0.01, n_estimators=5000, objective='binary', metric='auc', 
        max_depth=2, subsample=0.75, colsample_bytree=0.75, 
        save_binary=True, is_unbalance=True, random_state=seed
    )

    sa = SimulatedAnneal(gbm, params, scoring='roc_auc', T=10.0, T_min=0.001, alpha=0.75,
                         n_trans=5, max_iter=0.25, max_runtime=300, cv=5, 
                         random_state=seed, verbose=True, refit=True, n_jobs=1)
    sa.fit(train[predictors], train[target])

    if results:
        print('\nbest score:', '{:.6f}'.format(sa.best_score_),
              '\nbest parameters:', str({key: '{:.2f}'.format(value) for key, value in sa.best_params_.items()}))

    optimized_clf = sa.best_estimator_

    return optimized_clf


def bayesOpt(train, target, predictors, results=True, seed=2018):
    """ Hyper parameter tuning with Bayesian optimization.

    Employes the Bayesian optimization to find the optimal hyper parameters and 
    return an optimized classifier.

    Args:
        train: A training set of your machine learning project.
        target: The target variablet; limited to binary.
        predictors: The predictors.
        results: Whether print the progress out; default with True.
        seed: The random state.
        
    Returns:
        None.
    """
    print('optimizing...')
    
    y = dtrain[target]
    X = dtrain[predictors]

    Dtrain = lgb.Dataset(X, label = y)

    def lgb_evaluate(max_depth, colsample_bytree, subsample):
    
        params = {
            'eta': 0.01,
            'silent': 1,
            'num_boost_round':3000,
            'early_stopping_round':20,
            'n_fold':5,
            'verbose_eval': True,
            'seed': seed
        }

        params['max_depth'] = int(max_depth)
        params['cosample_bytree'] = max(min(colsample_bytree, 1), 0)   
        params['subsample'] = max(min(subsample, 1), 0)

        cv_result = lgb.cv(params, Dtrain, metrics='auc')
        return cv_result['auc-mean'][-1]

    lgbBO = BayesianOptimization(lgb_evaluate, {'max_depth': (1, 20),
                                                'colsample_bytree': (0.1, 1),
                                                'subsample': (0.1, 1)})
    lgbBO.maximize(init_points=5, n_iter=25)

    if results:
        print('\nbest score:', '{:.6f}'.format(lgbBO.res['max']['max_val']),
              '\nbest parameters:', str({key: '{:.2f}'.format(value) for key, value in lgbBO.res['max']['max_params'].items()}))

    return None