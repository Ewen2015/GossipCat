#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
from .SimAnneal import SimulatedAnneal


class Tune(object):
    """ Hyper parameter tuning."""
    def __init__(self, train, target, predictors, metric, scoring):
        """
        Args: 
            train: A training set of your machine learning project.
            target: The target variablet; limited to binary.
            predictors: The predictors.
            metric: for lightgbm, 'auc';  for bayes_opt, 'binary_logloss'
            scoring: for simulated annealing, 'roc_acu'; for sim_anneal, 'neg_log_loss'
        """
        super(Tune, self).__init__()
        self.train = train
        self.target = target
        self.predictors = predictors
        self.metric = metric    
        self.scoring = scoring

    def simpleTrain(self):

        train, test = train_test_split(self.train, test_size=0.2, random_state=0)
        lgb_train = lgb.Dataset(train[self.predictors], train[self.target], free_raw_data=False)
        lgb_valid = lgb.Dataset(test[self.predictors], test[self.target], reference=lgb_train, free_raw_data=False)
        
        params = {
        'boosting_type': 'gbdt',
        'objective': 'bianry',
        'metric': 'auc',
        'num_leaves': 64,
        'learning_rate': 0.01,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'verbose': 0
        }

        print('training...')
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=10000,
                        valid_set=lgb_valid,
                        early_stopping_round=200,
                        verbose_eval=100)
        return gbm

    def simAnneal(self, alpha=0.75, n_trans=10, results=True, seed=2017):
        """ Hyper parameter tuning with simulated annealing.

        Employes the simulated annealing to find the optimal hyper parameters and 
        return an optimized classifier.

        Args:
            alpha:  
            n_trans: 
            results: Whether print the progress out; default with True.
            seed: 2017

        Returns:
            A classifier generated from gbdt with simulated annealing hyper parameter tuning.
        """
        params = {
            'max_depth': [i for i in range(3, 11, 1)],
            'subsample': [i / 10.0 for i in range(1, 11, 1)],
            'colsample_bytree': [i / 10.0 for i in range(1, 11, 1)],
        }

        print('simulating...')

        gbm = LGBMClassifier(
            learning_rate=0.01, 
            n_estimators=5000, 
            objective='binary', 
            metric=self.metric,
            n_leaves=1025, 
            max_depth=2, 
            subsample=0.75, 
            colsample_bytree=0.75, 
            save_binary=True, 
            is_unbalance=True, 
            random_state=seed
        )
        sa = SimulatedAnneal(gbm, 
            params, 
            scoring=self.scoring, 
            T=10.0, 
            T_min=0.001, 
            alpha=alpha,
            n_trans=n_trans, 
            max_iter=0.25, 
            max_runtime=300, 
            cv=5, 
            andom_state=seed, 
            verbose=True, 
            refit=True, 
            n_jobs=1)
        
        sa.fit(self.train[self.predictors], self.train[self.target])

        if results:
            print('\nbest score:', '{:.6f}'.format(sa.best_score_),
                  '\nbest parameters:', str({key: '{:.2f}'.format(value) for key, value in sa.best_params_.items()}))

        return sa


    def bayesOpt(self, results=True, seed=2018):
        """ Hyper parameter tuning with Bayesian optimization.

        Employes the Bayesian optimization to find the optimal hyper parameters and 
        return an optimized classifier.

        Args:
            results: Whether print the progress out; default with True.
            seed: The random state.
            
        Returns:
            A classifier generated from gbdt with Bayesian optimization hyper parameter tuning.
        """
        print('optimizing...')

        Dtrain = lgb.Dataset(self.train[self.predictors], label = self.train[self.target])

        def lgb_evaluate(max_depth, colsample_bytree, subsample):
        
            params = {
                'eta': 0.01,
                'silent': 1,
                'num_boost_round':10000,
                'early_stopping_round':100,
                'n_fold':5,
                'verbose_eval': True,
                'seed': seed
            }

            params['max_depth'] = int(max_depth)
            params['cosample_bytree'] = max(min(colsample_bytree, 1), 0)   
            params['subsample'] = max(min(subsample, 1), 0)

            cv_result = lgb.cv(params, Dtrain, metrics=self.metric)
            if self.metric == 'binary_logloss':
                return -np.array(cv_result[list(cv_result.keys())[0]]).max()
            else:
                return np.array(cv_result[list(cv_result.keys())[0]]).max()

        lgbBO = BayesianOptimization(lgb_evaluate, {'max_depth': (3, 10),
                                                    'colsample_bytree': (0.1, 1),
                                                    'subsample': (0.1, 1)})
        lgbBO.maximize(init_points=5, n_iter=25)

        if results:
            print('\nbest score:', '{:.6f}'.format(lgbBO.res['max']['max_val']),
                  '\nbest parameters:', str({key: '{:.2f}'.format(value) for key, value in lgbBO.res['max']['max_params'].items()}))

        return lgbBO