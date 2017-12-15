#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wang.enqun@outlook.com
license:    Apache License 2.0
"""
import os
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR
from .SimAnneal import SimulatedAnneal


PARAM = {
    'max_depth': [i for i in range(1, 11, 1)],
    'subsample': [i / 10.0 for i in range(1, 11, 1)],
    'colsample_bytree': [i / 10.0 for i in range(1, 11, 1)],
}


def glimpse(data, target, predictors):
    """A glimpse at the dataset.

    Prints a general infomation of the dataset and plot the distribution 
    of target value.

    Args:
        data: A dataset you wanna glimpse.
        target: The target variable in your dataset; limited to binary.
        predictors: The predictors of your dataset.
    """
    print('\nInformation:')
    print(data.info())
    print('\nHead:')
    print(data.head())
    print('\nShape:')
    print(data.shape)
    print('\nTarget Rate:')
    print(data[target].values.sum() / data.shape[0])
    print('\nCorrelated Predictors(0.9999):')
    print(corr_pairs(data[predictors], gamma=0.9999))
    print('\nCorrelated Predictors(0.9):')
    print(corr_pairs(data[predictors], gamma=0.9))
    print('\nCorrelated Predictors(0.85):')
    print(corr_pairs(data[predictors], gamma=0.85))
    print('\nTarget Distribution:')
    data[target].plot.hist()

    return None


def simAnneal(train, target, predictors, cate_features=None, params=PARAM, results=True, seed=2017):
    """ Hyper parameter tuning with simulated annealing.

    Employes the simulated annealing to find the optimal hyper parameters and 
    return an optimized classifier.

    Args:
        train: A training set of your machine learning project.
        target: The target variablet; limited to binary.
        predictors: The predictors.
        cate_features: The categorical features.
        param: A hyper parameter dictionary for tuning task; default with param_1.
        results: Whether print the progress out; default with True.

    Returns:
        optimized_clf: An optimized classifier after hyper parameter tuning.
    """
    print('\nsimulating...')

    gbm = LGBMClassifier(
        learning_rate=0.01, n_estimators=5000, objective='binary', metric='auc', 
        max_depth=2, subsample=0.75, colsample_bytree=0.75, categorical_feature=cate_features, 
        save_binary=True, is_unbalance=True, random_state=seed
    )

    sa = SimulatedAnneal(gbm, params, scoring='roc_auc', T=10.0, T_min=0.001, alpha=0.75,
                         n_trans=5, max_iter=0.25, max_runtime=300, cv=5, 
                         random_state=seed, verbose=True, refit=True, n_jobs=1)
    sa.fit(train[predictors], train[target])

    if results:
        print('\nbest score: ', sa.best_score_,
              '\nbest parameters: ', sa.best_params_)

    optimized_clf = sa.best_estimator_

    return optimized_clf


def main():
    pass

if __name__ == '__main__':
    main()
