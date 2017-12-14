#!/usr/bin/env python
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


param_1 = {
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


def features_dup(data, n_head=5000, print_dup=False):
    """ Obtain duplicated features.

    Checks first n_head rows and obtains duplicated features.

    Args:
        data: A dataset you wanna check and return duplicated columns names.
        n_head: First n_head rows to be checked; default 5000.
        print_dup: Whether print duplicated columns names; default with False.

    Returns:
        A list of the names of duplicatec columns.
    """
    dup_features = []
    if data.head(n_head).T.duplicated().any():
        dup_list = np.where(data.head(n_head).T.duplicated())[0].tolist()
        dup_features = data.columns[dup_list]
        if print_dup:
            print(dup_features)

    return dup_features.tolist()


def features_clf(data, features):
    """ Feature classification.

    Divides features into sublists according to their data type.

    Args:
        data: A dataset which you wanna classify features into subsets 
            according to the data type.
        features: A list of column names.

    Returns:
        int_features: A list of column names of int features.
        float_features: A list of column names of float features.
        object_features: A list of column names of object features.
    """
    dtypes = data[features].dtypes.apply(lambda x: x.name).to_dict()
    int_features, float_features, object_features = [], [], []

    for col, dtype in dtypes.items():
        if dtype == 'int64':
            int_features.append(col)
        elif dtype == 'float64':
            float_features.append(col)
        elif dtype == 'object':
            object_features.append(col)

    return int_features, float_features, object_features


def corr_pairs(data, gamma=0.9):
    """ Detect corralted features.

    Computes correlated feature pairs with correlated coefficient larger than gamma.

    Args:
        data: A dataset which you wanna detect corralted features from.
        gamma: The correlated coefficiency; default at 0.9.

    Returns:
        pairs: A list of correlated features.
    """
    pairs = []
    corr_matrix = data.corr().abs()
    os = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
          .stack()
          .sort_values(ascending=False))
    pairs = os[os > gamma].index.values.tolist()
    return pairs


def features_new(data, corr_list, target, auc_score=0.75, silent=False):
    """ Build new features from correlated features.

    Builds new features based on correlated feature pairs if the new feature 
    has an auc greater than auc_score.

    Args:
        data: A dataset which you wanna generate new features from.
        corr_list: The correlated list to generate new features from.
        target: The target variable in your dataset; limited to binary.
        auc_score: The auc to decide whether generate features, default at 0.75.
        silent: Whether print the new features' names out; default with False.

    Returns:
        new: A dataset conatianing new features.
    """
    new = pd.DataFrame()

    for index, value in enumerate(corr_list):
        temp = data[corr_list[index][0]] - data[corr_list[index][1]]
        if len(temp.unique()) > 1:
            temp = pd.DataFrame(temp.fillna(temp.median()))
            lr = LR()
            lr.fit(temp, data[target])
            prob = lr.predict_proba(temp)[:, 1]
            auc = metrics.roc_auc_score(data[target], prob)
            if auc > auc_score:
                if silent:
                    print('-'.join(value), ' AUC (train): ', auc)
                new['-'.join(value)] = data[corr_list[index][0]] - data[corr_list[index][1]]

    return new


def simAnneal(train, target, predictors, cate_features=None, params=param_1, results=True, seed=2017):
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
        print('\n best score: ', sa.best_score_,
              '\n best parameters: ', sa.best_params_)
    optimized_clf = sa.best_estimator_

    return optimized_clf


def main():
    pass

if __name__ == '__main__':
    main()
