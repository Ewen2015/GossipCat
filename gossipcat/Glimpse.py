#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wang.enqun@outlook.com
license:    Apache License 2.0
"""
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