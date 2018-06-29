#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wang.enqun@outlook.com
license:    Apache License 2.0
"""
import matplotlib.pyplot as plt
import seaborn as sns

class Glimpse(object):
    """A glimpse at the dataset.
    Prints a general infomation of the dataset and plot the distribution 
    of target value.

    Args:
        data: A dataset you wanna glimpse.
        target: The target variable in your dataset; limited to binary.
        features: The features of your dataset.
    """
    def __init__(self, data, target, features):
        super(Glimpse, self).__init__()
        self.data = data
        self.target = target
        self.features = features

        self.targetList = self.data[self.target].unique().tolist()

    def Summary(self):
        print('\nInformation:')
        print(self.data.info())
        print('\nHead:')
        print(self.data.head())
        print('\nShape:')
        print(self.data.shape)
        print('\nTarget Rate:')
        print(round(len(self.data[self.data[self.target]==self.data[self.target].unique()[1]])/self.data.shape[0]*100, 2), '%')
        print('\nTarget Distribution:')
        self.data[self.target].plot.hist()
        return None

    def BiBoxplot(self):
        for f in self.features:
            plt.figure(figsize=(16, 1))
            sns.boxplot(y=self.target, x=f, data=self.data, orient='h', width=0.4, fliersize=0.3)
            plt.show()
        return None

    def BiDensity(self):
        for f in self.features:
            plt.figure()
            for cat in self.targetList:
              ax = sns.kdeplot(self.data[f][self.data[self.target]==cat], shade=True, label=cat)
              ax.set(xlabel=f, ylabel='density')
              ax.legend(title=self.target)
        return None

        