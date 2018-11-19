#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
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
        features_num: The features_num of your dataset.
    """
    def __init__(self, data, target, features_num, features_cat=None):
        super(Glimpse, self).__init__()
        self.data = data
        self.target = target
        self.features_num = features_num
        self.features_cat = features_num

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
        for f in self.features_num:
            plt.figure(figsize=(16, 1))
            sns.boxplot(y=self.target, x=f, data=self.data, orient='h', width=0.4, fliersize=0.3)
            plt.show()
        return None

    def BiDensity(self):
        for f in self.features_num:
            plt.figure()
            for cat in self.targetList:
              ax = sns.kdeplot(self.data[f][self.data[self.target]==cat], shade=True, label=cat)
              ax.set(xlabel=f, ylabel='density')
              ax.legend(title=self.target)
        return None

    def DenBox(self):
        for f in self.features_num:
            fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(16, 6))
            fig.suptitle('Distribution Comparison')
            
            for cat in self.targetList:
              sns.kdeplot(self.data[f][self.data[self.target]==cat], shade=True, label=cat, ax=ax1)
              ax1.set(ylabel='density')
              ax1.legend(title=self.target)

            sns.boxplot(y=self.target, x=f, data=self.data, orient='h', width=0.4, fliersize=0.3, ax=ax2)
            ax1.get_shared_x_axes().join(ax1, ax2)
            
            fig.subplots_adjust(hspace=0)
            plt.show()
        return None







        