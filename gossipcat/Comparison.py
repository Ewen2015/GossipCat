#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
import warnings 
warnings.filterwarnings('ignore')

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.preprocessing import Imputer

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neural_network import MLPClassifier

class Comparison(object):
    """docstring for Comparison
    
    Args:
        scoring: a string format score, sklearn.metrics. 'roc_auc', 'average_precision'
    """
    def __init__(self, data, target, features, scoring, record_file):
        super(Comparison, self).__init__()
        self.data = data
        self.target = target
        self.features = features
        self.scoring = scoring
        self.record_file = record_file

        self.df_prep = pd.DataFrame(Imputer(strategy='mean').fit_transform(self.data), columns=self.data.columns)

        self.AmongResults = pd.DataFrame(columns=['algorithm', 'score_mean', 'score_std', 'time'])
        self.results = []
        self.names = []
        self.cost = []
        self.means = []
        self.stds = []

    def AmongModels(self):
        
        models = []
        models.append(('LR', LogisticRegression()))
        models.append(('SDG', SGDClassifier()))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('CART', DecisionTreeClassifier()))

        models.append(('BAG', BaggingClassifier(DecisionTreeClassifier(), bootstrap=True, oob_score=True, n_jobs=-1)))
        models.append(('RF', RandomForestClassifier(n_jobs=-1)))
        models.append(('ERT', ExtraTreesClassifier(n_jobs=-1)))
        models.append(('ABDT', AdaBoostClassifier(DecisionTreeClassifier())))
        models.append(('GBDT', GradientBoostingClassifier()))

        models.append(('MLP', MLPClassifier()))
        with open(self.record_file, 'a') as file:
            file.write('\n'+'='*20+'\n')
        for name, model in models:
            start = time.time()
            kfold = model_selection.KFold(n_splits=10, random_state=0)
            cv_results = model_selection.cross_val_score(model, self.df_prep[self.features], self.df_prep[self.target], cv=kfold, scoring=self.scoring)
            time_cost = time.time()-start
            score_mean = cv_results.mean()
            score_std = cv_results.std()
            msg = "%s:\t%f (%f)\ttime: %f s" % (name, score_mean, score_std, time_cost)
            with open(self.record_file, 'a') as file:
                file.write(msg)
            print(msg)
            self.results.append(cv_results)
            self.names.append(name)
            self.means.append(score_mean)
            self.stds.append(score_std)
            self.cost.append(time_cost)

        self.AmongResults['algorithm'] = self.names
        self.AmongResults['score_mean'] = self.means
        self.AmongResults['score_std'] = self.stds
        self.AmongResults['time'] = self.cost
        self.AmongResults['ratio'] = np.power(self.AmongResults.score_mean, 2)*np.power(self.AmongResults.time, -1/10)
        self.AmongResults = self.AmongResults.sort_values(by='score_mean', ascending=False)
        return self.AmongResults

    def Visual(self, time=False):
        fig = plt.figure(figsize=(8, 8))
        if not time:        
            ax = fig.add_subplot(111)
            plt.boxplot(self.results)
            ax.set_xticklabels(self.names)
            plt.title('Algorithm Comparison')
        else:
            fig.suptitle('Algorithm Comparison')

            ax1=fig.add_subplot(111, label="1")
            ax2=fig.add_subplot(111, label="2", frame_on=False)

            ax1.errorbar(self.names, self.means, self.stds, color="C0", linestyle='None', marker='o')
            ax1.set_xlabel("algorithm", color="C0")
            ax1.set_ylabel("score mean", color="C0")
            ax1.tick_params(axis="algorithm", colors="C0")
            ax1.tick_params(axis="score mean", colors="C0")

            ax2.bar(self.names, self.cost, color="C1", alpha=0.3, width=0.5)
            ax2.xaxis.tick_top()
            ax2.yaxis.tick_right()
            ax2.set_xlabel('algorithm', color="C1") 
            ax2.set_ylabel('time', color="C1")   
            ax2.xaxis.set_label_position('top') 
            ax2.yaxis.set_label_position('right') 
            ax2.tick_params(axis='algorithm', colors="C1")
            ax2.tick_params(axis='time', colors="C1")
        plt.grid()
        plt.show()
        return None
        



