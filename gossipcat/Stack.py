#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
import numpy as np
from vecstack import stacking

from sklearn.linear_model import LogisticRegression
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

from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import log_loss, average_precision_score

import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')

class Stack(object):
    """docstring for Stack"""
    def __init__(self, train, test, target, features):
        super(Stack, self).__init__()
        self.train = train
        self.test = test 
        self.target = target
        self.features = features

        self.X_train = self.train[self.features]
        self.y_train = self.train[self.target]
        self.X_test = self.test[self.features]
        self.y_test = self.test[self.target]

    def Level_1(self, append_model=[]):
        self.models = [
            LogisticRegression(random_state=0),
            LinearDiscriminantAnalysis(),
            KNeighborsClassifier(),
            GaussianNB(),
            DecisionTreeClassifier(random_state=0),
            BaggingClassifier(DecisionTreeClassifier(random_state=0), bootstrap=True, oob_score=True, n_jobs=-1, random_state=0),
            RandomForestClassifier(n_jobs=-1, random_state=0),
            ExtraTreesClassifier(n_jobs=-1, random_state=0),
            AdaBoostClassifier(DecisionTreeClassifier(random_state=0), random_state=0),
            GradientBoostingClassifier(random_state=0),
            MLPClassifier(random_state=0)
        ]
        if len(append_model)==0:
            pass
        else:
            for m in append_model:
                self.models.append(append_model)

        self.S_train, self.S_test = stacking(self.models,
                                             np.array(self.X_train), np.array(self.y_train), np.array(self.X_test),
                                             regression=False,
                                             mode='oof_pred',
                                             needs_proba=True,
                                             save_dir='.',
                                             metric=log_loss,
                                             n_folds=5,
                                             stratified=True,
                                             shuffle=True,
                                             random_state=0,
                                             verbose=2)
        return None

    def Level_2(self):
        self.model = LogisticRegression(random_state=0)
        self.model.fit(self.S_train, self.y_train)
        return self.model

    def Visual(self):
        self.y_test_pred = self.model.predict(self.S_test)
        self.y_test_prob = self.model.predict_proba(self.S_test)[:,1]

        precision, recall, _ = precision_recall_curve(self.y_test, self.y_test_prob)
        average_precision = average_precision_score(self.y_test, self.y_test_prob)

        print('\nModel Report')
        print('Average Precision: {0:0.3f}'.format(average_precision))
        print(classification_report(self.y_test, self.y_test_pred))

        plt.figure(figsize=(8, 7))
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.5, color='red')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve: AP={0:0.3f}'.format(average_precision))
        plt.show()
        return None
        













        