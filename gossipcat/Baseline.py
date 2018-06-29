#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wang.enqun@outlook.com
license:    Apache License 2.0
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_slection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from .Report import Report

def Split(data):
    return train_test_split(data, test_size=0.2, random_state=0)

class Baseline(object):
    """Provide general machine learning models as baseline."""
    def __init__(self, train, valid, target, features):
        super(Baseline, self).__init__()
        self.target = target
        self.features = features

        self.train = train
        self.valid = valid

    def LR(self, feature_num=6, report='True'):
        """Logistic Regression.

        Args:
            feature_num: number of feaures to keep in the model.
            report: whether print out the model analysis report.
        Returns:
            Logistic regression model."""

        predictors=[]
        logreg_fs = LogisticRegression(n_jobs=-1)
        rfe = RFE(logreg_fs, feature_num)
        rfe = rfe.fit(self.train[self.features], self.train[self.target])
        for ind, val in enumerate(rfe.support_):
            if val: predictors.append(self.features[ind])

        logreg = LogisticRegression(n_jobs=-1)
        logreg.fit(self.train[predictors], self.train[self.target])

        if report:
            rpt = Report(logreg, self.train, self.valid, self.target, predictors)
            rpt.ALL()

        return logreg
    
    def RF(self, report='True'):
        """Random Forest.

        Args:
            feature_num: number of feaures to keep in the model.
            report: whether print out the model analysis report.
        Returns:
            Decision tree model generated from Random Forest."""
        rf = RandomForestClassifier(n_estimators=1000, 
                                    max_features='sqrt',
                                    max_depth=10,
                                    random_state=0, 
                                    n_jobs=-1)
        rf.fit(self.train[self.features], self.train[self.target])

        if report:
            rpt = Report(rf, self.train, self.valid, self.target, self.features)
            rpt.ALL()

        return rf

    def NN(self, report='True'):
        """Neutral Network.

        Args:
            report: whether print out the model analysis report.
        Returns:
            One layer neutral network model."""
        def baseline_model():
            model = Sequential()
            model.add(Dense(8, input_dim=len(self.features), activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model    

        estimator = KerasClassifier(build_fn=baseline_model, epochs=5, batch_size=5, verbose=1)
        estimator.fit(self.train[self.features], self.train[self.target])

        if report:
            rpt = Report(estimator, self.train, self.valid, self.target, self.features)
            rpt.ALL()

        return estimator