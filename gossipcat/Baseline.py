#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""

class Baseline(object):
    """Provide general machine learning models as baseline."""
    def __init__(self, train, test, target, features, impute=False):
        """
        Args:
            train: training set.
            test: test set.
            target: target variable, only support for binary classification issues.
            features: feature varibles.
            impurte: default False, whether the data needs imputation.
        """
        super(Baseline, self).__init__()
        self.target = target
        self.features = features

        self.train = train
        self.test = test
        
        if impute:
            import pandas as pd
            from sklearn.preprocessing import Imputer

            self.train_prep = pd.DataFrame(Imputer(strategy='mean').fit_transform(self.train), columns=self.train.columns)
            self.test_prep = pd.DataFrame(Imputer(strategy='mean').fit_transform(self.test), columns=self.test.columns)
        else:
            self.train_prep = self.train
            self.test_prep = self.test          

    def LR(self, report=False):
        """Logistic Regression.

        Args:
            feature_num: number of feaures to keep in the model.
            report: whether print out the model analysis report, default False.
        Returns:
            Logistic regression model.
        """
        from sklearn.linear_model import LogisticRegression

        self.lr = LogisticRegression(n_jobs=-1)
        self.lr.fit(self.train_prep[self.features], self.train_prep[self.target])

        if report:
            import matplotlib.pyplot as plt
            from .Report import Report
            rpt = Report(self.lr, self.train_prep, self.test_prep, self.target, self.features, predict=True, is_sklearn=True)
            rpt.ALL()
            plt.show()

        return self.lr
    
    def RF(self, report=False):
        """Random Forest.

        Args:
            report: whether print out the model analysis report.
        Returns:
            Decision tree model generated from Random Forest.
        """
        from sklearn.ensemble import RandomForestClassifier

        self.rf = RandomForestClassifier(n_estimators=1000, 
                                         max_features='sqrt',
                                         max_depth=10,
                                         random_state=0, 
                                         n_jobs=-1)
        self.rf.fit(self.train_prep[self.features], self.train_prep[self.target])

        if report:
            import matplotlib.pyplot as plt
            from .Report import Report
            rpt = Report(self.rf, self.train_prep, self.test_prep, self.target, self.features, predict=True, is_sklearn=True)
            rpt.ALL()
            plt.show()

        return self.rf

    def GBDT(self, report=False):
        """Gradient Boosting Decision Tree.

        Args:
            report: whether print out the model analysis report.
        Returns:
            Decision tree model generated from Gradient Boosting Decision Tree.
        """
        from xgboost.sklearn import XGBClassifier

        self.gbdt = XGBClassifier(objective='binary:logistic',
                                  booster='gbtree',
                                  learning_rate=0.01,
                                  n_estimators=5000,
                                  max_depth=3,
                                  subsample=0.75,
                                  colsample_bytree=0.75,
                                  n_jobs=4,
                                  random_state=2018)

        self.gbdt.fit(self.train_prep[self.features], self.train_prep[self.target])

        if report:
            import matplotlib.pyplot as plt
            from .Report import Report
            rpt = Report(self.gbdt, self.train, self.test, self.target, self.features, predict=True, is_sklearn=True)
            rpt.ALL()
            plt.show()

        return self.gbdt

    def NN(self, report=False):
        """Neutral Network.

        Args:
            report: whether print out the model analysis report.
        Returns:
            One layer neutral network model.
        """
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.wrappers.scikit_learn import KerasClassifier

        def baseline_model():
            model = Sequential()
            model.add(Dense(8, input_dim=len(self.features), activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model    

        self.nn = KerasClassifier(build_fn=baseline_model, epochs=1000, batch_size=100, verbose=1)
        self.nn.fit(self.train[self.features], self.train[self.target])

        if report:
            import matplotlib.pyplot as plt
            from .Report import Report
            rpt = Report(self.nn, self.train, self.test, self.target, self.features, predict=True, is_sklearn=True)
            rpt.ALL()
            plt.show()

        return self.nn