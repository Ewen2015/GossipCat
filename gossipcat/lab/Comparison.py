#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.impute import SimpleImputer

import warnings 
warnings.filterwarnings('ignore')

class Comparison(object):
    """ Machine Learning Algorithms Comparison
    """
    def __init__(self, df, target, features, metric=None, cv_n=10, log_path='algorithms_comparison.log'):
        """
  
        Args:
            df (pandas.DataFrame): A training set. 
            target (str): The target for supervised machine learning.
            features (list): The feature list for the model.
            metric (str): The metric used for cross-validation, sklearn.metrics. For classification, consider 'roc_auc', 'average_precision'; for regression, consider 'neg_root_mean_squared_error'.
            cv_n (int): The number of splits of cross-validation.
            log_path (str): The logging file. 

        """
        super(Comparison, self).__init__()
        self.df = df
        self.data = self.df[features+[target]]
        self.target = target
        self.features = features
        self.metric = metric
        self.cv_n = cv_n 
        self.log_path = log_path

        self.df_prep = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(self.data), columns=self.data.columns)

        self.Results = pd.DataFrame(columns=['algorithm', 'score_mean', 'score_std', 'time'])
        self.results = []
        self.names = []
        self.cost = []
        self.means = []
        self.stds = []

        if len(pd.unique(self.df[self.target])) == 2:
            self.classification = True 
        else:
            self.classification = False

        self.Results = self.compare(classification=self.classification)


    def classifers(self):
        """ Compare classification algorithms."""
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
        return models

    def regressors(self):
        """ Compare regression algorithms."""
        from sklearn.linear_model import LinearRegression
        from sklearn.linear_model import Ridge
        from sklearn.linear_model import Lasso
        from sklearn.linear_model import ElasticNet
        from sklearn.linear_model import Lars
        from sklearn.linear_model import BayesianRidge
        from sklearn.linear_model import SGDRegressor
        from sklearn.linear_model import PassiveAggressiveRegressor

        from sklearn.kernel_ridge import KernelRidge
        from sklearn.svm import SVR
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.tree import DecisionTreeRegressor

        from sklearn.ensemble import BaggingRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.ensemble import HistGradientBoostingRegressor

        models = []
        models.append(('LR', LinearRegression()))
        models.append(('RDG', Ridge()))
        models.append(('LSS', Lasso()))
        models.append(('ENR', ElasticNet()))
#         models.append(('LAR', Lars()))
        models.append(('BYS', BayesianRidge()))
#         models.append(('SDG', SGDRegressor()))
#         models.append(('PAR', PassiveAggressiveRegressor()))
        
        models.append(('KRNL', KernelRidge()))
        models.append(('SVM', SVR()))
        models.append(('KNB', KNeighborsRegressor()))
#         models.append(('GPR', GaussianProcessRegressor()))
        models.append(('PLS', PLSRegression()))
        models.append(('DTs', DecisionTreeRegressor()))

        models.append(('BAG', BaggingRegressor(n_jobs=-1)))
        models.append(('RF', RandomForestRegressor(n_jobs=-1)))
        models.append(('ABDT', AdaBoostRegressor()))
        models.append(('GBDT', GradientBoostingRegressor()))
        models.append(('HGB', HistGradientBoostingRegressor()))
        return models


    def compare(self, classification=True):
        """ Compare supervised machine learning algorithms. 

        Args:
            classification (bool): If the task is classification, default True.

        Return:
            Results: The comparison results.
        """
        self.classification = classification

        if self.classification:
            models = self.classifers()
        else:
            models = self.regressors()
        
        with open(self.log_path, 'a') as file:
            file.write('\n'+'='*20+'\n')
        for name, model in models:
            start = time.time()
            kfold = model_selection.KFold(n_splits=self.cv_n, shuffle=True, random_state=0)
            cv_results = model_selection.cross_val_score(model, self.df_prep[self.features], self.df_prep[self.target], cv=kfold, scoring=self.metric)
            time_cost = time.time()-start
            score_mean = cv_results.mean()
            score_std = cv_results.std()
            msg = "%s:\t%f (%f)\ttime: %f s" % (name, score_mean, score_std, time_cost)
            with open(self.log_path, 'a') as file:
                file.write(msg)
            print(msg)
            self.results.append(cv_results)
            self.names.append(name)
            self.means.append(score_mean)
            self.stds.append(score_std)
            self.cost.append(time_cost)

        self.Results['algorithm'] = self.names
        self.Results['score_mean'] = self.means
        self.Results['score_std'] = self.stds
        self.Results['time'] = self.cost
        self.Results['ratio'] = np.power(self.Results.score_mean, 2)*np.power(self.Results.time, -1/10)
        self.Results = self.Results.sort_values(by='score_mean', ascending=False)
        return self.Results

    def visualize(self, time=False, figsize=(8, 8)):
        """ Visualize the comparison results.

        Args:
            time (bool): Whether to include computing time in the chart, defualt False.
            figsize (tuple): The figure size for output, default (8, 8).

        Return:
            None
        """
        fig = plt.figure(figsize=figsize)
        color_score = 'darkslategray'
        color_time = 'firebrick'
        color_black = 'k'
        
        if not time:        
            ax = fig.add_subplot(111)
            plt.boxplot(self.results, 
                        patch_artist=True, 
                        boxprops=dict(facecolor='firebrick'),
            #             capprops=dict(color=c),
            #             whiskerprops=dict(color=c),
            #             flierprops=dict(color=c, markeredgecolor=c),
                        medianprops=dict(color='lightgray')
                       )
            ax.set_xticklabels(self.names)
            plt.ylabel(self.metric)
            plt.title('Algorithm Comparison')
        else:
            fig.suptitle('Algorithm Comparison')

            ax1=fig.add_subplot(111, label="1")
            ax2=fig.add_subplot(111, label="2", frame_on=False)

            ax1.errorbar(self.names, self.means, self.stds, color=color_score, linestyle='None', marker='o')
            ax1.set_xlabel("algorithm", color=color_black)
            ax1.set_ylabel(self.metric, color=color_black)
            ax1.tick_params(axis="both", colors=color_black)

            ax2.bar(self.names, self.cost, color=color_time, alpha=0.5, width=0.5)
            ax2.xaxis.tick_top()
            ax2.yaxis.tick_right()
            ax2.set_xlabel('algorithm', color=color_time) 
            ax2.set_ylabel('time', color=color_time)   
            ax2.xaxis.set_label_position('top') 
            ax2.yaxis.set_label_position('right') 
            ax2.tick_params(axis='both', colors=color_time)
        plt.grid()
        plt.show()
        return None

if __name__ == '__main__':
    main()