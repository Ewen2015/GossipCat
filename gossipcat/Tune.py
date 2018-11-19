#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
import pandas as pd
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

class Tune(object):
    """docstring for Tune"""
    def __init__(self, data, target, features, model, param_grid):
        super(Tune, self).__init__()
        self.data = data
        self.target = target
        self.features = features
        self.X = self.data[self.features]
        self.y = self.data[self.target]
        self.model = model
        self.param_grid = param_grid
        self.grid = GridSearchCV(estimator=self.model, param_grid=self.param_grid)
        self.grid_result = self.grid.fit(self.X, self.y)

    def GridLine(self, negative=False):
        cv_results = self.grid_result.cv_results_
        param_name = [*self.param_grid][0]
        ticks = [*self.param_grid.values()][0]

        self.scores_df = pd.DataFrame(cv_results).sort_values(by='rank_test_score')

        best_row = self.scores_df.iloc[0, :]
        best_param = best_row['param_' + param_name]
        best_mean = -best_row['mean_test_score'] if negative else best_row['mean_test_score']
        best_stdev = best_row['std_test_score']
        
        print("best parameters: {}".format(best_param))
        print("best score:      {:0.5f} (+/-{:0.5f})".format(best_mean, best_stdev))
        
        self.scores_df = self.scores_df.sort_values(by='param_' + param_name)
        
        params = self.scores_df['param_' + param_name]
        means = -self.scores_df['mean_test_score'] if negative else self.scores_df['mean_test_score']
        stds = self.scores_df['std_test_score']
        
        fig = plt.figure(figsize=(16, 6))
        ax = fig.add_subplot(1, 1, 1)
        plt.errorbar(params, means, yerr=stds)

        plt.axhline(y=best_mean + best_stdev, color='red', ls='--')
        plt.axhline(y=best_mean - best_stdev, color='red', ls='--')
        plt.plot(best_param, best_mean, 'or')

        plt.title(param_name + " vs Score\nBest Score {:0.5f}".format(best_mean))
        plt.xlabel(param_name)
        plt.ylabel('Score')
        ax.set_xticks(ticks)
        plt.grid()
        plt.show()
        return self.scores_df

    def GridHeat(self):
        p1_val, p2_val = [*self.param_grid.values()]
        p1_nam, p2_nam = [*self.param_grid]
        scores = self.grid_result.cv_results_['mean_test_score'].reshape(len(p1_val), len(p2_val))
        self.scores_df = pd.DataFrame(data=scores, columns=p2_val, index=p1_val)
        self.scores_df = self.scores_df.rename_axis(p1_nam).rename_axis(p2_nam, axis=1)

        plt.figure(figsize=(6,6))
        ax = sns.heatmap(self.scores_df, linewidth=.5)
        ax.set(title='Grid Search Score')
        plt.show()
        return self.scores_df