#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
import pandas as pd 
import xgboost as xgb
import warnings 
warnings.filterwarnings('ignore')


class GridSearch(object):
    """Perform a grid search for XGBoost hyper-parameter tuning, focusing on `max_depth`, `subsample`, and `colsample_bytree`.
    """
    def __init__(self, df=None, target=None, features=None, regression=False, if_visualize=False, log_path='grid_search.log'):
        """

        Args:
            df (pandas.DataFrame): A training set. 
            target (str): The target for supervised machine learning.
            features (list): The feature list for the model.
            regression (bool): Whether the machine learning task is regression.
            if_visualize (bool): Whether the task is to visualize, default False.
            log_path (str): The logging file. 
        """
        super(GridSearch, self).__init__()

        self.df = df
        self.target = target
        self.features = features
        self.regression = regression
        self.if_visualize = if_visualize
        self.log_path = log_path

        self.generalParams = {
            'nfold': 5,
            'learning_rate': 0.1,
            'n_rounds': 3000,
            'early_stopping_rounds': 100,
            'maximize': True,
            'verbose': 1,
            'seed': 123
        }

        self.treeParams = {
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'eval_metric': 'aucpr',
            'eta': self.generalParams['learning_rate'],
            'gamma': 0,
            'min_child_weight': 0.01,
            'max_depth': 3,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'colsample_bylevel': 0.7,
            'colsample_bynode': 1,
            'lambda': 5,
            'alpha': 0.2
        }

        if self.regression:
            self.treeParams['objective'] = 'reg:squarederror'
            self.treeParams['eval_metric'] = 'rmse'
            self.generalParams['maximize'] = False
            self.ascending = True

        if self.if_visualize:
            self.get_log()
        else:
            self.dtrain = xgb.DMatrix(data=self.df[self.features], label=self.df[self.target], silent=False, nthread=-1)


    def search(self, 
               range_max_depth=range(1, 10, 1),
               range_subsample=range(50, 91, 5),
               range_colsample_bytree=range(50, 91, 5)):
        """To search on the hyper-parameter space.

        Args:
            range_max_depth (list): The search space of `max_depth`, default range(1, 10, 1). 
            range_subsample (list): The search space of `subsample`, default range(50, 91, 5). 
            range_colsample_bytree (list): The search space of `colsample_bytree`, default range(50, 91, 5). 
        """
        self.range_max_depth = range_max_depth
        self.range_subsample = range_subsample
        self.range_colsample_bytree = range_colsample_bytree

        metric = self.treeParams['eval_metric']

        with open(self.log_path, 'w') as f:
            f.write('max_depth,subsample,colsample_bytree,best_round,train_{}_mean,train_{}_std,test_{}_mean,test_{}_std\n'\
                .format(metric, metric, metric, metric))

        for d in self.range_max_depth:
            for s in self.range_subsample:
                for c in self.range_colsample_bytree:
                    self.treeParams['max_depth'] = d
                    self.treeParams['subsample'] = s/100
                    self.treeParams['colsample_bytree'] = c/100

                    cvr = xgb.cv(params=self.treeParams,
                                 dtrain=self.dtrain,
                                 num_boost_round=self.generalParams['n_rounds'],
                                 nfold=self.generalParams['nfold'],
                                 stratified=True,
                                 metrics=self.treeParams['eval_metric'],
                                 maximize=self.generalParams['maximize'],
                                 early_stopping_rounds=self.generalParams['early_stopping_rounds'],
                                 verbose_eval=self.generalParams['verbose'],
                                 seed=self.generalParams['seed'])
                    with open(self.log_path, 'a') as f:
                        f.write('%d,%f,%f,%d,%f,%f,%f,%f\n' % (self.treeParams['max_depth'], 
                                                               self.treeParams['subsample'], 
                                                               self.treeParams['colsample_bytree'], 
                                                               cvr.index[-1],
                                                               cvr.tail(1)['train-{}-mean'.format(self.treeParams['eval_metric'])],
                                                               cvr.tail(1)['train-{}-std'.format(self.treeParams['eval_metric'])],
                                                               cvr.tail(1)['test-{}-mean'.format(self.treeParams['eval_metric'])],
                                                               cvr.tail(1)['test-{}-std'.format(self.treeParams['eval_metric'])]))
        print('done.')
        return None

    def get_log(self):
        self.data = pd.read_csv(self.log_path)
        self.get_best()
        return None

    def get_last(self):
        print('the lastest results:')
        return self.data.iloc[-1:]

    def get_best(self):
        print('the best results:')
        return self.data.sort_values(by='test_{}_mean'.format(self.treeParams['eval_metric']), ascending=self.ascending).head(1)

    def get_top(self, top):
        print('the top %d results:' % top)
        return self.data.sort_values(by='test_{}_mean'.format(self.treeParams['eval_metric']), ascending=self.ascending).head(top)

    def visualize(self, max_depth=1, top=1):
        """To visualize the grid search results in 3D format. The x-axis: `subsample`, the y-axis: `colsample_bytree`, and the z-axis: the mean of cross-validation test score.

        Args:
            max_depth (int): The `max_depth` for the 3D visualization.
            top (int): The top results to print out.

        Return:
            The top results of grid search.
        """
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt 
        from matplotlib import cm

        df = self.data[self.data.max_depth == max_depth]
        x = 'subsample'; y = 'colsample_bytree'; z = 'test_{}_mean'.format(self.treeParams['eval_metric'])

        fig = plt.figure(figsize=(8, 8))
        ax = Axes3D(fig)
        surf = ax.plot_trisurf(df[x], df[y], df[z], 
                               cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        fig.colorbar(surf, shrink=.5, aspect=5)
        plt.title('Grid Search Visualization (max_depth: %d)' %(max_depth))
        plt.show()
        return df.sort_values(by=z, ascending=self.ascending).head(top)


if __name__ == '__main__':
    main()


