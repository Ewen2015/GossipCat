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

from .Configure import getConfig

generalParams = {
    'nfold': 5,
    'learning_rate': 0.01,
    'n_rounds': 30000,
    'early_stopping': 200,
    'verbose': 1,
    'seed': 123
}

treeParams = {
    'objective': 'binary:logistic',
    'tree_method': 'gpu_hist',
    'eval_metric': 'aucpr',
    'eta': generalParams['learning_rate'],
    'gamma': 0,
    'min_child_weight': 0.01,
    'max_depth': 5,
    'max_delta_step': 1,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'colsample_bylevel': 0.7,
    'colsample_bynode': 1,
    'lambda': 5,
    'alpha': 0.2
}

def Search(train, features, target, general_params=generalParams, tree_params=treeParams, log_path=None):
    dtrain = xgb.DMatrix(data=train[features], label=train[target], silent=False, nthread=-1)

    with open(log_path, 'w') as f:
        f.write('max_depth,subsample,colsample_bytree,best_round,train_aucpr_mean,train_aucpr_std,test_aucpr_mean,test_aucpr_std')

    for d in range(2, 20, 1):
        for s in range(10, 100, 5):
            for c in range(10, 100, 5):
                treeParams['max_depth'] = d
                treeParams['subsample'] = s/100
                treeParams['colsample_bytree'] = c/100

                cvr = xgb.cv(params=treeParams,
                             dtrain=dtrain,
                             num_boost_round=generalParams['n_rounds'],
                             nfold=generalParams['nfold'],
                             stratified=True,
                             metrics=treeParams['eval_metric'],
                             maximize=True,
                             early_stopping=generalParams['early_stopping'],
                             verbose_eval=generalParams['verbose'],
                             seed=generalParams['seed'])
                with open(log_path, 'a') as f:
                    f.write('%d,%f,%f,%d,%f,%f,%f,%f\n' % (treeParams['max_depth'], treeParams['subsample'], treeParams['colsample_bytree'], 
                                                           cvr.index[-1],
                                                           cvr.index[-1,0],
                                                           cvr.index[-1,1],
                                                           cvr.index[-1,2],
                                                           cvr.index[-1,3]))
    print('fulfill.')
    return None

class Results(object):
    """docstring for Results"""
    def __init__(self, log_path):
        super(Results, self).__init__()
        self.log_path = log_path
        self.df = pd.read_csv(self.log_path)

    def getLast(self):
        print('the lastest results:')
        return self.df.iloc[-1:]

    def getBest(self):
        print('the best results:')
        return self.df.sort_values(by='test_aucpr_mean', ascending=False).head(1)

    def getTop(self, num):
        print('the top %d results:' % num)
        return self.df.sort_values(by='test_aucpr_mean', ascending=False).head(num)

    def getVisual(self, max_depth, num=1):
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt 
        from matplotlib import cm 

        df = self.df[self.df.max_depth == max_depth]
        x = 'subsample'; y = 'colsample_bytree'; z = 'test_aucpr_mean'

        fig = plt.figure()
        ax = Axes3D(fig)
        surf = ax.plot_trisurf(df[x], df[y], df[z], 
                               cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        fig.colorbar(surf, shrink=.5, aspect=5)
        plt.title('Grid Search Visualization (%s: %d)' %(z, max_depth))
        plt.show()
        return df.sort_values(by=z, ascending=False).head(num)

def main(): 
    config = getConfig()

    dir_train = config['dir_train']
    file_train = config['file_train']

    dir_log = config['dir_log']
    file_log = config['file_gs']

    target = config['target']
    drop = config['drop']

    train = pd.read_csv(dir_train+file_train)
    features = [x for x in train.columns if x not in drop]

    Search(train, features, target, 
           general_params=generalParams, 
           tree_params=treeParams, 
           log_path=dir_log+file_log)


if __name__ == '__main__':
    main()


