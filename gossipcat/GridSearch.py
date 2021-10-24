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
    'learning_rate': 0.1,
    'n_rounds': 100,
    'early_stopping_rounds': 10,
    'verbose': 1,
    'seed': 123
}

treeParams = {
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'eval_metric': 'aucpr',
    'eta': generalParams['learning_rate'],
    'gamma': 0,
    'min_child_weight': 0.01,
    'max_depth': 5,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'colsample_bylevel': 0.7,
    'colsample_bynode': 1,
    'lambda': 5,
    'alpha': 0.2
}

def Search(train, 
           target, 
           features, 
           general_params=generalParams, 
           tree_params=treeParams, 
           range_max_depth=range(3, 10, 1),
           range_subsample=range(50, 91, 5),
           range_colsample_bytree=range(50, 91, 5),
           log_path=None):
    dtrain = xgb.DMatrix(data=train[features], label=train[target], silent=False, nthread=-1)

    with open(log_path, 'w') as f:
        f.write('max_depth,subsample,colsample_bytree,best_round,train_aucpr_mean,train_aucpr_std,test_aucpr_mean,test_aucpr_std\n')

    for d in range_max_depth:
        for s in range_subsample:
            for c in range_colsample_bytree:
                tree_params['max_depth'] = d
                tree_params['subsample'] = s/100
                tree_params['colsample_bytree'] = c/100

                cvr = xgb.cv(params=tree_params,
                             dtrain=dtrain,
                             num_boost_round=general_params['n_rounds'],
                             nfold=general_params['nfold'],
                             stratified=True,
                             metrics=tree_params['eval_metric'],
                             maximize=True,
                             early_stopping_rounds=general_params['early_stopping_rounds'],
                             verbose_eval=general_params['verbose'],
                             seed=general_params['seed'])
                with open(log_path, 'a') as f:
                    f.write('%d,%f,%f,%d,%f,%f,%f,%f\n' % (tree_params['max_depth'], 
                                                           tree_params['subsample'], 
                                                           tree_params['colsample_bytree'], 
                                                           cvr.index[-1],
                                                           cvr.tail(1)['train-aucpr-mean'],
                                                           cvr.tail(1)['train-aucpr-std'],
                                                           cvr.tail(1)['test-aucpr-mean'],
                                                           cvr.tail(1)['test-aucpr-std']))
    print('done.')
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

    def getTop(self, top):
        print('the top %d results:' % top)
        return self.df.sort_values(by='test_aucpr_mean', ascending=False).head(top)

    def getVisual(self, max_depth, top=1):
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt 
        from matplotlib import cm 

        df = self.df[self.df.max_depth == max_depth]
        x = 'subsample'; y = 'colsample_bytree'; z = 'test_aucpr_mean'

        fig = plt.figure(figsize=(8, 8))
        ax = Axes3D(fig)
        surf = ax.plot_trisurf(df[x], df[y], df[z], 
                               cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        fig.colorbar(surf, shrink=.5, aspect=5)
        plt.title('Grid Search Visualization (%s: %d)' %(z, max_depth))
        plt.show()
        return df.sort_values(by=z, ascending=False).head(top)

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


