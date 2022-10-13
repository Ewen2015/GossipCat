#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
import warnings
warnings.filterwarnings('ignore')
import random
random.seed(0)

import time
import json
import pickle

import pandas as pd 
import matplotlib.pyplot as plt

import xgboost as xgb

class XGB(object):
    """Quickly develop a XGBoost model with best-practice parameters."""
    def __init__(self, data, indcol, target, features, regression=False, predicting=False, balanced=False, multi=False, gpu=False, seed=0):
        """
        Args:
            data (pandas.DataFrame): A DataFrame for modeling.
            indcol (str): The indicator column name for the dataset.
            target (str): The target column name.
            features (list): The feature list.
            predicting (bool): Whether a predicting task, default False.
            balance (bool): Whether the sample is balanced for binary classification task, default False.
            multi (bool): Whether a multi-category task, default False.
            gpu (bool): Whether to use GPU, default False.
            seed (int): The seed for randomness.
        """
        super(XGB, self).__init__()

        self.data = data
        self.indcol = indcol
        self.features = features
        self.regression = regression
        self.predicting = predicting

        if self.predicting:
            self.target = None
            self.dtest = xgb.DMatrix(self.data[self.features])
        else:
            self.target = target
            self.dtrain = xgb.DMatrix(self.data[self.features], label=self.data[self.target])
        
        self.multi = multi
        self.gpu = gpu
        self.seed = seed
        
        self.balanced = balanced
        self.params = {
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'eval_metric': 'aucpr',
            'eta': 0.01,
            'gamma': 0,
            'min_child_weight': 0.01,
            'max_depth': 3,
            'subsample': 0.85,
            'colsample_bytree': 0.75,
            'colsample_bylevel': 0.75,
            'colsample_bynode': 1.0,
            'lambda': 5,
            'alpha': 0.2
        }
        
        self.params_learning = {
            'maximize': True
        }
        
        if self.regression:
            self.params['objective'] = 'reg:squarederror'
            self.params['eval_metric'] = 'rmse'
            self.params_learning['maximize'] = False
        if self.balanced:
            self.params['eval_metric'] = 'auc'
        if self.gpu:
            self.params['tree_method'] = 'gpu_hist'
        if self.multi:
            self.params['objective'] = 'multi:softmax'
            self.params['eval_metric'] = 'mlogloss'

        self.cvr = pd.DataFrame()
        self.prediction = pd.DataFrame()
    
    def algorithm(self, learning_rate=0.01, nfold=5, n_rounds=3000, early_stopping=50, verbose=100):
        
        self.params['learning_rate'] = learning_rate
        self.nfold = nfold
        self.n_rounds = n_rounds
        self.early_stopping = early_stopping
        self.verbose = verbose

        start_time = time.time()
        message = 'cross validation started and will stop if performace did not improve in %d rounds.' % self.early_stopping
        print(message)

        self.cvr = xgb.cv(params=self.params,
                          dtrain=self.dtrain,
                          num_boost_round=self.n_rounds,
                          nfold=self.nfold,
                          stratified=True,
                          metrics=self.params['eval_metric'],
                          maximize=self.params_learning['maximize'],
                          early_stopping_rounds=self.early_stopping,
                          verbose_eval=self.verbose,
                          seed=self.seed)
        self.n_rounds = self.cvr.shape[0] - early_stopping

        duration = time.time() - start_time
        message = 'cross validation done with number of rounds: %d \tduration: %.3f s.' % (self.n_rounds, duration)
        print(message)
        message = 'test %s: %.3f' %(self.params['eval_metric'], self.cvr.iloc[-1, 2])
        print(message)
        return None

    def train(self, path_model=None):
        try:
            message = 'number of training rounds: %d.' % self.n_rounds
            print(message)
        except Exception as e:
            message = 'no hpyter parameters assigned and default assigned.'
            print(message)
            self.algorithm()
            print(json.dumps(self.params, indent=4))

        self.bst = xgb.train(params=self.params,
                             dtrain=self.dtrain,
                             evals=[(self.dtrain, 'train')],
                             num_boost_round=self.n_rounds,
                             verbose_eval=True)

        if path_model == None:
            pass
        else:
            pickle.dump(self.bst, open(path_model, 'wb'))
            print('model saved in path: %s' % path_model)

        self.prediction[self.indcol] = self.data[self.indcol]
        self.prediction['prob'] = self.bst.predict(self.dtrain)
        message = 'prediction done.'
        print(message)
        return None 

    def evaluate(self, path_model):
        self.bst = pickle.load(open(path_model, 'rb'))
        message = 'model loaded from path: %s' % path_model
        print(message)
        return self.bst.eval(self.dtrain)

    def predict(self, path_model, path_result=None):
        self.bst = pickle.load(open(path_model, 'rb'))
        message = 'model loaded from path: %s' % path_model
        print(message)

        self.prediction[self.indcol] = self.data[self.indcol]
        self.prediction['prob'] = self.bst.predict(self.dtest)
        message = 'prediction done.'
        print(message)

        if path_result == None:
            pass
        else:
            self.prediction.to_csv(path_result, index=False)
            message = 'results saved in path: %s' % path_result
            print(message)
        return None

    def retrain(self, path_model, path_model_update=None):
        try:
            message = 'number of training rounds: %d' % self.n_rounds
            print(message)
        except Exception as e:
            message = 'no hpyter parameters assigned and default assigned.'
            print(message)
            self.algorithm()
            print(json.dumps(self.params, indent=4))

        self.bst = pickle.load(open(path_model, 'rb'))
        message = 'model loaded from path: %s' % path_model
        print(message)

        self.bst.update(dtrain=self.dtrain, iteration=self.n_rounds)
        message = 'model updated.'
        print(message)

        if path_model_update == None:
            pass
        else:
            pickle.dump(self.bst, open(path_model_update, 'wb'))
            print('model saved in path: %s' % path_model_update)

        self.prediction[self.indcol] = self.data[self.indcol]
        self.prediction['prob'] = self.bst.predict(self.dtrain)
        message = 'prediction done.'
        print(message)
        return None

    def learning_curve(self, figsize=(10, 5)):
        if len(self.cvr) == 0:
            return 'no models trained, no learning curves.'

        plt.figure(figsize=figsize)
        plt.plot(self.cvr[self.cvr.columns[0]], label='train')
        plt.plot(self.cvr[self.cvr.columns[2]], label='test')
        plt.title('learning curve')
        plt.xlabel('number of rounds')
        plt.ylabel(self.params['eval_metric'])
        plt.legend(loc='lower right' if self.params_learning['maximize']==True else 'upper right', 
                   title='dataset')
        plt.grid() 
        plt.show()
        return None

    def report(self):
        try:
            from gossipcat.lab.Report import Visual
        except Exception as e:
            print('[WARNING] Package GossipCat not installed.')
            try:
                from Report import Visual
            except Exception as e:
                return '[ERROR] Package Report not installed.'

        test_target = self.data[self.target]

        prob = self.prediction['prob']

        plt.figure(figsize=(6, 5.5))
        self.prediction['prob'].hist(bins=100)
        plt.title('distribution of predictions')

        vis = Visual(test_target=test_target, test_predprob=prob)
        vis.combo()
        self.df_cap = vis.df_cap
        return None