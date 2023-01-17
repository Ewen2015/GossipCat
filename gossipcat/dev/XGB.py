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
    """Develop a XGBoost model with best-practice parameters.
    """
    def __init__(self, df, indcol, target, features, regression=False, predicting=False, balanced=False, multi=False, gpu=False, seed=0):
        """
        Args:
            df (pandas.DataFrame): A DataFrame for modeling.
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

        self.df = df
        self.indcol = indcol
        self.features = features
        self.regression = regression
        self.predicting = predicting

        if self.predicting:
            self.target = None
            self.dtest = xgb.DMatrix(self.df[self.features])
        else:
            self.target = target
            self.dtrain = xgb.DMatrix(self.df[self.features], label=self.df[self.target])
        
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
        self.prediction[self.indcol] = self.df[self.indcol]
    
    def algorithm(self, learning_rate=0.01, nfold=5, n_rounds=3000, early_stopping=50, verbose=100):
        """Perform cross-validation on the training set.

        Args:
            learning_rate (float): Boosting learning rate (xgb’s “eta”).
            n_fold (int): Number of folds in CV.
            n_rounds (int): Number of boosting iterations.
            early_stopping (int): Activates early stopping. Cross-Validation metric (average of validation metric computed over CV folds) needs to improve at least once in every early_stopping_rounds round(s) to continue training. The last entry in the evaluation history will represent the best iteration. If there’s more than one metric in the eval_metric parameter given in params, the last metric will be used for early stopping.
            verbose (bool, int, or None): Whether to display the progress. If None, progress will be displayed when np.ndarray is returned. If True, progress will be displayed at boosting stage. If an integer is given, progress will be displayed at every given verbose_eval boosting stage.
        """
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

    def load_model(self, path_model='model_xgb.pkl'):
        """Load a pretrained model.
        
        Args:
            path_model (str): Path of the model.
        """
        self.bst = pickle.load(open(path_model, 'rb'))
        message = 'model loaded from path: %s' % path_model
        print(message)
        return None

    def save_model(self, path_model='model_xgb.pkl'):
        """Load a pretrained model.
        
        Args:
            path_model (str): Path of the model.
        """
        if path_model == None:
            pass
        else:
            pickle.dump(self.bst, open(path_model, 'wb'))
            print('model saved in path: %s' % path_model)
        return None

    def train(self, path_model='model_xgb.pkl'):
        """Train a model with the best iteration rounds obtained from `algorithm`.

        Args:
            path_model (str): Path to save the model.
        """
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

        self.save_model(path_model=path_model)

        self.prediction['prediction'] = self.bst.predict(self.dtrain)
        self.prediction['target'] = self.df[self.target]
        message = 'prediction done.'
        print(message)
        return None 

    def evaluate(self, path_model='model_xgb.pkl'):
        """Evaluate a model loaded from the path.

        Args:
            path_model (str): Path of the model.

        Return:
            Model evaluation.
        """
        self.load_model(path_model=path_model)
        return self.bst.eval(self.dtrain)

    def predict(self, path_model='model_xgb.pkl', path_result='prediction.csv'):
        """Predict with model loaded from the path and save it as a CSV file.

        Args:
            path_model (str): Path of the model.
            path_result (str): Path of the prediction.
        """
        self.load_model(path_model=path_model)

        self.prediction['prediction'] = self.bst.predict(self.dtest)
        self.prediction['version'] = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
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
        """Retrain a model with the model from path and save to a new path.

        Args:
            path_model (str): Path to save the model.
            path_model_update (str): New path for the updated model.
        """
        try:
            message = 'number of training rounds: %d' % self.n_rounds
            print(message)
        except Exception as e:
            message = 'no hpyter parameters assigned and default assigned.'
            print(message)
            self.algorithm()
            print(json.dumps(self.params, indent=4))

        self.load_model(path_model=path_model)

        self.bst.update(dtrain=self.dtrain, iteration=self.n_rounds)
        message = 'model updated.'
        print(message)

        self.save_model(path_model=path_model_update)

        self.prediction[self.indcol] = self.df[self.indcol]
        self.prediction['prediction'] = self.bst.predict(self.dtrain)
        message = 'prediction done.'
        print(message)
        return None

    def learning_curve(self, figsize=(10, 5)):
        """Draw a learning curve of the cross-validation.

        Args:
            figsize (tupe): Figure size of the chart.
        """
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
        """Report for the binary classification task.
        """
        try:
            from gossipcat.lab.Report import Visual
        except Exception as e:
            print('[WARNING] Package GossipCat not installed.')
            try:
                from Report import Visual
            except Exception as e:
                return '[ERROR] Package Report not installed.'

        test_target = self.df[self.target]

        prob = self.prediction['prediction']

        plt.figure(figsize=(6, 5.5))
        self.prediction['prediction'].hist(bins=100)
        plt.title('distribution of predictions')

        vis = Visual(test_target=test_target, test_predprob=prob)
        vis.combo()
        self.df_cap = vis.df_cap
        return None

if __name__ == '__main__':
    main()