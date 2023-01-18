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

import pandas as pd 
import matplotlib.pyplot as plt

import catboost as cb

class CAT(object):
    """Quickly develop a CatBoost model with best-practice parameters."""
    def __init__(self, df, indcol, target, features, features_cat, regression=False, predicting=False, multi=0, balanced=0, gpu=0, seed=0):
        """
        Args:
            df (pandas.DataFrame): A DataFrame for modeling.
            indcol (str): The indicator column name for the dataset.
            target (str): The target column name.
            features (list): The feature list.
            features_cat (list): Categorical feature list.
            predicting (bool): Whether a predicting task, default False.
            balance (bool): Whether the sample is balanced for binary classification task, default False.
            multi (bool): Whether a multi-category task, default False.
            gpu (bool): Whether to use GPU, default False.
            seed (int): The seed for randomness.
        """
        super(CAT, self).__init__()
        
        self.df = df
        self.indcol = indcol
        self.features = features
        self.features_cat = features_cat
        self.regression = regression
        self.predicting = predicting
    
        self.df[self.features_cat] = self.df[self.features_cat].fillna('NaN')
        
        if self.predicting:
            self.target = None
            self.dtest = cb.Pool(data=self.df[self.features],
                                 cat_features=self.features_cat)
        else:    
            self.target = target
            self.dtrain = cb.Pool(data=self.df[self.features], 
                                  label=self.df[self.target],
                                  cat_features=self.features_cat)
        
        self.multi = multi
        self.balanced = balanced
        self.gpu = gpu
        self.seed = seed
        self.params = {}

        if self.regression:
            self.params['loss_function'] = 'RMSE'
        else:
            self.params['loss_function'] = 'Logloss'

        self.cvr = pd.DataFrame()
        self.prediction = pd.DataFrame()
        self.prediction[self.indcol] = self.df[self.indcol]

    def algorithm(self, learning_rate=0.01, iterations=100, early_stopping_rounds=20, nfold=10, verbose=100, plot=False):
        """Perform cross-validation on the training set.

        Args:
            learning_rate (float): Boosting learning rate (xgb’s “eta”).
            iterations (int): Number of boosting iterations.
            early_stopping (int): Activates early stopping. Cross-Validation metric (average of validation metric computed over CV folds) needs to improve at least once in every early_stopping_rounds round(s) to continue training. The last entry in the evaluation history will represent the best iteration. If there’s more than one metric in the eval_metric parameter given in params, the last metric will be used for early stopping.
            n_fold (int): Number of folds in CV.
            verbose (bool, int, or None): Whether to display the progress. If None, progress will be displayed when np.ndarray is returned. If True, progress will be displayed at boosting stage. If an integer is given, progress will be displayed at every given verbose_eval boosting stage.
            plot (bool): Whether plot the output, default False.
        """
        self.params['learning_rate'] = learning_rate
        self.params['iterations'] = iterations
        self.params['early_stopping_rounds'] = early_stopping_rounds
        self.params['verbose'] = verbose
        
        message = 'cross validation started and will stop if performace did not improve in {} rounds.'.format(early_stopping_rounds)
        print(message)
        self.cvr = cb.cv(dtrain=self.dtrain,
                         params=self.params,
                         nfold=nfold,
                         seed=self.seed,
                         plot=plot)
        
        col_loss = 'test-{}-mean'.format(self.params['loss_function'])
        self.n_rounds = self.cvr.sort_values([col_loss, 'iterations'])['iterations'].values[0]
        loss = self.cvr.sort_values([col_loss, 'iterations'])[col_loss].values[0]

        self.params['iterations'] = self.n_rounds
        message = 'cross validation done with number of rounds: {}.'.format(self.n_rounds)
        print(message)
        
        message = 'test {}: {:.3f}'.format(self.params['loss_function'], loss)
        print(message)
        return None

    def load_model(self, path_model='model_cb.json', format='json'):
        """Load a pretrained model.
        
        Args:
            path_model (str): Path of the model.
            format (str): Model format, default json.
        """
        if self.regression:
            self.bst = cb.CatBoostRegressor()
        else:
            self.bst = cb.CatBoostClassifier()
        self.bst = self.bst.load_model(fname=path_model, format=format)

        message = 'model loaded from path: %s' % path_model
        print(message)
        return None

    def save_model(self, path_model='model_cb.json', format='json'):
        """Load a pretrained model.
        
        Args:
            path_model (str): Path of the model.
            format (str): Model format, default json.
        """
        if path_model == None:
            pass
        else:
            self.bst.save_model(fname=path_model, format=format)
        return None

    def train(self, path_model='model_cb.json'):
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

        if self.regression:
            self.bst = cb.CatBoostRegressor(**self.params)
        else:
            self.bst = cb.CatBoostClassifier(**self.params)
        
        self.bst.fit(self.dtrain)

        self.save_model(path_model=path_model)

        self.prediction['pred'] = self.bst.predict(self.dtrain)

        if self.regression==False:
            self.prediction['prob'] = self.bst.predict_proba(self.dtrain)[:,1]

        message = 'prediction done.'
        print(message)
        return None
    
    def predict(self, path_model='model_cb.json', path_result='prediction.csv', model_format='json'):
        """Predict with model loaded from the path and save it as a CSV file.

        Args:
            path_model (str): Path of the model.
            path_result (str): Path of the prediction.
            model_format (str): Model format, default json.
        """
        self.load_model(path_model=path_model, format=model_format)

        self.prediction['pred'] = self.bst.predict(self.dtest)
        if self.regression==False:
            self.prediction['prob'] = self.bst.predict_proba(self.dtest)[:,1]

        message = 'prediction done.'
        print(message)

        if path_result == None:
            pass
        else:
            self.prediction.to_csv(path_result, index=False)
            message = 'results saved in path: %s' % path_result
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
        plt.plot(self.cvr[self.cvr.columns[1]], label='test')
        plt.plot(self.cvr[self.cvr.columns[3]], label='train')
        plt.title('learning curve')
        plt.xlabel('number of rounds')
        plt.ylabel(self.params['loss_function'])
        plt.legend(loc='upper right', title='dataset')
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

        prob = self.prediction['prob']

        plt.figure(figsize=(6, 5.5))
        self.prediction['prob'].hist(bins=100)
        plt.title('distribution of predictions')

        vis = Visual(test_target=test_target, test_predprob=prob)
        vis.combo()
        self.df_cap = vis.df_cap
        return None    

if __name__ == '__main__':
    main()