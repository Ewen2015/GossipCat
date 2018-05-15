#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wang.enqun@outlook.com
license:    Apache License 2.0
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pandas_profiling as pp
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from .Tune import Tune
from .Report import Report

class Machine(object):
	"""docstring for Machine"""
	def __init__(self, wd, input_file, target, drop_list):
		super(Machine, self).__init__()
		self.wd = wd
		self.data = pd.read_csv(input_file)
		self.train, self.test = train_test_split(self.data, test_size=0.2, random_state=0)
		self.target = target
		self.drop_list = drop_list
		self.features = [i for i in self.data.columns if i not in self.drop_list]

	def Summary(self):
		print('='*20)
		print('data shape: ', self.data.shape)
		pp.ProfileReport(self.data)
		return None

	def Model(self):
		print('='*20)
		self.clf = Tune(train=self.train, target=self.target, predictors=self.features, metric='auc', scoring='roc_acu').simpleTrain()

		print('='*20)
		print('plotting...')
		lgb.plot_importance(self.clf, figsize=(20, 16))
		
		print('='*20)
		print('reporting...')
		self.rpt = Report(self.clf, self.train, self.test, self.target, self.features, is_sklearn=False)
		self.rpt.ALL()
		return None

	def Result(self, threshold=0.5):
		print('='*20)
		print('predicting...')
		self.results = pd.DataFrame(columns=['truth', 'prediction', 'probability'])
		self.results['truth'] = self.test[self.target]
		self.results['prediction'] = self.pred_test = np.where(self.prob_test>=threshold, 1, 0)
		self.reuslts['probability'] = self.clf.predict(self.test[self.features])

		self.results = self.results.sort_values('probability', accending=False)

		print('='*20)
		print('plotting...')
		plt.figure(figsize=(16, 12))
		sns.violinplot(x='truth', y='probability', hue='prediction', data=self.reuslts,
			inner='quartile', split=True, palette='Set2')
		return self.reuslts

