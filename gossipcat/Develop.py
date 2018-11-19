#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
import pandas as pd
import numpy as np
import pandas_profiling as pp
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import lime.lime_tabular
from IPython.core import display
from IPython.core.display import HTML
from .Report import Report

class Machine(object):
	"""docstring for Machine"""
	def __init__(self, train, valid, target, features):
		super(Machine, self).__init__()
		self.train = train, 
		self.valid = valid 
		self.target = target
		self.features = features

	def Summary(self):
		print('='*20)
		print('data shape: ', self.data.shape)
		pp.ProfileReport(self.data)
		return None

	def Model(self, wd, model_file):
		print('='*20)
		self.clf = Tune(train=self.train, target=self.target, predictors=self.features, metric='auc', scoring='roc_acu').simpleTrain()

		print('='*20)
		print('model saving...')
		self.clf.save_model(wd+model_file)

		print('='*20)
		print('plotting...')
		lgb.plot_importance(self.clf, figsize=(20, 16))
		
		print('='*20)
		print('reporting...')
		self.rpt = Report(self.clf, self.train, self.valid, self.target, self.features, is_sklearn=False)
		self.rpt.ALL()
		return self.clf

	def Result(self, threshold=0.5):
		print('='*20)
		print('predicting...')
		self.results = pd.DataFrame(columns=['truth', 'prediction', 'probability'])
		self.results['truth'] = self.valid[self.target]
		self.results['prediction'] = np.where(self.prob_valid>=threshold, 1, 0)
		self.reuslts['probability'] = self.clf.predict(self.valid[self.features])

		self.results = self.results.sort_values('probability', accending=False)
		display(HTML(self.results.head().to_html()))
		print('='*20)
		print('plotting...')
		plt.figure(figsize=(16, 12))
		sns.violinplot(x='truth', y='probability', hue='prediction', data=self.reuslts,
			inner='quartile', split=True, palette='Set2')
		return self.reuslts

	def IniExplainer(self):
		self.explainer = lime.lime_tabular.LimeTabularExplainer(np.array(self.train[self.features]),
			feature_names=self.features, class_names=self.class_names, discretize_continous=True)
		return None

	def Explainer(self, id):
		exp = self.explainer.explain_instance(np.array(self.valid[self.features])[id], self.clf.predict, num_features=10, top_labels=1)
		exp.show_in_notebook(show_table=True, show_all=False)
		return None

	def LoadModel(self, wd, file):
		self.clf = lgb.Booster(model_file=wd+file)
		return self.clf

