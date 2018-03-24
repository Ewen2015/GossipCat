#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author: 	Ewen Wang
email: 		wang.enqun@outlook.com
license: 	Apache License 2.0
"""
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from .Report import Report

class Baseline(object):
	"""Provide general machine learning models as baseline."""
	def __init__(self, data, target, features):
		super(Baseline, self).__init__()
		self.data = data
		self.target = target
		self.features = features

		self.train, self.test = train_test_split(data, test_size=0.2, random_state=0)

	def LR(self, feature_num=6, verbose='True'):
		"""Logistic Regression.

		Args:
			feature_num: number of feaures to keep in the model.
			verbose: whether print out the model analysis report.
		Returns:
			Logistic regression model."""

		predictors=[]
		logreg_fs = LogisticRegression()
		rfe = RFE(logreg_fs, feature_num)
		rfe = rfe.fit(self.data[self.features], self.data[self.target])
		for ind, val in enumerate(rfe.support_):
			if val: predictors.append(self.features[ind])

		logreg = LogisticRegression()
		logreg.fit(self.train[predictors], self.train[self.target])

		if verbose:
			rpt = Report(logreg, self.train, self.test, self.target, predictors)
			rpt.ALL()

		return logreg
	
	def RF(self, verbose='True'):
		"""Random Forest.

		Args:
			feature_num: number of feaures to keep in the model.
			verbose: whether print out the model analysis report.
		Returns:
			Decision tree model generated from Random Forest."""
		tf = RandomForestClassifier(random_state=0)

		n_estimator = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
		max_features = ['log2', 'sqrt']
		max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
		max_depth.append(None)
		min_samples_split = [2, 5, 10]
		min_samples_leaf = [1, 2, 4]
		bootstrap = [True, False]

		random_grid = {'n_estimator': n_estimator, 
					   'max_features': max_features, 
					   'max_depth': max_depth, 
					   'min_samples_split': min_samples_split, 
					   'min_samples_leaf': min_samples_leaf, 
					   'bootstrap': bootstrap}
		rf_random = RandomizedSearchCV(estimator=tf,
			param_distributions=random_grid,
			n_iter=100, 
			cv=3, 
			verbose=1, 
			random_state=0, 
			n_jobs=-1)
		rf_random.fit(self.train[self.features], self.train[self.target])

		if verbose:
			rpt = Report(rf_random, self.train, self.test, self.target, self.features)
			rpt.ALL()

		return rf_random

