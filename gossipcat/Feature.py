#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author: 	Ewen Wang
email: 		wang.enqun@outlook.com
license: 	Apache License 2.0
"""
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


class Feature(object):

	def __init__(self, data, target, features):
		"""
		Args:
			data: A dataset you wanna play with.
			target: The target name of your dataset.
			features: The feature names of your dataset.
		"""
		self.data = data
		self.target = target
		self.features = features

		self.predictors = []
		self.dup = []
		self.int_lst, self.float_lst, self.object_lst = [], [], []
		self.corr_lst = []
		self.new_data = pd.DataFrame()
		self.new_col = []

	def duplicated(self, n_head=5000):
		""" Obtain duplicated features.

		Checks first n_head rows and obtains duplicated features.

		Args:
			n_head: First n_head rows to be checked; default 5000.
			print_dup: Whether print duplicated columns names; default with False.

		Returns:
			A list of the names of duplicatec columns, if any.
		"""
		dup_features = []
		print('checking...')
		if self.data.head(n_head).T.duplicated().any():
			dup_list = np.where(self.data.head(n_head).T.duplicated())[0].tolist()
			dup_features = self.data.columns[dup_list]
			print('duplicated features:\n', dup_features)
		else:
			print('no duplicated columns in first', n_head, 'rows.')

		return dup_features

	def classify(self):
		""" Feature classification.

		Divides features into sublists according to their data type.

		Returns:
			int_features: A list of column names of int features.
			float_features: A list of column names of float features.
			object_features: A list of column names of object features.
		"""
		int_features, float_features, object_features = [], [], []

		print('classifying...')
		dtypes = self.data[self.features].dtypes.apply(lambda x: x.name).to_dict()
		for col, dtype in dtypes.items():
			if dtype == 'int64':
				int_features.append(col)
			elif dtype == 'float64':
				float_features.append(col)
			elif dtype == 'object':
				object_features.append(col)
		print('int features count:', len(int_features),
			'\nfloat features count:', len(float_features),
			'\nobject features count:', len(object_features))

		return int_features, float_features, object_features

	def corr_pairs(self, col_list=None, gamma=0.99):
		""" Detect corralted features.

		Computes correlated feature pairs with correlated coefficient larger than gamma.

		Args:
			col_list: A column list to be calculated.
			gamma: The correlated coefficiency; default at 0.99.

		Returns:
			pairs: A list of correlated features.
		"""
		if col_list is None:
			col_list = self.features
		pairs = []

		corr_matrix = self.data[col_list].corr().abs()
		os = (corr_matrix
			  .where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
			  .stack()
			  .sort_values(ascending=False))
		pairs = os[os > gamma].index.values.tolist()

		return pairs

	def generate(self, corr_list=None, auc_score=0.7):
		""" Build new features from correlated features.

		Builds new features based on correlated feature pairs if the new feature
		has an auc greater than auc_score.

		Args:
			corr_list: The correlated list to generate new features from.
			auc_score: The auc to decide whether generate features, default at 0.75.

		Returns:
			new_data: A dataset conatianing new features.
			new_data.columns: The column list of the new_data.
		"""
		if corr_list is None:
			corr_list = self.corr_lst

		new_data = pd.DataFrame()

		if len(corr_list) > 1:
			print('generating...')
			for index, value in enumerate(corr_list):
				temp = self.data[corr_list[index][0]] - self.data[corr_list[index][1]]
				if len(temp.unique()) > 1:
					temp = pd.DataFrame(temp.fillna(temp.median()))
					lr = LogisticRegression()
					lr.fit(temp, self.data[self.target])
					prob = lr.predict_proba(temp)[:, 1]
					auc = metrics.roc_auc_score(self.data[self.target], prob)
					if auc > auc_score:
						print('-'.join(value), ' AUC (train): ', auc)
						new_data['-'.join(value)] = self.data[corr_list[index][0]] - self.data[corr_list[index][1]]
			if new_data.shape[1] > 0:
				print('\nnew features:\n', new_data.columns)
			else:
				print('no features meet the requirement.')
		else:
			print('no pair lists input.')

		return new_data, new_data.columns.tolist()

	def aut(self, n_head=5000, gamma=0.99, auc_score=0.7):
		""" Automatical feature engineering.

		Automatically detact new features and generate new data.

		Returns:
			predictors: A list of predictors.
			data/new_data: A new data with generated features, if any.
		"""
		self.dup = self.duplicated(n_head)
		self.predictors = [x for x in self.features if x not in self.dup]

		self.int_lst, self.float_lst, self.object_lst = self.classify()
		self.corr_p = self.corr_pairs(self.int_lst, gamma) + self.corr_pairs(self.float_lst, gamma)

		if len(self.corr_p)>0:
			self.new_data, self.new_col = self.generate(self.corr_p, auc_score)
			if len(self.new_col)>0:
				self.predictors = self.predictors + self.new_col
				self.new_data = pd.concat([self.data, self.new_data], axis=1)
				return self.predictors, self.new_data
		else:
			print('no correlated pairs.')
			print('no new features generated.')

		return self.predictors, self.data
