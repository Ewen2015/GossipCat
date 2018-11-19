#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author: 	Ewen Wang
email:      wolfgangwong2012@gmail.com
license: 	Apache License 2.0
"""
import numpy as np
import pandas as pd
import itertools
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

		self.dup = []
		self.int_lst, self.float_lst, self.object_lst, self.non_obj = [], [], [], []

		self.corr_lst = []
		self.new_data_corr = pd.DataFrame()

		self.result_COM = pd.DataFrame()
		self.result_LDA = pd.DataFrame()
		self.new_data_comb = pd.DataFrame()

		self.new_col_corr = []
		self.new_col_comb = []

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

	def generate_corr(self, corr_list=None, auc_score=0.7):
		""" Build new features from correlated features.

		Builds new features based on correlated feature pairs if the new feature
		has an auc greater than auc_score.

		Args:
			corr_list: The correlated list to generate new features from.
			auc_score: The auc to decide whether generate features, default at 0.7.

		Returns:
			new_data_corr: A dataset conatianing new features.
			new_data_corr.columns: The column list of the new_data_corr.
		"""
		if corr_list is None:
			corr_list = self.corr_lst

		new_data_corr = pd.DataFrame()

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
						new_data_corr['-'.join(value)] = self.data[corr_list[index][0]] - self.data[corr_list[index][1]]
			if new_data_corr.shape[1] > 0:
				print('\nnew features:\n', new_data_corr.columns)
			else:
				print('no features meet the requirement.')
		else:
			print('no pair lists input.')

		return new_data_corr, new_data_corr.columns.tolist()

	def generate_comb(self, features_max=100, kill=50, n_combinations=2, auc_score=0.7, n_print=200):
		""" Build new features from LDA filetering and combination.
		
		Builds new features based linear discriminat analysis.

		Args:
			features_max: The maximum features remined after LDA filtering.
			kill: The number of features to be killed at each round.
			n_combinations: The number of features each combination contains, default at 2.
			auc_score: The auc to decide whether generate features, default at 0.7.
			n_print: The number of iterations each printout contains.

		Returns:
			new_data_comb: A dataset conatianing new features.
			new_data_comb.columns: The column list of the new_data_corr.
		"""
		self.int_lst, self.float_lst, self.object_lst = self.classify()
		self.non_obj = self.int_lst + self.float_lst

		X = self.data[self.non_obj]
		y = self.data[self.target]

		features_temp = X.columns

		self.result_COM = pd.DataFrame(columns=['Features', 'AUC', 'Coefficients', 'Intercept'])
		self.new_data_comb = pd.DataFrame()

		print('\nfiltering....')
		while (len(features_temp) > features_max):
		    
		    lda = LinearDiscriminantAnalysis(n_components=2)
		    temp = StandardScaler().fit_transform(X[features_temp].fillna(X.median()))
		    lda.fit(temp, y)
		    self.result_LDA = pd.DataFrame({'title': X[features_temp].columns,
		                               'coff': np.abs(lda.coef_.reshape(X[features_temp].shape[1]))})
		    features_temp = list(self.result_LDA.sort_values(ascending=False, by=['coff']).iloc[:, 1].values)
		    del features_temp[-kill:]
		    print('Turns Left:', round(len(features_temp)/kill))

		print('\nNumber of features left: ', len(features_temp))
		print('\ncombinations checking...')

		features_comb = list(itertools.combinations(features_temp, n_combinations))

		for index, value in enumerate(features_comb):
		    golden_features = X[list(value)]
		    lda = LinearDiscriminantAnalysis(n_components=2)
		    temp = lda.fit_transform(golden_features.fillna(golden_features.median()), y)
		    prob = lda.predict_proba(golden_features.fillna(golden_features.median()))[:, 1]
		    auc = metrics.roc_auc_score(y, prob)
		    if auc > auc_score:
		        print('-'.join(value), ' AUC (train): ', auc)
		        self.new_data_comb['-'.join(value)] = temp.tolist()
		        self.result_COM=self.result_COM.append({'Features': value, 'AUC': auc,
		                                      'Coefficients': lda.coef_,'Intercept': lda.intercept_}, 
		                                     ignore_index=True)
		    if index % n_print == 0:
		        print('Turns Remaining: ', len(features_comb) - index)

		self.result_COM = self.result_COM.sort_values(by='AUC', ascending=False)

		return self.new_data_comb, self.new_data_comb.columns.tolist()

	def aut_corr(self, n_head=5000, gamma=0.99, auc_score=0.7):
		""" Automatical feature engineering with high correlated algorithm.

		Automatically detact new features and generate new data with high correlated algorithm.

		Returns:
			new_data_corr: A dataset conatianing new features.
			new_data_corr.columns: The column list of the new_data_corr.
		"""
		self.dup = self.duplicated(n_head)
		self.features = [x for x in self.features if x not in self.dup]

		self.int_lst, self.float_lst, self.object_lst = self.classify()
		self.corr_p = self.corr_pairs(self.int_lst, gamma) + self.corr_pairs(self.float_lst, gamma)

		if len(self.corr_p)>0:
			self.new_data_corr, self.new_col_corr = self.generate_corr(self.corr_p, auc_score)
			if len(self.new_col_corr)>0:
				print('\nNew features are found!')
		else:
			print('no correlated pairs.')
			print('no new features generated.')

		return self.new_data_corr, self.new_col_corr

	def aut_comb(self, n_head=5000, features_max=100, kill=50, n_combinations=2, auc_score=0.7, n_print=200):
		""" Automatical feature engineering with LDA.

		Automatically detact new features and generate new data with LDA algorithm.

		Returns:
			new_data_comb: A dataset conatianing new features.
			new_data_comb.columns: The column list of the new_data_comb.
		"""
		self.dup = self.duplicated(n_head)
		self.features = [x for x in self.features if x not in self.dup]

		self.new_data_comb, self.new_col_comb = self.generate_comb(features_max, kill, n_combinations, auc_score, n_print)

		if len(self.new_col_comb)>0:
			print('\nNew features are found!')
		else:
			print('no new features generated.')

		return self.new_data_comb, self.new_col_com







