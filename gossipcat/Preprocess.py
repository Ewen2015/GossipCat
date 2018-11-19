#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
import pandas as pd 	
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

class Preprocess(object):
	"""docstring for Preprocess"""
	def __init__(self, data, feature_num):
		super(Preprocess, self).__init__()
		self.data = data
		self.features_tr = feature_num
		

		num_pipeline = Pipeline([
		    ('imputer', Imputer(strategy='median')),
		    ('std_scalar', StandardScaler()),
		])

		self.features_tr = pd.DataFrame(num_pipeline.fit_transform(self.data[self.feature_num]), columns=self.feature_num)