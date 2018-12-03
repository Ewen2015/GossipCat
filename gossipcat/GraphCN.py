#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
import numpy as np 
import tensorflow as tf 


class GraphCN(object):
	"""docstring for GraphCN"""
	def __init__(self, arg):
		super(GraphCN, self).__init__()
		self.arg = arg
