#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0


Philosophy:

[load]
[trim] trim the dataset, columns first then rows.
	- dropVariables
	- selectGroup
	- dropDuplicates
	- dropColumns
[feature engnineering]
[type] understand data types.
	- understandType
	- dummyCat
	- imputeNum
[ensamble] ensamble pipelines.
	- edAnalysis
	- modeling
	- preprocessing
[run] run for feature engineering test or not.
"""
import pandas as pd

from .util import getConfig
from .FeatureEng import featureEngineering

def getList():
	""" To get file list.
	"""
	file_list = []
	for y in range(2016, 2019):
		for m in range(1, 13):
			if (y == 2016 and m < 7): continue
			if len(str(m)) == 1: m = '0'+str(m)
			file_list.append('train'+str(y)+str(m)+'.csv')
	return file_list

def concatFile(file_list):
	""" To combine files in file list.
	"""
	config = getConfig()

	print('[load]concating...')
	df_list = []
	for f in file_list:
		print(f)
		tmp = pd.read_csv(config['dir_raw']+f, index_col=None, header=0)
		df_list.append(tmp)
	df = pd.concat(df_list, axis=0, ignore_index=True)
	return df


def dropVariables(df):
	""" To drop unnecessary variables.
	"""
	config = getConfig()
	print('[trim]dropping unnecessary variables...')
	return df[[x for x in df.columns if x not in config['COL_DROP']]]

def selectGroup(df):
	""" To select a specific group.
	"""
	print('[trim]selecting a group...')	
	return df[() & ()]

def dropDuplicates(df):
	""" To drop duplicated rows.
	"""
	print('[trim]dropping duplicated rows...')	
	return df.drop_duplicates()

def dropColumns(df):
	""" To drop unusable variables.
	"""
	print('[trim]dropping unusable variables...')
	COL_NULL = df.columns[df.isnull().all()]
	COL_CONS = df.columns[df.apply(lambda x: (x.nunique()) == 1)]
	print('all null columns:\n', COL_NULL)
	print('constant columns:\n', COL_CONS)
	return df[[x for x in df.columns if x not in COL_NULL+COL_CONS]]


def understandType(df):
	""" To understand types of variables.
	"""
	COL_CAT_SPEC = []
	COL_NUM_SPEC = []

	COL_CAT_REG = [x for x in df.columns if any(s in x for s in ['level', 'code'])]
	COL_NUM_REG = [x for x in df.columns if any(s in x for s in ['rate', 'amt'])]

	COL_NUM = COL_NUM_REG + COL_NUM_SPEC
	COL_CAT = COL_CAT_REG + COL_CAT_SPEC
	return COL_NUM, COL_CAT

def numeric_encoder(df, features_cat, train=True, dict_code_path='dict_category_code.json'):
    def col2str(df, feature_list):
        for i in feature_list:
            df[i] = df[i].astype(str)
        return df
        
    df[features_cat] = col2str(df[features_cat], features_cat)
    
    if train:
        dict_code = dict.fromkeys(features_cat)

        for i, f in enumerate(features_cat):
            cat = list(pd.unique(df[f]))
            numeric = list(range(len(cat)))

            code = dict(zip(cat, numeric))
            dict_code[f] = code

            df[f] = df[f].map(code)
        with open(dict_code_path, 'w') as f:
            f.write(json.dumps(dict_code))
        print('code dictionary of categorical features saved to {}'.format(dict_code_path))
    else:
        with open(dict_code_path, 'r') as f:
            dict_code = json.load(f)
        for f in features_cat:
            df[f] = df[f].map(dict_code[f])
            df[f].fillna(-1, inplace=True)
            df[f] = df[f].astype(int, errors='ignore')
    return df
    
def dummyCat(df):
	""" To get dummies of category variables.
	"""
	COL_NUM, COL_CAT = understandType(df)
	df[COL_CAT] = df[COL_CAT].astype('category')
	return pd.get_dummies(df, columns=COL_CAT)

def imputeNum(df):
	""" To impute partial numeric variables. 
	Note: We don't impute all numeric variables, but the ones we know they should be 0 if Nan.
	"""
	COL_NUM, COL_CAT = understandType(df)
	COL_NULL_ANY = df.columns[df.isna().any()].tolist()
	COL_NUM_NULL_ANY = [x for x in COL_NUM if x in COL_NULL_ANY]
	COL_IMP_ZERO = [x for x in COL_NUM_NULL_ANY if any(s in x for s in ['cnt', 'amt'])]

	df[COL_IMP_ZERO] = df[COL_IMP_ZERO].fillna(0)
	return df


def edAnalysis(df):
	""" Setup pipeline for exploratory data analysis.
	"""
	df = (df.pipe(dropVariables)
			.pipe(selectGroup)
			.pipe(dropDuplicates)
			.pipe(dropColumns))
	return df 

def modeling(df):
	""" Setup pipeline for modeling.
	"""
	df = (df.pipe(featureEngineering)
			.pipe(dummyCat)
			.pipe(imputeNum))
	return df 

def preprocessing(df):
	""" Setup pipeline for the whole preprecessing.
	"""
	df = (df.pipe(edAnalysis)
			.pipe(modeling))
	return df


def splitting(df, target):
	""" Split data into train and test date sets.
	"""
	from sklearn.model_selection import train_test_split
	return train_test_split(df, test_size=0.2, stratify=df[target], random_state=2019)


def run(test=False):
	config = getConfig()

	df = concatFile(getList)

	df = edAnalysis(df)
	if not test:
		print('saving for eda...')
		df.to_csv(config['dir_tmp']+config['file_eda'], index=False)

	df = modeling(df)
	train, test = splitting(df, config['target'])
	if not test:
		print('saving for modeling...')
		train.to_csv(config['dir_train']+config['file_train'], index=False)
		test.to_csv(config['dir_test']+config['file_test'], index=False)
		print('done.')

	if test:
		from .FeatureEng import scoreIt
		print('testing...')
		features = [x for x in train.columns if x not in config['drop']]
		scoreIt(train, config['target'], features)
	print('done.')
	return None


def main():
	run(test=False)
	return None


if __name__ == '__main__':
	main()