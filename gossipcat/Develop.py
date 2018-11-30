#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
import warnings
warnings.filterwarnings('ignore')

import json
import time
import logging
import pickle 
import numpy as np 
import pandas as pd

import xgboost as xgb 

from keras.utils import np_utils
from keras.models import Sequential 
from keras.layers import Dense 
from keras.callbacks import EarlyStopping 

import .Configure

def xgb_train(train, target, features, xvalidation=0, num_round=3000, num_verbose=600, balanced=0, gpu=0, multi=0):
    dtrain = xgb.DMatrix(train[features], label=train[target])

    if xvalidation == 0:
        pass
    else:
        start = time.time()
        param_cv = {
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'eval_metric': 'auc',
            'maximize': True,
            'early_stoppong_rounds': 100,
            'eta': 0.01,
            'max_depth': 3,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'stratified': True,
            'nfold': 10,
            'seed': 2018
        }
        if balanced == 0:
            param_cv['eval_metric'] = 'aucpr'
        if gpu == 1:
            param_cv['tree_method'] = 'gpu_hist'
        if multi == 1:
            param_cv['objective'] = 'multi:softmax'
            param_cv['eval_metric'] = 'mlogloss'
        cvb = xgb.cv(params=param_cv,
                     dtrain=dtrain,
                     num_boost_round=num_round,
                     verbose_eval=num_verbose)
        duration = round((time.time()-start)/60, 2)
        try:
            logging.info('cross validation done.')
            logging.info('duration: '+str(duration)+' min.')
        except Exception as e:
            print('cross validation done.')
            print('duration: '+str(duration)+' min.')

    start = time.time()
    evallist = [(dtrain, 'eval'), (dtrain, 'train')]
    param_tr = {
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'eval_metric': 'auc',
        'eta': 0.01,
        'max_depth': 3,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
    }
    if balanced == 0:
        param_tr['eval_metric'] = 'aucpr'
    if gpu == 1:
        param_tr['tree_method'] = 'gpu_hist'
        param_tr['predictor'] = 'gpu_predictor'
    if multi == 1:
        param_tr['objective'] = 'multi:softmax'
        param_tr['eval_metric'] = 'mlogloss'
    bst = xgb.train(params=param_tr,
                    dtrain=dtrain,
                    evals=evallist,
                    num_boost_round=num_round if xvalidation==0 else cvb.shape[0],
                    verbose_eval=num_verbose)
    try:
        logging.info('cross validation done.')
        logging.info('duration: '+str(duration)+' min.')
    except Exception as e:
        print('cross validation done.')
        print('duration: '+str(duration)+' min.')
    return bst 

def xgb_predict(test, features, booster):
    dtest = xgb.DMatrix(test[features])
    pred = booster.predict(dtest)
    try:
        logging.info('prediction done.')
    except Exception as e:
        print('prediction done.')
    return pred 

def keras_train(train, target, features, batch_size=100, epochs=600, patience=30, verbose=1, validation_split=0.2, multi=0):
    if multi == 0:
        y = train[target]
    else:
        y = np_utils.to_categorical(train[target])
    X = train[features]
    n_features = len(features)

    start = time.time()
    model = Sequential()
    model.add(Dense(16, input_dim=n_features, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(4, kernel_initializer='uniform', activation='relu'))
    if multi == 0:
        model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model.add(Dense(multi, kernel_initializer='uniform', activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.fit(X, y, 
              batch_size=epochs,
              epochs=epochs,
              verbose=verbose,
              callbacks=[EarlyStopping(patience=patience)],
              validation_split=validation_split)
    duration = round((time.time()-start)/60, 2)
    try:
        logging.info('training done.')
        logging.info('duration: '+str(duration)+' min.')
    except Exception as e:
        print('cross validation done.')
        print('duration: '+str(duration)+' min.')
    return model 

def keras_predict(test, features, model, multi=0):
    if multi == 0:
        pred = model.predict(test[features])
        pred = np.array([item for sublist in pred for item in sublist])
    else:
        pred = model.predict_classes(test[features])
    try:
        logging.info('prediction done.')
    except Exception as e:
        print('prediction done.')
    return pred 

def Develop(algorithm):
    alias = str(algorithm)[:2]+'_'

    config = Configure.config()

    config['file_model'] = 'model_'+alias+config['version']+'.pkl'
    config['file_result'] = 'result_'+alias+config['version']+'.csv'

    logging.basicConfig(filename=config['wd_log']+config['file_log'],
                        level=logging.INFO,
                        format='%(asctime)s:%(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(config)

    train = pd.read_csv(config['wd_train']+config['file_train'], low_memory=False)
    logging.info('training data loaded from '+config['wd_train']+config['file_train'])

    target = config['target']
    features = [x for x in train.columns if x not in config['drop_list']]

    if algorithm == 'gbdt':
        bst = xgb_train(train, target, features,
                        xvalidation=config['xvalidation'],
                        num_round=config['num_round'],
                        num_verbose=config['num_verbose'],
                        balanced=config['balanced'],
                        gpu=config['gpu'], 
                        multi=config['multi'])
        pickle.dump(bst, open(config['wd_model']+config['file_model'], 'wb'))
        logging.info('model saved to '+config['wd_model']+config['file_model'])
    elif algorithm == 'dl':
        mod = keras_train(train, target, features,
                        batch_size=config['batch_size'],
                        epochs=config['epochs'],
                        patience=config['patience'],
                        verbose=config['verbose'],
                        validation_split=config['validation_split'],
                        multi=config['multi'])
        mod.save(config['wd_model']+config['file_model'])
        logging.info('model saved to '+config['wd_model']+config['file_model'])
    else:
        logging.info('oops!')
    return None



















