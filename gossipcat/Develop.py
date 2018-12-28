#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
import warnings
warnings.filterwarnings('ignore')

import time
import json
import logging
import numpy as np 
import pandas as pd

from keras.utils import np_utils
from keras.models import Sequential 
from keras.layers import Dense 
from keras.callbacks import EarlyStopping 

from .Configure import Configure

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

def Develop():
    alias = 'dense_'

    config = Configure()

    config['file_model'] = 'model_'+alias+config['version']+'.pkl'
    config['file_result'] = 'result_'+alias+config['version']+'.csv'

    logging.basicConfig(filename=config['wd_log']+config['file_log'],
                        level=logging.INFO,
                        format='%(asctime)s:%(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(json.dumps(config, indent=4))
    time.sleep(3)

    train = pd.read_csv(config['wd_train']+config['file_train'], low_memory=False)
    logging.info('training data loaded from '+config['wd_train']+config['file_train'])

    target = config['target']
    features = [x for x in train.columns if x not in config['drop_list']]


    mod = keras_train(train, target, features,
                    batch_size=config['batch_size'],
                    epochs=config['epochs'],
                    patience=config['patience'],
                    verbose=config['verbose'],
                    validation_split=config['validation_split'],
                    multi=config['multi'])
    mod.save(config['wd_model']+config['file_model'])
    logging.info('model saved to '+config['wd_model']+config['file_model'])

    return None



















