#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wang.enqun@outlook.com
license:    Apache License 2.0
"""
import os
import pandas as pd 
import datetime
import lightgbm as lgb 
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

def _record(msg):
    print(msg)
    with open(os.getcwd()+'/record/record_train.log', 'a') as file:
        file.write(msg)
    return None

def Power(target, drop_list):
    cwd = os.getcwd()
    train_file = os.listdir(cwd+'/data/train/')[0]
    train_data = pd.read_csv(cwd+'/data/train/'+train_file)

    msg = '\ntime:\t'+datetime.datetime.now().strftime('%Y-%m-%d %H:%M')+'\ndata:\t'+train_file
    _record(msg)

    features = [x for x in train_data.columns if x not in drop_list]

    train, test = train_test_split(train_data, test_size=0.2, random_state=0)

    lgb_train = lgb.Dataset(train[features], train[target], free_raw_data=False)
    lgb_valid = lgb.Dataset(test[features], test[target], reference=lgb_train, free_raw_data=False)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'bianry',
        'metric': 'auc',
        'num_leaves': 32,
        'learning_rate': 0.01,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'verbose': 0
        }

    print('\ntraining...')
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=10000,
                    valid_set=lgb_valid,
                    early_stopping_round=200,
                    verbose_eval=100)
    print('\nsaving...')
    now = datetime.datetime.now()
    model_file = 'model_'+str(now.year)+'_'+str(now.month)+'_'+str(now.day)+'.txt'
    gbm.save_model(cwd+'/model/'+model_file)

    average_precision = average_precision_score(test[target], gbm.predict(test[features]))
    msg = '\naverage precision:\t'+str(average_precision)
    _record(msg)

    return None

def main():
    target = ''
    drop_list = []
    Power(target, drop_list)
    return None

if __name__ == '__main__':
    main()