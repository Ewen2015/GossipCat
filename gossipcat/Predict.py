#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
import os
import pandas as pd 
import numpy as np
import datetime
import lightgbm as lgb 

def _record(msg):
    print(msg)
    with open(os.getcwd()+'/record/record_predict.log', 'a') as file:
        file.write(msg)
    return None

def Power(index, drop_list):
    cwd = os.getcwd()
    test_file = os.listdir(cwd+'/data/test/')[0]
    test_data = pd.read_csv(cwd+'/data/test/'+test_file)

    msg = '\ntime:\t'+datetime.datetime.now().strftime('%Y-%m-%d %H:%M')+'\ndata:\t'+test_file
    _record(msg)

    model_file = [x for x in os.listdir(cwd+'/model/') if 'model' in x][0]
    model = lgb.Booster(model_file=cwd+'/model/'+model_file)

    features = [x for x in test_data.columns if x not in drop_list]

    print('\npredicting...')
    pred = model.predict(test_data[features])
    prob = np.where(pred>=0.5, 1, 0)

    results = pd.DataFrame(columns=['index', 'prediction', 'probability'])
    results['index'] = test_data[index]
    results['prediction'] = prob
    results['probability'] = pred
    results = results.sort_values('probability', ascending=False)

    print('\nsaving...')
    now = datetime.datetime.now()
    result_file = 'result_'+str(now.year)+'_'+str(now.month)+'_'+str(now.day)+'.txt'
    results.to_csv(cwd+'/data/predict/'+result_file, index=False)
    return None

def main():
    index = ''
    drop_list = []
    Power(index, drop_list)
    return None

if __name__ == '__main__':
    main()