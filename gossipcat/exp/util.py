#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
import warnings
warnings.filterwarnings('ignore')

def getConfig(configFile):
    """To get configuration file in one step.

    Args:
        configFile: configuration file name, like 'config.json'.

    Returns:
        config: a dictionary contians configuration.
    """
    import json

    config = dict()
    try:
        with open(configFile, 'r') as f:
            config = json.load(f)
        return config 
    except Exception as e:
        print('[CRITIAL] NO CONFIGURATION FILE FOUND!')
        raise e

def install(package):
    """To install a Python package within Python.

    Args:
        package: package name, string.
    """
    import subprocess
    import sys

    try:
        subprocess.call([sys.executable, "-m", "pip", "install", package])
        print(package, ' successfuly installed.')
    except Exception as e:
        raise e

def flatten(df, feature_list, k_list):
    import ast
    for i, f in enumerate(feature_list):
        l = []
        for j in range(k_list[i]):
            l.append('{}_{}'.format(feature_list[i], j))
        df[feature_list[i]] = df[feature_list[i]].apply(lambda x: ast.literal_eval(x))
        df[l] = pd.DataFrame(df[feature_list[i]].tolist(), index=df.index)
        _ = df.pop(feature_list[i])
    return df

def as_keras_metric(method):
    """A metric decorator for Keras.

    example:
    import tensorflow as tf

    precision = as_keras_metric(tf.metrics.precision)
    recall = as_keras_metric(tf.metrics.recall)

    @as_keras_metric
    def auc_pr(y_true, y_pred, curve='PR'):
        return tf.metrics.auc(y_true, y_pred, curve=curve)

    ...[Keras model build]
    model.complile(loss='binary_crossentropy', optimizer='adam', metrics=[auc_pr])
    """
    import functools
    from keras import backend as K 
    import tensorflow as tf 

    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

def undersampling(df, target, num_class=2):
    """An algorithm for undersampling.

    Args:
        df: a Pandas dataframe.
        target: the target variable.
        num_class: the number of classes of the target, can only be 2 or 3.
        
    Return:
        df: an undersampled dataframe.
    """
    import numpy as np 
    
    if num_class==2:
        num_p = len(df[df[target] == 1])
        ind_n = df[df[target] == 0].index
        ind_p = df[df[target] == 1].index
    elif num_class == 3:
        num_p = len(df[df[target].isin([1,2])])
        ind_n = df[df[target] == 0].index
        ind_p = df[df[target].isin([1,2])].index
    else:
        print("[ERROR] num_class is not supported! It can only be 2 or 3.")

    ind_r = np.random.choice(ind_n, num_p, replace=False)
    ind_s = np.concatenate([ind_p, ind_r])
    return df.loc[ind_s]


def learningCurve(loss_ls):
    import numpy as np
    import matplotlib.pyplot as plt 

    loss_min = np.min(loss_ls)
    loss_max = np.max(loss_ls)
    loss_std = np.std(loss_ls)

    plt.figure(figsize=(10,8))
    plt.plot(loss_ls)
    plt.title('learning curves')
    plt.xlabel('number of iterations')
    plt.ylabel('reconstruction loss')
    plt.ylim(loss_min-loss_std, loss_max+loss_std)
    plt.grid()
    plt.show()

    return None


def stackPlot(df):
    df.groupby(by=['cate_1', 'cate_2']).size().unstack().plot(kind='bar', stacked=True)
    return None

def ListReporter(df, prob=None, label=None, rpt_num=100):
    import matplotlib.pyplot as plt

    df = df.sort_values(by=prob, ascending=False).reset_index()
    df['cum_cor'] = df[label].cumsum()
    df['cum_acc'] = round(df['cum_cor']/(df.index.values+1)*100, 2)
    del df['index']
    
    plt.figure(figsize=(10, 6))
    df['cum_acc'].plot(xlim=(-round(rpt_num/60), rpt_num))
    plt.grid()
    plt.title('Accuracy Along with Report List')
    plt.xlabel('Number of Reported')
    plt.ylabel('Accuracy (%)')
    plt.show()

    return None 