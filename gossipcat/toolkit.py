#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
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

