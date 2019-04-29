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


def beautiful_nx(g):
    """A wrapper to draw graph nets beautifully.

    Args:
        g: a networkx graph object.

    Source:
        https://gist.github.com/jg-you/144a35013acba010054a2cc4a93b07c7.js
    """
    import copy
    import networkx as nx
    import matplotlib.pyplot as plt 

    pos = nx.layout.spectral_layout(g)
    pos = nx.spring_layout(g, pos=pos, iterations=50)

    pos_shadow = copy.deepcopy(pos)
    shift_amount = 0.006
    for idx in pos_shadow:
        pos_shadow[idx][0] += shift_amount
        pos_shadow[idx][1] -= shift_amount

    fig = plt.figure(frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    nx.draw_networkx_nodes(g, pos_shadow, node_color='k', alpha=0.5)
    nx.draw_networkx_nodes(g, pos, with_labels=True, node_color="#3182bd", linewidths=1)
    nx.draw_networkx_edges(g, pos, width=1)
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















