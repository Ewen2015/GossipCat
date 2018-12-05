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
import logging

import numpy as np 
import pandas as pd 
import networkx as nx 
import tensorflow as tf 
from tensorflow.contrib.layers import dropout

def onehot_encoder(target_ls):
    classes = set(target_ls)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    target_onehot = np.array(list(map(classes_dict.get, target_ls)), dtype=np.int32)
    return target_onehot

def adjacency_normalizer(adj):
    adj = adj + np.eye(adj.shape[0])
    d = np.diagflat(np.power(np.array(adj.sum(1)), -1))
    a_norm = d.dot(adj)
    return a_norm

def graph_convolution(a, x, w):
    with tf.name_scope('gcn_layer'):
        kernel = tf.matmul(x, w)
        return tf.nn.relu(tf.matmul(a, kernel))

class GraphCN(object):
    """docstring for GraphCN"""
    def __init__(self, edgelist, data, nodecol, target, features):
        super(GraphCN, self).__init__()
        self.edgelist = edgelist
        self.data = data
        self.nodecol = nodecol
        self.target = target
        self.features = features

        self.g = nx.from_pandas_edgelist(self.edgelist, 
                                         source=self.edgelist.columns[0], 
                                         target=self.edgelist.columns[1], 
                                         create_using=nx.MultiDiGraph())
        self.a = adjacency_normalizer(nx.to_numpy_matrix(self.g))

        self.df = pd.DataFrame({self.nodecol: self.g.nodes()})
        self.df = self.df.join(self.data, how='left', on=self.nodecol, lsuffix='_l', rsuffix='_r')
        self.df = self.df.fillna(0)

        self.x = self.df[self.features]
        self.y = onehot_encoder(list(self.df[self.target]))

        self.n_nodes = self.g.number_of_nodes()
        self.n_features = len(self.features)
        self.n_classes = self.y.shape[1]

        self.loss_ls = []
        self.accuracy_ls = []
        self.prediction = []

    def model(self, keep_prob=0.5, learning_rate=0.001, n_epochs=100, verbose=1, path_model=None):
        if path_model == None:
            path_model = 'model.ckpt'

        adjacency = tf.placeholder(tf.float32, shape=(self.n_nodes, self.n_nodes))
        feature = tf.placeholder(tf.float32, shape=(self.n_nodes, self.n_features))
        label = tf.placeholder(tf.float32, shape=(self.n_nodes, self.n_classes))

        weight = tf.Variable(tf.random_normal([self.n_features, self.n_classes], stddev=1))
            
        hidden1 = graph_convolution(a=adjacency, x=feature, w=weight)
        hidden2 = dropout(hidden1, keep_prob=keep_prob)
        output = tf.nn.softmax(logits=hidden2, axis=self.n_classes)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=label))
        accuracy = tf.metrics.accuracy(labels=label, predictions=output)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(n_epochs):
                start_time = time.time()
                training_op.run(feed_dict={adjacency: self.a, feature: self.x, label: self.y})
                if verbose <= 0:
                    pass
                elif epoch % verbose == 0:
                    duration = time.time() - start_time
                    err = loss.eval(feed_dict={adjacency: self.a, feature: self.x, label: self.y})
                    acc = accuracy.eval(feed_dict={adjacency: self.a, feature: self.x, label: self.y})
                    self.loss_ls.append(err)
                    self.accuracy_ls.append(acc)
                    message = 'epoch: %d \tloss: %.6f \taccuracy: %.6f \tduration: %.2f s' % (epoch, err, acc, duration)
                    try:
                        logging.info(message)
                    except Exception as e:
                        print(message)
            self.prediction = output.eval(feed_dict={adjacency: self.a, feature: self.x, label: self.y})
            if path_model == None:
                pass
            else:
                save_path = saver.save(sess, path_model, write_meta_graph=False)
                message = 'model saved in path: %s' % save_path
                try:
                    logging.info(message)
                 except Exception as e:
                    print(message)
        return None

    def predict(self, path_model=None):
        tf.reset_default_graph()

        adjacency = tf.placeholder(tf.float32, shape=(self.n_nodes, self.n_nodes))
        feature = tf.placeholder(tf.float32, shape=(self.n_nodes, self.n_features))

        weight = tf.get_variable(dtype=tf.float32, shape=(self.n_features, self.n_classes))

        # hidden1 = graph_convolution(a=adjacency, x=feature, w=weight)
        # hidden2 = dropout(hidden1)
        # output = tf.nn.softmax(logits=hidden2, axis=self.n_classes)
        prediction = graph_convolution(a=adjacency, x=feature, w=weight)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, path_model)
            message = 'model restored from path: %s' % path_model
            try:
                logging.info(message)
            except Exception as e:
                print(message)
            self.prediction = prediction.eval(feed_dict={adjacency: self.a, feature: self.x})
            message = 'prediction done.'
            try:
                logging.info(message)
            except Exception as e:
                print(message)
        return self.prediction
        




