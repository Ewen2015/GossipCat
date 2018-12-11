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
from datetime import datetime
import logging

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import networkx as nx 
import tensorflow as tf 

def onehot_encoder(target_ls):
    classes = set(target_ls)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    target_onehot = np.array(list(map(classes_dict.get, target_ls)), dtype=np.int32)
    return target_onehot, [*classes]

def adjacency_normalizer(adj):
    adj = adj + np.eye(adj.shape[0])
    d = np.diagflat(np.power(np.array(adj.sum(1)), -1))
    a_norm = d.dot(adj)
    return a_norm

def graph_convolution(a, x, w, b, name=None):
    with tf.name_scope(name, 'gcn_layer', [a, x, w, b]):
        kernel = tf.add(tf.matmul(a, tf.matmul(x, w)), b)
        return tf.nn.tanh(kernel)


class GraphCN(object):
    """docstring for GraphCN"""
    def __init__(self, edgelist, data, nodecol, target, features, target_multi, multi_label=False, classes=None, seed=0):
        super(GraphCN, self).__init__()
        self.edgelist = edgelist
        self.data = data
        self.nodecol = nodecol
        self.features = features
        self.multi_label = multi_label
        try:
            self.target = target
        except Exception as e:
            self.target = None   
        try:
            self.target_multi = target_multi
        except Exception as e:
            self.target_multi = None
        try:
            self.classes = classes
        except Exception as e:
            self.classes = [0, 1]
        
        self.seed = seed

        self.g = nx.from_pandas_edgelist(self.edgelist, 
                                         source=self.edgelist.columns[0], 
                                         target=self.edgelist.columns[1], 
                                         create_using=nx.MultiDiGraph())
        self.a = adjacency_normalizer(nx.to_numpy_matrix(self.g))

        self.df = pd.DataFrame({self.nodecol: self.g.nodes()})
        self.df = self.df.merge(self.data, how='left', on=self.nodecol)

        self.x = self.df[self.features]
        
        if self.multi_label:
            self.y = self.df[self.target_multi]
            self.classes = self.target_multi
        else:
            self.y, self.classes = onehot_encoder(list(self.df[self.target]))
            
        self.n_nodes = self.g.number_of_nodes()
        self.n_features = len(self.features)
        self.n_classes = len(self.classes)

        self.loss_ls = []
        self.prediction = []


    def model(self, n_dense_1=64, n_dense_2=16, learning_rate=0.001, n_epochs=100, verbose=1, path_model=None):
        self.n_dense_1 = n_dense_1
        self.n_dense_2 = n_dense_2

        tf.set_random_seed(self.seed)

        adjacency = tf.placeholder(tf.float32, shape=(self.n_nodes, self.n_nodes))
        feature = tf.placeholder(tf.float32, shape=(self.n_nodes, self.n_features))
        label = tf.placeholder(tf.float32, shape=(self.n_nodes, self.n_classes))

        weight1 = tf.Variable(tf.random_normal([self.n_features, self.n_features]), name='weight1')
        bais1 = tf.Variable(tf.random_normal([self.n_nodes, self.n_features]), name='bais1')

        weight2 = tf.Variable(tf.random_normal([self.n_features, self.n_dense_1]), name='weight2')
        bais2 = tf.Variable(tf.random_normal([self.n_nodes, self.n_dense_1]), name='bais2')

        gcn_1 = graph_convolution(a=adjacency, x=feature, w=weight1, b=bais1, name='gcn_1')
        gcn_2 = graph_convolution(a=adjacency, x=gcn_1, w=weight2, b=bais2, name='gcn_2')

        with tf.contrib.framwork.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu):
            dense_1 = tf.contrib.layers.fully_connected(inputs=gcn_2, num_outputs=self.n_dense_2, scope='dense_1')
            dense_2 = tf.contrib.layers.fully_connected(inputs=dense_1, num_outputs=self.n_classes, scope='dense_2')
        
        if self.multi_label:
            # output = 
            # loss = 
            pass
        else:
            output = tf.nn.softmax(logits=dense_2, axis=-1, name='output')
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=label))

        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(n_epochs):
                start_time = time.time()
                training_op.run(feed_dict={adjacency: self.a, feature: self.x, label: self.y})
                cost = loss.eval(feed_dict={adjacency: self.a, feature: self.x, label: self.y})
                self.loss_ls.append(cost)
                if verbose <= 0:
                    pass
                elif epoch % verbose == 0:
                    duration = time.time() - start_time
                    message = '%s: epoch: %d \tloss: %.6f \tduration: %.2f s' % (datetime.now(), epoch, cost, duration)
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

    def evaluate(self, path_model=None):
        self.evaluate_loss = None

        tf.reset_default_graph()

        adjacency = tf.placeholder(tf.float32, shape=(self.n_nodes, self.n_nodes))
        feature = tf.placeholder(tf.float32, shape=(self.n_nodes, self.n_features))
        label = tf.placeholder(tf.float32, shape=(self.n_nodes, self.n_classes))

        weight = tf.get_variable(name='weight', dtype=tf.float32, shape=(self.n_features, self.n_classes))

        hidden = graph_convolution(a=adjacency, x=feature, w=weight)

        if self.multi_label:
            pass
        else:
            output = tf.nn.softmax(logits=hidden, axis=self.n_classes)

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=label))

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, path_model)
            message = 'model restored from path: %s' % path_model
            try:
                logging.info(message)
            except Exception as e:
                print(message)
            self.prediction = output.eval(feed_dict={adjacency: self.a, feature: self.x})
            message = 'prediction done.'
            try:
                logging.info(message)
            except Exception as e:
                print(message)
            self.evaluate_loss = loss.eval(feed_dict={adjacency: self.a, feature: self.x, label: self.y})
            message = '\nevaluation: '+\
                      '\n\tloss: %.6f' % self.evaluate_loss
            try:
                logging.info(message)
            except Exception as e:
                print(message)
        return None

    def predict(self, path_model=None, path_result=None):
        tf.reset_default_graph()

        adjacency = tf.placeholder(tf.float32, shape=(self.n_nodes, self.n_nodes))
        feature = tf.placeholder(tf.float32, shape=(self.n_nodes, self.n_features))

        weight = tf.get_variable(name='weight', dtype=tf.float32, shape=(self.n_features, self.n_classes))

        hidden = graph_convolution(a=adjacency, x=feature, w=weight)
        if self.multi_label:
            pass
        else:
            output = tf.nn.softmax(logits=hidden, axis=self.n_classes)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, path_model)
            message = 'model restored from path: %s' % path_model
            try:
                logging.info(message)
            except Exception as e:
                print(message)
            self.prediction = output.eval(feed_dict={adjacency: self.a, feature: self.x})
            message = 'prediction done.'
            try:
                logging.info(message)
            except Exception as e:
                print(message)

        if path_result == None:
            pass
        else:
            prediction = pd.DataFrame(self.prediction, columns=self.classes)
            prediction.to_csv(path_result, index=False)
            message = 'results saved in path: %s' % path_result
            try:
                logging.info(message)
            except Exception as e:
                print(message)            
        return self.prediction

    def retrain(self, keep_prob=0.5, learning_rate=0.001, n_epochs=100, verbose=1, path_model=None, path_model_update=None):
        tf.reset_default_graph()

        adjacency = tf.placeholder(tf.float32, shape=(self.n_nodes, self.n_nodes))
        feature = tf.placeholder(tf.float32, shape=(self.n_nodes, self.n_features))
        label = tf.placeholder(tf.float32, shape=(self.n_nodes, self.n_classes))

        weight = tf.get_variable(name='weight', dtype=tf.float32, shape=(self.n_features, self.n_classes))
            
        hidden1 = graph_convolution(a=adjacency, x=feature, w=weight)
        hidden2 = tf.nn.dropout(hidden1, keep_prob=keep_prob)
        
        if self.multi_label:
            pass
        else:
            output = tf.nn.softmax(logits=hidden2, axis=self.n_classes)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=label))
            
        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, path_model)
            message = 'model restored from path: %s' % path_model

            for epoch in range(n_epochs):
                start_time = time.time()
                training_op.run(feed_dict={adjacency: self.a, feature: self.x, label: self.y})
                cost = loss.eval(feed_dict={adjacency: self.a, feature: self.x, label: self.y})
                self.loss_ls.append(cost)
                if verbose <= 0:
                    pass
                elif epoch % verbose == 0:
                    duration = time.time() - start_time
                    message = '%s: epoch: %d \tloss: %.6f \tduration: %.2f s' % (datetime.now(), epoch, cost, duration)
                    try:
                        logging.info(message)
                    except Exception as e:
                        print(message)
            self.prediction = output.eval(feed_dict={adjacency: self.a, feature: self.x, label: self.y})
            if path_model_update == None:
                pass
            else:
                save_path = saver.save(sess, path_model_update)
                message = 'model saved in path: %s' % save_path
                try:
                    logging.info(message)
                except Exception as e:
                    print(message)
        return None

    def learning_curve(self):
        self.loss_min, self.loss_max, self.loss_std, self.loss_avg = np.min(self.loss_ls), np.max(self.loss_ls), np.std(self.loss_ls), np.average(self.loss_ls)

        plt.figure(figsize=(10,8))
        plt.plot(self.loss_ls)
        plt.title('learning curves')
        plt.xlabel('number of iterations')
        plt.ylabel('reconstruction loss')
        plt.ylim(self.loss_min-self.loss_std, self.loss_max+self.loss_std)
        plt.grid()
        plt.show()

        summary = 'summary:\n'+\
                  'average loss: %.6f (std: %.6f)' % (self.loss_avg, self.loss_std)
        print(summary)
        return None
        

