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

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import networkx as nx 
import tensorflow as tf 
import sklearn as sk

def onehot_encoder(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot, [*classes]

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
    def __init__(self, edgelist, data, nodecol, target, features, target_multi, multi_label=False, seed=0):
        super(GraphCN, self).__init__()
        self.edgelist = edgelist
        self.data = data 
        self.nodecol = nodecol
        try:
            self.target = target
        except Exception as e:
            self.target = None
        self.features = features
        try:
            self.target_multi = target_multi  
        except Exception as e:
            self.target_multi = None 
        self.multi_label = multi_label
        self.seed = seed
        
        self.g = nx.from_pandas_edgelist(df=self.edgelist, 
                                         source=self.edgelist.columns[0],
                                         target=self.edgelist.columns[1],
                                         create_using=nx.MultiDiGraph())
        self.a = adjacency_normalizer(nx.to_numpy_matrix(G=self.g))

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
        self.evaluate_loss = None
        self.loss_min, self.loss_max = None, None
        self.loss_std, self.loss_avg = None, None

    def model(self, learning_rate=0.001):
        tf.reset_default_graph()
        tf.set_random_seed(self.seed)
        n_dense1 = 64 
        n_dense2 = 16

        adjacency = tf.placeholder(dtype=tf.float32, shape=(self.n_nodes, self.n_nodes))
        feature = tf.placeholder(dtype=tf.float32, shape=(self.n_nodes, self.n_features))
        label = tf.placeholder(dtype=tf.float32, shape=(self.n_nodes, self.n_classes))

        with tf.name_scope('gcn1'):
            weight1 = tf.get_variable(dtype=tf.float32, initializer=tf.random_normal([self.n_features, self.n_features]), name='weight1')
            bais1 = tf.get_variable(dtype=tf.float32, initializer=tf.random_normal([self.n_nodes, self.n_features]), name='baise1')
            gcn1 = graph_convolution(a=adjacency, x=feature, w=weight1, b=bais1, name='gcn_layer1')
        
        with tf.name_scope('gcn1'):
            weight2 = tf.get_variable(dtype=tf.float32, initializer=tf.random_normal([self.n_features, n_dense1]), name='weight2')
            bais2 = tf.get_variable(dtype=tf.float32, initializer=tf.random_normal([self.n_nodes, n_dense1]), name='baise2')
            gcn2 = graph_convolution(a=adjacency, x=gcn1, w=weight2, b=bais2, name='gcn_layer2')

        with tf.contrib.framework.arg_scope([tf.contrib.layers.fully_connected], 
                                            activation_fn=tf.nn.relu):
            dense1 = tf.contrib.layers.fully_connected(inputs=gcn2, num_outputs=n_dense2, scope='dense1')
            dense2 = tf.contrib.layers.fully_connected(inputs=dense1, num_outputs=self.n_classes, scope='dense5')

        with tf.name_scope('output'):
            if self.multi_label:
                output = tf.nn.sigmoid(x=dense2, name='output')
            else:
                output = tf.nn.softmax(logits=dense2, axis=-1, name='output')

        with tf.name_scope('loss'):
            if self.multi_label:
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=label))
            else:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=label))

        training_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return adjacency, feature, label, output, loss, training_op

    def train(self, learning_rate=0.001, n_epochs=100, verbose=1, early_stopping=10, path_model=None):
        adjacency, feature, label, output, loss, training_op = self.model(learning_rate)

        init = tf.global_variables_initializer()
        saver = saver = tf.train.Saver()
        
        self.best_loss = 9999
        stopping_step = 0

        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(1, n_epochs+1):
                start_time = time.time()
                training_op.run(feed_dict={adjacency: self.a, feature: self.x, label: self.y})
                cost = loss.eval(feed_dict={adjacency: self.a, feature: self.x, label: self.y})
                self.loss_ls.append(cost)

                if verbose <= 0:
                    pass 
                elif epoch % verbose == 0:
                    duration = time.time() - start_time
                    message = '%s: epoch: %d \tloss: %.6f \tduration: %.2f s' % (datetime.now(), epoch, cost, duration)
                    print(message)

                if self.best_loss > cost:
                    stopping_step = 0
                    self.best_loss = cost
                else:
                    stopping_step += 1
                if stopping_step >= early_stopping:
                    print('early stopping triggered at epoch: %d with best loss: %.6f.' % (epoch, self.best_loss))
                    break

            self.prediction = pd.DataFrame(output.eval(feed_dict={adjacency: self.a, feature: self.x}), columns=self.classes)
            if path_model == None:
                pass
            else:
                save_path = saver.save(sess, path_model)
                print('model saved in path: %s' % save_path)
        return None

    def evaluate(self, path_model):
        adjacency, feature, label, output, loss, training_op = self.model()

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, path_model)
            message = 'model loaded from path: %s' % path_model
            print(message)
            self.prediction = pd.DataFrame(output.eval(feed_dict={adjacency: self.a, feature: self.x}), columns=self.classes)
            message = 'prediction done.'
            print(message)

            self.evaluate_loss = loss.eval(feed_dict={adjacency: self.a, feature: self.x, label: self.y})

            if self.multi_label:
                message = '\nevaluation:'+\
                          '\n\tloss: %.6f' % self.evaluate_loss
                return self.evaluate_loss
            else:
                predprob = self.prediction[self.prediction.columns[1]]
                self.evaluate_auc_roc = sk.metrics.roc_auc_score(self.df[self.target], predprob)
                self.evaluate_auc_pr = sk.metrics.average_precision_score(self.df[self.target], predprob)

                message = '\nevaluation:'+\
                          '\n\tloss: %.6f' % self.evaluate_loss+\
                          '\n\tauc_roc: %.6f' % self.evaluate_auc_roc+\
                          '\n\tauc_pr: %.6f' % self.evaluate_auc_pr
        print(message)
        

    def predict(self, path_model=None, path_result=None):
        adjacency, feature, label, output, loss, training_op = self.model()

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, path_model)
            message = 'model loaded from path: %s' % path_model
            print(message)
            self.prediction = pd.DataFrame(output.eval(feed_dict={adjacency: self.a, feature: self.x}), columns=self.classes)
            message = 'prediction done.'
            print(message)

        if path_result == None:
            pass
        else:
            self.prediction.to_csv(path_result, index=False)
            message = 'results saved in path: %s' % path_result
            print(message)
        return self.prediction

    def retrain(self, learning_rate=0.001, n_epochs=100, verbose=1, early_stopping=10, path_model=None, path_model_update=None):
        adjacency, feature, label, output, loss, training_op = self.model(learning_rate)

        saver = tf.train.Saver()

        self.best_loss = 9999
        stopping_step = 0

        with tf.Session() as sess:
            saver.restore(sess, path_model)
            message = 'model loaded from path: %s' % path_model
            print(message)
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
                    print(message)

                if self.best_loss > cost:
                    stopping_step = 0
                    self.best_loss = cost
                else:
                    stopping_step += 1
                if stopping_step >= early_stopping:
                    print('early stopping triggered at epoch: %d with best loss: %.6f.' % (epoch, self.best_loss))
                    break

            self.prediction = pd.DataFrame(output.eval(feed_dict={adjacency: self.a, feature: self.x}), columns=self.classes)
            if path_model_update == None:
                pass
            else:
                save_path = saver.save(sess, path_model_update)
                print('model saved in path: %s' % save_path)
        return None

    def learning_curve(self):
        if len(self.loss_ls) == 0:
            return 'no models trained, no learning curve.'

        self.loss_min, self.loss_max = np.min(self.loss_ls), np.max(self.loss_ls)
        self.loss_std, self.loss_avg = np.std(self.loss_ls), np.average(self.loss_ls)

        plt.figure(figsize=(10, 4))
        plt.plot(self.loss_ls)
        plt.title('learning curve')
        plt.xlabel('number of epochs')
        plt.ylabel('loss')
        plt.ylim(self.loss_min-self.loss_std, self.loss_max+self.loss_std)
        plt.grid() 
        plt.show()

        summary = 'summary:\n'+\
                  'average loss: %.6f (std: %.6f)' % (self.loss_avg, self.loss_std)
        print(summary)
        return None   

    def report(self):
        if len(self.prediction) == 0:
            return 'no models trained, no learning curve.'

        try:
            from gossipcat.Report import Visual
        except Exception as e:
            print('[WARNING] Package GossipCat not installed.')
            try:
                from Report import Visual
            except Exception as e:
                return '[ERROR] Package Report not installed.'

        for ind, val in enumerate(self.prediction.columns):
            if self.multi_label:
                print(ind, val)
                test_target=self.df[val]
            else:
                test_target=self.df[self.target]
                if ind == 0:
                    continue
            prob = self.prediction[val]

            plt.figure(figsize=(6, 5.5))
            self.prediction[val].hist(bins=100)
            plt.title('distribution of predictions')

            vis = Visual(test_target=test_target, test_predprob=prob)
            vis.CM()
            vis.ROC()
            vis.PR()
        return None









        