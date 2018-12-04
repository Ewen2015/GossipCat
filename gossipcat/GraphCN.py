#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
import numpy as np 
import networkx as nx 
import tensorflow as tf 
from tensorflow.contrib.layers import dropout
from tensorflow.contrib.layers import fully_connected

def graph_convolution(features, adjacency, degree, weights, N):
    with tf.name_scope('gcn_layer'):
        d_ = tf.pow(degree + tf.eye(N), -0.5)
        y = tf.matmul(d_, tf.matmul(adjacency + tf.eye(N), d_))
        kernel = tf.matmul(features, weights)
        return tf.nn.relu(tf.matmul(y, kernel))

class GraphCN(object):
    """docstring for GraphCN"""
    def __init__(self, edgelist, data, nodecol, target, features):
        super(GraphCN, self).__init__()
        self.edgelist = edgelist
        self.data = data
        self.nodecol = nodecol
        self.target = target
        self.features = features

        self.G = nx.from_pandas_edgelist(self.edgelist, 
                                         source=self.edgelist.column[0], 
                                         target=self.edgelist.column[1], 
                                         create_using=nx.MultiDiGraph())

        self.T = 2
        self.P = len(self.features)
        self.N = self.G.number_of_nodes()

        self.A = nx.to_numpy_matrix(self.G)
        self.D = np.array(np.sum(self.A, axis=0))[0]
        self.D = np.matrix(np.diag(self.D))
        self.X = self.data[self.features]
        self.L = self.data[self.target]

    def build(self, is_training=True, keep_prob=0.5, learning_rate=0.01, n_iterations=100, verbose=1):
        self.loss_ls = []

        # instantiate tensors
        Adjacency = tf.placeholder(tf.float32, shape=(self.N, self.N))
        Degree = tf.placeholder(tf.float32, shape=(self.N, self.N))
        Features = tf.placeholder(tf.float32, shape=(self.N, self.P))
        Labels = tf.placeholder(tf.float32, shape=(self.N, self.T))

        # weight matrix: standard normal
        weights1 = tf.Variable(tf.random_normal([self.P, self.N]))
        weights2 = tf.Variable(tf.random_normal([self.N, self.N]))
            
        hidden1 = graph_convolution(Features, Adjacency, Degree, weights1, self.N)
        hidden2 = graph_convolution(hidden1, Adjacency, Degree, weights2, self.N)
        hidden3 = dropout(hidden2, keep_prob=keep_prob, is_training=is_training)
        output = fully_connected(hidden3, self.T, activation_fn=None)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=Labels))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            for iteration in range(n_iterations):
                training_op.run(feed_dict={Features: self.X, Adjacency: self.A, Degree: self.D, Labels: self.L})
                if verbose <= 0:
                    pass
                elif iteration % verbose == 0:
                    err = loss.eval(feed_dict={Features: self.X, Adjacency: self.A, Degree: self.D, Labels: self.L})
                    self.loss_ls.append(err)
                    print(iteration, '\tloss: %.6f' %err)
            # Save the variables to disk.
            save_path = saver.save(sess, "../model/model.ckpt")
            print("Model saved in path: %s" % save_path)
        return None
        




