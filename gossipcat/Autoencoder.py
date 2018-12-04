#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

class Autoencoder(object):
    """docstring for Autoencoder"""
    def __init__(self, train, test, features, target=None):
        super(Autoencoder, self).__init__()
        self.train = train
        self.test = test
        self.features = features
        self.target = target

        self.X_train = self.train[self.features]
        self.X_test = self.test[self.features]

        self.y_train = self.train[self.target]
        self.y_test = self.test[self.target]

        self.codings_val_l = None
        self.codings_val_s = None
        
    def simplelinear(self, n_inputs=16, n_hidden=2, learning_rate=0.01, n_iterations=100, verbose=10):
        n_inputs = len(self.features)
        n_outputs = n_inputs
        self.loss = []

        X = tf.placeholder(tf.float32, shape=[None, n_inputs])

        hidden = fully_connected(X, n_hidden, activation_fn=None)
        outputs = fully_connected(hidden, n_outputs, activation_fn=None)

        reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_op = optimizer.minimize(reconstruction_loss)

        init = tf.global_variables_initializer()

        codings = hidden

        with tf.Session() as sess: 
            sess.run(init)
            for iteration in range(n_iterations):
                training_op.run(feed_dict={X: self.X_train})
                if verbose <= 0:
                    pass
                elif iteration % verbose == 0:
                    err = reconstruction_loss.eval(feed_dict={X: self.X_train})
                    self.loss.append(err)
                    print(iteration, '\treconstruction loss: %.6f' %err)
            self.codings_val_l = codings.eval(feed_dict={X: self.X_test})
        self.loss_min = np.min(self.loss)
        self.loss_max = np.max(self.loss)
        self.index_min = self.loss.index(self.loss_min)
        return self.codings_val_l

    def stacked(self, n_inputs=16, n_hidden1=64, n_hidden2=16, n_hidden3=2, n_iterations=100, verbose=10, learning_rate=0.01, l2_reg=0.001):
        n_inputs = len(self.features)
        n_hidden4 = n_hidden2
        n_hidden5 = n_hidden1
        n_outputs = n_inputs
        self.loss_total = []
        self.loss_recon = []

        X = tf.placeholder(tf.float32, shape=[None, n_inputs])

        with tf.contrib.framework.arg_scope([fully_connected], 
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                            weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg)):
            hidden1 = fully_connected(X, n_hidden1)
            hidden2 = fully_connected(hidden1, n_hidden2)
            hidden3 = fully_connected(hidden2, n_hidden3)
            hidden4 = fully_connected(hidden3, n_hidden4)
            hidden5 = fully_connected(hidden4, n_hidden5)
            outputs = fully_connected(hidden5, n_outputs, activation_fn=None)

        reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.add_n([reconstruction_loss] + reg_losses)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        
        codings = hidden3

        with tf.Session() as sess: 
            sess.run(init)
            for iteration in range(n_iterations):
                training_op.run(feed_dict={X: self.X_train})
                if verbose <= 0:
                    pass
                elif iteration % verbose == 0:
                    err_t = loss.eval(feed_dict={X: self.X_train})
                    err_r = reconstruction_loss.eval(feed_dict={X: self.X_train})
                    self.loss_total.append(err_t)
                    self.loss_recon.append(err_r)
                    print(iteration, '\ttotal loss: %.6f' %err_t, '\treconstrctn loss: %.6f' %err_r)
            self.codings_val_s = codings.eval(feed_dict={X: self.X_test})
        self.loss_t_min = np.min(self.loss_total)
        self.loss_t_max = np.max(self.loss_total)
        self.index_t_min = loss.index(self.loss_t_min)

        self.loss_r_min = np.min(self.loss_recon)
        self.loss_r_max = np.max(self.loss_recon)
        self.index_r_min = loss.index(self.loss_r_min)
        return self.codings_val_s

    def learningCurve(self):
        try:
            plt.figure(figsize=(10,8))
            plt.plot(self.loss)
            plt.title('learning curves')
            plt.xlabel('number of iterations')
            plt.ylabel('reconstruction loss')
            plt.ylim(self.loss_min-10, self.loss_max+10)
            plt.grid()
        except Exception as e:
            pass

        try:
            plt.figure(figsize=(10,8))
            plt.plot(self.loss_recon)
            plt.title('learning curves')
            plt.xlabel('number of iterations')
            plt.ylabel('reconstruction loss')
            plt.ylim(self.loss_r_min-10, self.loss_r_max+10)
            plt.grid()
        except Exception as e:
            pass
        return None

    def visualize(self, bokeh=False, linear=True):
        if linear:
            val = self.codings_val_l
        else:
            val = self.codings_val_s

        if bokeh:
            from bokeh.plotting import figure
            from bokeh.io import show, output_notebook
            from bokeh.resources import INLINE
            output_notebook(INLINE)

            df = pd.DataFrame(val, columns=['neuron_1', 'neuron_2'])
            df['target'] = self.target
            df['color'] = df.target.apply(lambda x: '#1f77b4' if x==0 else '#2ca01c')

            plot = figure(plot_width=800, plot_height=800)
            plot.cirvle(df['neuron_1'], df['neuron_2'], size=12, color=df['color'], alpha=0.4)

            output_notebook()
            show(plot)
        else:
            x_min, x_max = val[:, 0].min() - 100, val[:, 0].max() + 100
            y_min, y_max = val[:, 1].min() - 100, val[:, 1].max() + 100

            plt.figure(figsize=(12, 10))
            plt.scatter(val[:,0], val[:,1], c=self.y_test, cmap=plt.cm.Set1, edgecolors='k', alpha=0.4, s=60)
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.show()
        return None






