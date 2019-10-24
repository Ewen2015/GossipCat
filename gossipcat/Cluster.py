#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
import numpy as np
import pandas as pd 
from scipy.spatial.distance import cdist

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import datasets

import matplotlib.pyplot as plt 
import seaborn as sns 


class Cluster(object):
    """docstring for Cluster"""
    def __init__(self, data, target, features):
        super(Cluster, self).__init__()
        self.X = data[features]
        self.y = data[target]
        self.n_classes = len(self.y.unique())

        self.X_prep = Imputer(strategy='mean').fit_transform(self.X)
        self.X_prep = StandardScaler().fit_transform(self.X_prep)

    def PCAna(self):
        pca = PCA().fit(self.X_prep)
        features = range(pca.n_components_)

        plt.figure(figsize=(16, 6))
        plt.bar(features, pca.explained_variance_)
        plt.xlabel('PCA feature')
        plt.ylabel('variance')
        plt.xticks(features)
        plt.show()
        return None 

    def KMeansAna(self):
        self.pca_features = PCA(n_components=self.n_classes).fit_transform(self.X_prep)
        self.cluster = KMeans(n_clusters=self.n_classes).fit(self.X_prep).labels_
        self.comb = pd.DataFrame({
            'pca_1': self.pca_features[:,0],
            'pca_2': self.pca_features[:,1],
            'cluster': self.cluster,
            'target': self.y
            })

        sns.lmplot(x='pca_1', y='pca_2', hue='cluster', col='target', data=self.comb, fit_reg=False)
        sns.lmplot(x='pca_1', y='pca_2', hue='target', col='cluster', data=self.comb, fit_reg=False)
        return None 

    def Iris(self):
        iris = datasets.load_iris()
        target = 'target'
        features = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']

        data = pd.DataFrame(iris.data, columns=features)
        data[target] = iris.target

        cluster = Cluster(data, target, features)
        cluster.PCAna()
        cluster.KMeansAna()
        return None 

def elbow(Xdata, clusters_range=range(1, 10)):
    clusters_range = clusters_range
    meanddist = []

    for k in clusters_range:
        model = KMeans(n_clusters=k)
        model.fit(Xdata)
        meanddist.append(sum(np.min(cdist(Xdata, model.cluster_centers_, 'euclidean'), axis=1))/Xdata.shape[0])

    plt.figure(figsize=(8, 7))
    plt.plot(clusters_range, meanddist)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average distance')
    plt.title('Selecting k with the Elbow Method')
    plt.show()
    return None 





















