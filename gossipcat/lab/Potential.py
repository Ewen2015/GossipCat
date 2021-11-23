#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
from sklearn.cluster import KMeans

def Potential(data, id_col, potential_target, features, n_clusters=100, shreshold=0, seed=0):
    """ Potential items detector with KMeans.

    Args:
      data:             a pandas.DataFrame for detection.
      id:               id of items.
      potential_target: potential target to detect.
      features:         features used for dectation.
      n_clusters:       the number of clusters used in KMeans.
      shreshold:        the shreshold to detect items as potential items.
      seed:             random state.

    Returns:
      potential_list:   a list of potential items.
    """

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(data[features])

    data['kmeans_label'] = kmeans.labels_

    pt_ptg = (data.groupby(by='kmeans_label')[potential_target]
                  .agg('mean')
                  .rename('potentialTarget_ptg')
                  .reset_index()
                  .sort_values(by='potentialTarget_ptg', ascending=False))
                                                                                
    data = data.merge(pt_ptg, how='left', on='kmeans_label')

    filters = (data[potential_target]==0) & (data['potentialTarget_ptg']>shreshold) & (data['potentialTarget_ptg']<1)
    potential_list = data[filters][[id_col]]

    return potential_list 