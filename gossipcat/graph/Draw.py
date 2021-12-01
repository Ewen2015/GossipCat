#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
import itertools 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import networkx as nx


def beautiful_nx(g):
    """A wrapper to draw graph nets beautifully.

    Args:
        g: a networkx graph object.

    Source:
        https://gist.github.com/jg-you/144a35013acba010054a2cc4a93b07c7.js
    """
    import copy


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

def graph_with_label(G, df_node, metric, shreshold, figsize=(8, 6)):
    """
    Arg:
        G: a graph, networkx
        df_node: a node attribute dataframe, gossipcat.GraphFE
        metric: a topology attribute for labelling node
        shreshold: a shreshold to select nodes to be labbeled
    Return:
        None
    """
    hubs = df_node[(df_node[metric] > shreshold)].node.to_list()

    labels = {}    
    for node in G.nodes():
        if node in hubs:
            #set the node name as the key and the label as its value 
            labels[node] = node

    plt.figure(figsize=figsize)
    #set the argument 'with labels' to False so you have unlabeled graph
    pos = nx.spring_layout(G)
    nx.draw(G, 
            pos=pos, 
            node_size=50,
            node_color='lightblue',
            alpha=0.8,
            width=0.5,
            edge_color='gray')
    #Now only add labels to the nodes you require (the hubs in my case)
    nx.draw_networkx_labels(G,
                            pos=pos,
                            labels=labels,
                            font_size=12,
                            font_color='r')
    return None

def graph_with_scale(G, weight='wt', node_scalar=40000, edge_scalar=0.002, seed=2021, figsize=(20, 20)):
    #set canvas size
    plt.figure(figsize=figsize)

    sizes  = [node_scalar * nx.closeness_centrality(G)[x] for x in G.nodes]
    widths = [edge_scalar * x[2] for x in G.edges.data(weight, default=1)]

    #draw the graph
    pos = nx.spring_layout(G, k=1, iterations=100, seed=seed)

    nx.draw(G, 
            pos,
            node_color='lightblue',
            with_labels=True, 
            font_size=12, 
            font_weight='bold', 
            node_size=sizes, 
            width=widths,
            edge_color='black')
    return None