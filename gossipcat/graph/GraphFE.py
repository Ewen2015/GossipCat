#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
import pandas as pd
import numpy as np
import networkx as nx
import itertools 

def link_pred_generator(function):
    def link_pred(graph, source, target):
        for u, v, p in function(graph, [(source, target)]):
            return p
    return link_pred

def hitting_time(nodelist, adj, source, target):
    hit_ind = (nodelist.index(source), nodelist.index(target))
    A = adj.copy()
    A[hit_ind[1],:] = 0
    A[hit_ind[1], hit_ind[1]] = 1
    A = (A.T/A.sum(axis=1)).T
    B = A.copy()
    prob = 0
    n = 0
    while prob < 0.99 and n < 100:
        prob = B[hit_ind]
        B = np.dot(B, A)
        n += 1
    return n
    
def edge(graph):
    edge_attr = []
    nodelist = list(graph)
    adj = nx.adj_matrix(graph)
    edge_attr = pd.DataFrame(list(itertools.combinations(list(graph.nodes.keys()), r=2)), columns=['source', 'target'])
    edge_attr['hitting_time'] = edge_attr.apply(lambda x: hitting_time(nodelist, adj, x[0], x[1]), axis=1)
    edge_attr['shortest_path_length'] = edge_attr.apply(lambda x: nx.shortest_path_length(graph, x[0], x[1]) if nx.has_path(graph, x[0], x[1]) else 0, axis=1)
    edge_attr['efficiency'] = edge_attr.apply(lambda x: nx.efficiency(graph, x[0], x[1]), axis=1)
    edge_attr['jaccard_coefficient'] = edge_attr.apply(lambda x: link_pred_generator(nx.jaccard_coefficient)(graph, x[0], x[1]), axis=1)
    edge_attr['resource_allocation_index'] = edge_attr.apply(lambda x: link_pred_generator(nx.resource_allocation_index)(graph, x[0], x[1]), axis=1)
    edge_attr['adamic_adar_index'] = edge_attr.apply(lambda x: link_pred_generator(nx.adamic_adar_index)(graph, x[0], x[1]), axis=1)
    edge_attr['preferential_attachment'] = edge_attr.apply(lambda x: link_pred_generator(nx.preferential_attachment)(graph, x[0], x[1]), axis=1)
    return edge_attr


class Attribute(object):
    """Generate all node-based, edge-based, and graph-based attributes of all connected components in a whole graph.
    """
    def __init__(self, graph):
        """Initialize the class and generate graph attributes"""
        self.graphs = list(graph.subgraph(c) for c in nx.connected_components(graph))
        self.graph = self.graphs[0]
        if len(self.graphs)>1:
            self.all_attr = pd.DataFrame()
            print("Note: "+str(len(self.graphs))+" connected components are contained.")

        self.graph_attr = pd.DataFrame()
        self.node_attr = pd.DataFrame()
        self.edge_attr = pd.DataFrame()
        self.pair_attr = pd.DataFrame()

    def _graph(self):
        """Generate graph-based attributes."""
        self.graph_attr['number_of_nodes'] = [nx.number_of_nodes(self.graph)]
        self.graph_attr['number_of_edges'] = [nx.number_of_edges(self.graph)]
        self.graph_attr['number_of_selfloops'] = [nx.number_of_selfloops(self.graph)]
        self.graph_attr['graph_number_of_cliques'] = [nx.graph_number_of_cliques(self.graph)]
        self.graph_attr['graph_clique_number'] = [nx.graph_clique_number(self.graph)]
        self.graph_attr['density'] = [nx.density(self.graph)]
        self.graph_attr['transitivity'] = [nx.transitivity(self.graph)]
        self.graph_attr['average_clustering'] = [nx.average_clustering(self.graph)]
        self.graph_attr['radius'] = [nx.radius(self.graph)]
        self.graph_attr['is_tree'] = [1 if nx.is_tree(self.graph) else 0]
        self.graph_attr['wiener_index'] = [nx.wiener_index(self.graph)]
        return self.graph_attr

    def _node(self):
        """Generate node-based attributes."""
        degree_cent = pd.DataFrame(list(nx.degree_centrality(self.graph).items()), columns=['node', 'degree_centrality'])
        closenessCent = pd.DataFrame(list(nx.closeness_centrality(self.graph).items()), columns=['node', 'closeness_centrality'])
        betweennessCent = pd.DataFrame(list(nx.betweenness_centrality(self.graph).items()), columns=['node', 'betweenness_centrality'])
        pagerank = pd.DataFrame(list(nx.pagerank(self.graph).items()), columns=['node', 'pagerank'])

        self.node_attr = degree_cent
        self.node_attr['closeness_centrality'] = closenessCent['closeness_centrality']
        self.node_attr['betweenness_centrality'] = betweennessCent['betweenness_centrality']
        self.node_attr['pagerank'] = pagerank['pagerank']
        return self.node_attr

    def _edge(self):
        """Generate edge-based attributes."""
        self.edge_attr = edge(self.graph)
        return self.edge_attr

    def sigTabular(self):
        """Combine all node-based, edge-based, and graph-based attributes of a single connected component."""
        self.node_attr = self._node(self.graph)
        self.edge_attr = self._edge(self.graph)
        self.graph_attr = self._graph(self.graph)
        self.pair_attr = self.edge_attr.merge(self.node_attr, how='left', left_on='source', right_on='node').merge(self.node_attr, how='left', left_on='target', right_on='node') 
        self.pair_attr = self.pair_attr.drop(['node_x', 'node_y'], axis=1)

        graph_attr_l = ['number_of_nodes', 'number_of_edges', 'number_of_selfloops', 'graph_number_of_cliques', 
                        'graph_clique_number', 'density', 'transitivity', 'average_clustering', 'radius', 'is_tree', 'wiener_index']
        for i in graph_attr_l:
            self.pair_attr[i] = self.graph_attr[i][0]
        return self.pair_attr

    def mulTabular(self):
        """Combine all node-based, edge-based, and graph-based attributes of all connected components in the whole graph."""
        for ind, graph in enumerate(self.graphs):
            self.graph = graph
            self.pair_attr = self.sigTabular()
            if ind==0:
                self.all_attr = self.pair_attr
            else:
                self.all_attr = pd.concat([self.all_attr, self.pair_attr])
        return self.all_attr


class GFeature(object):
    """Feature engineering to add all node-based and graph-based attributes of all connected components in a whole graph.
    """
    def __init__(self, df, source, target):
        """
        Args:
            df: dataframe with source and target nodes.
            source: source node name.
            target: target node name.

        Returns:
            A DataFrame with graph features.
        """
        self.df = df 
        self.source = source
        self.target = target

        self.g = nx.from_pandas_edgelist(df=self.df, source=self.source, target=self.target) 
        self.graphs = list(self.g.subgraph(c) for c in nx.connected_components(self.g))
        print("Note: "+str(len(self.graphs))+" connected components are contained.")      
        
        self.graph_attr = pd.DataFrame()
        self.node_attr = pd.DataFrame()
        self.df_r = pd.DataFrame()

    def _graph(self, graph):
        """Generate graph-based attributes."""
        graph_attr = pd.DataFrame()
        graph_attr['number_of_nodes'] = [nx.number_of_nodes(graph)]
        graph_attr['number_of_edges'] = [nx.number_of_edges(graph)]
        graph_attr['number_of_selfloops'] = [nx.number_of_selfloops(graph)]
        graph_attr['graph_number_of_cliques'] = [nx.graph_number_of_cliques(graph)]
        graph_attr['graph_clique_number'] = [nx.graph_clique_number(graph)]
        graph_attr['density'] = [nx.density(graph)]
        graph_attr['transitivity'] = [nx.transitivity(graph)]
        graph_attr['average_clustering'] = [nx.average_clustering(graph)]
        graph_attr['radius'] = [nx.radius(graph)]
        graph_attr['is_tree'] = [1 if nx.is_tree(graph) else 0]
        graph_attr['wiener_index'] = [nx.wiener_index(graph)]
        return graph_attr

    def _node(self, graph):
        """Generate node-based attributes."""
        node_attr = pd.DataFrame()
        degree_cent = pd.DataFrame(list(nx.degree_centrality(graph).items()), columns=['node', 'degree_centrality'])
        closenessCent = pd.DataFrame(list(nx.closeness_centrality(graph).items()), columns=['node', 'closeness_centrality'])
        betweennessCent = pd.DataFrame(list(nx.betweenness_centrality(graph).items()), columns=['node', 'betweenness_centrality'])
        pagerank = pd.DataFrame(list(nx.pagerank(graph).items()), columns=['node', 'pagerank'])

        node_attr = degree_cent
        node_attr['closeness_centrality'] = closenessCent['closeness_centrality']
        node_attr['betweenness_centrality'] = betweennessCent['betweenness_centrality']
        node_attr['pagerank'] = pagerank['pagerank']
        return node_attr 
    
    def signleGraphFeatures(self, graph, df):
        """Combine all node-based, edge-based, and graph-based attributes of a single connected component."""
        node_attr = self._node(graph)
        graph_attr = self._graph(graph)
        df = df.merge(node_attr, how='left', left_on='srcIp', right_on='node')
        df = df.drop(['node'], axis=1)
        graph_attr_l = ['number_of_nodes', 'number_of_edges', 'number_of_selfloops', 'graph_number_of_cliques', 
                        'graph_clique_number', 'density', 'transitivity', 'average_clustering', 'radius', 'is_tree', 'wiener_index']
        for i in graph_attr_l:
            node_attr[i] = graph_attr[i][0]
        df = df.merge(node_attr, how='left', left_on='destIp', right_on='node') 
        df = df.drop(['node'], axis=1)
        return df

    def graphFeaturesUpdate(self, graph, df, d_r):
        """Combine all node-based, edge-based, and graph-based attributes of a single connected component."""
        t = self.signleGraphFeatures(graph, df)
        d_r.update(t, overwrite=False)
        return d_r

    def generate(self):
        for ind, graph in enumerate(self.graphs):
            if ind == 0:
                self.d_r = self.signleGraphFeatures(graph, self.df)
            else:
                self.d_r = self.graphFeaturesUpdate(graph, self.df, self.d_r)
        return self.d_r

# def _graph(graph):
#     """Generate graph-based attributes."""
#     graph_attr = pd.DataFrame()
#     graph_attr['number_of_nodes'] = [nx.number_of_nodes(graph)]
#     graph_attr['number_of_edges'] = [nx.number_of_edges(graph)]
#     graph_attr['number_of_selfloops'] = [nx.number_of_selfloops(graph)]
#     graph_attr['graph_number_of_cliques'] = [nx.graph_number_of_cliques(graph)]
#     graph_attr['graph_clique_number'] = [nx.graph_clique_number(graph)]
#     graph_attr['density'] = [nx.density(graph)]
#     graph_attr['transitivity'] = [nx.transitivity(graph)]
#     graph_attr['average_clustering'] = [nx.average_clustering(graph)]
#     graph_attr['radius'] = [nx.radius(graph)]
#     graph_attr['is_tree'] = [1 if nx.is_tree(graph) else 0]
#     graph_attr['wiener_index'] = [nx.wiener_index(graph)]
#     return graph_attr

# def _node(graph):
#     """Generate node-based attributes."""
#     node_attr = pd.DataFrame()
#     degree_cent = pd.DataFrame(list(nx.degree_centrality(graph).items()), columns=['node', 'degree_centrality'])
#     closenessCent = pd.DataFrame(list(nx.closeness_centrality(graph).items()), columns=['node', 'closeness_centrality'])
#     betweennessCent = pd.DataFrame(list(nx.betweenness_centrality(graph).items()), columns=['node', 'betweenness_centrality'])
#     pagerank = pd.DataFrame(list(nx.pagerank(graph).items()), columns=['node', 'pagerank'])

#     node_attr = degree_cent
#     node_attr['closeness_centrality'] = closenessCent['closeness_centrality']
#     node_attr['betweenness_centrality'] = betweennessCent['betweenness_centrality']
#     node_attr['pagerank'] = pagerank['pagerank']
#     return node_attr

# def signleGraphFeatures(graph, df):
#     """Combine all node-based, edge-based, and graph-based attributes of a single connected component."""
#     node_attr = _node(graph)
#     graph_attr = _graph(graph)
#     df = df.merge(node_attr, how='left', left_on='srcIp', right_on='node')
#     df = df.drop(['node'], axis=1)
#     graph_attr_l = ['number_of_nodes', 'number_of_edges', 'number_of_selfloops', 'graph_number_of_cliques', 
#                     'graph_clique_number', 'density', 'transitivity', 'average_clustering', 'radius', 'is_tree', 'wiener_index']
#     for i in graph_attr_l:
#         node_attr[i] = graph_attr[i][0]
#     df = df.merge(node_attr, how='left', left_on='destIp', right_on='node') 
#     df = df.drop(['node'], axis=1)
#     return df

# def graphFeaturesUpdate(graph, df, d_r):
#     """Combine all node-based, edge-based, and graph-based attributes of a single connected component."""
#     t = signleGraphFeatures(graph, df)
#     d_r.update(t, overwrite=False)
#     return d_r

# def generate(df, source, target):
#     """Feature engineering to add all node-based and graph-based attributes of all connected components in a whole graph.
    
#     Args:
#         df: dataframe with source and target nodes.
#         source: source node name.
#         target: target node name.

#     Returns:
#         A DataFrame with graph features.
#     """
#     g = nx.from_pandas_edgelist(df=df, source=source, target=target) 
#     graphs = list(g.subgraph(c) for c in nx.connected_components(g))
#     print("Note: "+str(len(graphs))+" connected components are contained.")

#     for ind, graph in enumerate(graphs):
#         if ind == 0:
#             d_r = signleGraphFeatures(graph, df)
#         else:
#             d_r = graphFeaturesUpdate(graph, df, d_r)
#     return d_r