#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author: 	Ewen Wang
email: 		wang.enqun@outlook.com
license: 	Apache License 2.0
"""
import pandas as pd
import networkx as nx
import itertools 

def link_pred_generator(function):
	def link_pred(graph, source, target):
		for u, v, p in function(graph, [(source, target)]):
			return p
	return link_pred

def edge(graph):
	edge_attr = []
	edge_attr = pd.DataFrame(list(itertools.combinations(list(graph.nodes.keys()), r=2)), columns=['source', 'target'])
	edge_attr['shortest_path_length'] = edge_attr.apply(lambda x: nx.shortest_path_length(graph, x[0], x[1]) if nx.has_path(graph, x[0], x[1]) else 0, axis=1)
	edge_attr['efficiency'] = edge_attr.apply(lambda x: nx.efficiency(graph, x[0], x[1]), axis=1)
	edge_attr['jaccard_coefficient'] = edge_attr.apply(lambda x: link_pred_generator(nx.jaccard_coefficient)(graph, x[0], x[1]), axis=1)
	edge_attr['resource_allocation_index'] = edge_attr.apply(lambda x: link_pred_generator(nx.resource_allocation_index)(graph, x[0], x[1]), axis=1)
	edge_attr['adamic_adar_index'] = edge_attr.apply(lambda x: link_pred_generator(nx.adamic_adar_index)(graph, x[0], x[1]), axis=1)
	edge_attr['preferential_attachment'] = edge_attr.apply(lambda x: link_pred_generator(nx.preferential_attachment)(graph, x[0], x[1]), axis=1)
	return edge_attr

class Attribute(object):
	"""Generate bunch of attributes of a single connected graph."""
	def __init__(self, graph):
		"""Initialize the class and generate graph attributes"""
		self.graph = graph
		self.node_attr = pd.DataFrame()
		self.edge_attr = pd.DataFrame()
		self.pair_attr = pd.DataFrame()

		self.number_of_nodes = nx.number_of_nodes(graph)
		self.number_of_edges = nx.number_of_edges(graph)
		self.number_of_selfloops = nx.number_of_selfloops(graph)
		self.graph_number_of_cliques = nx.graph_number_of_cliques(graph)
		self.graph_clique_number = nx.graph_clique_number(graph)
		self.chordal_graph_treewidth = nx.chordal_graph_treewidth(graph)
		self.density = nx.density(graph)
		self.transitivity = nx.transitivity(graph)
		self.average_clustering = nx.average_clustering(graph)
		self.radius = nx.radius(graph)
		self.is_tree = 1 if nx.is_tree(graph) else 0
		self.wiener_index = nx.wiener_index(graph)

	def node(self):
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

	def edge(self):
		"""Generate edge-based attributes."""
		self.edge_attr = edge(self.graph)
		return self.edge_attr

	def pair_attributes(self):
		"""Combine all node-based, edge-based, and graph-based attributes."""
		self.node_attr = self.node()
		self.edge_attr = self.edge()
		self.pair_attr = self.edge_attr.merge(self.node_attr, how='left', left_on='source', right_on='node').merge(self.node_attr, how='left', left_on='target', right_on='node')	
		self.pair_attr = self.pair_attr.drop(['node_x', 'node_y'], axis=1)
		
		self.pair_attr['number_of_nodes'] = self.number_of_nodes
		self.pair_attr['number_of_edges'] = self.number_of_edges
		self.pair_attr['number_of_selfloops'] = self.number_of_selfloops
		self.pair_attr['graph_number_of_cliques'] = self.graph_number_of_cliques
		self.pair_attr['graph_clique_number'] = self.graph_clique_number
		self.pair_attr['chordal_graph_treewidth'] = self.chordal_graph_treewidth
		self.pair_attr['density'] = self.density
		self.pair_attr['transitivity'] = self.transitivity
		self.pair_attr['average_clustering'] = self.average_clustering
		self.pair_attr['radius'] = self.radius
		self.pair_attr['is_tree'] = self.is_tree
		self.pair_attr['wiener_index'] = self.wiener_index	

		return self.pair_attr

def NodePairs(data, index, source, target, node_min=3, node_max=100, num_graph=100, verbose=True):
	"""Generate node pairs attributes dataframe from edge dataframe.

	Args:
		data: A dataframe contains edge infomation of multiple graphs.
		index: The index of each single connected graph.
		source: The column name of source of an edge.
		target: The column name of target of an edge.
		node_min: The minimum number of nodes should be included in the results.
		node_max: The maximum number of nodes should be included in the results.
		num_graph: The graph number should be included in the the results.
		verbose: Whether print out the process while running.
	Returns:
		A dataframe can be used for machine learning."""
	index_list = list(data[index].unique())
	final_df = pd.DataFrame()
	graph_count = 0

	for ind, val in enumerate(index_list):
		temp_data = data[(data[index] == val)]
		G = nx.from_pandas_edgelist(df=temp_data, source=source, target=target)
		if G.number_of_nodes > node_max or G.number_of_nodes < node_min:
			continue
		GA = Attribute(G)
		temp_df = GA.pair_attributes()
		temp_df[index] = val
		if ind == 0:
			final_df = temp_df
		else:
			final_df = final_df.append(temp_df)
		graph_count += 1
		if verbose:
			print('Graph Number: ', graph_count, '\tNodes Count: ', GA.number_of_nodes, '\tEdges Count: ', GA.number_of_edges)
		if graph_count>=num_graph:
			break

	return final_df 