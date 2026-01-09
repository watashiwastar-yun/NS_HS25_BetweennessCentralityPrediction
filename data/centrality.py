import networkx as nx
import pandas as pd
import os
import sys

file ='avg_travel_graph2.gml'
weight='weight_AKA_average_time'

G=nx.read_gml(file)

#closeness C
print("calculating closeness C")
close=nx.closeness_centrality(G, distance= weight)
print("done")
#betweeness
print("calculate betweeness c")
between=nx.betweenness_centrality(G, weight= weight, normalized=True)
print("done")

#put in the gml file

nx.set_node_attributes(G, close, 'closenss')
nx.set_node_attributes(G, between, 'betweenness')
nx.write_gml(G,"graph_with_centrality.gml")