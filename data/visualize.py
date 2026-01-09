import pandas as pd
import networkx as nx
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

if len(sys.argv)>1:
    citybase=sys.argv[1]
else:
    citybase= 'C:/Users/16409/Downloads/city/canberra'
nodes_Path=os.path.join(citybase,'network_nodes.csv')
graph='graph_with_centrality.gml'

nodeCol='stop_I'
latCol='lat'
lonCol='lon'

G=nx.read_gml(graph)
nodes=pd.read_csv(nodes_Path,sep=";")
nodes[nodeCol]=nodes[nodeCol].astype(str)

pos={}
#get stop geo location
for index,row in nodes.iterrows():
    stopID=row['stop_I']
    if stopID in G.nodes:
        pos[stopID]=(row['lon'],row['lat'])




subgraph_pos={}  #dictionary for pos, if node exist in kown coord, retrieve its coord, add nodeID and its geo location to new dictionary
for nodeID in G.nodes:
    if nodeID in pos:
        coordinates=pos[nodeID]
        subgraph_pos[nodeID]=coordinates


#loop through every single node, store their coord in tuple, extract/add long/lat
xCoords=[]
yCoords=[]
for nodeID in G.nodes:
    coordinatesT=subgraph_pos[nodeID]
    lon,lat=coordinatesT[0],coordinatesT[1]
    xCoords.append(coordinatesT[0])
    yCoords.append(coordinatesT[1])

#centrality data, only betweeness is logNormed
betweennessRaw=np.array(list(nx.get_node_attributes(G,'betweenness').values()))
mapBetween=plt.cm.get_cmap('Reds')
#normalize, need to determine lowest value should map to make sure all are visible, even if near 0
minimum=np.max([betweennessRaw[betweennessRaw>0].min()*0.1,1e-6])
normBetweenness=mcolors.LogNorm(vmin=minimum,vmax=betweennessRaw.max())


#closenessRaw=np.array(list(nx.get_node_attributes(G,'closeness').values()))
#mapCloseness=plt.cm.get_cmap('YlGnBu')
#normCloseness=mcolors.Normalize(vmin=closenessRaw.min(),vmax=closenessRaw.max())#only linear norm

#draw between
plt.figure(figsize=(30,30), facecolor='black')
plt.title("Betweenness centrality with LogNorm",fontsize=20,color='white')
#draw edge/nodes
nx.draw_networkx_edges(G,subgraph_pos,edge_color='lightgray',width=0.5,alpha=0.15,arrows=False)
ScatterPoint_between=plt.scatter(xCoords,yCoords,s=70,c=betweennessRaw,cmap=mapBetween,norm=normBetweenness,alpha=0.8)
#color bar
cbar_B=plt.colorbar(ScatterPoint_between, orientation='vertical', label='Betweenness centrality,logNormed')
cbar_B.ax.yaxis.label.set_color('white')
cbar_B.ax.tick_params(colors='white')
plt.axis('off')
plt.savefig(' betweeness centrality normalized', facecolor='black')
print("between done")


#closeness, no logNorm needed
closenessRaw=np.array(list(nx.get_node_attributes(G,'closenss').values()))
mapCloseness=plt.cm.get_cmap('YlGnBu')
normCloseness=mcolors.Normalize(vmin=closenessRaw.min(),vmax=closenessRaw.max())#only linear norm

plt.figure(figsize=(30,30), facecolor='black')
plt.title("Closeness Centrality",fontsize=20,color='white')
#draw edge/nodes
nx.draw_networkx_edges(G,subgraph_pos,edge_color='lightgray',width=0.5,alpha=0.15,arrows=False)
ScatterPoint_between=plt.scatter(xCoords,yCoords,s=70,c=closenessRaw,cmap=mapCloseness,norm=normCloseness,alpha=0.8)
#color bar
cbar_B=plt.colorbar(ScatterPoint_between, orientation='vertical', label='Closeness centrality')
cbar_B.ax.yaxis.label.set_color('white')
cbar_B.ax.tick_params(colors='white')
plt.axis('off')
plt.savefig(' closeness centrality ', facecolor='black')
print("close done")





