import pandas as pd
import networkx as nx

path='data/berlin/network_temporal.csv' 
start='from_stop_I'
target='to_stop_I'
departureTimeC='dep_time_ut'
arrivalTimeC='arr_time_ut'

dataframe=pd.read_csv(path,sep=",", skipinitialspace=True, engine='python', encoding='latin1')

if departureTimeC not in dataframe.columns or arrivalTimeC not in dataframe.columns:#check validity before processing
    print("col error")
    exit()

#calculate average,combined multiple rides between two edges into 1 single ride(Eg: there are eight rides from Zurich HB to oerlikon per day, this steps reduce the number of ride to 1, calculating the average travel distance)
dataframe['duration']=dataframe[arrivalTimeC]-dataframe[departureTimeC]
routes=dataframe.groupby([start,target])
avg=routes['duration'].mean()
result=avg.reset_index()
dfCombined= result.rename(columns={'duration':'avg_travel_time'})



print("there are "+str(len(dataframe))+" trips， now aggregate to" +str(len(dfCombined))+" static routes")


G=nx.MultiDiGraph()
#graphing
for row in dfCombined.itertuples():
    u=row.from_stop_I
    v=row.to_stop_I
    w=row.avg_travel_time
    G.add_edge(u, v, weight_AKA_average_time=w)


import pickle
import numpy as np

print(f" Constructed a static DiGraph with {G.number_of_nodes()} stops and {G.number_of_edges()} average routes.")

# Configuration for model compatibility
MODEL_SIZE = 10000  # Matches main.py
print("Calculating Betweenness Centrality (Ground Truth)... this may take a while.")
# Note: 'weight_AKA_average_time' is cost/distance, which is what nx.betweenness_centrality expects.
bc_dict = nx.betweenness_centrality(G, weight='weight_AKA_average_time')

print("Formatting data for model...")
# 1. Node Sequence
node_sequence = list(G.nodes())
num_nodes = len(node_sequence)

# 2. BC Matrix (padded to model_size)
bc_mat = np.zeros((MODEL_SIZE, 1), dtype=np.float32)
for idx, node in enumerate(node_sequence):
    bc_mat[idx, 0] = bc_dict[node]

# 3. Lists (Dataset usually contains multiple graphs, here we have just one)
list_graph = [G]
list_n_seq = [node_sequence]
list_num_node = [num_nodes]

output_pickle = 'data/berlin/processed_data.pickle'

print(f"Saving to {output_pickle}...")
with open(output_pickle, 'wb') as f:
    # Format matches main.py loading: list_graph_train, list_n_seq_train, list_num_node_train, bc_mat_train
    pickle.dump((list_graph, list_n_seq, list_num_node, bc_mat), f)

print("Done.")


