import pandas as pd
import networkx as nx
import pickle
import numpy as np
import os
import random
import glob

# Configuration
DATA_DIR = 'data/'
MODEL_SIZE = 10000
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

def process_single_csv(file_path):
    print(f"Processing {file_path}...")
    try:
        dataframe = pd.read_csv(file_path, sep=",", skipinitialspace=True, engine='python', encoding='latin1')
        
        start = 'from_stop_I'
        target = 'to_stop_I'
        departureTimeC = 'dep_time_ut'
        arrivalTimeC = 'arr_time_ut'
        
        if departureTimeC not in dataframe.columns or arrivalTimeC not in dataframe.columns:
            print(f"Skipping {file_path}: Missing columns")
            return None

        # Calculate average travel time
        dataframe['duration'] = dataframe[arrivalTimeC] - dataframe[departureTimeC]
        routes = dataframe.groupby([start, target])
        avg = routes['duration'].mean()
        result = avg.reset_index()
        dfCombined = result.rename(columns={'duration': 'avg_travel_time'})

        G = nx.MultiDiGraph()
        for row in dfCombined.itertuples():
            u = row.from_stop_I
            v = row.to_stop_I
            w = row.avg_travel_time
            G.add_edge(u, v, weight_AKA_average_time=w)
            
        return G
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def generate_dataset_lists(graph_list):
    list_graph = []
    list_n_seq = []
    list_num_node = []
    
    # Pre-allocate BC Matrix: (MODEL_SIZE, num_graphs)
    num_graphs = len(graph_list)
    bc_mat = np.zeros((MODEL_SIZE, num_graphs), dtype=np.float32)

    for i, G in enumerate(graph_list):
        print(f"  Calculating BC for graph {i+1}/{num_graphs} (Nodes: {G.number_of_nodes()})...")
        
        # Calculate Ground Truth BC
        bc_dict = nx.betweenness_centrality(G, weight='weight_AKA_average_time')
        
        node_sequence = list(G.nodes())
        num_nodes = len(node_sequence)
        
        # Fill BC Matrix column
        for idx, node in enumerate(node_sequence):
            if idx < MODEL_SIZE:
                 bc_mat[idx, i] = bc_dict[node]
        
        list_graph.append(G)
        list_n_seq.append(node_sequence)
        list_num_node.append(num_nodes)
        
    return list_graph, list_n_seq, list_num_node, bc_mat

# 1. Find all temporal CSV files
# Recursive search for *temporal.csv in all subdirectories of 'data/'
csv_files = glob.glob(os.path.join(DATA_DIR, '**/*_temporal.csv'), recursive=True)

print(f"Found {len(csv_files)} files.")

if not csv_files:
    print("No files found! Exiting.")
    exit()

# 2. Shuffle and split
random.shuffle(csv_files)
split_idx = int(len(csv_files) * TRAIN_RATIO)
train_files = csv_files[:split_idx]
test_files = csv_files[split_idx:]

print(f"Training Data: {len(train_files)} files")
print(f"Testing Data: {len(test_files)} files")

# 3. Process Graphs
train_graphs = [g for g in (process_single_csv(f) for f in train_files) if g is not None]
test_graphs = [g for g in (process_single_csv(f) for f in test_files) if g is not None]

# 4. Generate Lists and Matrices
print("\nGenerating Training Set...")
train_data = generate_dataset_lists(train_graphs)

print("\nGenerating Test Set...")
test_data = generate_dataset_lists(test_graphs)

# 5. Save to Pickle
with open(os.path.join(DATA_DIR, 'training.pickle'), 'wb') as f:
    pickle.dump(train_data, f)
print(f"Saved training.pickle with {len(train_graphs)} graphs.")

with open(os.path.join(DATA_DIR, 'test.pickle'), 'wb') as f:
    pickle.dump(test_data, f)
print(f"Saved test.pickle with {len(test_graphs)} graphs.")

print("All done.")


