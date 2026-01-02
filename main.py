 
import numpy as np
import pickle
import networkx as nx
import torch
from utils import *
import random
import torch.nn as nn
from model import GNN_Bet
torch.manual_seed(20)

#Loading graph data

data_path = "./data/"

#Load training data
print(f"Loading data...")
with open(data_path+"training.pickle","rb") as fopen:
    train_data = pickle.load(fopen)

with open(data_path+"test.pickle","rb") as fopen:
    test_data = pickle.load(fopen)

# Handle both old (4-element) and new (5-element with features) formats
if len(train_data) == 4:
    print("Detected old format (no node features)")
    list_graph_train, list_n_seq_train, list_num_node_train, bc_mat_train = train_data
    list_graph_test, list_n_seq_test, list_num_node_test, bc_mat_test = test_data
    
    # Create empty feature matrices
    num_train = len(list_graph_train)
    num_test = len(list_graph_test)
    node_feat_train = np.zeros((10000, num_train, 0), dtype=np.float32)
    node_feat_test = np.zeros((10000, num_test, 0), dtype=np.float32)
    node_feat_dim = 0
    print("  No node features available. Use feature_manager.py to add features.")
    
elif len(train_data) == 5:
    print("Detected new format (with node features)")
    list_graph_train, list_n_seq_train, list_num_node_train, bc_mat_train, node_feat_train = train_data
    list_graph_test, list_n_seq_test, list_num_node_test, bc_mat_test, node_feat_test = test_data
    
    # Automatically detect number of features
    node_feat_dim = node_feat_train.shape[2]
    print(f"  Detected {node_feat_dim} node feature(s)")
    
else:
    raise ValueError(f"Unexpected data format: {len(train_data)} elements")

model_size = 10000
#Get adjacency matrices from graphs
print(f"Graphs to adjacency conversion.")

list_adj_train,list_adj_t_train = graph_to_adj(list_graph_train,list_n_seq_train,list_num_node_train,model_size)
list_adj_test,list_adj_t_test = graph_to_adj(list_graph_test,list_n_seq_test,list_num_node_test,model_size)



def train(list_adj_train,list_adj_t_train,list_num_node_train,bc_mat_train,node_feat_train):
    model.train()
    total_count_train = list()
    loss_train = 0
    num_samples_train = len(list_adj_train)
    for i in range(num_samples_train):
        adj = list_adj_train[i]
        num_nodes = list_num_node_train[i]
        adj_t = list_adj_t_train[i]
        adj = adj.to(device)
        adj_t = adj_t.to(device)
        
        # Extract node features for this graph
        node_feat = torch.from_numpy(node_feat_train[:, i, :]).float().to(device)

        optimizer.zero_grad()
            
        y_out = model(adj,adj_t,node_feat)
        true_arr = torch.from_numpy(bc_mat_train[:,i]).float()
        true_val = true_arr.to(device)
        
        loss_rank = loss_cal(y_out,true_val,num_nodes,device,model_size)
        loss_train = loss_train + float(loss_rank)
        loss_rank.backward()
        optimizer.step()

def test(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test,node_feat_test):
    model.eval()
    loss_val = 0
    list_kt = list()
    num_samples_test = len(list_adj_test)
    for j in range(num_samples_test):
        adj = list_adj_test[j]
        adj_t = list_adj_t_test[j]
        adj=adj.to(device)
        adj_t = adj_t.to(device)
        num_nodes = list_num_node_test[j]
        
        # Extract node features for this graph
        node_feat = torch.from_numpy(node_feat_test[:, j, :]).float().to(device)
        
        y_out = model(adj,adj_t,node_feat)
    
        
        true_arr = torch.from_numpy(bc_mat_test[:,j]).float()
        true_val = true_arr.to(device)
    
        kt = ranking_correlation(y_out,true_val,num_nodes,model_size)
        list_kt.append(kt)

    print(f"Average kendall tau score is: {np.mean(np.array(list_kt))} and std: {np.std(np.array(list_kt))}")



#Model parameters
hidden = 20
n_layers = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model with detected feature dimension
# If node_feat_dim = 0, model will use only structure (old behavior)
# If node_feat_dim > 0, model will combine structure + features
if node_feat_dim == 0:
    print(f"Initializing model WITHOUT node features (structure only)")
    # For backward compatibility, we still pass node_feat_dim but with empty features
    model = GNN_Bet(ninput=model_size, nhid=hidden, node_feat_dim=1, dropout=0.6)
else:
    print(f"Initializing model WITH {node_feat_dim} node feature(s)")
    model = GNN_Bet(ninput=model_size, nhid=hidden, node_feat_dim=node_feat_dim, dropout=0.6)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)
num_epoch = 10

print(f"Number of epoches: {num_epoch}")
for e in range(num_epoch):
    print(f"Epoch number: {e}")
    train(list_adj_train,list_adj_t_train,list_num_node_train,bc_mat_train,node_feat_train)
#test on 10 test graphs and print average KT Score and its stanard deviation
with torch.no_grad():
    test(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test,node_feat_test)


    