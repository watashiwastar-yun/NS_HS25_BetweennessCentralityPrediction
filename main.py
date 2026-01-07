 
import numpy as np
import pickle
import networkx as nx
import torch
from utils import *
import random
import torch.nn as nn
from model import GNN_Bet
import json
from datetime import datetime
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
    node_feat_train_full = np.zeros((10000, num_train, 0), dtype=np.float32)
    node_feat_test_full = np.zeros((10000, num_test, 0), dtype=np.float32)
    node_feat_dim = 0
    print("  No node features available. Use feature_manager.py to add features.")
    
elif len(train_data) == 5:
    print("Detected new format (with node features)")
    list_graph_train, list_n_seq_train, list_num_node_train, bc_mat_train, node_feat_train_full = train_data
    list_graph_test, list_n_seq_test, list_num_node_test, bc_mat_test, node_feat_test_full = test_data
    
    # Automatically detect number of features
    node_feat_dim = node_feat_train_full.shape[2]
    print(f"  Detected {node_feat_dim} node feature(s)")
    
else:
    raise ValueError(f"Unexpected data format: {len(train_data)} elements")

model_size = 10000
#Get adjacency matrices from graphs
print(f"Graphs to adjacency conversion.")

list_adj_train,list_adj_t_train = graph_to_adj(list_graph_train,list_n_seq_train,list_num_node_train,model_size)
list_adj_test,list_adj_t_test = graph_to_adj(list_graph_test,list_n_seq_test,list_num_node_test,model_size)


def train_epoch(model, optimizer, list_adj_train, list_adj_t_train, list_num_node_train, bc_mat_train, node_feat_train, device):
    """Train for one epoch"""
    model.train()
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
    
    return loss_train / num_samples_train

def evaluate(model, list_adj_test, list_adj_t_test, list_num_node_test, bc_mat_test, node_feat_test, device):
    """Evaluate model and return Kendall's Tau scores"""
    model.eval()
    list_kt = list()
    num_samples_test = len(list_adj_test)
    
    with torch.no_grad():
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

    mean_kt = np.mean(np.array(list_kt))
    std_kt = np.std(np.array(list_kt))
    
    return mean_kt, std_kt, list_kt


def run_experiment(experiment_name, node_feat_train, node_feat_test, num_epochs=10):
    """Run a single experiment with given features"""
    print(f"\n{'='*60}")
    print(f"Running: {experiment_name}")
    print(f"{'='*60}")
    
    current_feat_dim = node_feat_train.shape[2]
    print(f"Feature dimension: {current_feat_dim}")
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if current_feat_dim == 0:
        model = GNN_Bet(ninput=model_size, nhid=hidden, node_feat_dim=1, dropout=0.6)
    else:
        model = GNN_Bet(ninput=model_size, nhid=hidden, node_feat_dim=current_feat_dim, dropout=0.6)
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    
    # Training loop
    for e in range(num_epochs):
        avg_loss = train_epoch(model, optimizer, list_adj_train, list_adj_t_train, 
                               list_num_node_train, bc_mat_train, node_feat_train, device)
        if (e + 1) % 5 == 0:
            print(f"  Epoch {e+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")
    
    # Evaluation
    mean_kt, std_kt, all_kt = evaluate(model, list_adj_test, list_adj_t_test, 
                                       list_num_node_test, bc_mat_test, node_feat_test, device)
    
    print(f"Results: Kendall's Tau = {mean_kt:.4f} ± {std_kt:.4f}")
    
    return {
        'experiment_name': experiment_name,
        'feature_dim': current_feat_dim,
        'mean_kendall_tau': float(mean_kt),
        'std_kendall_tau': float(std_kt),
        'all_kendall_tau': [float(x) for x in all_kt],
        'num_epochs': num_epochs
    }


#Model parameters
hidden = 20
n_layers = 6
num_epoch = 10

# Storage for all results
all_results = []

print("\n" + "="*60)
print("ABLATION STUDY")
print("="*60)

# Experiment 1: All features (baseline)
print("\n[1/{}] Baseline: All Features".format(node_feat_dim + 1))
result = run_experiment("All Features", node_feat_train_full, node_feat_test_full, num_epoch)
all_results.append(result)

# Ablation experiments: Remove one feature at a time
if node_feat_dim > 0:
    for feature_idx in range(node_feat_dim):
        print(f"\n[{feature_idx + 2}/{node_feat_dim + 1}] Ablation: Removing Feature {feature_idx}")
        
        # Create feature matrices without feature_idx
        # Keep all features except the one at feature_idx
        indices = [i for i in range(node_feat_dim) if i != feature_idx]
        
        if len(indices) > 0:
            node_feat_train_ablated = node_feat_train_full[:, :, indices]
            node_feat_test_ablated = node_feat_test_full[:, :, indices]
        else:
            # If removing the only feature, use empty features
            num_train = node_feat_train_full.shape[1]
            num_test = node_feat_test_full.shape[1]
            node_feat_train_ablated = np.zeros((10000, num_train, 0), dtype=np.float32)
            node_feat_test_ablated = np.zeros((10000, num_test, 0), dtype=np.float32)
        
        result = run_experiment(
            f"Without Feature {feature_idx}", 
            node_feat_train_ablated, 
            node_feat_test_ablated, 
            num_epoch
        )
        all_results.append(result)

# Save results to JSON
output_file = f"ablation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print("\n" + "="*60)
print("ABLATION STUDY SUMMARY")
print("="*60)

for result in all_results:
    print(f"{result['experiment_name']:30s} | KT: {result['mean_kendall_tau']:.4f} ± {result['std_kendall_tau']:.4f}")

print(f"\nDetailed results saved to: {output_file}")

# Calculate feature importance (drop in performance when removed)
if len(all_results) > 1:
    baseline_kt = all_results[0]['mean_kendall_tau']
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE (Performance Drop)")
    print("="*60)
    
    for i, result in enumerate(all_results[1:]):
        drop = baseline_kt - result['mean_kendall_tau']
        print(f"Feature {i}: {drop:+.4f} (removing it {'hurts' if drop > 0 else 'helps'} performance)")