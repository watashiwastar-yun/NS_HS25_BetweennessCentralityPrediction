"""
Feature Manager for GNN Node Features

This script manages node features for the GNN model. It can:
- Add new features (closeness, degree, etc.) to existing pickle files
- Remove features by name
- List current features

Usage:
    # Add closeness centrality feature
    python feature_manager.py --add closeness --input data/training.pickle --output data/training.pickle
    
    # Add degree feature
    python feature_manager.py --add degree --input data/training.pickle --output data/training.pickle
    
    # Remove a feature
    python feature_manager.py --remove closeness --input data/training.pickle --output data/training.pickle
    
    # List current features
    python feature_manager.py --list --input data/training.pickle
"""

import argparse
import pickle
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict

MODEL_SIZE = 10000

# Feature computation functions
def compute_closeness_centrality(G: nx.Graph) -> Dict:
    """Compute weighted closeness centrality."""
    return nx.closeness_centrality(G, distance='weight_AKA_average_time')

def compute_degree_centrality(G: nx.Graph) -> Dict:
    """Compute degree centrality (normalized)."""
    return nx.degree_centrality(G)

def compute_in_degree_centrality(G: nx.Graph) -> Dict:
    """Compute in-degree centrality."""
    return nx.in_degree_centrality(G)

def compute_out_degree_centrality(G: nx.Graph) -> Dict:
    """Compute out-degree centrality."""
    return nx.out_degree_centrality(G)

def compute_pagerank(G: nx.Graph) -> Dict:
    """Compute PageRank."""
    return nx.pagerank(G, weight='weight_AKA_average_time')

# Feature registry
FEATURE_REGISTRY = {
    'closeness': compute_closeness_centrality,
    'degree': compute_degree_centrality,
    'in_degree': compute_in_degree_centrality,
    'out_degree': compute_out_degree_centrality,
    'pagerank': compute_pagerank,
}

def load_pickle_data(filepath: str) -> Tuple:
    """Load data from pickle file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle_data(filepath: str, data: Tuple):
    """Save data to pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def get_feature_names(node_features_mat: np.ndarray) -> List[str]:
    """
    Get feature names from metadata or return default names.
    For now, returns generic names based on feature dimension.
    """
    num_features = node_features_mat.shape[2] if len(node_features_mat.shape) == 3 else 0
    return [f"feature_{i}" for i in range(num_features)]

def add_feature(data: Tuple, feature_name: str) -> Tuple:
    """
    Add a new feature to the dataset.
    
    Args:
        data: (list_graph, list_n_seq, list_num_node, bc_mat, node_features_mat)
        feature_name: Name of feature to add
        
    Returns:
        Updated data tuple with new feature
    """
    if feature_name not in FEATURE_REGISTRY:
        raise ValueError(f"Unknown feature: {feature_name}. Available: {list(FEATURE_REGISTRY.keys())}")
    
    list_graph, list_n_seq, list_num_node, bc_mat, node_features_mat = data
    
    compute_fn = FEATURE_REGISTRY[feature_name]
    num_graphs = len(list_graph)
    
    # Create new feature column
    new_feature = np.zeros((MODEL_SIZE, num_graphs, 1), dtype=np.float32)
    
    print(f"Computing {feature_name} for {num_graphs} graphs...")
    for i, G in enumerate(list_graph):
        print(f"  Graph {i+1}/{num_graphs}...", end='\r')
        
        feat_dict = compute_fn(G)
        node_sequence = list_n_seq[i]
        
        for idx, node in enumerate(node_sequence):
            if idx < MODEL_SIZE:
                new_feature[idx, i, 0] = feat_dict[node]
    
    print(f"\nAdded {feature_name} feature.")
    
    # Concatenate with existing features
    if node_features_mat.shape[2] == 0:
        # No existing features
        updated_features = new_feature
    else:
        updated_features = np.concatenate([node_features_mat, new_feature], axis=2)
    
    return (list_graph, list_n_seq, list_num_node, bc_mat, updated_features)

def remove_feature(data: Tuple, feature_index: int) -> Tuple:
    """
    Remove a feature by index.
    
    Args:
        data: (list_graph, list_n_seq, list_num_node, bc_mat, node_features_mat)
        feature_index: Index of feature to remove (0-based)
        
    Returns:
        Updated data tuple without the specified feature
    """
    list_graph, list_n_seq, list_num_node, bc_mat, node_features_mat = data
    
    num_features = node_features_mat.shape[2]
    
    if feature_index < 0 or feature_index >= num_features:
        raise ValueError(f"Invalid feature index: {feature_index}. Valid range: 0-{num_features-1}")
    
    # Remove feature by slicing
    indices = [i for i in range(num_features) if i != feature_index]
    updated_features = node_features_mat[:, :, indices]
    
    print(f"Removed feature at index {feature_index}.")
    
    return (list_graph, list_n_seq, list_num_node, bc_mat, updated_features)

def list_features(data: Tuple):
    """List all current features in the dataset."""
    list_graph, list_n_seq, list_num_node, bc_mat, node_features_mat = data
    
    num_features = node_features_mat.shape[2]
    print(f"\nDataset contains {num_features} feature(s):")
    print(f"  Shape: {node_features_mat.shape}")
    print(f"  Number of graphs: {len(list_graph)}")
    
    if num_features > 0:
        print("\nFeature statistics (first graph):")
        for i in range(num_features):
            feat_vals = node_features_mat[:list_num_node[0], 0, i]
            print(f"  Feature {i}: min={feat_vals.min():.4f}, max={feat_vals.max():.4f}, mean={feat_vals.mean():.4f}")

def main():
    parser = argparse.ArgumentParser(description='Manage node features for GNN model')
    parser.add_argument('--input', type=str, required=True, help='Input pickle file')
    parser.add_argument('--output', type=str, help='Output pickle file (default: same as input)')
    parser.add_argument('--add', type=str, help=f'Add feature: {list(FEATURE_REGISTRY.keys())}')
    parser.add_argument('--remove', type=int, help='Remove feature by index (0-based)')
    parser.add_argument('--list', action='store_true', help='List current features')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    data = load_pickle_data(args.input)
    
    # Ensure data has 5 elements (add empty features if needed)
    if len(data) == 4:
        print("Converting old format (4 elements) to new format (5 elements)...")
        list_graph, list_n_seq, list_num_node, bc_mat = data
        num_graphs = len(list_graph)
        node_features_mat = np.zeros((MODEL_SIZE, num_graphs, 0), dtype=np.float32)
        data = (list_graph, list_n_seq, list_num_node, bc_mat, node_features_mat)
    
    # Execute command
    if args.list:
        list_features(data)
    
    elif args.add:
        data = add_feature(data, args.add)
        output_path = args.output or args.input
        save_pickle_data(output_path, data)
        print(f"Saved to {output_path}")
        list_features(data)
    
    elif args.remove is not None:
        data = remove_feature(data, args.remove)
        output_path = args.output or args.input
        save_pickle_data(output_path, data)
        print(f"Saved to {output_path}")
        list_features(data)
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
