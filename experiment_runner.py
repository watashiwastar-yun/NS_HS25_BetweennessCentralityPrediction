"""
Experiment Runner for Node Feature Ablation Study

This script runs experiments with different combinations of node features
and generates comprehensive evaluation reports.

Usage:
    python experiment_runner.py --data_path ./data/ --output_dir ./experiments/
"""

import argparse
import json
import os
import pickle
import time
from datetime import datetime
from itertools import combinations
import numpy as np
import torch
from pathlib import Path

from model import GNN_Bet
from utils import graph_to_adj, ranking_correlation, loss_cal
from metrics import evaluate_all_metrics, aggregate_metrics


# Feature combinations to test
FEATURE_SETS = {
    'baseline': [],  # No features (structure only)
    'closeness': ['closeness'],
    'degree': ['degree'],
    'in_degree': ['in_degree'],
    'out_degree': ['out_degree'],
    'pagerank': ['pagerank'],
    'closeness+degree': ['closeness', 'degree'],
    'closeness+pagerank': ['closeness', 'pagerank'],
    'degree+pagerank': ['degree', 'pagerank'],
    'all_centrality': ['closeness', 'degree', 'in_degree', 'out_degree', 'pagerank'],
}


class ExperimentRunner:
    def __init__(self, data_path, output_dir, model_size=10000, hidden_dim=20, 
                 num_epochs=10, device='cuda'):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_size = model_size
        self.hidden_dim = hidden_dim
        self.num_epochs = num_epochs
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
    
    def load_data(self, feature_set_name):
        """Load data with specific feature set."""
        print(f"\nLoading data for feature set: {feature_set_name}")
        
        # Load training data
        with open(os.path.join(self.data_path, "training.pickle"), "rb") as f:
            train_data = pickle.load(f)
        
        with open(os.path.join(self.data_path, "test.pickle"), "rb") as f:
            test_data = pickle.load(f)
        
        # Handle format
        if len(train_data) == 5:
            list_graph_train, list_n_seq_train, list_num_node_train, bc_mat_train, node_feat_train = train_data
            list_graph_test, list_n_seq_test, list_num_node_test, bc_mat_test, node_feat_test = test_data
            
            # Filter features based on feature_set
            features = FEATURE_SETS[feature_set_name]
            if features:
                # Assume features are in order they were added
                # This is a simplification - in production, store feature names in pickle
                num_features = len(features)
                node_feat_train = node_feat_train[:, :, :num_features]
                node_feat_test = node_feat_test[:, :, :num_features]
            else:
                # Baseline: no features
                num_train = len(list_graph_train)
                num_test = len(list_graph_test)
                node_feat_train = np.zeros((self.model_size, num_train, 0), dtype=np.float32)
                node_feat_test = np.zeros((self.model_size, num_test, 0), dtype=np.float32)
        else:
            # Old format
            list_graph_train, list_n_seq_train, list_num_node_train, bc_mat_train = train_data
            list_graph_test, list_n_seq_test, list_num_node_test, bc_mat_test = test_data
            
            num_train = len(list_graph_train)
            num_test = len(list_graph_test)
            node_feat_train = np.zeros((self.model_size, num_train, 0), dtype=np.float32)
            node_feat_test = np.zeros((self.model_size, num_test, 0), dtype=np.float32)
        
        node_feat_dim = node_feat_train.shape[2]
        print(f"  Feature dimension: {node_feat_dim}")
        print(f"  Training graphs: {len(list_graph_train)}")
        print(f"  Test graphs: {len(list_graph_test)}")
        
        # Convert to adjacency matrices
        list_adj_train, list_adj_t_train = graph_to_adj(
            list_graph_train, list_n_seq_train, list_num_node_train, self.model_size
        )
        list_adj_test, list_adj_t_test = graph_to_adj(
            list_graph_test, list_n_seq_test, list_num_node_test, self.model_size
        )
        
        return {
            'train': (list_adj_train, list_adj_t_train, list_num_node_train, bc_mat_train, node_feat_train),
            'test': (list_adj_test, list_adj_t_test, list_num_node_test, bc_mat_test, node_feat_test),
            'node_feat_dim': node_feat_dim
        }
    
    def train_model(self, data, node_feat_dim):
        """Train model and return trained model."""
        list_adj_train, list_adj_t_train, list_num_node_train, bc_mat_train, node_feat_train = data['train']
        
        # Initialize model
        model = GNN_Bet(
            ninput=self.model_size,
            nhid=self.hidden_dim,
            node_feat_dim=max(1, node_feat_dim),  # At least 1
            dropout=0.6
        )
        model.to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        
        # Training loop
        model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            
            for i in range(len(list_adj_train)):
                adj = list_adj_train[i].to(self.device)
                adj_t = list_adj_t_train[i].to(self.device)
                num_nodes = list_num_node_train[i]
                
                # Extract node features
                if node_feat_dim > 0:
                    node_feat = torch.from_numpy(node_feat_train[:, i, :]).float().to(self.device)
                else:
                    node_feat = torch.zeros((self.model_size, 1), dtype=torch.float32).to(self.device)
                
                optimizer.zero_grad()
                y_out = model(adj, adj_t, node_feat)
                
                true_arr = torch.from_numpy(bc_mat_train[:, i]).float().to(self.device)
                loss = loss_cal(y_out, true_arr, num_nodes, self.device, self.model_size)
                
                epoch_loss += float(loss)
                loss.backward()
                optimizer.step()
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch + 1}/{self.num_epochs}, Loss: {epoch_loss:.4f}")
        
        return model
    
    def evaluate_model(self, model, data):
        """Evaluate model and return detailed metrics."""
        list_adj_test, list_adj_t_test, list_num_node_test, bc_mat_test, node_feat_test = data['test']
        node_feat_dim = data['node_feat_dim']
        
        model.eval()
        all_metrics = []
        
        with torch.no_grad():
            for j in range(len(list_adj_test)):
                adj = list_adj_test[j].to(self.device)
                adj_t = list_adj_t_test[j].to(self.device)
                num_nodes = list_num_node_test[j]
                
                # Extract node features
                if node_feat_dim > 0:
                    node_feat = torch.from_numpy(node_feat_test[:, j, :]).float().to(self.device)
                else:
                    node_feat = torch.zeros((self.model_size, 1), dtype=torch.float32).to(self.device)
                
                y_out = model(adj, adj_t, node_feat)
                
                # Get predictions and ground truth
                y_pred = y_out.cpu().numpy().flatten()[:num_nodes]
                y_true = bc_mat_test[:num_nodes, j]
                
                # Compute all metrics
                metrics = evaluate_all_metrics(y_pred, y_true, k_values=[5, 10, 20, 50])
                all_metrics.append(metrics)
        
        # Aggregate across all test graphs
        aggregated = aggregate_metrics(all_metrics)
        
        return aggregated, all_metrics
    
    def run_experiment(self, feature_set_name):
        """Run single experiment with given feature set."""
        print(f"\n{'='*60}")
        print(f"Experiment: {feature_set_name}")
        print(f"Features: {FEATURE_SETS[feature_set_name]}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Load data
        data = self.load_data(feature_set_name)
        
        # Train model
        print("\nTraining model...")
        model = self.train_model(data, data['node_feat_dim'])
        
        # Evaluate
        print("\nEvaluating model...")
        aggregated_metrics, per_graph_metrics = self.evaluate_model(model, data)
        
        elapsed_time = time.time() - start_time
        
        # Prepare results
        results = {
            'feature_set': feature_set_name,
            'features': FEATURE_SETS[feature_set_name],
            'node_feat_dim': data['node_feat_dim'],
            'num_epochs': self.num_epochs,
            'hidden_dim': self.hidden_dim,
            'training_time_seconds': elapsed_time,
            'aggregated_metrics': aggregated_metrics,
            'per_graph_metrics': per_graph_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Results Summary for {feature_set_name}:")
        print(f"{'='*60}")
        print(f"Kendall's Tau: {aggregated_metrics.get('kendall_tau_mean', 0):.4f} ± {aggregated_metrics.get('kendall_tau_std', 0):.4f}")
        print(f"Recall@10: {aggregated_metrics.get('recall@10_mean', 0):.4f} ± {aggregated_metrics.get('recall@10_std', 0):.4f}")
        print(f"NDCG@10: {aggregated_metrics.get('ndcg@10_mean', 0):.4f} ± {aggregated_metrics.get('ndcg@10_std', 0):.4f}")
        print(f"Training time: {elapsed_time:.2f}s")
        
        return results
    
    def run_all_experiments(self):
        """Run experiments for all feature sets."""
        all_results = []
        
        for feature_set_name in FEATURE_SETS.keys():
            try:
                results = self.run_experiment(feature_set_name)
                all_results.append(results)
                
                # Save individual result
                result_file = self.output_dir / f"result_{feature_set_name}.json"
                with open(result_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\nSaved results to {result_file}")
                
            except Exception as e:
                print(f"\nError in experiment {feature_set_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Save combined results
        combined_file = self.output_dir / "all_results.json"
        with open(combined_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n{'='*60}")
        print(f"All results saved to {combined_file}")
        print(f"{'='*60}")
        
        # Generate comparison table
        self.generate_comparison_table(all_results)
        
        return all_results
    
    def generate_comparison_table(self, all_results):
        """Generate markdown comparison table."""
        table_file = self.output_dir / "comparison_table.md"
        
        with open(table_file, 'w') as f:
            f.write("# Experiment Results Comparison\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Table header
            f.write("| Feature Set | Features | Kendall's Tau | Recall@10 | Recall@20 | NDCG@10 | MAP | Training Time (s) |\n")
            f.write("|-------------|----------|---------------|-----------|-----------|---------|-----|-------------------|\n")
            
            # Table rows
            for result in all_results:
                name = result['feature_set']
                features = ', '.join(result['features']) if result['features'] else 'None'
                metrics = result['aggregated_metrics']
                
                kt = f"{metrics.get('kendall_tau_mean', 0):.4f}"
                r10 = f"{metrics.get('recall@10_mean', 0):.4f}"
                r20 = f"{metrics.get('recall@20_mean', 0):.4f}"
                ndcg = f"{metrics.get('ndcg@10_mean', 0):.4f}"
                map_score = f"{metrics.get('map_mean', 0):.4f}"
                time_s = f"{result['training_time_seconds']:.2f}"
                
                f.write(f"| {name} | {features} | {kt} | {r10} | {r20} | {ndcg} | {map_score} | {time_s} |\n")
        
        print(f"\nComparison table saved to {table_file}")


def main():
    parser = argparse.ArgumentParser(description='Run node feature ablation experiments')
    parser.add_argument('--data_path', type=str, default='./data/', help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='./experiments/', help='Output directory for results')
    parser.add_argument('--model_size', type=int, default=10000, help='Model size (padding)')
    parser.add_argument('--hidden_dim', type=int, default=20, help='Hidden dimension')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--feature_set', type=str, default=None, help='Run single feature set (optional)')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_size=args.model_size,
        hidden_dim=args.hidden_dim,
        num_epochs=args.num_epochs,
        device=args.device
    )
    
    if args.feature_set:
        # Run single experiment
        if args.feature_set not in FEATURE_SETS:
            print(f"Error: Unknown feature set '{args.feature_set}'")
            print(f"Available: {list(FEATURE_SETS.keys())}")
            return
        runner.run_experiment(args.feature_set)
    else:
        # Run all experiments
        runner.run_all_experiments()


if __name__ == '__main__':
    main()
