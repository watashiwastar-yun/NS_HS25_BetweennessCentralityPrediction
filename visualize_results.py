"""
Results Visualization and Analysis

This script generates visualizations from experiment results:
- Comparison bar charts
- Heatmaps
- Performance trends

Usage:
    python visualize_results.py --results_dir ./experiments/
"""

import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


def load_all_results(results_dir):
    """Load all experiment results."""
    results_file = Path(results_dir) / "all_results.json"
    
    if not results_file.exists():
        print(f"Error: {results_file} not found")
        return None
    
    with open(results_file, 'r') as f:
        return json.load(f)


def create_comparison_plot(results, output_dir):
    """Create bar chart comparing key metrics across feature sets."""
    feature_sets = [r['feature_set'] for r in results]
    
    metrics_to_plot = [
        ('kendall_tau_mean', 'Kendall\'s Tau'),
        ('recall@10_mean', 'Recall@10'),
        ('recall@20_mean', 'Recall@20'),
        ('ndcg@10_mean', 'NDCG@10'),
        ('map_mean', 'MAP')
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, (metric_key, metric_name) in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        values = [r['aggregated_metrics'].get(metric_key, 0) for r in results]
        stds = [r['aggregated_metrics'].get(metric_key.replace('_mean', '_std'), 0) for r in results]
        
        bars = ax.bar(range(len(feature_sets)), values, yerr=stds, capsize=5, alpha=0.7)
        
        # Color bars
        colors = plt.cm.viridis(np.linspace(0, 1, len(feature_sets)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel('Feature Set')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} Comparison')
        ax.set_xticks(range(len(feature_sets)))
        ax.set_xticklabels(feature_sets, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
    
    # Remove extra subplot
    fig.delaxes(axes[-1])
    
    plt.tight_layout()
    output_file = Path(output_dir) / 'comparison_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {output_file}")
    plt.close()


def create_heatmap(results, output_dir):
    """Create heatmap of all metrics across feature sets."""
    feature_sets = [r['feature_set'] for r in results]
    
    # Select metrics for heatmap
    metric_keys = [
        'kendall_tau_mean', 'spearman_rho_mean',
        'recall@5_mean', 'recall@10_mean', 'recall@20_mean',
        'precision@5_mean', 'precision@10_mean', 'precision@20_mean',
        'ndcg@5_mean', 'ndcg@10_mean', 'ndcg@20_mean',
        'map_mean'
    ]
    
    metric_names = [
        'Kendall Tau', 'Spearman Rho',
        'Recall@5', 'Recall@10', 'Recall@20',
        'Precision@5', 'Precision@10', 'Precision@20',
        'NDCG@5', 'NDCG@10', 'NDCG@20',
        'MAP'
    ]
    
    # Build data matrix
    data = []
    for result in results:
        row = [result['aggregated_metrics'].get(key, 0) for key in metric_keys]
        data.append(row)
    
    df = pd.DataFrame(data, index=feature_sets, columns=metric_names)
    
    # Create heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(df, annot=True, fmt='.3f', cmap='YlGnBu', cbar_kws={'label': 'Score'})
    plt.title('Performance Heatmap Across Feature Sets and Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Feature Sets')
    plt.tight_layout()
    
    output_file = Path(output_dir) / 'heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {output_file}")
    plt.close()


def create_top_k_plot(results, output_dir):
    """Plot Recall@K and Precision@K for different K values."""
    k_values = [5, 10, 20, 50]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for result in results:
        feature_set = result['feature_set']
        metrics = result['aggregated_metrics']
        
        recall_values = [metrics.get(f'recall@{k}_mean', 0) for k in k_values]
        precision_values = [metrics.get(f'precision@{k}_mean', 0) for k in k_values]
        
        ax1.plot(k_values, recall_values, marker='o', label=feature_set)
        ax2.plot(k_values, precision_values, marker='s', label=feature_set)
    
    ax1.set_xlabel('K')
    ax1.set_ylabel('Recall@K')
    ax1.set_title('Recall@K vs K')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(alpha=0.3)
    
    ax2.set_xlabel('K')
    ax2.set_ylabel('Precision@K')
    ax2.set_title('Precision@K vs K')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    output_file = Path(output_dir) / 'top_k_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved top-K plot to {output_file}")
    plt.close()


def create_summary_table(results, output_dir):
    """Create detailed summary table as CSV."""
    rows = []
    
    for result in results:
        row = {
            'Feature Set': result['feature_set'],
            'Features': ', '.join(result['features']) if result['features'] else 'None',
            'Num Features': result['node_feat_dim'],
            'Training Time (s)': result['training_time_seconds'],
        }
        
        # Add all aggregated metrics
        for key, value in result['aggregated_metrics'].items():
            row[key] = value
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    output_file = Path(output_dir) / 'summary_table.csv'
    df.to_csv(output_file, index=False)
    print(f"Saved summary table to {output_file}")
    
    return df


def print_best_performers(results):
    """Print best performing feature sets for each metric."""
    print("\n" + "="*60)
    print("BEST PERFORMERS")
    print("="*60)
    
    metrics_of_interest = [
        ('kendall_tau_mean', 'Kendall\'s Tau', 'max'),
        ('recall@10_mean', 'Recall@10', 'max'),
        ('recall@20_mean', 'Recall@20', 'max'),
        ('ndcg@10_mean', 'NDCG@10', 'max'),
        ('map_mean', 'MAP', 'max'),
        ('training_time_seconds', 'Training Time', 'min'),
    ]
    
    for metric_key, metric_name, mode in metrics_of_interest:
        if metric_key == 'training_time_seconds':
            values = [(r['feature_set'], r[metric_key]) for r in results]
        else:
            values = [(r['feature_set'], r['aggregated_metrics'].get(metric_key, 0)) for r in results]
        
        if mode == 'max':
            best = max(values, key=lambda x: x[1])
        else:
            best = min(values, key=lambda x: x[1])
        
        print(f"{metric_name:20s}: {best[0]:20s} ({best[1]:.4f})")


def main():
    parser = argparse.ArgumentParser(description='Visualize experiment results')
    parser.add_argument('--results_dir', type=str, default='./experiments/', 
                       help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (default: same as results_dir)')
    
    args = parser.parse_args()
    
    output_dir = args.output_dir or args.results_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load results
    print(f"Loading results from {args.results_dir}...")
    results = load_all_results(args.results_dir)
    
    if not results:
        return
    
    print(f"Found {len(results)} experiment results")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    create_comparison_plot(results, output_dir)
    create_heatmap(results, output_dir)
    create_top_k_plot(results, output_dir)
    
    # Create summary table
    print("\nGenerating summary table...")
    df = create_summary_table(results, output_dir)
    
    # Print best performers
    print_best_performers(results)
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print(f"All outputs saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
