#!/usr/bin/env python3
"""
topology_analysis.py - Reusable Transport Network Topology Analysis Tool

This script analyzes the structural properties of transport networks and compares
them to scale-free networks. Designed for the GNN-Bet transport network project.

Metrics based on course textbooks:
- Barabási, Network Science (2016)
- Menczer et al., A First Course in Network Science (2020)

Usage:
    python topology_analysis.py --csv path/to/network_temporal.csv --name "City Name"
    python topology_analysis.py --gml path/to/network.gml --name "City Name"

Author: Generated for Network Science Course Project
"""

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class TransportNetworkAnalyzer:
    """Analyzer for transport network topology."""
    
    def __init__(self):
        self.networks = {}
        self.scale_free = None
        
    def load_from_csv(self, csv_path, name):
        """Load transport network from GTFS-style network_temporal.csv"""
        df = pd.read_csv(csv_path)
        df['duration'] = df['arr_time_ut'] - df['dep_time_ut']
        df.loc[df['duration'] <= 0, 'duration'] = 60
        edge_weights = df.groupby(['from_stop_I', 'to_stop_I'])['duration'].mean().reset_index()
        
        G = nx.DiGraph()
        for _, row in edge_weights.iterrows():
            G.add_edge(int(row['from_stop_I']), int(row['to_stop_I']), weight=row['duration'])
        
        stats, degrees = self._compute_statistics(G, name)
        self.networks[name] = (G, degrees, stats)
        print(f"Loaded {name}: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
        return G
    
    def load_from_gml(self, gml_path, name):
        """Load network from GML file"""
        G = nx.read_gml(gml_path)
        if not G.is_directed():
            G = G.to_directed()
        stats, degrees = self._compute_statistics(G, name)
        self.networks[name] = (G, degrees, stats)
        print(f"Loaded {name}: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
        return G
    
    def generate_scale_free_baseline(self, n_nodes=None, m=3, seed=42):
        """Generate Barabási-Albert scale-free graph for comparison."""
        if n_nodes is None:
            n_nodes = int(np.mean([G.number_of_nodes() for G, _, _ in self.networks.values()]))
        G = nx.barabasi_albert_graph(n_nodes, m, seed=seed).to_directed()
        name = f"Scale-Free (BA)"
        stats, degrees = self._compute_statistics(G, name)
        self.scale_free = (G, degrees, stats)
        print(f"Generated {name}: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
        return G
    
    def _compute_statistics(self, G, name):
        """Compute network statistics."""
        degrees = [d for n, d in G.degree()]
        G_und = G.to_undirected()
        clustering = nx.average_clustering(G_und)
        
        stats = {
            'Network': name,
            'Nodes': G.number_of_nodes(),
            'Edges': G.number_of_edges(),
            'Mean Degree': round(np.mean(degrees), 2),
            'Max Degree': int(np.max(degrees)),
            'Min Degree': int(np.min(degrees)),
            'Std Degree': round(np.std(degrees), 2),
            'Clustering': round(clustering, 4),
            'Density': round(nx.density(G), 6)
        }
        return stats, degrees
    
    def get_statistics_table(self):
        """Return DataFrame with all network statistics"""
        all_stats = [stats for _, _, stats in self.networks.values()]
        if self.scale_free:
            all_stats.append(self.scale_free[2])
        return pd.DataFrame(all_stats)
    
    def print_statistics(self):
        """Print formatted statistics table"""
        df = self.get_statistics_table()
        print("\n" + "="*80)
        print("NETWORK TOPOLOGY COMPARISON")
        print("="*80)
        print(df.to_string(index=False))
        return df
    
    def create_comparison_figure(self, output_path='topology_comparison.png'):
        """Create comparison figure."""
        if not self.scale_free:
            self.generate_scale_free_baseline()
        
        t_deg = []
        for _, (G, degrees, _) in self.networks.items():
            t_deg.extend(degrees)
        sf_deg = self.scale_free[1]
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        tc, sfc = '#2E86AB', '#E94F37'
        
        # (a) Histogram
        axes[0].hist(t_deg, bins=range(0, min(max(t_deg)+2, 50)), alpha=0.7,
                     label='Transport', density=True, color=tc, edgecolor='white')
        axes[0].hist(sf_deg, bins=range(0, 60, 2), alpha=0.5,
                     label='Scale-Free', density=True, color=sfc, edgecolor='white')
        axes[0].set_xlabel('Degree (k)')
        axes[0].set_ylabel('P(k)')
        axes[0].set_title('(a) Degree Distribution')
        axes[0].legend()
        axes[0].set_xlim(0, 50)
        
        # (b) Log-log
        t_counts = Counter(t_deg)
        t_k = sorted(t_counts.keys())
        t_pk = [t_counts[k]/len(t_deg) for k in t_k]
        sf_counts = Counter(sf_deg)
        sf_k = sorted(sf_counts.keys())
        sf_pk = [sf_counts[k]/len(sf_deg) for k in sf_k]
        
        axes[1].loglog(t_k, t_pk, 'o', ms=6, alpha=0.7, label='Transport', color=tc)
        axes[1].loglog(sf_k, sf_pk, 's', ms=5, alpha=0.7, label='Scale-Free', color=sfc)
        k_ref = np.array([6, 100])
        axes[1].loglog(k_ref, 0.5*(k_ref/6)**(-2.5), '--', color='gray', label=r'$\gamma$=2.5')
        axes[1].set_xlabel('Degree (k)')
        axes[1].set_ylabel('P(k)')
        axes[1].set_title('(b) Log-Log Plot')
        axes[1].legend(loc='lower left')
        axes[1].grid(True, alpha=0.3, which='both', ls=':')
        
        # (c) Stats
        metrics = ['Mean', 'Std', 'Max']
        t_vals = [np.mean(t_deg), np.std(t_deg), np.max(t_deg)]
        sf_vals = [np.mean(sf_deg), np.std(sf_deg), np.max(sf_deg)]
        x = np.arange(3)
        axes[2].bar(x-0.175, t_vals, 0.35, label='Transport', color=tc)
        axes[2].bar(x+0.175, sf_vals, 0.35, label='Scale-Free', color=sfc)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(metrics)
        axes[2].set_ylabel('Value')
        axes[2].set_title('(c) Statistics')
        axes[2].legend()
        axes[2].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"Figure saved: {output_path}")


# Example usage
if __name__ == "__main__":
    analyzer = TransportNetworkAnalyzer()
    # analyzer.load_from_csv("path/to/network_temporal.csv", "CityName")
    # analyzer.generate_scale_free_baseline()
    # analyzer.print_statistics()
    # analyzer.create_comparison_figure("output.png")
    print("TransportNetworkAnalyzer ready. Import and use in your code.")
