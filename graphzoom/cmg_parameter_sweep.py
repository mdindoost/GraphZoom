#!/usr/bin/env python3
"""
CMG++ Parameter Sweep: Comprehensive Connectivity Analysis
Test all parameter combinations to find connectivity-preserving settings
"""

import numpy as np
import scipy.sparse as sp
import networkx as nx
import json
import torch
from torch_geometric.data import Data
import pandas as pd
import itertools
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

class CMGParameterSweep:
    def __init__(self, output_dir="cmg_parameter_sweep"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
    
    def load_cora_data(self):
        """Load Cora dataset as PyG Data object."""
        print("📂 Loading Cora dataset...")
        
        with open("dataset/cora/cora-G.json", 'r') as f:
            data_json = json.load(f)
        
        edges = data_json['links']
        n_nodes = len(data_json['nodes'])
        
        # Build PyG data
        edge_list = []
        for edge in edges:
            src, tgt = edge['source'], edge['target']
            edge_list.append((src, tgt))
            edge_list.append((tgt, src))
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Load features
        try:
            features = np.load("dataset/cora/cora-feats.npy")
            x = torch.tensor(features, dtype=torch.float)
        except FileNotFoundError:
            x = torch.eye(n_nodes, dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, num_nodes=n_nodes)
        
        print(f"   ✅ Original: {n_nodes} nodes, {len(edges)} edges")
        return data
    
    def analyze_connectivity_detailed(self, adjacency_matrix):
        """Comprehensive connectivity analysis."""
        
        # NetworkX connectivity
        try:
            if hasattr(nx, 'from_scipy_sparse_array'):
                G = nx.from_scipy_sparse_array(adjacency_matrix)
            else:
                G = nx.from_scipy_sparse_matrix(adjacency_matrix)
            
            n_components = nx.number_connected_components(G)
            components_list = list(nx.connected_components(G))
            component_sizes = sorted([len(c) for c in components_list], reverse=True)
            
            # Calculate statistics
            largest_component = component_sizes[0] if component_sizes else 0
            largest_component_pct = (largest_component / adjacency_matrix.shape[0]) * 100
            
            # Component size distribution
            size_distribution = {
                'sizes': component_sizes,
                'mean_size': np.mean(component_sizes) if component_sizes else 0,
                'median_size': np.median(component_sizes) if component_sizes else 0,
                'std_size': np.std(component_sizes) if component_sizes else 0,
                'singleton_count': sum(1 for size in component_sizes if size == 1),
                'small_components': sum(1 for size in component_sizes if size <= 5),
                'large_components': sum(1 for size in component_sizes if size > 10)
            }
            
            return {
                'success': True,
                'n_components': n_components,
                'largest_component': largest_component,
                'largest_component_pct': largest_component_pct,
                'component_sizes': component_sizes[:10],  # Top 10 for storage
                'size_distribution': size_distribution
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'n_components': -1,
                'largest_component': 0,
                'largest_component_pct': 0,
                'component_sizes': [],
                'size_distribution': {}
            }
    
    def run_cmg_single_config(self, data, threshold, k, d, run_id):
        """Run CMG++ with specific parameters."""
        
        print(f"      🔄 Config {run_id}: threshold={threshold}, k={k}, d={d}")
        
        try:
            # Set random seeds for reproducibility
            np.random.seed(42 + run_id)
            torch.manual_seed(42 + run_id)
            
            # Import CMG
            from filtered import cmg_filtered_clustering
            
            # Run CMG clustering
            clusters, n_clusters, phi_stats, lambda_crit = cmg_filtered_clustering(
                data, k=k, d=d, threshold=threshold
            )
            
            # Build original adjacency (for coarsening)
            n_nodes = data.num_nodes
            A_orig = sp.lil_matrix((n_nodes, n_nodes))
            edge_index = data.edge_index.cpu().numpy()
            
            for i in range(edge_index.shape[1]):
                src, tgt = edge_index[0, i], edge_index[1, i]
                A_orig[src, tgt] = 1
            
            A_orig = A_orig.tocsr()
            
            # Build assignment matrix
            P = sp.lil_matrix((n_nodes, n_clusters))
            for i, cluster_id in enumerate(clusters):
                P[i, cluster_id] = 1.0
            P = P.tocsr()
            
            # Coarsen graph
            A_coarse = P.T @ A_orig @ P
            A_coarse.eliminate_zeros()
            
            # Analyze connectivity
            connectivity = self.analyze_connectivity_detailed(A_coarse)
            
            # Calculate metrics
            nodes = A_coarse.shape[0]
            edges = A_coarse.nnz // 2
            reduction_ratio = n_nodes / nodes
            
            # Check for multi-edges
            multi_edge_count = np.sum(A_coarse.data > 1)
            max_edge_weight = A_coarse.data.max()
            
            result = {
                'run_id': run_id,
                'threshold': threshold,
                'k': k,
                'd': d,
                'success': True,
                'original_nodes': n_nodes,
                'coarse_nodes': nodes,
                'coarse_edges': edges,
                'reduction_ratio': reduction_ratio,
                'lambda_critical': lambda_crit,
                'multi_edge_count': multi_edge_count,
                'max_edge_weight': max_edge_weight,
                **connectivity
            }
            
            print(f"         ✅ {nodes} nodes, {connectivity['n_components']} components")
            print(f"         📊 Largest: {connectivity['largest_component_pct']:.1f}%")
            
            return result
            
        except Exception as e:
            print(f"         ❌ Failed: {e}")
            return {
                'run_id': run_id,
                'threshold': threshold,
                'k': k,
                'd': d,
                'success': False,
                'error': str(e),
                'original_nodes': data.num_nodes,
                'coarse_nodes': 0,
                'coarse_edges': 0,
                'reduction_ratio': 0,
                'n_components': -1,
                'largest_component': 0,
                'largest_component_pct': 0
            }
    
    def run_comprehensive_sweep(self):
        """Run complete parameter sweep."""
        
        print("🚀 CMG++ COMPREHENSIVE PARAMETER SWEEP")
        print("="*60)
        
        # Load data
        data = self.load_cora_data()
        
        # Parameter ranges (as you specified)
        thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]
        k_values = [5, 10, 15, 20]
        d_values = [10, 15, 20, 30]
        
        total_configs = len(thresholds) * len(k_values) * len(d_values)
        print(f"📊 Testing {total_configs} parameter combinations")
        print(f"   Thresholds: {thresholds}")
        print(f"   k values: {k_values}")
        print(f"   d values: {d_values}")
        print()
        
        # Run all combinations
        run_id = 0
        
        for threshold in thresholds:
            print(f"\n🎯 THRESHOLD = {threshold}")
            print("-" * 40)
            
            for k in k_values:
                print(f"   📈 k = {k}")
                
                for d in d_values:
                    run_id += 1
                    result = self.run_cmg_single_config(data, threshold, k, d, run_id)
                    self.results.append(result)
        
        print(f"\n✅ Sweep complete! Tested {len(self.results)} configurations")
        
        # Save results
        self.save_results()
        
        return self.results
    
    def save_results(self):
        """Save results in multiple formats."""
        
        # Save as CSV
        df = pd.DataFrame(self.results)
        csv_file = self.output_dir / "cmg_parameter_sweep_results.csv"
        df.to_csv(csv_file, index=False)
        
        # Save as pickle for full data
        pickle_file = self.output_dir / "cmg_parameter_sweep_full.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(self.results, f)
        
        print(f"💾 Results saved:")
        print(f"   📄 CSV: {csv_file}")
        print(f"   📦 Pickle: {pickle_file}")
    
    def analyze_results(self):
        """Analyze parameter sweep results."""
        
        print("\n📊 PARAMETER SWEEP ANALYSIS")
        print("="*60)
        
        df = pd.DataFrame(self.results)
        
        # Filter successful runs
        success_df = df[df['success'] == True]
        failed_df = df[df['success'] == False]
        
        print(f"✅ Successful runs: {len(success_df)}/{len(df)}")
        print(f"❌ Failed runs: {len(failed_df)}")
        
        if len(success_df) == 0:
            print("No successful runs to analyze!")
            return
        
        # Find best connectivity results
        print(f"\n🏆 BEST CONNECTIVITY RESULTS:")
        print("-" * 40)
        
        # Sort by connectivity (fewer components = better)
        best_connectivity = success_df.nsmallest(10, 'n_components')
        
        for _, row in best_connectivity.iterrows():
            print(f"🥇 {row['n_components']:3d} components: threshold={row['threshold']}, k={row['k']}, d={row['d']}")
            print(f"    {row['coarse_nodes']} nodes, {row['largest_component_pct']:.1f}% in largest")
        
        # Analyze by parameter
        print(f"\n📈 PARAMETER EFFECTS:")
        print("-" * 40)
        
        # Group by threshold
        print("By Threshold:")
        threshold_analysis = success_df.groupby('threshold')['n_components'].agg(['mean', 'min', 'max'])
        for threshold, stats in threshold_analysis.iterrows():
            print(f"   {threshold:4.2f}: {stats['mean']:5.1f} avg, {stats['min']:3.0f}-{stats['max']:3.0f} range")
        
        # Group by k
        print("\nBy k:")
        k_analysis = success_df.groupby('k')['n_components'].agg(['mean', 'min', 'max'])
        for k_val, stats in k_analysis.iterrows():
            print(f"   k={k_val:2d}: {stats['mean']:5.1f} avg, {stats['min']:3.0f}-{stats['max']:3.0f} range")
        
        # Group by d
        print("\nBy d:")
        d_analysis = success_df.groupby('d')['n_components'].agg(['mean', 'min', 'max'])
        for d_val, stats in d_analysis.iterrows():
            print(f"   d={d_val:2d}: {stats['mean']:5.1f} avg, {stats['min']:3.0f}-{stats['max']:3.0f} range")
        
        return success_df
    
    def create_visualizations(self, df):
        """Create comprehensive visualizations."""
        
        print(f"\n📊 Creating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('CMG++ Parameter Sweep: Connectivity Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Components vs Threshold
        ax1 = axes[0, 0]
        threshold_stats = df.groupby('threshold')['n_components'].agg(['mean', 'std']).reset_index()
        ax1.errorbar(threshold_stats['threshold'], threshold_stats['mean'], 
                    yerr=threshold_stats['std'], marker='o', capsize=5)
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Number of Components')
        ax1.set_title('Components vs Threshold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # Plot 2: Components vs k
        ax2 = axes[0, 1]
        k_stats = df.groupby('k')['n_components'].agg(['mean', 'std']).reset_index()
        ax2.errorbar(k_stats['k'], k_stats['mean'], 
                    yerr=k_stats['std'], marker='s', capsize=5)
        ax2.set_xlabel('Filter Order (k)')
        ax2.set_ylabel('Number of Components')
        ax2.set_title('Components vs Filter Order')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Components vs d
        ax3 = axes[0, 2]
        d_stats = df.groupby('d')['n_components'].agg(['mean', 'std']).reset_index()
        ax3.errorbar(d_stats['d'], d_stats['mean'], 
                    yerr=d_stats['std'], marker='^', capsize=5)
        ax3.set_xlabel('Embedding Dimension (d)')
        ax3.set_ylabel('Number of Components')
        ax3.set_title('Components vs Embedding Dimension')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Heatmap of threshold vs k (fixed seaborn issue)
        ax4 = axes[1, 0]
        pivot_data = df.pivot_table(values='n_components', index='threshold', columns='k', aggfunc='mean')
        im = ax4.imshow(pivot_data, cmap='viridis_r', aspect='auto')
        ax4.set_title('Components: Threshold vs k')
        ax4.set_xlabel('k')
        ax4.set_ylabel('threshold')
        ax4.set_xticks(range(len(pivot_data.columns)))
        ax4.set_xticklabels(pivot_data.columns)
        ax4.set_yticks(range(len(pivot_data.index)))
        ax4.set_yticklabels([f'{x:.2f}' for x in pivot_data.index])
        
        # Add text annotations
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                ax4.text(j, i, f'{pivot_data.iloc[i,j]:.0f}', 
                        ha='center', va='center', color='white', fontweight='bold')
        
        # Plot 5: Largest component percentage
        ax5 = axes[1, 1]
        ax5.scatter(df['n_components'], df['largest_component_pct'], alpha=0.6)
        ax5.set_xlabel('Number of Components')
        ax5.set_ylabel('Largest Component %')
        ax5.set_title('Component Count vs Size Distribution')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Best configurations
        ax6 = axes[1, 2]
        best_configs = df.nsmallest(20, 'n_components')
        colors = plt.cm.viridis(np.linspace(0, 1, len(best_configs)))
        
        bars = ax6.bar(range(len(best_configs)), best_configs['n_components'], color=colors)
        ax6.set_xlabel('Configuration Rank')
        ax6.set_ylabel('Number of Components')
        ax6.set_title('Top 20 Best Configurations')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "cmg_parameter_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ Saved: {plot_file}")
        
        return fig

def main():
    """Run complete parameter sweep and analysis."""
    
    # Run sweep
    sweeper = CMGParameterSweep()
    results = sweeper.run_comprehensive_sweep()
    
    # Analyze results
    df = sweeper.analyze_results()
    
    # Create visualizations
    if df is not None and len(df) > 0:
        sweeper.create_visualizations(df)
    
    print("\n🎯 PARAMETER SWEEP COMPLETE!")
    print("="*60)
    print("Results show the effect of threshold, k, and d on CMG++ connectivity")
    print("Use these insights to find optimal parameters for connectivity preservation")

if __name__ == "__main__":
    main()
