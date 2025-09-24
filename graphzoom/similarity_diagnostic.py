#!/usr/bin/env python3
"""
Similarity Diagnostic: Understand why threshold doesn't affect connectivity
"""

import numpy as np
import scipy.sparse as sp
import networkx as nx
import json
import torch
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class SimilarityDiagnostic:
    def __init__(self, output_dir="similarity_debug"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def load_cora_data(self):
        """Load Cora dataset."""
        with open("dataset/cora/cora-G.json", 'r') as f:
            data_json = json.load(f)
        
        edges = data_json['links']
        n_nodes = len(data_json['nodes'])
        
        edge_list = []
        for edge in edges:
            src, tgt = edge['source'], edge['target']
            edge_list.append((src, tgt))
            edge_list.append((tgt, src))
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        try:
            features = np.load("dataset/cora/cora-feats.npy")
            x = torch.tensor(features, dtype=torch.float)
        except FileNotFoundError:
            x = torch.eye(n_nodes, dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, num_nodes=n_nodes)
        return data
    
    def build_normalized_laplacian(self, A):
        """Build normalized Laplacian from your filtered.py."""
        d = np.array(A.sum(axis=1)).flatten()
        d_safe = d + 1e-12
        d_inv_sqrt = sp.diags(1.0 / np.sqrt(d_safe))
        L = sp.diags(d) - A
        L_norm = d_inv_sqrt @ L @ d_inv_sqrt
        return L_norm.tocsr()
    
    def apply_spectral_filter(self, X, L_norm, k):
        """Apply spectral filter from your filtered.py."""
        if not sp.isspmatrix_csr(L_norm):
            L_norm = L_norm.tocsr()
        
        I = sp.identity(L_norm.shape[0], format='csr')
        filter_matrix = I - 0.5 * L_norm
        Y = np.zeros_like(X)
        
        for j in range(X.shape[1]):
            x = X[:, j].copy()
            power_k = x.copy()
            for _ in range(k):
                power_k = filter_matrix @ power_k
            power_k_plus_1 = filter_matrix @ power_k
            Y[:, j] = power_k_plus_1 - power_k
        
        return Y
    
    def analyze_similarity_step_by_step(self, threshold=0.1, k=10, d=20):
        """Step-by-step analysis of similarity filtering."""
        
        print(f"🔍 STEP-BY-STEP SIMILARITY ANALYSIS")
        print(f"   Threshold: {threshold}, k: {k}, d: {d}")
        print("="*60)
        
        # Load data
        data = self.load_cora_data()
        edge_index = data.edge_index.cpu().numpy()
        n = data.num_nodes
        
        print(f"1️⃣ Original graph: {n} nodes, {edge_index.shape[1]} edges")
        
        # Build original adjacency
        from torch_geometric.utils import to_scipy_sparse_matrix
        A = to_scipy_sparse_matrix(data.edge_index, num_nodes=n).tocsr()
        print(f"   Original A: {A.nnz} nonzeros")
        
        # Build normalized Laplacian for filtering
        L_norm = self.build_normalized_laplacian(A)
        print(f"   Normalized Laplacian built")
        
        # Generate random vectors and filter
        np.random.seed(42)
        X = np.random.randn(n, d)
        Y = self.apply_spectral_filter(X, L_norm, k)
        print(f"   Filtered embeddings Y: {Y.shape}")
        
        # CRITICAL: Analyze the similarity matrix
        print(f"\n2️⃣ SIMILARITY MATRIX ANALYSIS:")
        sim = cosine_similarity(Y)
        print(f"   Similarity matrix shape: {sim.shape}")
        print(f"   Similarity range: [{sim.min():.4f}, {sim.max():.4f}]")
        print(f"   Similarity mean: {sim.mean():.4f}")
        print(f"   Similarity std: {sim.std():.4f}")
        
        # Count similarities above threshold
        above_threshold = np.sum(sim > threshold)
        total_pairs = sim.shape[0] * sim.shape[1]
        print(f"   Similarities > {threshold}: {above_threshold}/{total_pairs} ({above_threshold/total_pairs*100:.1f}%)")
        
        # Check diagonal (should be 1.0)
        diagonal_min = np.diag(sim).min()
        diagonal_max = np.diag(sim).max()
        print(f"   Diagonal range: [{diagonal_min:.4f}, {diagonal_max:.4f}]")
        
        # 3. Analyze edge-specific similarities
        print(f"\n3️⃣ EDGE-SPECIFIC SIMILARITY ANALYSIS:")
        if edge_index.ndim == 1:
            edge_index = edge_index.reshape(2, -1)
        
        rows, cols = edge_index[0], edge_index[1]
        edge_similarities = sim[rows, cols]
        
        print(f"   Edge similarities range: [{edge_similarities.min():.4f}, {edge_similarities.max():.4f}]")
        print(f"   Edge similarities mean: {edge_similarities.mean():.4f}")
        print(f"   Edge similarities std: {edge_similarities.std():.4f}")
        
        # Count edges above threshold
        edges_above_threshold = np.sum(edge_similarities > threshold)
        total_edges = len(edge_similarities)
        print(f"   Edges > {threshold}: {edges_above_threshold}/{total_edges} ({edges_above_threshold/total_edges*100:.1f}%)")
        
        # 4. Build reweighted adjacency
        print(f"\n4️⃣ REWEIGHTING ANALYSIS:")
        weights = np.where(edge_similarities > threshold, edge_similarities, 0.0)
        
        nonzero_weights = np.sum(weights > 0)
        print(f"   Nonzero weights: {nonzero_weights}/{total_edges} ({nonzero_weights/total_edges*100:.1f}%)")
        print(f"   Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
        print(f"   Weight mean (nonzero): {weights[weights > 0].mean():.4f}")
        
        # Build final adjacency
        A_reweighted = sp.coo_matrix((weights, (rows, cols)), shape=(n, n))
        A_reweighted = A_reweighted.maximum(A_reweighted.T).tocsr()
        
        print(f"   Reweighted A: {A_reweighted.nnz} nonzeros")
        print(f"   Sparsity reduction: {A.nnz} → {A_reweighted.nnz} ({A_reweighted.nnz/A.nnz*100:.1f}%)")
        
        # 5. Connectivity analysis
        print(f"\n5️⃣ CONNECTIVITY IMPACT:")
        
        # Original connectivity
        if hasattr(nx, 'from_scipy_sparse_array'):
            G_orig = nx.from_scipy_sparse_array(A)
            G_reweighted = nx.from_scipy_sparse_array(A_reweighted)
        else:
            G_orig = nx.from_scipy_sparse_matrix(A)
            G_reweighted = nx.from_scipy_sparse_matrix(A_reweighted)
        
        orig_components = nx.number_connected_components(G_orig)
        reweighted_components = nx.number_connected_components(G_reweighted)
        
        print(f"   Original components: {orig_components}")
        print(f"   Reweighted components: {reweighted_components}")
        print(f"   Component change: {orig_components} → {reweighted_components}")
        
        # Save diagnostic data
        diagnostic_data = {
            'threshold': threshold,
            'similarity_stats': {
                'min': float(sim.min()),
                'max': float(sim.max()),
                'mean': float(sim.mean()),
                'std': float(sim.std()),
                'above_threshold_pct': float(above_threshold/total_pairs*100)
            },
            'edge_stats': {
                'min': float(edge_similarities.min()),
                'max': float(edge_similarities.max()),
                'mean': float(edge_similarities.mean()),
                'std': float(edge_similarities.std()),
                'above_threshold_pct': float(edges_above_threshold/total_edges*100)
            },
            'connectivity': {
                'original_components': int(orig_components),
                'reweighted_components': int(reweighted_components),
                'original_edges': int(A.nnz),
                'reweighted_edges': int(A_reweighted.nnz)
            }
        }
        
        return diagnostic_data, sim, edge_similarities, A_reweighted
    
    def compare_multiple_thresholds(self, thresholds=[0.001, 0.01, 0.1, 0.5, 0.9]):
        """Compare similarity effects across multiple thresholds."""
        
        print(f"\n🔄 MULTI-THRESHOLD COMPARISON")
        print("="*60)
        
        results = []
        
        for threshold in thresholds:
            print(f"\n--- Threshold {threshold} ---")
            data, sim, edge_sim, A_reweighted = self.analyze_similarity_step_by_step(threshold)
            results.append(data)
        
        # Summary comparison
        print(f"\n📊 THRESHOLD COMPARISON SUMMARY:")
        print("-"*60)
        print("Threshold | Edge%>T | Reweight% | Components | SparsityChange")
        print("-"*60)
        
        for result in results:
            edge_pct = result['edge_stats']['above_threshold_pct']
            sparsity_change = result['connectivity']['reweighted_edges'] / result['connectivity']['original_edges'] * 100
            components = result['connectivity']['reweighted_components']
            
            print(f"{result['threshold']:8.3f} | {edge_pct:6.1f}% | {sparsity_change:8.1f}% | {components:9d} | {sparsity_change:11.1f}%")
        
        return results
    
    def create_similarity_distribution_plot(self, sim, edge_similarities, threshold=0.1):
        """Visualize similarity distributions."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Similarity Analysis (threshold={threshold})', fontsize=16)
        
        # 1. Full similarity matrix histogram
        ax1 = axes[0, 0]
        ax1.hist(sim.flatten(), bins=50, alpha=0.7, density=True)
        ax1.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold={threshold}')
        ax1.set_xlabel('Cosine Similarity')
        ax1.set_ylabel('Density')
        ax1.set_title('All Pairwise Similarities')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Edge-specific similarities
        ax2 = axes[0, 1]
        ax2.hist(edge_similarities, bins=50, alpha=0.7, density=True, color='orange')
        ax2.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold={threshold}')
        ax2.set_xlabel('Cosine Similarity')
        ax2.set_ylabel('Density')
        ax2.set_title('Edge-Specific Similarities')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Similarity matrix heatmap (subset)
        ax3 = axes[1, 0]
        subset_size = min(100, sim.shape[0])
        sim_subset = sim[:subset_size, :subset_size]
        im = ax3.imshow(sim_subset, cmap='viridis', aspect='auto')
        ax3.set_title(f'Similarity Matrix (first {subset_size}×{subset_size})')
        plt.colorbar(im, ax=ax3)
        
        # 4. Threshold effect visualization
        ax4 = axes[1, 1]
        thresholds_test = np.linspace(0.001, 0.999, 50)
        edge_survival_rates = []
        
        for t in thresholds_test:
            survival_rate = np.sum(edge_similarities > t) / len(edge_similarities) * 100
            edge_survival_rates.append(survival_rate)
        
        ax4.plot(thresholds_test, edge_survival_rates, 'b-', linewidth=2)
        ax4.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Current={threshold}')
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel('Edge Survival Rate (%)')
        ax4.set_title('Edge Preservation vs Threshold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_file = self.output_dir / f"similarity_analysis_t{threshold}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Similarity plot saved: {plot_file}")


def main():
    """Run similarity diagnostic."""
    
    print("🔬 SIMILARITY FILTERING DIAGNOSTIC")
    print("="*60)
    
    diagnostic = SimilarityDiagnostic()
    
    # Single threshold analysis
    print("\n🎯 DETAILED ANALYSIS FOR THRESHOLD = 0.1")
    data, sim, edge_sim, A_reweighted = diagnostic.analyze_similarity_step_by_step(threshold=0.1)
    
    # Create visualization
    diagnostic.create_similarity_distribution_plot(sim, edge_sim, threshold=0.1)
    
    # Multi-threshold comparison
    print("\n🔄 COMPARING MULTIPLE THRESHOLDS")
    results = diagnostic.compare_multiple_thresholds()
    
    print(f"\n🔍 DIAGNOSTIC COMPLETE!")
    print(f"Check the plots and analysis to understand why threshold doesn't affect connectivity")

if __name__ == "__main__":
    main()
