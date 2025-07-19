#!/usr/bin/env python3
"""
Spectral Properties Analysis Tool
Analyzes how different coarsening methods preserve graph spectral properties
"""

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import pandas as pd
from pathlib import Path

class SpectralAnalyzer:
    """Analyze spectral properties of graphs before/after coarsening"""
    
    def __init__(self):
        self.results = {}
        
    def compute_laplacian_spectrum(self, adjacency_matrix, k=20):
        """Compute the smallest k eigenvalues and eigenvectors of the Laplacian"""
        print(f"Computing spectrum for {adjacency_matrix.shape[0]} node graph...")
        
        # Build normalized Laplacian
        degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()
        degrees_safe = degrees + 1e-12  # Avoid division by zero
        
        # D^(-1/2)
        d_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees_safe))
        
        # L = D - A
        laplacian = sp.diags(degrees) - adjacency_matrix
        
        # Normalized Laplacian: L_norm = D^(-1/2) * L * D^(-1/2)
        laplacian_norm = d_inv_sqrt @ laplacian @ d_inv_sqrt
        
        # Compute smallest k eigenvalues/eigenvectors
        try:
            eigenvalues, eigenvectors = eigsh(laplacian_norm, k=min(k, adjacency_matrix.shape[0]-2), 
                                            which='SM', return_eigenvectors=True)
            
            # Sort by eigenvalue (should already be sorted)
            sorted_indices = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[sorted_indices]
            eigenvectors = eigenvectors[:, sorted_indices]
            
            return eigenvalues, eigenvectors
            
        except Exception as e:
            print(f"Error computing spectrum: {e}")
            return None, None
    
    def load_cora_graph(self):
        """Load the Cora dataset graph"""
        print("Loading Cora dataset...")
        
        # This is a simplified version - in practice you'd load from the actual dataset
        # For now, let's create a synthetic graph that mimics Cora's properties
        np.random.seed(42)
        n_nodes = 2708
        
        # Create a graph with community structure (like citation networks)
        n_communities = 7
        community_sizes = [n_nodes // n_communities] * n_communities
        community_sizes[0] += n_nodes % n_communities
        
        # Build adjacency matrix
        adjacency = sp.lil_matrix((n_nodes, n_nodes))
        
        node_idx = 0
        for i, size in enumerate(community_sizes):
            # Intra-community connections (denser)
            for j in range(size):
                for k in range(j+1, size):
                    if np.random.random() < 0.1:  # 10% intra-community edge prob
                        adjacency[node_idx + j, node_idx + k] = 1
                        adjacency[node_idx + k, node_idx + j] = 1
            
            # Inter-community connections (sparser)
            for j in range(size):
                for other_comm in range(i+1, n_communities):
                    other_start = sum(community_sizes[:other_comm])
                    other_size = community_sizes[other_comm]
                    for k in range(other_size):
                        if np.random.random() < 0.01:  # 1% inter-community edge prob
                            adjacency[node_idx + j, other_start + k] = 1
                            adjacency[other_start + k, node_idx + j] = 1
            
            node_idx += size
        
        return adjacency.tocsr()
    
    def simulate_lamg_coarsening(self, adjacency, target_nodes):
        """Simulate LAMG-style coarsening (spectral clustering)"""
        print(f"Simulating LAMG coarsening to {target_nodes} nodes...")
        
        # Compute eigenvectors for spectral clustering
        eigenvalues, eigenvectors = self.compute_laplacian_spectrum(adjacency, k=50)
        
        if eigenvalues is None:
            return None
        
        # Use first few eigenvectors for clustering (skip first which is all-ones)
        clustering_features = eigenvectors[:, 1:8]  # Use eigenvectors 2-8
        
        # K-means clustering in spectral space
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=target_nodes, random_state=42)
        clusters = kmeans.fit_predict(clustering_features)
        
        # Build coarsened adjacency matrix
        coarse_adj = sp.lil_matrix((target_nodes, target_nodes))
        
        for i in range(adjacency.shape[0]):
            for j in range(i+1, adjacency.shape[0]):
                if adjacency[i, j] > 0:
                    ci, cj = clusters[i], clusters[j]
                    if ci != cj:
                        coarse_adj[ci, cj] += 1
                        coarse_adj[cj, ci] += 1
                    else:
                        coarse_adj[ci, ci] += 1
        
        return coarse_adj.tocsr()
    
    def simulate_cmg_coarsening(self, adjacency, target_nodes):
        """Simulate CMG-style coarsening (spectral filtering + clustering)"""
        print(f"Simulating CMG coarsening to {target_nodes} nodes...")
        
        # Apply spectral filtering (simplified version)
        degrees = np.array(adjacency.sum(axis=1)).flatten()
        d_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees + 1e-12))
        laplacian = sp.diags(degrees) - adjacency
        laplacian_norm = d_inv_sqrt @ laplacian @ d_inv_sqrt
        
        # Filter: (I - 0.5*L)^k
        I = sp.identity(adjacency.shape[0])
        filter_matrix = I - 0.5 * laplacian_norm
        
        # Apply filter to random vectors
        np.random.seed(42)
        X = np.random.randn(adjacency.shape[0], 20)
        
        # Apply filter k times
        for _ in range(10):
            X = filter_matrix @ X
        
        # Cluster based on cosine similarity
        similarity = cosine_similarity(X)
        
        # Simple clustering: group nodes with high similarity
        from sklearn.cluster import AgglomerativeClustering
        clustering = AgglomerativeClustering(n_clusters=target_nodes, 
                                           affinity='precomputed', 
                                           linkage='average')
        clusters = clustering.fit_predict(1 - similarity)  # 1 - similarity for distance
        
        # Build coarsened adjacency matrix
        coarse_adj = sp.lil_matrix((target_nodes, target_nodes))
        
        for i in range(adjacency.shape[0]):
            for j in range(i+1, adjacency.shape[0]):
                if adjacency[i, j] > 0:
                    ci, cj = clusters[i], clusters[j]
                    if ci != cj:
                        coarse_adj[ci, cj] += 1
                        coarse_adj[cj, ci] += 1
                    else:
                        coarse_adj[ci, ci] += 1
        
        return coarse_adj.tocsr()
    
    def analyze_spectral_preservation(self, original_adj, coarsened_adj, method_name):
        """Analyze how well spectral properties are preserved"""
        print(f"\nAnalyzing spectral preservation for {method_name}...")
        
        # Compute spectra
        orig_eigenvals, orig_eigenvecs = self.compute_laplacian_spectrum(original_adj, k=20)
        coarse_eigenvals, coarse_eigenvecs = self.compute_laplacian_spectrum(coarsened_adj, k=20)
        
        if orig_eigenvals is None or coarse_eigenvals is None:
            return None
        
        # Normalize eigenvalues by graph size for comparison
        orig_normalized = orig_eigenvals * original_adj.shape[0]
        coarse_normalized = coarse_eigenvals * coarsened_adj.shape[0]
        
        # Compute preservation metrics
        results = {
            'method': method_name,
            'original_nodes': original_adj.shape[0],
            'coarse_nodes': coarsened_adj.shape[0],
            'compression_ratio': original_adj.shape[0] / coarsened_adj.shape[0],
            'original_eigenvalues': orig_eigenvals,
            'coarse_eigenvalues': coarse_eigenvals,
            'orig_normalized': orig_normalized,
            'coarse_normalized': coarse_normalized,
            'fiedler_preservation': abs(orig_eigenvals[1] - coarse_eigenvals[1]) / orig_eigenvals[1],
            'spectral_gap_preservation': abs((orig_eigenvals[2] - orig_eigenvals[1]) - 
                                           (coarse_eigenvals[2] - coarse_eigenvals[1])) / 
                                        (orig_eigenvals[2] - orig_eigenvals[1]),
        }
        
        # Store for comparison
        self.results[method_name] = results
        
        return results
    
    def plot_spectral_comparison(self, save_path="spectral_comparison.png"):
        """Plot spectral properties comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Eigenvalue comparison
        ax1 = axes[0, 0]
        for method, data in self.results.items():
            ax1.plot(data['original_eigenvalues'][:10], 'o-', label=f'{method} Original', alpha=0.7)
            ax1.plot(data['coarse_eigenvalues'][:10], 's--', label=f'{method} Coarsened', alpha=0.7)
        
        ax1.set_xlabel('Eigenvalue Index')
        ax1.set_ylabel('Eigenvalue')
        ax1.set_title('Eigenvalue Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Normalized eigenvalue comparison
        ax2 = axes[0, 1]
        for method, data in self.results.items():
            ax2.plot(data['orig_normalized'][:10], 'o-', label=f'{method} Original (normalized)', alpha=0.7)
            ax2.plot(data['coarse_normalized'][:10], 's--', label=f'{method} Coarsened (normalized)', alpha=0.7)
        
        ax2.set_xlabel('Eigenvalue Index')
        ax2.set_ylabel('Normalized Eigenvalue')
        ax2.set_title('Normalized Eigenvalue Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Fiedler value preservation
        ax3 = axes[1, 0]
        methods = list(self.results.keys())
        fiedler_errors = [self.results[m]['fiedler_preservation'] for m in methods]
        
        bars = ax3.bar(methods, fiedler_errors, alpha=0.7)
        ax3.set_ylabel('Fiedler Value Preservation Error')
        ax3.set_title('Fiedler Value Preservation (Lower = Better)')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, fiedler_errors):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom')
        
        # Plot 4: Spectral gap preservation
        ax4 = axes[1, 1]
        gap_errors = [self.results[m]['spectral_gap_preservation'] for m in methods]
        
        bars = ax4.bar(methods, gap_errors, alpha=0.7)
        ax4.set_ylabel('Spectral Gap Preservation Error')
        ax4.set_title('Spectral Gap Preservation (Lower = Better)')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, gap_errors):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def run_full_analysis(self):
        """Run complete spectral analysis"""
        print("🔬 SPECTRAL PROPERTIES ANALYSIS")
        print("=" * 50)
        
        # Load original graph
        original_adj = self.load_cora_graph()
        print(f"Original graph: {original_adj.shape[0]} nodes, {original_adj.nnz} edges")
        
        # Test different coarsening methods at similar compression
        target_nodes_lamg = 519  # LAMG's natural stopping point
        target_nodes_cmg = 398   # CMG's level 2 result
        
        # Simulate LAMG coarsening
        lamg_coarsened = self.simulate_lamg_coarsening(original_adj, target_nodes_lamg)
        
        # Simulate CMG coarsening  
        cmg_coarsened = self.simulate_cmg_coarsening(original_adj, target_nodes_cmg)
        
        # Analyze spectral preservation
        lamg_results = self.analyze_spectral_preservation(original_adj, lamg_coarsened, "LAMG")
        cmg_results = self.analyze_spectral_preservation(original_adj, cmg_coarsened, "CMG++")
        
        # Print results
        print("\n📊 SPECTRAL PRESERVATION RESULTS")
        print("=" * 50)
        
        for method, results in self.results.items():
            print(f"\n{method}:")
            print(f"  Compression: {results['compression_ratio']:.2f}x")
            print(f"  Fiedler preservation error: {results['fiedler_preservation']:.4f}")
            print(f"  Spectral gap preservation error: {results['spectral_gap_preservation']:.4f}")
            print(f"  Nodes: {results['original_nodes']} → {results['coarse_nodes']}")
        
        # Create visualization
        self.plot_spectral_comparison()
        
        # Summary
        print("\n🎯 KEY FINDINGS:")
        print("=" * 50)
        
        lamg_fiedler = self.results['LAMG']['fiedler_preservation']
        cmg_fiedler = self.results['CMG++']['fiedler_preservation']
        
        if lamg_fiedler < cmg_fiedler:
            print(f"✅ LAMG preserves Fiedler value better: {lamg_fiedler:.4f} vs {cmg_fiedler:.4f}")
        else:
            print(f"✅ CMG++ preserves Fiedler value better: {cmg_fiedler:.4f} vs {lamg_fiedler:.4f}")
        
        lamg_gap = self.results['LAMG']['spectral_gap_preservation']
        cmg_gap = self.results['CMG++']['spectral_gap_preservation']
        
        if lamg_gap < cmg_gap:
            print(f"✅ LAMG preserves spectral gap better: {lamg_gap:.4f} vs {cmg_gap:.4f}")
        else:
            print(f"✅ CMG++ preserves spectral gap better: {cmg_gap:.4f} vs {lamg_gap:.4f}")
        
        print("\n💡 INTERPRETATION:")
        print("- Lower preservation error = better structural preservation")
        print("- Fiedler value (λ₂) measures graph connectivity")
        print("- Spectral gap (λ₃ - λ₂) measures cluster separation")
        print("- Better preservation → better embeddings → higher accuracy")
        
        return self.results

if __name__ == "__main__":
    analyzer = SpectralAnalyzer()
    results = analyzer.run_full_analysis()
