#!/usr/bin/env python3
"""
Matrix Construction Diagnostic Script
Diagnose why eigenvalue counting disagrees with NetworkX connectivity
"""

import numpy as np
import scipy.sparse as sp
import networkx as nx
import json
import torch
from torch_geometric.data import Data
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import matplotlib.pyplot as plt

class MatrixDiagnostic:
    def __init__(self):
        self.results = {}
    
    def load_cora_data(self):
        """Load Cora dataset and create PyG data object."""
        print("🔍 Loading Cora dataset...")
        
        with open("dataset/cora/cora-G.json", 'r') as f:
            data_json = json.load(f)
        
        edges = data_json['links']
        n_nodes = len(data_json['nodes'])
        
        print(f"   Original graph: {n_nodes} nodes, {len(edges)} edges")
        
        # Build original adjacency matrix
        A_orig = sp.lil_matrix((n_nodes, n_nodes))
        for edge in edges:
            i, j = edge['source'], edge['target']
            A_orig[i, j] = 1
            A_orig[j, i] = 1
        A_orig = A_orig.tocsr()
        
        # Check original connectivity
        G_orig = nx.from_scipy_sparse_matrix(A_orig)
        orig_components = nx.number_connected_components(G_orig)
        print(f"   Original connectivity: {orig_components} components")
        
        # Build edge_index for PyG
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
        
        return data, A_orig, n_nodes
    
    def run_cmg_step_by_step(self, data, A_orig, n_nodes, k=10, d=10):
        """Run CMG clustering with detailed diagnostics at each step."""
        print("\n🔬 Running CMG with step-by-step diagnostics...")
        
        # Import CMG
        from filtered import cmg_filtered_clustering
        
        # Run CMG clustering
        print(f"   Parameters: k={k}, d={d}")
        clusters, n_clusters, phi_stats, lambda_crit = cmg_filtered_clustering(
            data, k=k, d=d, threshold=0.1
        )
        
        print(f"   CMG result: {n_clusters} clusters, λ_critical={lambda_crit:.6f}")
        
        # Analyze cluster assignments
        self.analyze_cluster_assignments(clusters, n_clusters, n_nodes)
        
        # Build assignment matrix P step by step
        P = self.build_assignment_matrix_detailed(clusters, n_clusters, n_nodes)
        
        # Perform coarsening step by step
        A_coarse = self.perform_coarsening_detailed(A_orig, P, n_clusters)
        
        # Build Laplacian step by step
        L_coarse = self.build_laplacian_detailed(A_coarse)
        
        # Analyze final matrices
        self.analyze_final_matrices(A_coarse, L_coarse)
        
        return A_coarse, L_coarse, P, clusters
    
    def analyze_cluster_assignments(self, clusters, n_clusters, n_nodes):
        """Analyze the cluster assignments from CMG."""
        print("\n📊 Analyzing cluster assignments...")
        
        clusters = np.array(clusters)
        
        print(f"   Cluster range: {clusters.min()} to {clusters.max()}")
        print(f"   Expected clusters: 0 to {n_clusters-1}")
        print(f"   Total nodes: {len(clusters)} (should be {n_nodes})")
        
        # Check for missing or extra clusters
        unique_clusters = np.unique(clusters)
        expected_clusters = np.arange(n_clusters)
        
        missing = set(expected_clusters) - set(unique_clusters)
        extra = set(unique_clusters) - set(expected_clusters)
        
        if missing:
            print(f"   ⚠️  Missing clusters: {missing}")
        if extra:
            print(f"   ⚠️  Extra clusters: {extra}")
        
        # Cluster size distribution
        cluster_sizes = np.bincount(clusters, minlength=n_clusters)
        print(f"   Cluster sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}, mean={cluster_sizes.mean():.1f}")
        print(f"   Empty clusters: {np.sum(cluster_sizes == 0)}")
        
        # Show first few assignments
        print(f"   First 10 assignments: {clusters[:10].tolist()}")
        print(f"   Last 10 assignments: {clusters[-10:].tolist()}")
        
        return cluster_sizes
    
    def build_assignment_matrix_detailed(self, clusters, n_clusters, n_nodes):
        """Build assignment matrix with detailed diagnostics."""
        print("\n🔧 Building assignment matrix P...")
        
        clusters = np.array(clusters)
        
        # Build assignment matrix
        P = sp.lil_matrix((n_nodes, n_clusters))
        for i, cluster_id in enumerate(clusters):
            P[i, cluster_id] = 1.0
        P = P.tocsr()
        
        print(f"   P shape: {P.shape}")
        print(f"   P nnz: {P.nnz}")
        print(f"   Expected nnz: {n_nodes} (one entry per node)")
        
        # Check if P is proper one-hot
        row_sums = np.array(P.sum(axis=1)).flatten()
        col_sums = np.array(P.sum(axis=0)).flatten()
        
        print(f"   Row sums (should all be 1): min={row_sums.min()}, max={row_sums.max()}")
        print(f"   Col sums (cluster sizes): min={col_sums.min()}, max={col_sums.max()}")
        
        # Check for anomalies
        bad_rows = np.sum(row_sums != 1)
        empty_clusters = np.sum(col_sums == 0)
        
        if bad_rows > 0:
            print(f"   ❌ {bad_rows} nodes with incorrect assignments!")
            bad_indices = np.where(row_sums != 1)[0][:5]
            print(f"      First bad indices: {bad_indices}")
        
        if empty_clusters > 0:
            print(f"   ⚠️  {empty_clusters} empty clusters!")
        
        # Test P^T P (should be diagonal)
        PtP = P.T @ P
        if sp.issparse(PtP):
            PtP = PtP.toarray()
        
        # Check if P^T P is diagonal
        diagonal = np.diag(PtP)
        off_diagonal = PtP - np.diag(diagonal)
        max_off_diag = np.abs(off_diagonal).max()
        
        print(f"   P^T P diagonal: {diagonal[:5]}... (cluster sizes)")
        print(f"   P^T P max off-diagonal: {max_off_diag} (should be 0)")
        
        if max_off_diag > 1e-10:
            print(f"   ❌ P is not properly orthogonal!")
        
        return P
    
    def perform_coarsening_detailed(self, A_orig, P, n_clusters):
        """Perform graph coarsening with detailed diagnostics."""
        print("\n⚗️  Performing graph coarsening: A_coarse = P^T @ A_orig @ P...")
        
        print(f"   A_orig shape: {A_orig.shape}")
        print(f"   A_orig nnz: {A_orig.nnz}")
        print(f"   A_orig type: {type(A_orig)}")
        
        # Check A_orig properties
        A_orig_symmetric = np.allclose((A_orig - A_orig.T).data, 0)
        A_orig_binary = np.all((A_orig.data == 0) | (A_orig.data == 1))
        
        print(f"   A_orig symmetric: {A_orig_symmetric}")
        print(f"   A_orig binary: {A_orig_binary}")
        print(f"   A_orig data range: {A_orig.data.min()} to {A_orig.data.max()}")
        
        # Perform coarsening
        print("   Computing P^T @ A_orig...")
        temp = P.T @ A_orig
        print(f"   Temp shape: {temp.shape}")
        
        print("   Computing (P^T @ A_orig) @ P...")
        A_coarse = temp @ P
        print(f"   A_coarse shape: {A_coarse.shape}")
        print(f"   A_coarse nnz: {A_coarse.nnz}")
        
        # Clean up numerical errors
        A_coarse.eliminate_zeros()
        
        # Check A_coarse properties
        A_coarse_data = A_coarse.data
        print(f"   A_coarse data range: {A_coarse_data.min()} to {A_coarse_data.max()}")
        print(f"   A_coarse integer values: {np.allclose(A_coarse_data, np.round(A_coarse_data))}")
        
        # Check symmetry
        diff = A_coarse - A_coarse.T
        max_asymmetry = np.abs(diff.data).max() if diff.nnz > 0 else 0
        print(f"   A_coarse max asymmetry: {max_asymmetry}")
        
        if max_asymmetry > 1e-10:
            print("   🔧 Symmetrizing A_coarse...")
            A_coarse = (A_coarse + A_coarse.T) / 2
            A_coarse.eliminate_zeros()
        
        # Check for negative values
        negative_values = np.sum(A_coarse.data < 0)
        if negative_values > 0:
            print(f"   ⚠️  {negative_values} negative values found! (should be 0)")
            print(f"   Min value: {A_coarse.data.min()}")
            
            # Fix negative values
            print("   🔧 Setting negative values to zero...")
            A_coarse.data = np.maximum(A_coarse.data, 0)
            A_coarse.eliminate_zeros()
        
        # Check diagonal
        diagonal = A_coarse.diagonal()
        print(f"   A_coarse diagonal range: {diagonal.min()} to {diagonal.max()}")
        
        # Calculate actual connectivity using NetworkX
        if A_coarse.shape[0] < 2000:  # Only for reasonably sized matrices
            G_coarse = nx.from_scipy_sparse_matrix(A_coarse)
            true_components = nx.number_connected_components(G_coarse)
            print(f"   🎯 TRUE connectivity (NetworkX): {true_components} components")
        else:
            print("   ⚠️  Matrix too large for NetworkX connectivity check")
            true_components = -1
        
        return A_coarse
    
    def build_laplacian_detailed(self, A_coarse):
        """Build Laplacian matrix with detailed diagnostics."""
        print("\n🏗️  Building Laplacian: L = D - A...")
        
        # Calculate degrees
        degrees = np.array(A_coarse.sum(axis=1)).flatten()
        print(f"   Degree range: {degrees.min()} to {degrees.max()}")
        print(f"   Isolated nodes (degree 0): {np.sum(degrees == 0)}")
        
        if np.any(degrees < 0):
            print(f"   ❌ Negative degrees found! Min: {degrees.min()}")
        
        # Build degree matrix
        D = sp.diags(degrees)
        print(f"   D shape: {D.shape}")
        print(f"   D nnz: {D.nnz}")
        
        # Build Laplacian
        L = D - A_coarse
        print(f"   L shape: {L.shape}")
        print(f"   L nnz: {L.nnz}")
        
        # Check Laplacian properties
        L_diagonal = L.diagonal()
        print(f"   L diagonal range: {L_diagonal.min()} to {L_diagonal.max()}")
        print(f"   L diagonal equals degrees: {np.allclose(L_diagonal, degrees)}")
        
        # Check row sums (should be 0 for Laplacian)
        row_sums = np.array(L.sum(axis=1)).flatten()
        max_row_sum = np.abs(row_sums).max()
        print(f"   L row sums (should be 0): max |sum| = {max_row_sum}")
        
        if max_row_sum > 1e-10:
            print(f"   ❌ Laplacian row sums are not zero!")
        
        # Check symmetry
        diff = L - L.T
        max_asymmetry = np.abs(diff.data).max() if diff.nnz > 0 else 0
        print(f"   L max asymmetry: {max_asymmetry}")
        
        return L
    
    def analyze_final_matrices(self, A_coarse, L_coarse):
        """Analyze final coarsened matrices."""
        print("\n🔬 Final matrix analysis...")
        
        # NetworkX connectivity (ground truth)
        if A_coarse.shape[0] < 2000:
            G = nx.from_scipy_sparse_matrix(A_coarse)
            true_components = nx.number_connected_components(G)
            print(f"   🎯 NetworkX components: {true_components}")
            
            # Show component sizes
            components = list(nx.connected_components(G))
            component_sizes = sorted([len(c) for c in components], reverse=True)
            print(f"   Component sizes: {component_sizes[:10]}{'...' if len(component_sizes) > 10 else ''}")
        else:
            true_components = -1
            print("   ⚠️  Matrix too large for NetworkX analysis")
        
        # Eigenvalue analysis
        print("\n   🧮 Computing eigenvalues...")
        try:
            # Try to compute first 20 eigenvalues
            k = min(20, L_coarse.shape[0] - 2)
            eigenvals, _ = eigsh(L_coarse, k=k, which='SM', sigma=0.0)
            eigenvals = np.sort(eigenvals)
            
            print(f"   First 12 eigenvalues:")
            for i, ev in enumerate(eigenvals[:12]):
                print(f"      λ_{i}: {ev:.2e}")
            
            # Count zero eigenvalues with different thresholds
            thresholds = [1e-16, 1e-14, 1e-12, 1e-10, 1e-8]
            for thresh in thresholds:
                zero_count = np.sum(np.abs(eigenvals) < thresh)
                print(f"   Zero eigenvalues (threshold {thresh:.0e}): {zero_count}")
            
            # Find the right threshold
            if true_components > 0:
                print(f"\n   🎯 Should have {true_components} zero eigenvalues")
                
                # Check if we can find the right threshold
                sorted_eigenvals = np.sort(np.abs(eigenvals))
                if len(sorted_eigenvals) >= true_components:
                    gap_after_zeros = sorted_eigenvals[true_components] - sorted_eigenvals[true_components-1]
                    suggested_threshold = sorted_eigenvals[true_components-1] * 10
                    print(f"   Gap after {true_components} eigenvalues: {gap_after_zeros:.2e}")
                    print(f"   Suggested threshold: {suggested_threshold:.2e}")
        
        except Exception as e:
            print(f"   ❌ Eigenvalue computation failed: {e}")
            
            # Try dense computation for small matrices
            if L_coarse.shape[0] < 500:
                print("   🔄 Trying dense eigenvalue computation...")
                try:
                    L_dense = L_coarse.toarray()
                    eigenvals_dense = eigh(L_dense, eigvals_only=True)
                    eigenvals_dense = np.sort(eigenvals_dense)
                    
                    print(f"   Dense computation - first 12 eigenvalues:")
                    for i, ev in enumerate(eigenvals_dense[:12]):
                        print(f"      λ_{i}: {ev:.2e}")
                    
                    # Count zeros with different thresholds
                    for thresh in [1e-16, 1e-14, 1e-12, 1e-10, 1e-8]:
                        zero_count = np.sum(np.abs(eigenvals_dense) < thresh)
                        print(f"   Zero eigenvalues (threshold {thresh:.0e}): {zero_count}")
                
                except Exception as e2:
                    print(f"   ❌ Dense computation also failed: {e2}")
    
    def plot_eigenvalue_analysis(self, eigenvals, true_components):
        """Plot eigenvalue analysis."""
        if len(eigenvals) == 0:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Eigenvalue magnitude
        ax1 = axes[0]
        ax1.semilogy(range(len(eigenvals)), np.abs(eigenvals), 'o-')
        ax1.axhline(y=1e-12, color='r', linestyle='--', label='Threshold 1e-12')
        ax1.axhline(y=1e-10, color='orange', linestyle='--', label='Threshold 1e-10')
        if true_components > 0:
            ax1.axvline(x=true_components-0.5, color='green', linestyle='-', 
                       label=f'Expected zeros: {true_components}')
        ax1.set_xlabel('Eigenvalue Index')
        ax1.set_ylabel('|Eigenvalue|')
        ax1.set_title('Eigenvalue Magnitudes')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Eigenvalue gaps
        ax2 = axes[1]
        if len(eigenvals) > 1:
            gaps = np.diff(np.abs(eigenvals))
            ax2.semilogy(range(len(gaps)), gaps, 'o-')
            if true_components > 0 and true_components < len(gaps):
                ax2.axvline(x=true_components-1, color='green', linestyle='-',
                           label=f'Expected gap at: {true_components-1}')
        ax2.set_xlabel('Gap Index')
        ax2.set_ylabel('Eigenvalue Gap')
        ax2.set_title('Eigenvalue Gaps')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('eigenvalue_diagnostic.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def run_full_diagnostic(self):
        """Run complete diagnostic analysis."""
        print("🚀 MATRIX CONSTRUCTION DIAGNOSTIC")
        print("="*50)
        
        # Load data
        data, A_orig, n_nodes = self.load_cora_data()
        
        # Run CMG with diagnostics
        A_coarse, L_coarse, P, clusters = self.run_cmg_step_by_step(data, A_orig, n_nodes)
        
        print("\n" + "="*50)
        print("📊 DIAGNOSTIC SUMMARY")
        print("="*50)
        
        # Final verification
        if A_coarse.shape[0] < 2000:
            G = nx.from_scipy_sparse_matrix(A_coarse)
            true_components = nx.number_connected_components(G)
            print(f"✅ TRUE connectivity: {true_components} components")
        else:
            true_components = -1
        
        # Try final eigenvalue computation
        try:
            k = min(15, L_coarse.shape[0] - 2)
            eigenvals, _ = eigsh(L_coarse, k=k, which='SM', sigma=0.0)
            eigenvals = np.sort(eigenvals)
            
            print(f"✅ Computed {len(eigenvals)} eigenvalues")
            
            # Test different thresholds
            if true_components > 0:
                for thresh in [1e-16, 1e-14, 1e-12, 1e-10]:
                    zero_count = np.sum(np.abs(eigenvals) < thresh)
                    match = "✅" if zero_count == true_components else "❌"
                    print(f"{match} Threshold {thresh:.0e}: {zero_count} zeros (expected {true_components})")
            
            # Plot analysis
            if true_components > 0:
                self.plot_eigenvalue_analysis(eigenvals, true_components)
        
        except Exception as e:
            print(f"❌ Final eigenvalue computation failed: {e}")
        
        print("\n🎯 RECOMMENDATIONS:")
        if true_components > 0:
            print(f"1. Use NetworkX as ground truth: {true_components} components")
            print("2. Fix eigenvalue threshold based on gap analysis")
            print("3. Verify Laplacian construction if eigenvalues still disagree")
        else:
            print("1. Matrix too large - use sampling or different approach")
            print("2. Focus on NetworkX connectivity for validation")
        
        return {
            'true_components': true_components,
            'A_coarse': A_coarse,
            'L_coarse': L_coarse,
            'P': P,
            'clusters': clusters
        }

if __name__ == "__main__":
    diagnostic = MatrixDiagnostic()
    results = diagnostic.run_full_diagnostic()
