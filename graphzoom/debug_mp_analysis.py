#!/usr/bin/env python3
"""
Debug analysis: Why is naive better than MP-aware?
Let's examine the matrices and clustering quality
"""

import numpy as np
import scipy.sparse as sp
import networkx as nx
from scipy.sparse import csr_matrix, diags
from scipy.io import mmread
import matplotlib.pyplot as plt

def create_test_graph():
    """Create our standard 12-node test graph"""
    edges = [
        (0, 1), (1, 2), (1, 4), (2, 3), (3, 4), (4, 9),
        (4, 5), (5, 6), (5, 7), (6, 8), (8, 9),
        (9, 10), (9, 11), (10, 11)
    ]
    
    G = nx.Graph()
    G.add_edges_from(edges)
    
    # Ensure all nodes 0-11 are present
    for i in range(12):
        if i not in G.nodes():
            G.add_node(i)
    
    return G

def analyze_cmg_clustering(Q, Q_plus):
    """Analyze the quality of CMG clustering"""
    print("CMG CLUSTERING ANALYSIS")
    print("=" * 50)
    
    print(f"Coarsening matrix Q shape: {Q.shape}")
    print(f"Number of original nodes: {Q.shape[1]}")
    print(f"Number of coarse nodes: {Q.shape[0]}")
    print(f"Reduction ratio: {Q.shape[1] / Q.shape[0]:.2f}x")
    
    # Show cluster assignments
    print(f"\nCluster assignments:")
    for coarse_id in range(Q.shape[0]):
        # Find which original nodes belong to this cluster
        original_nodes = []
        for orig_id in range(Q.shape[1]):
            if Q[coarse_id, orig_id] > 0:
                original_nodes.append(orig_id)
        print(f"  Cluster {coarse_id}: nodes {original_nodes} (size: {len(original_nodes)})")
    
    # Check if clusters are balanced
    cluster_sizes = []
    for coarse_id in range(Q.shape[0]):
        size = np.sum(Q[coarse_id, :] > 0)
        cluster_sizes.append(size)
    
    print(f"\nCluster size distribution:")
    print(f"  Min size: {min(cluster_sizes)}")
    print(f"  Max size: {max(cluster_sizes)}")
    print(f"  Mean size: {np.mean(cluster_sizes):.1f}")
    print(f"  Std size: {np.std(cluster_sizes):.1f}")

def analyze_propagation_matrices(S_original, S_c_MP, S_c_naive, A_original, A_c):
    """Analyze the propagation matrices in detail"""
    print("\nPROPAGATION MATRIX ANALYSIS")
    print("=" * 50)
    
    # Original matrix stats
    print(f"Original adjacency A:")
    print(f"  Shape: {A_original.shape}")
    print(f"  Edges: {A_original.nnz}")
    print(f"  Density: {A_original.nnz / A_original.size:.3f}")
    
    print(f"\nOriginal propagation S:")
    print(f"  Shape: {S_original.shape}")
    print(f"  Non-zeros: {S_original.nnz}")
    print(f"  Density: {S_original.nnz / S_original.size:.3f}")
    print(f"  Min value: {S_original.data.min():.6f}")
    print(f"  Max value: {S_original.data.max():.6f}")
    
    # Coarsened adjacency
    print(f"\nCoarsened adjacency A_c:")
    print(f"  Shape: {A_c.shape}")
    print(f"  Edges: {A_c.nnz}")
    print(f"  Density: {A_c.nnz / A_c.size:.3f}")
    print(f"  Values range: [{A_c.data.min():.3f}, {A_c.data.max():.3f}]")
    
    # MP-aware matrix
    print(f"\nMP-Aware S_c^MP:")
    print(f"  Shape: {S_c_MP.shape}")
    print(f"  Non-zeros: {S_c_MP.nnz}")
    print(f"  Density: {S_c_MP.nnz / S_c_MP.size:.3f}")
    print(f"  Values range: [{S_c_MP.data.min():.6f}, {S_c_MP.data.max():.6f}]")
    
    # Naive matrix
    print(f"\nNaive S_c:")
    print(f"  Shape: {S_c_naive.shape}")
    print(f"  Non-zeros: {S_c_naive.nnz}")
    print(f"  Density: {S_c_naive.nnz / S_c_naive.size:.3f}")
    print(f"  Values range: [{S_c_naive.data.min():.6f}, {S_c_naive.data.max():.6f}]")

def print_matrix(matrix, name, max_size=8):
    """Print small matrices for inspection"""
    if matrix.shape[0] <= max_size and matrix.shape[1] <= max_size:
        if sp.issparse(matrix):
            dense = matrix.toarray()
        else:
            dense = matrix
        
        print(f"\n{name} matrix ({dense.shape}):")
        for i, row in enumerate(dense):
            print(f"  Row {i}: " + " ".join([f"{val:7.3f}" for val in row]))

def compute_propagation_matrix_gcn(adj_matrix):
    """Compute GCN propagation matrix with detailed steps"""
    print(f"\nGCN PROPAGATION MATRIX COMPUTATION")
    print("-" * 40)
    
    print(f"Step 1: Add self-loops")
    A_self = adj_matrix + sp.identity(adj_matrix.shape[0])
    print(f"  A + I shape: {A_self.shape}, nnz: {A_self.nnz}")
    
    print(f"Step 2: Compute degrees")
    degrees = np.array(A_self.sum(axis=1)).flatten()
    print(f"  Degrees: {degrees}")
    
    print(f"Step 3: Compute D^(-1/2)")
    degrees_safe = degrees + 1e-12
    D_inv_sqrt_diag = 1.0 / np.sqrt(degrees_safe)
    print(f"  D^(-1/2) diagonal: {D_inv_sqrt_diag}")
    
    D_inv_sqrt = sp.diags(D_inv_sqrt_diag)
    S = D_inv_sqrt @ A_self @ D_inv_sqrt
    
    print(f"Step 4: Final S = D^(-1/2) (A + I) D^(-1/2)")
    print(f"  Final S shape: {S.shape}, nnz: {S.nnz}")
    
    return S

def main():
    """Debug analysis of MP-aware vs naive results"""
    print("DEBUG ANALYSIS: MP-AWARE vs NAIVE")
    print("=" * 60)
    
    # Load data
    G = create_test_graph()
    A = nx.adjacency_matrix(G).tocsr()
    Q = mmread("experiment_results/cmg_projection.mtx").tocsr().T
    Q_plus = Q.T
    
    # Analyze clustering
    analyze_cmg_clustering(Q, Q_plus)
    
    # Compute matrices with detailed analysis
    print(f"\nDETAILED MATRIX COMPUTATION")
    print("=" * 50)
    
    # Original GCN propagation
    S_original = compute_propagation_matrix_gcn(A)
    
    # MP-aware: S_c^MP = Q S Q^+
    print(f"\nMP-AWARE computation: S_c^MP = Q @ S @ Q^+")
    S_c_MP = Q @ S_original @ Q_plus
    print(f"Result: {S_c_MP.shape} matrix with {S_c_MP.nnz} non-zeros")
    
    # Naive: A_c = Q^+^T A Q^+, then S_c = f(A_c)
    print(f"\nNAIVE computation: A_c = Q^+^T @ A @ Q^+, then S_c = f(A_c)")
    A_c = Q_plus.T @ A @ Q_plus
    S_c_naive = compute_propagation_matrix_gcn(A_c)
    print(f"Result: {S_c_naive.shape} matrix with {S_c_naive.nnz} non-zeros")
    
    # Detailed analysis
    analyze_propagation_matrices(S_original, S_c_MP, S_c_naive, A, A_c)
    
    # Print small matrices for inspection
    print_matrix(A, "Original adjacency A")
    print_matrix(A_c, "Coarsened adjacency A_c") 
    print_matrix(S_c_MP, "MP-aware S_c^MP")
    print_matrix(S_c_naive, "Naive S_c")
    
    # Analyze why MP-aware might be worse
    print(f"\nWHY MP-AWARE MIGHT PERFORM WORSE")
    print("=" * 50)
    print("1. Both matrices are dense (density = 1.0)")
    print("2. Small graph + aggressive coarsening → all nodes connected")
    print("3. MP-aware preserves original propagation weights")
    print("4. Naive recomputes weights based on coarsened structure")
    print("5. For small dense graphs, recomputing might be more appropriate")
    
    # Theoretical insight
    print(f"\nTHEORETICAL INSIGHT")
    print("=" * 50)
    print("The paper's S_c^MP = Q S Q^+ preserves the MESSAGE PASSING,")
    print("but it might not preserve the NORMALIZATION appropriately.")
    print("On small dense graphs, this could lead to over-weighted connections.")
    print("The naive method recomputes normalization on the coarsened graph,")
    print("which might be more appropriate for this scale.")

if __name__ == "__main__":
    main()
