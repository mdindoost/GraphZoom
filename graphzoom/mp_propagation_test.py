#!/usr/bin/env python3
"""
Simple test to compare naive coarsening vs message-passing aware coarsening
Tests the S_c^MP = Q S Q^+ approach from the paper
"""

import numpy as np
import scipy.sparse as sp
import networkx as nx
from scipy.sparse import csr_matrix, diags
from scipy.io import mmwrite, mmread
import os

def create_test_graph():
    """Create the same 12-node test graph from before"""
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

def load_coarsening_matrices():
    """Load Q and Q^+ from previous experiment"""
    try:
        # Load CMG coarsening results
        cmg_projection = mmread("experiment_results/cmg_projection.mtx").tocsr()
        
        print("Loaded CMG coarsening matrices:")
        print(f"Projection matrix shape: {cmg_projection.shape}")
        print(f"Q^+ (projection): {cmg_projection.shape[0]} nodes → {cmg_projection.shape[1]} clusters")
        
        # Q^+ is the projection matrix (nodes → clusters)
        Q_plus = cmg_projection
        
        # Q is the coarsening matrix (clusters → nodes)
        # For uniform coarsening: Q[cluster, node] = 1/cluster_size if node in cluster
        Q = create_coarsening_matrix_from_projection(Q_plus)
        
        return Q, Q_plus
        
    except FileNotFoundError:
        print("Coarsening matrices not found. Creating simple example...")
        return create_simple_coarsening_example()

def create_coarsening_matrix_from_projection(Q_plus):
    """Create Q matrix from Q^+ assuming uniform coarsening"""
    n_clusters = Q_plus.shape[1]
    n_nodes = Q_plus.shape[0]
    
    # Count nodes per cluster
    cluster_sizes = np.array(Q_plus.sum(axis=0)).flatten()
    
    # Create Q matrix
    Q_data = []
    Q_row = []
    Q_col = []
    
    for cluster_id in range(n_clusters):
        cluster_size = cluster_sizes[cluster_id]
        if cluster_size > 0:
            # Find nodes in this cluster
            nodes_in_cluster = Q_plus[:, cluster_id].nonzero()[0]
            for node_id in nodes_in_cluster:
                Q_data.append(1.0 / cluster_size)  # Uniform weight
                Q_row.append(cluster_id)
                Q_col.append(node_id)
    
    Q = csr_matrix((Q_data, (Q_row, Q_col)), shape=(n_clusters, n_nodes))
    return Q

def create_simple_coarsening_example():
    """Create simple coarsening for testing"""
    # 6 nodes → 3 clusters: {0,1}, {2,3}, {4,5}
    n_nodes = 6
    n_clusters = 3
    
    # Q^+ matrix (nodes → clusters)
    Q_plus_data = [1, 1, 1, 1, 1, 1]
    Q_plus_row = [0, 1, 2, 3, 4, 5]
    Q_plus_col = [0, 0, 1, 1, 2, 2]
    Q_plus = csr_matrix((Q_plus_data, (Q_plus_row, Q_plus_col)), shape=(n_nodes, n_clusters))
    
    # Q matrix (clusters → nodes)  
    Q_data = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    Q_row = [0, 0, 1, 1, 2, 2]
    Q_col = [0, 1, 2, 3, 4, 5]
    Q = csr_matrix((Q_data, (Q_row, Q_col)), shape=(n_clusters, n_nodes))
    
    return Q, Q_plus

def compute_propagation_matrices(adjacency, gnn_type='gcn'):
    """Compute different GNN propagation matrices"""
    
    if gnn_type == 'gcn':
        # GCN: S = D^(-1/2) (A + I) D^(-1/2)
        A_self = adjacency + sp.identity(adjacency.shape[0])
        degrees = np.array(A_self.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1  # Avoid division by zero
        D_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees))
        S = D_inv_sqrt @ A_self @ D_inv_sqrt
        
    elif gnn_type == 'graphsage':
        # GraphSAGE: S = D^(-1) A (mean aggregation)
        degrees = np.array(adjacency.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1  # Avoid division by zero
        D_inv = sp.diags(1.0 / degrees)
        S = D_inv @ adjacency
        
    elif gnn_type == 'raw':
        # Raw adjacency
        S = adjacency.copy()
        
    else:
        raise ValueError(f"Unknown GNN type: {gnn_type}")
    
    return S

def compute_naive_coarsened_propagation(Q_plus, S_original, gnn_type='gcn'):
    """Naive approach: coarsen adjacency then compute propagation matrix"""
    
    # Extract original adjacency from S (approximate)
    if gnn_type == 'gcn':
        # For GCN, S ≈ D^(-1/2) A D^(-1/2), hard to extract A exactly
        # Use simple approximation
        A_original = S_original.copy()
    elif gnn_type == 'graphsage':
        # For GraphSAGE, S = D^(-1) A, so A ≈ D S
        degrees = np.array(S_original.sum(axis=1)).flatten()
        D = sp.diags(degrees)
        A_original = D @ S_original
    else:
        A_original = S_original.copy()
    
    # Coarsen adjacency: A_c = (Q^+)^T A Q^+
    A_coarsened = Q_plus.T @ A_original @ Q_plus
    
    # Compute propagation matrix on coarsened graph
    S_naive = compute_propagation_matrices(A_coarsened, gnn_type)
    
    return S_naive, A_coarsened

def compute_mp_aware_propagation(Q, Q_plus, S_original):
    """Message-passing aware approach: S_c^MP = Q S Q^+"""
    S_c_MP = Q @ S_original @ Q_plus
    return S_c_MP

def compare_propagation_methods(Q, Q_plus, S_original, gnn_type='gcn'):
    """Compare naive vs message-passing aware propagation"""
    
    print(f"\n{'='*60}")
    print(f"COMPARING PROPAGATION METHODS - {gnn_type.upper()}")
    print(f"{'='*60}")
    
    print(f"Original graph: {S_original.shape[0]} nodes")
    print(f"Coarsened graph: {Q.shape[0]} clusters")
    print(f"Q matrix shape: {Q.shape}")
    print(f"Q^+ matrix shape: {Q_plus.shape}")
    
    # Method 1: Naive coarsening
    S_naive, A_coarsened = compute_naive_coarsened_propagation(Q_plus, S_original, gnn_type)
    
    # Method 2: Message-passing aware
    S_c_MP = compute_mp_aware_propagation(Q, Q_plus, S_original)
    
    print(f"\nNaive S_c shape: {S_naive.shape}")
    print(f"MP-aware S_c^MP shape: {S_c_MP.shape}")
    
    # Compare matrix properties
    print(f"\nMatrix Properties:")
    print(f"Naive S_c - symmetric: {is_symmetric(S_naive)}")
    print(f"MP S_c^MP - symmetric: {is_symmetric(S_c_MP)}")
    
    print(f"Naive S_c - sum of rows: {np.array(S_naive.sum(axis=1)).flatten()}")
    print(f"MP S_c^MP - sum of rows: {np.array(S_c_MP.sum(axis=1)).flatten()}")
    
    # Test message passing preservation
    test_message_passing_preservation(Q, Q_plus, S_original, S_naive, S_c_MP)
    
    return S_naive, S_c_MP

def is_symmetric(matrix, tol=1e-10):
    """Check if matrix is symmetric"""
    if matrix.shape[0] != matrix.shape[1]:
        return False
    diff = matrix - matrix.T
    return np.abs(diff.data).max() < tol

def test_message_passing_preservation(Q, Q_plus, S_original, S_naive, S_c_MP):
    """Test how well each method preserves message passing"""
    
    print(f"\nMessage Passing Preservation Test:")
    print("-" * 40)
    
    # Create test signals
    n_nodes = S_original.shape[0]
    n_tests = 3
    
    for i in range(n_tests):
        # Random test signal
        np.random.seed(i)
        x = np.random.randn(n_nodes)
        
        # Original message passing
        Sx_original = S_original @ x
        
        # Coarsen signal
        x_c = Q @ x
        
        # Method 1: Naive propagation + lift
        Sx_naive_lifted = Q_plus @ (S_naive @ x_c)
        
        # Method 2: MP-aware propagation + lift  
        Sx_mp_lifted = Q_plus @ (S_c_MP @ x_c)
        
        # Compute errors
        error_naive = np.linalg.norm(Sx_original - Sx_naive_lifted)
        error_mp = np.linalg.norm(Sx_original - Sx_mp_lifted)
        
        print(f"Test {i+1}:")
        print(f"  Naive method error:    {error_naive:.6f}")
        print(f"  MP-aware method error: {error_mp:.6f}")
        print(f"  Improvement ratio:     {error_naive/error_mp:.2f}x")

def print_matrix_sample(matrix, title, max_size=5):
    """Print a sample of the matrix for inspection"""
    print(f"\n{title}:")
    if matrix.shape[0] <= max_size and matrix.shape[1] <= max_size:
        if sp.issparse(matrix):
            dense = matrix.toarray()
        else:
            dense = matrix
        
        for i, row in enumerate(dense):
            print(f"  Row {i}: " + " ".join([f"{val:6.3f}" for val in row]))
    else:
        print(f"  Shape: {matrix.shape} (too large to print)")
        print(f"  nnz: {matrix.nnz if sp.issparse(matrix) else 'dense'}")

def main():
    """Run the propagation matrix comparison test"""
    
    print("MESSAGE-PASSING AWARE COARSENING TEST")
    print("="*60)
    
    # Load coarsening matrices
    Q, Q_plus = load_coarsening_matrices()
    
    # Print matrix samples
    print_matrix_sample(Q, "Q (Coarsening Matrix)", max_size=6)
    print_matrix_sample(Q_plus, "Q^+ (Lifting Matrix)", max_size=6)
    
    # Create test graph
    G = create_test_graph()
    A = nx.adjacency_matrix(G).tocsr()
    
    # Test different GNN types
    gnn_types = ['gcn', 'graphsage']
    
    results = {}
    
    for gnn_type in gnn_types:
        # Compute original propagation matrix
        S_original = compute_propagation_matrices(A, gnn_type)
        
        print_matrix_sample(S_original, f"Original {gnn_type.upper()} Propagation Matrix", max_size=6)
        
        # Compare methods
        S_naive, S_c_MP = compare_propagation_methods(Q, Q_plus, S_original, gnn_type)
        
        results[gnn_type] = {
            'S_original': S_original,
            'S_naive': S_naive, 
            'S_c_MP': S_c_MP
        }
        
        print_matrix_sample(S_naive, f"Naive {gnn_type.upper()} Coarsened Matrix", max_size=4)
        print_matrix_sample(S_c_MP, f"MP-Aware {gnn_type.upper()} Matrix", max_size=4)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print("Key observations:")
    print("1. MP-aware matrices (S_c^MP) are often asymmetric")
    print("2. MP-aware method should show lower message passing errors")
    print("3. The improvement depends on the coarsening quality")
    print("4. Different GNN types benefit differently from MP-aware coarsening")
    
    # Save results for further analysis
    os.makedirs("mp_test_results", exist_ok=True)
    for gnn_type, matrices in results.items():
        mmwrite(f"mp_test_results/{gnn_type}_S_naive.mtx", matrices['S_naive'])
        mmwrite(f"mp_test_results/{gnn_type}_S_MP.mtx", matrices['S_c_MP'])
    
    print(f"\nResults saved to mp_test_results/ directory")

if __name__ == "__main__":
    main()
