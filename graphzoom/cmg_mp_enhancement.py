#!/usr/bin/env python3
"""
CMG Message-Passing Enhancement
Implements the S_c^MP = Q S Q^+ approach from the paper to improve CMG performance with GNNs
"""

import numpy as np
import scipy.sparse as sp
import networkx as nx
from scipy.sparse import csr_matrix, diags
import time

def compute_mp_propagation_matrix(Q, Q_plus, original_adj, gnn_type='gcn', add_self_loops=True):
    """
    Compute message-passing aware propagation matrix for coarsened graph.
    
    This is the key innovation from the paper: instead of computing S_c = f_S(A_c),
    we compute S_c^MP = Q S Q^+ which preserves message passing better.
    
    Args:
        Q: Coarsening matrix (n_coarse, n_original)  
        Q_plus: Lifting matrix (n_original, n_coarse)
        original_adj: Original adjacency matrix
        gnn_type: 'gcn', 'graphsage_mean', 'sage_mean'
        add_self_loops: Whether to add self-loops before normalization
    
    Returns:
        S_c_MP: Message-passing aware propagation matrix for coarsened graph
        S_original: Original propagation matrix (for comparison)
    """
    
    print(f"[MP] Computing message-passing matrix for {gnn_type}")
    
    # Add self-loops if requested (common in GCN)
    if add_self_loops:
        A_with_loops = original_adj + sp.identity(original_adj.shape[0])
    else:
        A_with_loops = original_adj.copy()
    
    # Compute original propagation matrix based on GNN type
    if gnn_type.lower() in ['gcn', 'gcnconv']:
        # S = D^(-1/2) (A + I) D^(-1/2) - symmetric normalization
        degrees = np.array(A_with_loops.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1  # Avoid division by zero
        D_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees))
        S_original = D_inv_sqrt @ A_with_loops @ D_inv_sqrt
        
    elif gnn_type.lower() in ['graphsage', 'sage_mean', 'graphsage_mean']:
        # S = D^(-1) A - row normalization (mean aggregation)
        degrees = np.array(A_with_loops.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1  # Avoid division by zero
        D_inv = sp.diags(1.0 / degrees)
        S_original = D_inv @ A_with_loops
        
    elif gnn_type.lower() in ['raw', 'none']:
        # S = A - no normalization
        S_original = A_with_loops
        
    else:
        raise ValueError(f"Unknown GNN type: {gnn_type}. Supported: 'gcn', 'graphsage_mean', 'raw'")
    
    # THE KEY STEP: S_c^MP = Q S Q^+
    # This preserves message passing better than naive S_c = f_S(A_c)
    S_c_MP = Q @ S_original @ Q_plus
    
    print(f"[MP] Original S shape: {S_original.shape}, Coarsened S_MP shape: {S_c_MP.shape}")
    print(f"[MP] S_MP sparsity: {S_c_MP.nnz}/{S_c_MP.shape[0]*S_c_MP.shape[1]} = {S_c_MP.nnz/(S_c_MP.shape[0]*S_c_MP.shape[1]):.3f}")
    
    # Check if result is asymmetric (expected from paper)
    if S_c_MP.shape[0] == S_c_MP.shape[1]:
        asymmetry = np.abs((S_c_MP - S_c_MP.T).data).max() if S_c_MP.nnz > 0 else 0
        print(f"[MP] Asymmetry measure: {asymmetry:.6f} (>0 means directed)")
    
    return S_c_MP, S_original


def cmg_coarse_with_mp(laplacian, levels, gnn_type='gcn', k=10, d=20, threshold=0.1):
    """
    Enhanced CMG coarsening that returns both standard and message-passing aware matrices.
    
    Args:
        laplacian: Original Laplacian matrix
        levels: Number of coarsening levels
        gnn_type: Type of GNN for message passing ('gcn', 'graphsage_mean')
        k, d, threshold: CMG parameters
    
    Returns:
        G: Final coarsened NetworkX graph
        projections: List of projection matrices 
        laplacians: List of Laplacian matrices at each level
        levels: Number of levels achieved
        mp_matrices: List of message-passing aware propagation matrices
        comparison_data: Data for comparing naive vs MP approach
    """
    
    print(f"[CMG-MP] Starting enhanced CMG coarsening for {gnn_type}")
    
    # Import your existing CMG function
    from cmg_coarsening_timed import cmg_coarse
    
    # Run standard CMG coarsening first
    G, projections, laplacians, actual_levels = cmg_coarse(laplacian, levels, k, d, threshold)
    
    print(f"[CMG-MP] Standard CMG completed: {actual_levels} levels")
    
    # Extract original adjacency matrix from Laplacian
    degree_diag = diags(laplacian.diagonal(), 0)
    original_adj = degree_diag - laplacian
    
    # Ensure non-negative (handle numerical issues)
    original_adj.data = np.abs(original_adj.data)
    
    # Compute message-passing matrices for each level
    mp_matrices = []
    naive_matrices = []
    comparison_data = {'mp_errors': [], 'naive_errors': [], 'improvements': []}
    
    current_adj = original_adj.copy()
    
    for i, projection in enumerate(projections):
        print(f"\n[CMG-MP] Processing level {i+1}/{len(projections)}")
        
        # Get projection matrices (Q and Q^+)
        Q = projection.T  # coarse-to-fine mapping
        Q_plus = projection  # fine-to-coarse mapping
        
        print(f"[CMG-MP] Q shape: {Q.shape}, Q^+ shape: {Q_plus.shape}")
        
        # Compute MP-aware propagation matrix
        S_c_MP, S_original = compute_mp_propagation_matrix(Q, Q_plus, current_adj, gnn_type)
        mp_matrices.append(S_c_MP)
        
        # For comparison: compute naive approach
        A_c_naive = Q_plus.T @ current_adj @ Q_plus  # Standard coarsened adjacency
        
        if gnn_type.lower() in ['gcn', 'gcnconv']:
            # Naive: apply GCN normalization to coarsened adjacency
            A_c_naive_loops = A_c_naive + sp.identity(A_c_naive.shape[0])
            degrees_naive = np.array(A_c_naive_loops.sum(axis=1)).flatten()
            degrees_naive[degrees_naive == 0] = 1
            D_inv_sqrt_naive = sp.diags(1.0 / np.sqrt(degrees_naive))
            S_c_naive = D_inv_sqrt_naive @ A_c_naive_loops @ D_inv_sqrt_naive
        elif gnn_type.lower() in ['graphsage', 'sage_mean', 'graphsage_mean']:
            A_c_naive_loops = A_c_naive + sp.identity(A_c_naive.shape[0])
            degrees_naive = np.array(A_c_naive_loops.sum(axis=1)).flatten()
            degrees_naive[degrees_naive == 0] = 1
            D_inv_naive = sp.diags(1.0 / degrees_naive)
            S_c_naive = D_inv_naive @ A_c_naive_loops
        else:
            S_c_naive = A_c_naive
        
        naive_matrices.append(S_c_naive)
        
        # Measure how well each preserves message passing
        # Ideal: Q^+ S_c Q ≈ S_original
        reconstruction_mp = Q_plus @ S_c_MP @ Q
        reconstruction_naive = Q_plus @ S_c_naive @ Q
        
        # Compute errors (Frobenius norm)
        mp_error = sp.linalg.norm(S_original - reconstruction_mp, 'fro')
        naive_error = sp.linalg.norm(S_original - reconstruction_naive, 'fro')
        improvement = (naive_error - mp_error) / naive_error * 100 if naive_error > 0 else 0
        
        comparison_data['mp_errors'].append(mp_error)
        comparison_data['naive_errors'].append(naive_error)
        comparison_data['improvements'].append(improvement)
        
        print(f"[CMG-MP] Message passing reconstruction error:")
        print(f"[CMG-MP]   Naive approach: {naive_error:.6f}")
        print(f"[CMG-MP]   MP approach:    {mp_error:.6f}")
        print(f"[CMG-MP]   Improvement:    {improvement:.2f}%")
        
        # Update adjacency for next level
        current_adj = Q_plus.T @ current_adj @ Q_plus
    
    # Summary
    avg_improvement = np.mean(comparison_data['improvements'])
    print(f"\n[CMG-MP] ===== SUMMARY =====")
    print(f"[CMG-MP] Average MP preservation improvement: {avg_improvement:.2f}%")
    print(f"[CMG-MP] Enhanced CMG completed with {len(mp_matrices)} MP-aware matrices")
    
    return G, projections, laplacians, actual_levels, mp_matrices, comparison_data


def create_test_graph():
    """Create test graph for validation"""
    edges = [
        (0, 1), (1, 2), (1, 4), (2, 3), (3, 4), (4, 9),
        (4, 5), (5, 6), (5, 7), (6, 8), (8, 9),
        (9, 10), (9, 11), (10, 11)
    ]
    
    G = nx.Graph()
    G.add_edges_from(edges)
    for i in range(12):
        if i not in G.nodes():
            G.add_node(i)
    
    return G


def test_mp_enhancement():
    """Test the MP enhancement on small graph"""
    print("="*60)
    print("TESTING CMG MESSAGE-PASSING ENHANCEMENT")
    print("="*60)
    
    # Create test graph
    G = create_test_graph()
    L = nx.laplacian_matrix(G, nodelist=sorted(G.nodes()))
    
    print(f"Test graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Test both GNN types
    gnn_types = ['gcn', 'graphsage_mean']
    
    for gnn_type in gnn_types:
        print(f"\n{'='*40}")
        print(f"Testing with {gnn_type.upper()}")
        print(f"{'='*40}")
        
        try:
            # Run enhanced CMG
            G_coarse, projections, laplacians, levels, mp_matrices, comparison = cmg_coarse_with_mp(
                L, levels=1, gnn_type=gnn_type, k=5, d=10, threshold=0.1
            )
            
            print(f"\n[RESULT] {gnn_type.upper()} MP Enhancement:")
            print(f"  - Levels achieved: {levels}")
            print(f"  - Final graph size: {G_coarse.number_of_nodes()} nodes")
            print(f"  - Average improvement: {np.mean(comparison['improvements']):.2f}%")
            
            # Check if MP matrices are asymmetric (expected)
            for i, mp_mat in enumerate(mp_matrices):
                if mp_mat.shape[0] == mp_mat.shape[1]:
                    asymmetry = np.abs((mp_mat - mp_mat.T).data).max() if mp_mat.nnz > 0 else 0
                    print(f"  - Level {i+1} asymmetry: {asymmetry:.6f}")
            
        except Exception as e:
            print(f"[ERROR] Failed to test {gnn_type}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Test completed!")


if __name__ == "__main__":
    test_mp_enhancement()
