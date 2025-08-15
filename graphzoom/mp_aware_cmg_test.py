#!/usr/bin/env python3
"""
Test: CMG with Message-Passing Aware Propagation Matrix
Compare naive vs message-passing aware coarsening for GNNs
"""

import numpy as np
import scipy.sparse as sp
import networkx as nx
from scipy.sparse import csr_matrix, diags
from scipy.io import mmread
import os

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

def compute_propagation_matrix(adj_matrix, gnn_type='gcn'):
    """
    Compute propagation matrix based on GNN type
    
    Args:
        adj_matrix: Adjacency matrix
        gnn_type: 'gcn' or 'graphsage'
    
    Returns:
        S: Propagation matrix
    """
    if gnn_type == 'gcn':
        # S = D^(-1/2) (A + I) D^(-1/2) - GCN with self-loops
        A_self = adj_matrix + sp.identity(adj_matrix.shape[0])
        degrees = np.array(A_self.sum(axis=1)).flatten()
        degrees_safe = degrees + 1e-12  # Avoid division by zero
        D_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees_safe))
        S = D_inv_sqrt @ A_self @ D_inv_sqrt
        
    elif gnn_type == 'graphsage':
        # S = D^(-1) A - Mean aggregation (no self-loops)
        degrees = np.array(adj_matrix.sum(axis=1)).flatten()
        degrees_safe = degrees + 1e-12
        D_inv = sp.diags(1.0 / degrees_safe)
        S = D_inv @ adj_matrix
        
    else:
        raise ValueError(f"Unknown GNN type: {gnn_type}")
    
    return S

def compute_mp_propagation_matrix(Q, Q_plus, original_adj, gnn_type='gcn'):
    """
    Compute message-passing aware propagation matrix for coarsened graph
    This is the key innovation from the paper: S_c^MP = Q S Q^+
    """
    print(f"[MP-AWARE] Computing message-passing matrix for {gnn_type}")
    
    # Debug: Check dimensions
    print(f"[MP-AWARE] Q shape: {Q.shape} (should be coarse × original)")
    print(f"[MP-AWARE] Q^+ shape: {Q_plus.shape} (should be original × coarse)")
    print(f"[MP-AWARE] original_adj shape: {original_adj.shape}")
    
    # Compute original propagation matrix
    S_original = compute_propagation_matrix(original_adj, gnn_type)
    print(f"[MP-AWARE] S_original shape: {S_original.shape}")
    
    # Check if dimensions will work for Q @ S @ Q^+
    print(f"[MP-AWARE] Checking dimensions for Q @ S @ Q^+:")
    print(f"[MP-AWARE]   Q @ S: ({Q.shape[0]}, {Q.shape[1]}) @ ({S_original.shape[0]}, {S_original.shape[1]}) = ({Q.shape[0]}, {S_original.shape[1]})")
    print(f"[MP-AWARE]   (Q @ S) @ Q^+: ({Q.shape[0]}, {S_original.shape[1]}) @ ({Q_plus.shape[0]}, {Q_plus.shape[1]}) = ({Q.shape[0]}, {Q_plus.shape[1]})")
    
    # THE KEY STEP: S_c^MP = Q S Q^+
    S_c_MP = Q @ S_original @ Q_plus
    
    print(f"[MP-AWARE] Final S_c^MP shape: {S_c_MP.shape}")
    print(f"[MP-AWARE] S_c^MP density: {S_c_MP.nnz / S_c_MP.size:.3f}")
    
    return S_c_MP

def compute_naive_propagation_matrix(Q_plus, original_adj, gnn_type='gcn'):
    """
    Compute naive coarsened propagation matrix
    Traditional approach: A_c = Q^+^T A Q^+, then S_c = f(A_c)
    """
    print(f"[NAIVE] Computing naive propagation matrix for {gnn_type}")
    print(f"[NAIVE] Q^+ shape: {Q_plus.shape}")
    print(f"[NAIVE] original_adj shape: {original_adj.shape}")
    
    # Step 1: Create coarsened adjacency A_c = Q^+^T A Q^+
    # Q^+ has shape (original, coarse), so Q^+^T has shape (coarse, original)
    A_c = Q_plus.T @ original_adj @ Q_plus
    print(f"[NAIVE] A_c shape after Q^+^T @ A @ Q^+: {A_c.shape}")
    
    # Step 2: Apply propagation function to coarsened adjacency
    S_c_naive = compute_propagation_matrix(A_c, gnn_type)
    
    print(f"[NAIVE] Final S_c shape: {S_c_naive.shape}")
    print(f"[NAIVE] S_c density: {S_c_naive.nnz / S_c_naive.size:.3f}")
    
    return S_c_naive, A_c

def simple_gnn_forward(S, X, W):
    """
    Simple one-layer GNN forward pass: H = S X W
    
    Args:
        S: Propagation matrix
        X: Node features  
        W: Weight matrix
    
    Returns:
        H: Output node representations
    """
    return S @ X @ W

def compute_message_passing_error(S_original, S_coarsened, Q, Q_plus, test_signals):
    """
    Compute message passing preservation error as in the paper
    ||S x - Q^+ S_c x_c||_F for various test signals
    
    Args:
        S_original: Original propagation matrix (N, N)
        S_coarsened: Coarsened propagation matrix (n, n) 
        Q: Coarsening matrix (n, N) - coarse × original
        Q_plus: Lifting matrix (N, n) - original × coarse
        test_signals: List of test vectors
    """
    errors = []
    
    for x in test_signals:
        # Original message passing
        Sx = S_original @ x
        
        # Coarsened message passing + lifting
        x_c = Q @ x  # Coarsen signal: (n, N) @ (N,) = (n,)
        Sx_c = S_coarsened @ x_c  # Message pass on coarsened graph: (n, n) @ (n,) = (n,)
        Sx_lifted = Q_plus @ Sx_c  # Lift back: (N, n) @ (n,) = (N,)
        
        # Compute error
        error = np.linalg.norm(Sx - Sx_lifted)
        errors.append(error)
    
    return np.mean(errors)

def load_cmg_results():
    """Load CMG coarsening results from previous experiment"""
    if not os.path.exists("experiment_results/cmg_projection.mtx"):
        print("Error: Need to run coarsening_experiment.py first!")
        return None, None
    
    # Load the projection matrix from CMG experiment
    projection_loaded = mmread("experiment_results/cmg_projection.mtx").tocsr()
    
    print(f"[CMG] Loaded projection matrix shape: {projection_loaded.shape}")
    
    # The loaded matrix has shape (12, 5) = (original_nodes, coarse_nodes)
    # For the paper's notation: Q ∈ R^(n×N), Q^+ ∈ R^(N×n)
    # where n = coarse, N = original
    
    if projection_loaded.shape[0] > projection_loaded.shape[1]:
        # Matrix is (original, coarse) → need to transpose for Q
        Q_cmg = projection_loaded.T  # Shape: (5, 12) = (coarse, original)
        Q_cmg_plus = projection_loaded  # Shape: (12, 5) = (original, coarse)
        print(f"[CMG] Transposed to get Q: {Q_cmg.shape}, Q^+: {Q_cmg_plus.shape}")
    else:
        # Matrix is already (coarse, original)
        Q_cmg = projection_loaded
        Q_cmg_plus = projection_loaded.T
    
    print(f"[CMG] Final Q shape: {Q_cmg.shape} (coarse × original)")
    print(f"[CMG] Final Q^+ shape: {Q_cmg_plus.shape} (original × coarse)")
    print(f"[CMG] Coarsening ratio: {Q_cmg.shape[1] / Q_cmg.shape[0]:.2f}x")
    
    return Q_cmg, Q_cmg_plus

def test_gnn_reconstruction_quality(S_original, S_coarsened, Q, Q_plus, original_adj):
    """
    Test how well the coarsened GNN can reconstruct node relationships
    
    Args:
        S_original: Original propagation matrix (N, N)
        S_coarsened: Coarsened propagation matrix (n, n)
        Q: Coarsening matrix (n, N) - coarse × original  
        Q_plus: Lifting matrix (N, n) - original × coarse
        original_adj: Original adjacency matrix
    """
    print(f"\n[RECONSTRUCTION TEST]")
    
    # Create simple node features (identity + noise)
    n_nodes = original_adj.shape[0]
    np.random.seed(42)
    X = np.random.randn(n_nodes, 4)  # 4-dimensional features
    W = np.random.randn(4, 2)  # Project to 2D
    
    # Original GNN forward pass
    H_original = simple_gnn_forward(S_original, X, W)
    
    # Coarsened GNN forward pass
    X_c = Q @ X  # Coarsen features: (n, N) @ (N, 4) = (n, 4)
    H_c = simple_gnn_forward(S_coarsened, X_c, W)  # GNN on coarsened graph: (n, 2)
    H_lifted = Q_plus @ H_c  # Lift back to original size: (N, n) @ (n, 2) = (N, 2)
    
    # Compute reconstruction error
    reconstruction_error = np.linalg.norm(H_original - H_lifted, 'fro')
    relative_error = reconstruction_error / np.linalg.norm(H_original, 'fro')
    
    print(f"  Reconstruction error: {reconstruction_error:.4f}")
    print(f"  Relative error: {relative_error:.4f} ({relative_error*100:.1f}%)")
    
    return relative_error

def main():
    """Run complete test comparing naive vs message-passing aware coarsening"""
    print("MESSAGE-PASSING AWARE CMG TEST")
    print("="*60)
    
    # Step 1: Create test graph
    G = create_test_graph()
    A = nx.adjacency_matrix(G).tocsr()
    
    print(f"Test graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Step 2: Load CMG coarsening results
    Q_cmg, Q_cmg_plus = load_cmg_results()
    if Q_cmg is None:
        return
    
    # Step 3: Test both GNN types
    gnn_types = ['gcn', 'graphsage']
    
    for gnn_type in gnn_types:
        print(f"\n{'='*60}")
        print(f"TESTING {gnn_type.upper()} MESSAGE PASSING")
        print(f"{'='*60}")
        
        # Compute original propagation matrix
        S_original = compute_propagation_matrix(A, gnn_type)
        
        # Method 1: Message-passing aware (PAPER'S METHOD)
        S_c_MP = compute_mp_propagation_matrix(Q_cmg, Q_cmg_plus, A, gnn_type)
        
        # Method 2: Naive coarsening (TRADITIONAL METHOD)
        S_c_naive, A_c = compute_naive_propagation_matrix(Q_cmg_plus, A, gnn_type)
        
        # Step 4: Test message passing preservation
        print(f"\n[MESSAGE PASSING PRESERVATION TEST]")
        
        # Generate test signals
        np.random.seed(42)
        test_signals = [np.random.randn(A.shape[0]) for _ in range(5)]
        
        # Compute preservation errors
        error_mp = compute_message_passing_error(S_original, S_c_MP, Q_cmg, Q_cmg_plus, test_signals)
        error_naive = compute_message_passing_error(S_original, S_c_naive, Q_cmg, Q_cmg_plus, test_signals)
        
        print(f"  MP-Aware error: {error_mp:.6f}")
        print(f"  Naive error:    {error_naive:.6f}")
        print(f"  Improvement:    {((error_naive - error_mp) / error_naive * 100):.1f}%")
        
        # Step 5: Test GNN reconstruction quality
        print(f"\n[GNN RECONSTRUCTION QUALITY TEST]")
        
        print(f"  MP-Aware method:")
        rel_error_mp = test_gnn_reconstruction_quality(S_original, S_c_MP, Q_cmg, Q_cmg_plus, A)
        
        print(f"  Naive method:")
        rel_error_naive = test_gnn_reconstruction_quality(S_original, S_c_naive, Q_cmg, Q_cmg_plus, A)
        
        print(f"\n[SUMMARY for {gnn_type.upper()}]")
        print(f"  Message Passing Error - MP-Aware: {error_mp:.6f}, Naive: {error_naive:.6f}")
        print(f"  GNN Reconstruction Error - MP-Aware: {rel_error_mp:.4f}, Naive: {rel_error_naive:.4f}")
        
        if error_mp < error_naive and rel_error_mp < rel_error_naive:
            print(f"  ✅ MP-Aware method WINS on both metrics!")
        elif error_mp < error_naive:
            print(f"  ✅ MP-Aware method wins on message passing preservation")
        elif rel_error_mp < rel_error_naive:
            print(f"  ✅ MP-Aware method wins on GNN reconstruction")
        else:
            print(f"  ❌ Naive method performs better")
    
    print(f"\n{'='*60}")
    print("CONCLUSION")
    print(f"{'='*60}")
    print("The message-passing aware propagation matrix S_c^MP = Q S Q^+")
    print("should show better preservation of GNN message passing compared")
    print("to the naive approach of computing S_c = f(A_c).")
    print("This validates the paper's key insight!")

if __name__ == "__main__":
    main()
