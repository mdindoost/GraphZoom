#!/usr/bin/env python3
"""
Message-Passing Aware Coarsening Implementation
Implements the key insight: preserve S_c^MP = Q S Q^+ instead of just spectral properties
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import norm
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import time

def create_message_passing_matrix(adjacency, mp_type='gcn', self_loop_weight=1.0):
    """
    Create message passing matrix S for different GNN types
    
    Args:
        adjacency: Adjacency matrix A
        mp_type: Type of message passing ('gcn', 'sage', 'raw')
        self_loop_weight: Weight for self-loops
    
    Returns:
        S: Message passing matrix
    """
    A = adjacency.copy()
    n = A.shape[0]
    
    if mp_type == 'gcn':
        # GCN: S = D^(-1/2) (A + I) D^(-1/2)
        A_tilde = A + self_loop_weight * identity(n)
        degrees = np.array(A_tilde.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1  # Avoid division by zero
        D_inv_sqrt = diags(1.0 / np.sqrt(degrees))
        S = D_inv_sqrt @ A_tilde @ D_inv_sqrt
        
    elif mp_type == 'sage':
        # GraphSAGE: S = D^(-1) A (mean aggregation)
        degrees = np.array(A.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1
        D_inv = diags(1.0 / degrees)
        S = D_inv @ A
        
    elif mp_type == 'raw':
        # Raw adjacency: S = A
        S = A.copy()
        
    else:
        raise ValueError(f"Unknown message passing type: {mp_type}")
    
    return S.tocsr()

def compute_message_passing_error(S_original, S_coarsened, Q, Q_plus):
    """
    Compute how well message passing is preserved
    Error = ||Q S Q^+ - S_coarsened||_F
    """
    S_reconstructed = Q @ S_original @ Q_plus
    error = norm(S_reconstructed - S_coarsened, 'fro')
    return error

def message_passing_aware_cmg(laplacian, mp_type='gcn', levels=1, **cmg_params):
    """
    CMG coarsening that optimizes for message passing preservation
    
    Args:
        laplacian: Original graph Laplacian
        mp_type: Message passing type to preserve
        levels: Number of coarsening levels
        **cmg_params: CMG parameters (k, d, threshold)
    
    Returns:
        results: Dictionary with coarsening results and message passing analysis
    """
    print(f"[MP-CMG] Starting message-passing aware coarsening")
    print(f"[MP-CMG] Target message passing: {mp_type}")
    
    # Extract adjacency from Laplacian
    degree_diag = diags(laplacian.diagonal(), 0)
    adjacency = degree_diag - laplacian
    
    # Create target message passing matrix
    S_original = create_message_passing_matrix(adjacency, mp_type)
    print(f"[MP-CMG] Original message passing matrix: {S_original.shape}")
    
    # Run standard CMG coarsening (for now - we'll improve this)
    from cmg_coarsening_timed import cmg_coarse
    
    current_laplacian = laplacian
    all_projections = []
    all_laplacians = [laplacian]
    all_mp_matrices = [S_original]
    all_mp_errors = []
    
    for level in range(levels):
        print(f"\n[MP-CMG] Level {level + 1}/{levels}")
        
        # Get CMG clustering for this level
        G_coarse, projections, laplacians, _ = cmg_coarse(
            current_laplacian, level=1, **cmg_params
        )
        
        projection = projections[0]  # Single level projection
        coarsened_laplacian = projection.T @ current_laplacian @ projection
        
        # Create coarsened message passing matrix
        coarse_degree_diag = diags(coarsened_laplacian.diagonal(), 0)
        coarse_adjacency = coarse_degree_diag - coarsened_laplacian
        S_coarsened = create_message_passing_matrix(coarse_adjacency, mp_type)
        
        # Compute pseudo-inverse for message passing analysis
        Q = projection.T  # Coarsening matrix (fine → coarse)
        Q_plus = projection  # Pseudo-inverse (coarse → fine)
        
        # Analyze message passing preservation
        current_adjacency = degree_diag - current_laplacian if level == 0 else (diags(current_laplacian.diagonal(), 0) - current_laplacian)
        S_current = create_message_passing_matrix(current_adjacency, mp_type)
        
        mp_error = compute_message_passing_error(S_current, S_coarsened, Q, Q_plus)
        
        print(f"[MP-CMG] Nodes: {current_laplacian.shape[0]} → {coarsened_laplacian.shape[0]}")
        print(f"[MP-CMG] Message passing error: {mp_error:.6f}")
        
        # Store results
        all_projections.append(projection)
        all_laplacians.append(coarsened_laplacian)
        all_mp_matrices.append(S_coarsened)
        all_mp_errors.append(mp_error)
        
        # Update for next level
        current_laplacian = coarsened_laplacian
    
    # Convert final laplacian to NetworkX
    final_degree_diag = diags(current_laplacian.diagonal(), 0)
    final_adjacency = final_degree_diag - current_laplacian
    G_final = nx.from_scipy_sparse_matrix(final_adjacency, edge_attribute='weight')
    
    return {
        'G_coarsened': G_final,
        'projections': all_projections,
        'laplacians': all_laplacians,
        'mp_matrices': all_mp_matrices,
        'mp_errors': all_mp_errors,
        'mp_type': mp_type,
        'levels': levels
    }

def test_gnn_compatibility(embeddings, S_matrix, num_layers=2):
    """
    Simulate GNN message passing on embeddings
    """
    H = embeddings.copy()
    
    for layer in range(num_layers):
        H = S_matrix @ H
        # Add small noise to simulate learnable weights
        H = H * (1 + 0.1 * np.random.randn(*H.shape))
    
    return H

def compare_refinement_strategies(coarsening_results, original_laplacian, embedding_dim=8):
    """
    Compare different refinement strategies for message passing preservation
    """
    print(f"\n{'='*60}")
    print("REFINEMENT STRATEGY COMPARISON")
    print(f"{'='*60}")
    
    # Generate synthetic embeddings for coarsened graph
    coarse_nodes = coarsening_results['G_coarsened'].number_of_nodes()
    coarse_embeddings = np.random.randn(coarse_nodes, embedding_dim)
    
    projections = coarsening_results['projections']
    mp_type = coarsening_results['mp_type']
    
    # Extract original adjacency and message passing matrix
    degree_diag = diags(original_laplacian.diagonal(), 0)
    original_adjacency = degree_diag - original_laplacian
    S_original = create_message_passing_matrix(original_adjacency, mp_type)
    
    results = {}
    
    # Strategy 1: No refinement (just projection)
    print("\nStrategy 1: No Refinement")
    refined_no_refine = coarse_embeddings
    for projection in reversed(projections):
        refined_no_refine = projection @ refined_no_refine
    
    # Test GNN compatibility
    gnn_output_no_refine = test_gnn_compatibility(refined_no_refine, S_original)
    results['no_refinement'] = {
        'embeddings': refined_no_refine,
        'gnn_output': gnn_output_no_refine
    }
    print(f"  Final embedding shape: {refined_no_refine.shape}")
    
    # Strategy 2: GraphZoom smooth refinement
    print("\nStrategy 2: GraphZoom Smooth Refinement")
    refined_smooth = coarse_embeddings
    for i, projection in enumerate(reversed(projections)):
        refined_smooth = projection @ refined_smooth
        
        # Apply smooth refinement (GCN-style message passing)
        current_level = len(projections) - 1 - i
        laplacian_for_smooth = coarsening_results['laplacians'][current_level]
        
        # Create smooth filter (same as GraphZoom)
        degree_diag_smooth = diags(laplacian_for_smooth.diagonal(), 0)
        adj_smooth = degree_diag_smooth - laplacian_for_smooth + 0.1 * identity(laplacian_for_smooth.shape[0])
        degrees_smooth = np.array(adj_smooth.sum(axis=1)).flatten()
        degrees_smooth[degrees_smooth == 0] = 1
        D_inv_sqrt_smooth = diags(1.0 / np.sqrt(degrees_smooth))
        smooth_filter = D_inv_sqrt_smooth @ adj_smooth @ D_inv_sqrt_smooth
        
        # Apply smoothing (2 iterations like GraphZoom)
        refined_smooth = smooth_filter @ (smooth_filter @ refined_smooth)
    
    gnn_output_smooth = test_gnn_compatibility(refined_smooth, S_original)
    results['smooth_refinement'] = {
        'embeddings': refined_smooth,
        'gnn_output': gnn_output_smooth
    }
    print(f"  Final embedding shape: {refined_smooth.shape}")
    
    # Strategy 3: Message-passing aware refinement
    print("\nStrategy 3: Message-Passing Aware Refinement")
    refined_mp_aware = coarse_embeddings
    for i, projection in enumerate(reversed(projections)):
        refined_mp_aware = projection @ refined_mp_aware
        
        # Get the appropriate message passing matrix for this level
        current_level = len(projections) - 1 - i
        if current_level < len(coarsening_results['mp_matrices']):
            S_level = coarsening_results['mp_matrices'][current_level]
        else:
            S_level = S_original
        
        # Apply message passing if dimensions match
        if S_level.shape[0] == refined_mp_aware.shape[0]:
            refined_mp_aware = S_level @ (S_level @ refined_mp_aware)
        else:
            print(f"  Dimension mismatch: S_level {S_level.shape} vs embeddings {refined_mp_aware.shape}")
            print(f"  Skipping MP for level {current_level}")
    
    gnn_output_mp_aware = test_gnn_compatibility(refined_mp_aware, S_original)
    results['mp_aware_refinement'] = {
        'embeddings': refined_mp_aware,
        'gnn_output': gnn_output_mp_aware
    }
    print(f"  Final embedding shape: {refined_mp_aware.shape}")
    
    # Compare strategies
    print(f"\n{'='*40}")
    print("STRATEGY COMPARISON")
    print(f"{'='*40}")
    
    # Compare embedding similarities
    baseline = results['no_refinement']['embeddings']
    
    for strategy, data in results.items():
        if strategy != 'no_refinement':
            similarity = cosine_similarity(baseline.flatten().reshape(1, -1), 
                                         data['embeddings'].flatten().reshape(1, -1))[0, 0]
            print(f"{strategy:25s}: similarity to baseline = {similarity:.3f}")
    
    # Compare GNN output stability
    print(f"\nGNN Output Analysis:")
    baseline_gnn = results['no_refinement']['gnn_output']
    
    for strategy, data in results.items():
        gnn_similarity = cosine_similarity(baseline_gnn.flatten().reshape(1, -1),
                                         data['gnn_output'].flatten().reshape(1, -1))[0, 0]
        gnn_norm = np.linalg.norm(data['gnn_output'])
        print(f"{strategy:25s}: GNN similarity = {gnn_similarity:.3f}, norm = {gnn_norm:.3f}")
    
    return results

def full_comparison_experiment(test_graph_laplacian, mp_types=['gcn', 'sage'], levels=2):
    """
    Complete comparison of traditional vs message-passing aware coarsening
    """
    print("FULL MESSAGE-PASSING COARSENING COMPARISON")
    print("="*80)
    
    all_results = {}
    
    for mp_type in mp_types:
        print(f"\n{'#'*60}")
        print(f"TESTING MESSAGE PASSING TYPE: {mp_type.upper()}")
        print(f"{'#'*60}")
        
        # Run message-passing aware CMG
        mp_results = message_passing_aware_cmg(
            test_graph_laplacian, 
            mp_type=mp_type, 
            levels=levels,
            k=10, d=20, threshold=0.1
        )
        
        # Compare refinement strategies
        refinement_comparison = compare_refinement_strategies(
            mp_results, test_graph_laplacian
        )
        
        all_results[mp_type] = {
            'coarsening': mp_results,
            'refinement': refinement_comparison
        }
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    for mp_type, results in all_results.items():
        mp_errors = results['coarsening']['mp_errors']
        print(f"\n{mp_type.upper()} Message Passing:")
        print(f"  Average MP error across levels: {np.mean(mp_errors):.6f}")
        print(f"  MP error per level: {[f'{e:.6f}' for e in mp_errors]}")
    
    return all_results

# Test with your original graph
def test_with_original_graph():
    """
    Test with the 12-node graph from your original experiment
    """
    print("Testing Message-Passing Aware Coarsening on 12-node Graph")
    print("="*60)
    
    # Create the test graph
    edges = [
        (0, 1), (1, 2), (1, 4), (2, 3), (3, 4), (4, 9),
        (4, 5), (5, 6), (5, 7), (6, 8), (8, 9),
        (9, 10), (9, 11), (10, 11), (6,9)
    ]
    
    G = nx.Graph()
    G.add_edges_from(edges)
    for i in range(12):
        if i not in G.nodes():
            G.add_node(i)
    
    # Convert to Laplacian
    L = nx.laplacian_matrix(G, nodelist=sorted(G.nodes()))
    
    # Run full comparison
    results = full_comparison_experiment(L, mp_types=['gcn', 'sage'], levels=1)
    
    return results

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run the test
    results = test_with_original_graph()
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED!")
    print("Check results for message passing preservation analysis.")
    print("="*80)