#!/usr/bin/env python3
"""
Enhanced Refinement for CMG: Message-Passing Aware Refinement
Overcomes CMG's limitations without changing CMG itself
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import norm as sparse_norm
import networkx as nx

def build_mp_preserving_matrix(clusters, n):
    """
    Build message-passing preserving matrix Q from CMG clusters
    Following NeurIPS 2024 paper: Q[k,i] = 1/sqrt(|C_k|) if i in C_k
    """
    num_clusters = len(clusters)
    Q = np.zeros((num_clusters, n))
    
    for k, cluster in enumerate(clusters):
        cluster_size = len(cluster)
        weight = 1.0 / np.sqrt(cluster_size)
        
        for node in cluster:
            Q[k, node] = weight
    
    return csr_matrix(Q)

def compute_message_passing_error(S, Q):
    """
    Compute how badly CMG clustering violates message-passing preservation
    Error = ||S - Q^T (Q S Q^T) Q||_F
    """
    QT = Q.T
    S_c = Q @ S @ QT  # Coarse propagation matrix
    S_approx = QT @ S_c @ Q  # Reconstructed propagation
    
    error_matrix = S - S_approx
    
    # Use scipy.sparse.linalg.norm for sparse matrices
    try:
        frobenius_error = sparse_norm(error_matrix, 'fro')
        s_norm = sparse_norm(S, 'fro')
    except:
        # Fallback to dense computation for small matrices
        error_dense = error_matrix.toarray() if sp.issparse(error_matrix) else error_matrix
        s_dense = S.toarray() if sp.issparse(S) else S
        frobenius_error = np.linalg.norm(error_dense, 'fro')
        s_norm = np.linalg.norm(s_dense, 'fro')
    
    relative_error = frobenius_error / s_norm if s_norm > 0 else 0
    
    return relative_error, S_approx, S_c

def enhanced_refinement_mp_aware(levels, projections, laplacians, embeddings, 
                                lda=0.1, power=False, mp_correction=True):
    """
    Enhanced refinement that corrects for message-passing violations
    
    Args:
        mp_correction: If True, apply message-passing correction
    """
    print(f"[ENHANCED REFINEMENT] Starting with mp_correction={mp_correction}")
    
    for i in reversed(range(levels)):
        print(f"[ENHANCED REFINEMENT] Level {i}")
        
        # Step 1: Standard GraphZoom projection
        embeddings = projections[i] @ embeddings
        print(f"  After projection: embeddings shape {embeddings.shape}")
        
        if mp_correction:
            # Step 2: Message-passing correction
            laplacian = laplacians[i]
            
            # Get clustering from projection matrix
            clusters = extract_clusters_from_projection(projections[i])
            
            # Build message-passing preserving matrix
            Q = build_mp_preserving_matrix(clusters, laplacian.shape[0])
            
            # Compute message-passing error
            # Convert Laplacian to propagation matrix (normalized)
            S = laplacian_to_propagation(laplacian)
            mp_error, S_approx, S_c = compute_message_passing_error(S, Q)
            
            print(f"  Message-passing error: {mp_error:.6f}")
            
            if mp_error > 0.1:  # Only correct if error is significant
                print(f"  Applying message-passing correction...")
                
                # Correction: Use S_approx instead of original S for smoothing
                correction_filter = build_correction_filter(S_approx, lda)
                embeddings = correction_filter @ embeddings
                print(f"  After MP correction: embeddings shape {embeddings.shape}")
        
        # Step 3: Standard spectral smoothing (possibly with corrected filter)
        if not mp_correction:  # Use original GraphZoom smoothing
            filter_ = smooth_filter(laplacians[i], lda)
        else:  # Use message-passing aware smoothing
            S = laplacian_to_propagation(laplacians[i])
            filter_ = build_mp_aware_filter(S, lda)
        
        if power or i == 0:
            embeddings = filter_ @ (filter_ @ embeddings)
            print(f"  After smoothing: embeddings shape {embeddings.shape}")
    
    return embeddings

def extract_clusters_from_projection(projection):
    """Extract cluster assignments from GraphZoom projection matrix"""
    clusters = []
    n_nodes, n_clusters = projection.shape
    
    for cluster_id in range(n_clusters):
        cluster = []
        for node_id in range(n_nodes):
            if projection[node_id, cluster_id] > 0:
                cluster.append(node_id)
        if cluster:  # Only add non-empty clusters
            clusters.append(cluster)
    
    return clusters

def laplacian_to_propagation(laplacian, method='normalized'):
    """Convert Laplacian to propagation matrix"""
    # Get adjacency: A = D - L
    degree_diag = diags(laplacian.diagonal(), 0)
    adjacency = degree_diag - laplacian
    
    if method == 'normalized':
        # D^{-1/2} A D^{-1/2}
        degrees = np.array(adjacency.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1
        D_inv_sqrt = diags(1.0 / np.sqrt(degrees))
        S = D_inv_sqrt @ adjacency @ D_inv_sqrt
    else:
        # Just use adjacency
        S = adjacency
    
    return S.tocsr()

def build_correction_filter(S_approx, lda):
    """Build smoothing filter from message-passing corrected propagation"""
    n = S_approx.shape[0]
    I = sp.identity(n)
    # Simple smoothing: (1-lda) I + lda S_approx
    filter_matrix = (1 - lda) * I + lda * S_approx
    return filter_matrix.tocsr()

def build_mp_aware_filter(S, lda):
    """Build message-passing aware smoothing filter"""
    n = S.shape[0]
    I = sp.identity(n)
    # Enhanced smoothing that respects message-passing structure
    filter_matrix = (1 - lda) * I + lda * S
    return filter_matrix.tocsr()

def original_refinement(levels, projections, coarse_laplacian, embeddings, lda, power):
    """
    Original GraphZoom refinement function (copied from graphzoom_timed.py)
    """
    for i in reversed(range(levels)):
        embeddings = projections[i] @ embeddings
        filter_    = smooth_filter(coarse_laplacian[i], lda)

        ## power controls whether smoothing intermediate embeddings,
        ## preventing over-smoothing
        if power or i == 0:
            embeddings = filter_ @ (filter_ @ embeddings)
    return embeddings

def smooth_filter(laplacian_matrix, lda):
    """
    Original GraphZoom smooth filter (copied from utils.py)
    """
    try:
        from utils import smooth_filter as original_smooth_filter
        return original_smooth_filter(laplacian_matrix, lda)
    except ImportError:
        # Fallback implementation if utils not available
        print("  [WARNING] Using fallback smooth_filter implementation")
        dim = laplacian_matrix.shape[0]
        adj_matrix = diags(laplacian_matrix.diagonal(), 0) - laplacian_matrix + lda * sp.identity(dim)
        degree_vec = adj_matrix.sum(axis=1)
        
        with np.errstate(divide='ignore'):
            d_inv_sqrt = np.squeeze(np.asarray(np.power(degree_vec, -0.5)))
        d_inv_sqrt[np.isinf(d_inv_sqrt)|np.isnan(d_inv_sqrt)] = 0
        degree_matrix = diags(d_inv_sqrt, 0)
        norm_adj = degree_matrix @ (adj_matrix @ degree_matrix)
        return norm_adj

def compare_refinement_methods(G, embeddings_coarse, projections, laplacians):
    """Compare original vs enhanced refinement"""
    print(f"\n" + "="*60)
    print("COMPARING REFINEMENT METHODS")
    print("="*60)
    
    # Method 1: Original GraphZoom refinement
    print(f"\n--- ORIGINAL GRAPHZOOM REFINEMENT ---")
    embeddings_original = original_refinement(
        len(projections), projections, laplacians, 
        embeddings_coarse.copy(), lda=0.1, power=False
    )
    
    # Method 2: Enhanced refinement without MP correction
    print(f"\n--- ENHANCED REFINEMENT (NO MP CORRECTION) ---")
    embeddings_enhanced_no_mp = enhanced_refinement_mp_aware(
        len(projections), projections, laplacians, 
        embeddings_coarse.copy(), lda=0.1, power=False, mp_correction=False
    )
    
    # Method 3: Enhanced refinement with MP correction
    print(f"\n--- ENHANCED REFINEMENT (WITH MP CORRECTION) ---")
    embeddings_enhanced_mp = enhanced_refinement_mp_aware(
        len(projections), projections, laplacians, 
        embeddings_coarse.copy(), lda=0.1, power=False, mp_correction=True
    )
    
    # Compare results
    print(f"\n" + "="*40)
    print("REFINEMENT COMPARISON")
    print("="*40)
    
    methods = [
        ("Original", embeddings_original),
        ("Enhanced (No MP)", embeddings_enhanced_no_mp), 
        ("Enhanced (MP)", embeddings_enhanced_mp)
    ]
    
    for name, embeddings in methods:
        # Basic statistics
        mean_val = np.mean(embeddings)
        std_val = np.std(embeddings)
        norm_val = np.linalg.norm(embeddings)
        
        print(f"{name:20s}: mean={mean_val:.4f}, std={std_val:.4f}, norm={norm_val:.4f}")
    
    # Compute differences
    diff_original_enhanced = np.linalg.norm(embeddings_original - embeddings_enhanced_no_mp)
    diff_enhanced_mp = np.linalg.norm(embeddings_enhanced_no_mp - embeddings_enhanced_mp)
    
    print(f"\nDifferences:")
    print(f"  Original vs Enhanced (No MP): {diff_original_enhanced:.6f}")
    print(f"  Enhanced vs Enhanced (MP):    {diff_enhanced_mp:.6f}")
    
    # Test message-passing preservation
    print(f"\nMessage-Passing Analysis:")
    S = laplacian_to_propagation(laplacians[0])
    
    # Extract clusters from projection
    clusters = extract_clusters_from_projection(projections[0])
    Q = build_mp_preserving_matrix(clusters, laplacians[0].shape[0])
    
    mp_error, S_approx, S_c = compute_message_passing_error(S, Q)
    print(f"  Message-passing error in coarsening: {mp_error:.6f}")
    
    if mp_error > 0.5:
        print(f"  ⚠️  High message-passing error detected!")
        print(f"  📊 Enhanced refinement should help correct this")
    else:
        print(f"  ✅ Low message-passing error - coarsening is good")
    
    return {
        'original': embeddings_original,
        'enhanced_no_mp': embeddings_enhanced_no_mp,
        'enhanced_mp': embeddings_enhanced_mp,
        'mp_error': mp_error
    }

def test_enhanced_refinement():
    """Test the enhanced refinement on path graph"""
    print("TESTING ENHANCED REFINEMENT FOR CMG")
    print("="*50)
    
    # Create test case
    G = nx.path_graph(12)
    
    # Simulate CMG coarsening result (unbalanced clusters)
    cmg_clusters = [[0], [1, 2, 3, 4], [5, 6, 7], [8, 9, 10, 11]]
    print(f"Simulated CMG clusters: {cmg_clusters}")
    
    # Build projection matrix
    n = 12
    projection = np.zeros((n, len(cmg_clusters)))
    for node_id in range(n):
        for cluster_id, cluster in enumerate(cmg_clusters):
            if node_id in cluster:
                projection[node_id, cluster_id] = 1.0
    
    projection = csr_matrix(projection)
    
    # Create test Laplacian
    L = nx.laplacian_matrix(G).astype(float)
    
    # Create dummy coarse embeddings
    embeddings_coarse = np.random.randn(len(cmg_clusters), 64)  # 64-dim embeddings
    
    # Test refinement
    projections = [projection]
    laplacians = [L]
    
    results = compare_refinement_methods(G, embeddings_coarse, projections, laplacians)
    
    return results

if __name__ == "__main__":
    test_enhanced_refinement()