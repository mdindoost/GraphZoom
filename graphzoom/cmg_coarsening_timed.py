# Complete cmg_coarsening_timed.py - Combines both fusion mapping and coarsening
import time
import numpy as np
import scipy.sparse as sp
import networkx as nx
from scipy.sparse import csr_matrix, diags
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
import torch

# Import your CMG functions (adjust paths as needed)
from filtered_timed import cmg_filtered_clustering, save_timing_data


def scipy_to_pyg_data(laplacian_matrix):
    """Convert scipy sparse Laplacian to PyTorch Geometric Data object"""
    # Convert Laplacian to adjacency matrix
    degree_diag = diags(laplacian_matrix.diagonal(), 0)
    adjacency = degree_diag - laplacian_matrix
    
    # Make sure adjacency is symmetric and non-negative
    adjacency = (adjacency + adjacency.T) / 2
    adjacency.data = np.abs(adjacency.data)
    
    # Convert to PyG format
    edge_index, edge_weight = from_scipy_sparse_matrix(adjacency)
    
    # Create PyG Data object
    num_nodes = laplacian_matrix.shape[0]
    data = Data(edge_index=edge_index, edge_attr=edge_weight, num_nodes=num_nodes)
    
    return data


def laplacian_to_pyg_data(laplacian):
    """
    Convert scipy sparse Laplacian to PyTorch Geometric Data object.
    Alternative implementation for fusion mapping.
    """
    # Convert Laplacian to adjacency: A = D - L
    if hasattr(laplacian, 'toarray'):
        laplacian_dense = laplacian.toarray()
    else:
        laplacian_dense = laplacian
    
    # Extract adjacency matrix
    adjacency_dense = -laplacian_dense.copy()
    np.fill_diagonal(adjacency_dense, 0)  # Remove diagonal
    adjacency_dense = np.maximum(adjacency_dense, 0)  # Keep only positive entries
    
    # Convert to sparse
    adjacency = sp.csr_matrix(adjacency_dense)
    
    # Convert to COO format for PyG
    adjacency_coo = adjacency.tocoo()
    
    # Create edge_index
    edge_index = np.vstack([adjacency_coo.row, adjacency_coo.col])
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    
    # Create PyG Data object
    data = Data(
        edge_index=edge_index,
        num_nodes=laplacian.shape[0]
    )
    
    return data


def clusters_to_mapping_matrix(cluster_assignments, num_clusters, num_nodes):
    """
    Convert cluster assignments to mapping matrix for fusion.
    """
    # Create mapping matrix: rows = clusters, cols = nodes
    row_indices = []
    col_indices = []
    data = []
    
    for node_id, cluster_id in enumerate(cluster_assignments):
        row_indices.append(cluster_id)
        col_indices.append(node_id)
        data.append(1.0)
    
    mapping = sp.csr_matrix(
        (data, (row_indices, col_indices)), 
        shape=(num_clusters, num_nodes)
    )
    
    return mapping


def create_identity_mapping(num_nodes):
    """Create identity mapping as fallback when CMG fails."""
    print(f"[CMG FUSION] Creating identity mapping for {num_nodes} nodes")
    mapping = sp.identity(num_nodes, format='csr')
    return mapping


def create_projection_matrix(cluster_assignments, num_nodes, num_clusters):
    """Create projection matrix for graph coarsening."""
    # Create projection matrix: rows = nodes, cols = clusters
    row_indices = []
    col_indices = []
    data = []
    
    for node_id, cluster_id in enumerate(cluster_assignments):
        row_indices.append(node_id)
        col_indices.append(cluster_id)
        data.append(1.0)
    
    projection = sp.csr_matrix(
        (data, (row_indices, col_indices)), 
        shape=(num_nodes, num_clusters)
    )
    
    return projection


# ========================= FUSION MAPPING FUNCTION =========================
def cmg_fusion_mapping(laplacian, k=10, d=20, threshold=0.1):
    """
    CMG fusion mapping function for graph fusion step.
    
    Purpose: Run CMG clustering to get smart node groupings for feature edge creation.
    This is NOT for graph reduction - just for determining which nodes should be 
    considered together when creating feature-based edges.
    
    Args:
        laplacian: Original graph Laplacian matrix (scipy sparse)
        k: CMG filter order
        d: CMG embedding dimension  
        threshold: CMG cosine similarity threshold
    
    Returns:
        mapping: Mapping matrix (num_clusters, num_nodes) where 
                mapping[cluster_id, node_id] = 1 if node belongs to cluster
    """
    
    print(f"[CMG FUSION] Running CMG clustering for fusion mapping")
    print(f"[CMG FUSION] Input: {laplacian.shape[0]} nodes")
    print(f"[CMG FUSION] Parameters: k={k}, d={d}, threshold={threshold}")
    
    # Convert laplacian to PyG data format
    data = laplacian_to_pyg_data(laplacian)
    
    # Run CMG clustering
    try:
        cluster_assignments, num_clusters, phi_stats, lambda_crit = cmg_filtered_clustering(
            data, k=k, d=d, threshold=threshold
        )
        
        print(f"[CMG FUSION] CMG clustering completed")
        print(f"[CMG FUSION] Found {num_clusters} clusters")
        print(f"[CMG FUSION] Average conductance: {phi_stats.get('avg_phi', 'N/A')}")
        
        # Convert cluster assignments to mapping matrix
        mapping = clusters_to_mapping_matrix(cluster_assignments, num_clusters, laplacian.shape[0])
        
        print(f"[CMG FUSION] Created mapping matrix: {mapping.shape}")
        print(f"[CMG FUSION] Reduction ratio: {laplacian.shape[0] / num_clusters:.2f}x")
        
        return mapping
        
    except Exception as e:
        print(f"[CMG FUSION] Error in CMG clustering: {e}")
        print(f"[CMG FUSION] Falling back to identity mapping")
        return create_identity_mapping(laplacian.shape[0])


# ========================= COARSENING FUNCTION =========================
def cmg_coarse(laplacian, level=1, k=10, d=20, threshold=0.1):
    """
    CMG coarsening function that matches GraphZoom's sim_coarse interface.
    
    Args:
        laplacian: scipy sparse Laplacian matrix (original or fused)
        level: number of coarsening levels
        k: CMG filter order
        d: CMG embedding dimension  
        threshold: CMG cosine similarity threshold
        
    Returns:
        G: NetworkX graph of coarsened graph
        projections: list of projection matrices
        laplacians: list of Laplacian matrices at each level
        level: number of levels
        all_cluster_assignments: list of cluster assignments for each level (NEW)
    """
    print(f"[CMG] Starting CMG coarsening with k={k}, d={d}, threshold={threshold}")
    print(f"[CMG] Input graph: {laplacian.shape[0]} nodes, {int((laplacian.nnz - laplacian.shape[0])/2)} edges")
    total_start_time = time.time()
    
    projections = []
    laplacians = []
    all_cluster_assignments = []  # NEW: Store cluster assignments for each level
    current_laplacian = laplacian.copy()
    
    for i in range(level):
        print(f"[CMG] Coarsening Level: {i+1}")
        print(f"[CMG] Current nodes: {current_laplacian.shape[0]}, edges: {int((current_laplacian.nnz - current_laplacian.shape[0])/2)}")
        
        # Store current Laplacian
        laplacians.append(current_laplacian.copy())
        
        # Convert to PyTorch Geometric format for CMG
        data = scipy_to_pyg_data(current_laplacian)
        
        # Run CMG clustering
        try:
            clusters, nc, phi_stats, lambda_crit = cmg_filtered_clustering(
                data, k=k, d=d, threshold=threshold
            )
            print(f"[CMG] Found {nc} clusters, λ_critical ≈ {lambda_crit:.4f}")
            
            # NEW: Store cluster assignments
            all_cluster_assignments.append(clusters)
            
        except Exception as e:
            print(f"[CMG] Error in CMG clustering: {e}")
            # Fallback to simple clustering if CMG fails
            print("[CMG] Falling back to simple spectral clustering")
            from utils import smooth_filter, spec_coarsen
            filter_ = smooth_filter(current_laplacian, 0.1)
            current_laplacian, mapping = spec_coarsen(filter_, current_laplacian)
            projections.append(mapping)
            
            # NEW: Create fallback cluster assignments
            n_nodes = current_laplacian.shape[0]
            fallback_assignments = list(range(n_nodes))  # Each node in its own cluster
            all_cluster_assignments.append(fallback_assignments)
            continue
        
        # Build projection matrix from CMG clusters
        num_nodes = current_laplacian.shape[0]
        mapping = create_projection_matrix(clusters, num_nodes, nc)
        projections.append(mapping)
        
        # Create coarsened Laplacian
        current_laplacian = mapping.T @ current_laplacian @ mapping
        
        print(f"[CMG] Coarsened to {nc} nodes, {int((current_laplacian.nnz - current_laplacian.shape[0])/2)} edges")
    
    # Convert final Laplacian to NetworkX graph
    degree_diag = diags(current_laplacian.diagonal(), 0)
    adjacency = degree_diag - current_laplacian
    
    # Ensure non-negative weights
    adjacency.data = np.abs(adjacency.data)
    
    # Create NetworkX graph
    G = nx.from_scipy_sparse_matrix(adjacency, edge_attribute='wgt')
    
    print(f"[CMG] Final graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
    
    total_cmg_time = time.time() - total_start_time
    print(f"[CMG] Total coarsening time: {total_cmg_time:.3f}s")
    
    # Save detailed timing data
    import os
    timing_file = f"results/timing_results/cmg_detailed_{level}level.json"
    os.makedirs("results/timing_results", exist_ok=True)
    save_timing_data(timing_file)
    
    # NEW: Return cluster assignments as well
    return G, projections, laplacians, level, all_cluster_assignments


# ========================= LEGACY FUNCTION =========================
def cmg_coarse_fusion(laplacian, k=10, d=20, threshold=0.1):
    """
    Legacy function - kept for backward compatibility.
    Now just calls standard GraphZoom fusion.
    """
    print("[CMG] Using GraphZoom's standard fusion (not custom CMG fusion)")
    
    # Import GraphZoom's standard fusion
    from utils import sim_coarse_fusion
    return sim_coarse_fusion(laplacian)


# ========================= TEST FUNCTION =========================
def test_cmg_fusion_mapping():
    """Test function to verify cmg_fusion_mapping works correctly."""
    print("Testing CMG fusion mapping...")
    
    # Create a simple test Laplacian (4-node cycle)
    n = 4
    edges = [(0,1), (1,2), (2,3), (3,0)]
    
    # Build adjacency
    adj = sp.lil_matrix((n, n))
    for i, j in edges:
        adj[i, j] = 1
        adj[j, i] = 1
    
    # Build Laplacian
    degrees = np.array(adj.sum(axis=1)).flatten()
    laplacian = sp.diags(degrees) - adj.tocsr()
    
    print(f"Test Laplacian shape: {laplacian.shape}")
    
    # Test the mapping function
    try:
        mapping = cmg_fusion_mapping(laplacian, k=5, d=10, threshold=0.1)
        print(f"✅ Success! Mapping shape: {mapping.shape}")
        print(f"Mapping matrix:\n{mapping.toarray()}")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    test_cmg_fusion_mapping()